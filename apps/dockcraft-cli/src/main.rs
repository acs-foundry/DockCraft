#![forbid(unsafe_code)]

mod mcp;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use dock_engine::{DockingConfig, ScoringFunction};
use dock_pipeline::{
    DEFAULT_MAX_FILE_BYTES, PathSandbox, campaign_report_to_csv, load_ligand_library_bytes,
    run_screening_campaign,
};
use hyde_scorer::score_pose_hyde;
use pocket_detect::GridBox;
use struct_core::{
    parse_mmcif, parse_mol2, parse_pdb, parse_pdbqt, parse_sdf_3d_with_limit, prepare_receptor,
    write_pdbqt_receptor,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliScoring {
    Vina,
    Vinardo,
    Hybrid,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum McpTransport {
    Stdio,
    Tcp,
}

#[derive(Parser)]
#[command(
    name = "dockcraft-cli",
    version,
    about = "DockCraft headless virtual screening, receptor preparation, HYDE scoring, and hardened MCP server"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Prepare a protein receptor (strip waters, infer bonds, assign Gasteiger charges & AD4 types)
    PrepareReceptor {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = true)]
        strip_waters: bool,
    },
    /// Run a PyRx-style parallel virtual screening campaign
    Screen {
        #[arg(long)]
        receptor: PathBuf,
        #[arg(long)]
        library: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_delimiter = ',')]
        center: Vec<f64>,
        #[arg(long, value_delimiter = ',')]
        size: Vec<f64>,
        #[arg(long, value_enum, default_value_t = CliScoring::Vina)]
        scoring: CliScoring,
        #[arg(long, default_value_t = 6)]
        exhaustiveness: usize,
        #[arg(long, default_value_t = 1000)]
        max_ligands: usize,
    },
    /// Score a docked complex with Vina + HYDE per-atom desolvation/H-bond/steric breakdown
    ScorePose {
        #[arg(long)]
        receptor: PathBuf,
        #[arg(long)]
        ligand: PathBuf,
    },
    /// Start the hardened JSON-RPC 2.0 MCP server
    Mcp {
        #[arg(long)]
        read_root: Option<PathBuf>,
        #[arg(long)]
        write_root: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = McpTransport::Stdio)]
        transport: McpTransport,
        #[arg(long, default_value = "127.0.0.1:7879")]
        bind: String,
        #[arg(long)]
        auth_token: Option<String>,
    },
}

fn read_receptor_file(
    path: &std::path::Path,
    strip_waters: bool,
) -> Result<struct_core::Molecule3D, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("pdb")
        .to_ascii_lowercase();
    let mut rec = match ext.as_str() {
        "cif" | "mmcif" => parse_mmcif(&raw, strip_waters)?,
        "pdbqt" => parse_pdbqt(&raw)?,
        "mol2" => parse_mol2(&raw)?,
        _ => parse_pdb(&raw, strip_waters)?,
    };
    prepare_receptor(&mut rec, strip_waters);
    Ok(rec)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::PrepareReceptor {
            input,
            output,
            strip_waters,
        } => {
            let rec = read_receptor_file(&input, strip_waters)?;
            std::fs::write(&output, write_pdbqt_receptor(&rec))?;
            println!(
                "Prepared receptor '{}': {} atoms ({} heavy) -> {}",
                rec.name,
                rec.atoms.len(),
                rec.heavy_atom_count(),
                output.display()
            );
        }
        Commands::Screen {
            receptor,
            library,
            out,
            center,
            size,
            scoring,
            exhaustiveness,
            max_ligands,
        } => {
            if center.len() != 3 || size.len() != 3 {
                return Err("--center and --size must be comma-separated 3-vectors x,y,z".into());
            }
            let rec = read_receptor_file(&receptor, true)?;
            let gbox = GridBox::from_center_size(
                [center[0], center[1], center[2]],
                [size[0], size[1], size[2]],
                0.375,
            )?;
            let bytes = std::fs::read(&library)?;
            let ext = library
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("smi");
            let lib = load_ligand_library_bytes(ext, &bytes, max_ligands)?;
            let cfg = DockingConfig {
                scoring: match scoring {
                    CliScoring::Vina => ScoringFunction::Vina,
                    CliScoring::Vinardo => ScoringFunction::Vinardo,
                    CliScoring::Hybrid => ScoringFunction::DockCraftHybrid,
                },
                exhaustiveness,
                ..Default::default()
            };
            let report = run_screening_campaign(&rec, &gbox, &lib, &cfg)?;
            std::fs::write(&out, campaign_report_to_csv(&report))?;
            println!(
                "Screened {} compounds in {} ms -> {}",
                report.results.len(),
                report.elapsed_ms,
                out.display()
            );
        }
        Commands::ScorePose { receptor, ligand } => {
            let rec = read_receptor_file(&receptor, true)?;
            let lig_raw = std::fs::read_to_string(&ligand)?;
            let mut ligs = parse_sdf_3d_with_limit(&lig_raw, 1);
            let mut lig = ligs.pop().ok_or("empty ligand SDF")??;
            struct_core::perceive_all(&mut lig);
            let hyde = score_pose_hyde(&rec, &lig)?;
            println!("{}", serde_json::to_string_pretty(&hyde)?);
        }
        Commands::Mcp {
            read_root,
            write_root,
            transport,
            bind,
            auth_token,
        } => {
            let sandbox = PathSandbox::new(read_root, write_root, DEFAULT_MAX_FILE_BYTES)?;
            match transport {
                McpTransport::Stdio => mcp::serve_stdio(&sandbox)?,
                McpTransport::Tcp => {
                    let tok = auth_token
                        .filter(|s| !s.trim().is_empty())
                        .ok_or("--auth-token is mandatory when --transport tcp")?;
                    mcp::serve_tcp(sandbox, &bind, &tok)?;
                }
            }
        }
    }
    Ok(())
}

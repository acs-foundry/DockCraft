#![forbid(unsafe_code)]

mod mcp;

use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use dock_engine::{
    DockingConfig, DockingConstraint, PharmacophoreKind, ScoringFunction, dock_ligand,
    score_complex,
};
use dock_pipeline::{
    CampaignSpec, ChimeraXRest, DEFAULT_MAX_FILE_BYTES, MdPlatform, MdValidateConfig, PathSandbox,
    SafetyPanelConfig, SafetyPanelManifest, campaign_report_to_csv, check_openmm_env,
    default_bowes_tier1_catalog, default_cdk2_campaign_template, default_chemcraft_results_dir,
    default_chimerax_export_dir, default_safety_panel_dir, default_safety_panel_manifest_path,
    grid_box_from_chimerax_selection, is_chimerax_available, launch_chimerax_for_scene,
    load_ligand_library_bytes, run_campaign, run_md_validation, run_safety_panel,
    run_screening_campaign, write_chimerax_session_bundle, write_results_sidecar,
};
use flate2::read::GzDecoder;
use hyde_scorer::{
    analyze_torsions, consensus_affinity_kcal, score_pose_hyde, validate_pose_geometry,
};
use pharm_core::{
    OffTargetPocketInput, PharmScreenConfig, PocketSelectivityConfig, SiteAnalysisConfig,
    SiteBlueprint, analyze_binding_site, analyze_pocket_selectivity, craft_data_dir,
    screen_molecules_parallel, screen_smiles_stream,
};
use pocket_detect::{GridBox, PocketDetectConfig, detect_pockets};
use prot_prep::{
    EnsembleAggregation, EnsembleReceptor, PrepConfig, prepare_protein_file, run_ensemble_docking,
};
use struct_core::{
    detect_bridging_waters, parse_mmcif, parse_mol2, parse_pdb, parse_pdbqt,
    parse_sdf_3d_with_limit, prepare_receptor, prepare_receptor_keeping_waters, write_pdb,
    write_pdbqt_receptor, write_sdf_3d,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliScoring {
    Vina,
    Vinardo,
    Hybrid,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliAggregation {
    Boltzmann,
    Best,
    Mean,
    Rrf,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliMdPlatform {
    Auto,
    Cuda,
    Opencl,
    Cpu,
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
    /// Rigorous receptor preparation & structural audit: PDB/mmCIF/AlphaFold audit, altloc occupancy
    /// selection, PROPKA-inspired local pKa protonation (HID/HIE/HIP/ASH/GLH/LYN/CYM/CYX),
    /// 180° ASN/GLN/HIS H-bond flips, polar-H orientation, and ligand/cofactor separation
    ProtPrep {
        #[arg(long)]
        input: PathBuf,
        /// Write clean prepared receptor PDB
        #[arg(long)]
        out_pdb: Option<PathBuf>,
        /// Optional prepared receptor AutoDock PDBQT
        #[arg(long)]
        out_pdbqt: Option<PathBuf>,
        /// Optional JSON structural audit + pKa + H-bond optimization report
        #[arg(long)]
        out_report: Option<PathBuf>,
        /// Optional human-readable Markdown Structural Audit Report
        #[arg(long)]
        out_md: Option<PathBuf>,
        /// Optional output path (`.sdf` or `.pdb`) for extracted primary cocrystallized organic ligand
        #[arg(long)]
        extract_ligand: Option<PathBuf>,
        /// Comma-separated chain IDs to retain (e.g. `A,B`; omit to keep all chains)
        #[arg(long, value_delimiter = ',')]
        chains: Vec<char>,
        /// Target physiological pH for local pKa assignment (default 7.4)
        #[arg(long, default_value_t = 7.4)]
        ph: f64,
        /// Strip recognized biological cofactors (ATP/NAD/FAD/HEM/...) instead of keeping them
        #[arg(long)]
        no_cofactors: bool,
        /// Strip catalytic divalent metal ions (Zn/Mg/Fe/Mn/Ca) instead of keeping them
        #[arg(long)]
        no_metals: bool,
        /// Keep cocrystallized organic ligands inside the receptor (default strips & extracts them)
        #[arg(long)]
        keep_ligands: bool,
        /// Automatically retain buried structural bridging waters (>= 2 receptor H-bonds inside pocket)
        #[arg(long)]
        keep_bridging_waters: bool,
        /// Skip 180° ASN/GLN/HIS flip & polar-H torsion optimization
        #[arg(long)]
        no_flips: bool,
        /// Optional active-site center `x,y,z` for pocket-focused audit & bridging-water retention
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        center: Vec<f64>,
        /// Pocket neighborhood radius in Å (default 10.0)
        #[arg(long, default_value_t = 10.0)]
        radius: f64,
    },
    /// Multi-conformation Ensemble Docking across 2..N receptor states with Boltzmann / Best / Mean / RRF consensus
    EnsembleDock {
        /// Comma-separated or repeated receptor PDB/PDBQT files representing distinct target conformations
        #[arg(long, value_delimiter = ',', required = true)]
        receptors: Vec<PathBuf>,
        /// Single ligand file (`.sdf`/`.mol2`/`.pdb`/`.pdbqt`)
        #[arg(long)]
        ligand: Option<PathBuf>,
        /// Or screening library (`.sdf`/`.smi`/`.csv`/`.chemcraft`)
        #[arg(long)]
        library: Option<PathBuf>,
        /// Single shared center `x,y,z` OR per-receptor centers `x1,y1,z1;x2,y2,z2` (omit to auto-detect top pocket per receptor)
        #[arg(long, allow_hyphen_values = true)]
        center: Option<String>,
        #[arg(
            long,
            value_delimiter = ',',
            allow_hyphen_values = true,
            default_values_t = vec![20.0, 20.0, 20.0]
        )]
        size: Vec<f64>,
        #[arg(long, value_enum, default_value_t = CliScoring::Vina)]
        scoring: CliScoring,
        #[arg(long, value_enum, default_value_t = CliAggregation::Boltzmann)]
        aggregation: CliAggregation,
        #[arg(long, default_value_t = 6)]
        exhaustiveness: usize,
        /// Use raw docking score without HYDE consensus blending
        #[arg(long)]
        no_hyde_consensus: bool,
        #[arg(long, default_value_t = 250)]
        max_ligands: usize,
        /// Optional ranked ensemble CSV output
        #[arg(long)]
        out_csv: Option<PathBuf>,
        /// Optional best-conformation 3D poses SDF output
        #[arg(long)]
        out_sdf: Option<PathBuf>,
        /// Optional Markdown ensemble report output
        #[arg(long)]
        out_md: Option<PathBuf>,
    },
    /// Run a PyRx-style parallel virtual screening campaign
    Screen {
        #[arg(long)]
        receptor: PathBuf,
        #[arg(long)]
        library: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        center: Vec<f64>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        size: Vec<f64>,
        /// Take the box from what is selected right now in a running ChimeraX, instead of
        /// --center/--size. Select the binding-site residues on screen first.
        #[arg(long)]
        box_from_chimerax: bool,
        /// Padding in angstroms around the selected atoms when --box-from-chimerax is used
        #[arg(long, default_value_t = 5.0)]
        box_padding: f64,
        #[arg(long, value_enum, default_value_t = CliScoring::Vina)]
        scoring: CliScoring,
        #[arg(long, default_value_t = 6)]
        exhaustiveness: usize,
        #[arg(long, default_value_t = 1000)]
        max_ligands: usize,
        /// Optional `.pharm.json` blueprint from `analyze-site` used as soft pharmacophore guidance
        #[arg(long)]
        pharmacophore_file: Option<PathBuf>,
        /// Soft restraint weight multiplier when `--pharmacophore-file` is supplied
        #[arg(long, default_value_t = 0.45)]
        pharm_weight: f64,
        /// Also write a `*.dockcraft.json` results sidecar plus the docked poses, for ChemCraft
        /// to fold back into its SAR table. Default directory:
        /// $XDG_DATA_HOME/ChemCraft/dockcraft_results
        #[arg(long)]
        sidecar: bool,
        /// Directory for the sidecar, overriding the default
        #[arg(long)]
        sidecar_dir: Option<PathBuf>,
    },
    /// Profile an active site with 7-probe 3D MIFs, PCA dimensions, optimal inter-feature distance
    /// matrix, wall exclusion spheres, and water assessment (emits `.pharm.json` + Markdown blueprint)
    AnalyzeSite {
        #[arg(long)]
        receptor: PathBuf,
        /// Optional cocrystallized reference ligand(s) for hybrid Apo+Holo hotspot consensus
        #[arg(long)]
        ligand: Vec<PathBuf>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        center: Vec<f64>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        size: Vec<f64>,
        /// Automatically select the top druggable pocket when `--center`/`--size` are omitted
        #[arg(long)]
        auto_pocket: bool,
        /// Retain crystallographic waters in receptor for ConservedBridge vs Displaceable audit
        #[arg(long)]
        keep_waters: bool,
        /// MIF grid spacing in Å
        #[arg(long, default_value_t = 0.65)]
        spacing: f64,
        /// Maximum number of pharmacophore features in the blueprint
        #[arg(long, default_value_t = 10)]
        max_features: usize,
        /// Write versioned `.pharm.json` blueprint file
        #[arg(long)]
        out_json: Option<PathBuf>,
        /// Write human-readable Markdown Target Binding Blueprint
        #[arg(long)]
        out_md: Option<PathBuf>,
    },
    /// Ultra-fast streaming 3D pharmacophore + exclusion-volume funnel (`10^6 -> 10^4`) against a
    /// `.pharm.json` blueprint, emitting ranked CSV and pre-aligned 3D SDF hits for docking
    PharmScreen {
        #[arg(long)]
        blueprint: PathBuf,
        /// Chemical library (`.smi`, `.ism`, `.smi.gz`, `.sdf`, `.sdf.gz`, `.chemcraft`)
        #[arg(long)]
        library: PathBuf,
        #[arg(long)]
        out_csv: Option<PathBuf>,
        /// Optional output SDF containing 3D Kabsch-aligned hit poses inside the active site
        #[arg(long)]
        out_sdf: Option<PathBuf>,
        /// Minimum enabled blueprint features that must match simultaneously
        #[arg(long, default_value_t = 3)]
        min_match: usize,
        /// Maximum feature alignment RMSD (Å)
        #[arg(long, default_value_t = 1.65)]
        max_rmsd: f64,
        /// Maximum heavy-atom clashes against receptor exclusion spheres
        #[arg(long, default_value_t = 3)]
        max_clashes: usize,
        /// Conformers sampled per flexible ligand (`1` = fastest, `3–5` = higher recall)
        #[arg(long, default_value_t = 3)]
        conformers: usize,
        /// Keep top N hits sorted by pharmacophore score (`0` = keep all)
        #[arg(long, default_value_t = 5000)]
        top: usize,
        /// Optional hard cap on input molecules read (`0` = stream entire file)
        #[arg(long, default_value_t = 0)]
        max_ligands: usize,
    },
    /// Compare binding pocket vs 1..N off-target/paralog structures via 7-probe ΔMIF & steric-exclusivity analysis
    ComparePockets {
        #[arg(long)]
        target: PathBuf,
        /// Comma-separated off-target/paralog PDB/mmCIF/PDBQT files
        #[arg(long, value_delimiter = ',', required = true)]
        off_targets: Vec<PathBuf>,
        /// Optional cocrystallized ligand in target pocket
        #[arg(long)]
        ligand: Option<PathBuf>,
        /// Target pocket center `x,y,z` (auto-detected if omitted)
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        target_center: Vec<f64>,
        /// Optional semicolon-separated off-target pocket centers `x1,y1,z1;x2,y2,z2` (auto-detected if omitted)
        #[arg(long, allow_hyphen_values = true)]
        off_target_centers: Option<String>,
        #[arg(
            long,
            value_delimiter = ',',
            allow_hyphen_values = true,
            default_values_t = vec![22.0, 22.0, 22.0]
        )]
        size: Vec<f64>,
        /// MIF grid spacing in Å (default 0.75)
        #[arg(long, default_value_t = 0.75)]
        spacing: f64,
        /// Skip rigid pocket superposition when structures are already aligned
        #[arg(long)]
        pre_aligned: bool,
        /// Optional existing `.pharm.json` blueprint to augment with selectivity weights
        #[arg(long)]
        in_blueprint: Option<PathBuf>,
        /// Optional output path for selectivity-augmented `.pharm.json`
        #[arg(long)]
        out_blueprint: Option<PathBuf>,
        #[arg(long)]
        out_json: Option<PathBuf>,
        #[arg(long)]
        out_md: Option<PathBuf>,
        /// Optional ChimeraX `.cxc` overlay script visualizing target, aligned off-targets, and ΔMIF hotspots
        #[arg(long)]
        out_cxc: Option<PathBuf>,
        /// Optional directory where superposed off-target PDB files are written
        #[arg(long)]
        out_aligned_dir: Option<PathBuf>,
    },
    /// Run Inverse Docking Safety Pharmacology Panel (Bowes Tier-1 15 human anti-targets) or prepare panel PDBs
    SafetyPanel {
        /// Prepare all raw PDBs found in this directory via `prot-prep` and write `panel.json`
        #[arg(long)]
        prepare_raw_dir: Option<PathBuf>,
        /// Panel manifest JSON path (defaults to CRAFT_DATA_DIR/safety_panel/panel.json)
        #[arg(long)]
        manifest: Option<PathBuf>,
        /// Lead ligand library (`.smi`, `.sdf`, `.chemcraft`) to screen across the safety panel
        #[arg(long)]
        ligands: Option<PathBuf>,
        /// Reference primary target affinity (kcal/mol) used to compute selectivity margin ΔΔG
        #[arg(long, allow_hyphen_values = true)]
        target_dg: Option<f64>,
        /// Minimum selectivity margin ΔΔG (kcal/mol) required for Clear verdict (default 1.8)
        #[arg(long, default_value_t = 1.8)]
        min_margin: f64,
        #[arg(long, default_value_t = 4)]
        exhaustiveness: usize,
        #[arg(long)]
        out_json: Option<PathBuf>,
        #[arg(long)]
        out_md: Option<PathBuf>,
        /// Optional directory to export 3D SDF poses of flagged off-target hits for ChimeraX inspection
        #[arg(long)]
        out_poses_dir: Option<PathBuf>,
    },
    /// Detect druggable binding pockets blindly from a receptor structure
    DetectPockets {
        #[arg(long)]
        receptor: PathBuf,
        #[arg(long, default_value_t = 8)]
        max_pockets: usize,
    },
    /// Detect buried structural bridging waters inside a binding pocket
    DetectWaters {
        #[arg(long)]
        receptor: PathBuf,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        center: Vec<f64>,
        #[arg(long, default_value_t = 10.0)]
        radius: f64,
        #[arg(long, default_value_t = 2)]
        min_hbonds: usize,
    },
    /// Audit a docked pose with PoseBusters geometric checks and SeeSAR-style torsion traffic-lights
    ValidatePose {
        #[arg(long)]
        receptor: PathBuf,
        #[arg(long)]
        ligand: PathBuf,
        #[arg(long)]
        keep_waters: bool,
    },
    /// Dock a single ligand and emit multi-mode 3D poses SDF + JSON summary
    Dock {
        #[arg(long)]
        receptor: PathBuf,
        #[arg(long)]
        ligand: PathBuf,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        center: Vec<f64>,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        size: Vec<f64>,
        #[arg(long, value_enum, default_value_t = CliScoring::Vina)]
        scoring: CliScoring,
        #[arg(long, default_value_t = 8)]
        exhaustiveness: usize,
        #[arg(long, default_value_t = 9)]
        num_modes: usize,
        #[arg(long, default_value_t = 20260401)]
        seed: u64,
        /// Retain all crystallographic waters present in the receptor file
        #[arg(long)]
        keep_waters: bool,
        /// Automatically retain buried structural bridging waters (>= 2 receptor H-bonds inside box)
        #[arg(long)]
        auto_waters: bool,
        /// Disable AutoDock4Zn catalytic metal coordination potential (use raw literature Vina/Vinardo)
        #[arg(long)]
        no_metal_coord: bool,
        /// Re-rank docked modes using SeeSAR-style Vina + directional HYDE + torsion consensus
        #[arg(long)]
        rerank_hyde: bool,
        /// Pharmacophore sphere constraint `x,y,z,radius,kind` (kind: hdonor|hacceptor|aromatic|hydrophobic|pos|neg|metal|any)
        #[arg(long, allow_hyphen_values = true)]
        pharmacophore: Vec<String>,
        /// Optional `.pharm.json` blueprint from `analyze-site` used as soft pharmacophore guidance
        #[arg(long)]
        pharmacophore_file: Option<PathBuf>,
        /// Soft restraint weight multiplier when `--pharmacophore-file` is supplied
        #[arg(long, default_value_t = 0.45)]
        pharm_weight: f64,
        /// Flexible receptor sidechain residue specs (e.g. `A:ASP189,A:TRP79`)
        #[arg(long, value_delimiter = ',')]
        flex_residues: Vec<String>,
        /// Covalent attachment constraint `rec_atom_idx,lig_atom_idx,bond_len_a`
        #[arg(long)]
        covalent: Option<String>,
        #[arg(long)]
        out_sdf: Option<PathBuf>,
    },
    /// Score a docked complex with Vina + HYDE per-atom desolvation/H-bond/steric breakdown
    ScorePose {
        #[arg(long)]
        receptor: PathBuf,
        #[arg(long)]
        ligand: PathBuf,
        /// Include Vina, Vinardo, and DockCraftHybrid energy breakdowns alongside HYDE
        #[arg(long)]
        full: bool,
        /// Retain crystallographic waters present in the receptor file
        #[arg(long)]
        keep_waters: bool,
    },
    /// Write a self-contained UCSF ChimeraX scene (.cxc + receptor.pdb + ligand.sdf) into the
    /// export directory ChimeraX's sandbox can read, and optionally launch ChimeraX on it
    Chimerax {
        #[arg(long)]
        receptor: PathBuf,
        #[arg(long)]
        ligand: PathBuf,
        /// Name of the scene subdirectory; defaults to the ligand's own name
        #[arg(long)]
        name: Option<String>,
        /// Export root override. Default: $XDG_DATA_HOME/ChimeraX/dockcraft_export
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Spawn `chimerax` on the scene once it is written
        #[arg(long)]
        launch: bool,
    },
    /// Opt-in OpenMM GPU Molecular Dynamics validation (Amber14 + OpenFF Sage + TIP3P-FB + 150 mM NaCl)
    /// using the isolated venv in CRAFT_DATA_DIR/envs/openmm (tuned for 12 GB RTX 4080 GPUs)
    MdValidate {
        /// Probe the isolated OpenMM GPU environment and exit
        #[arg(long)]
        check_env: bool,
        #[arg(long)]
        receptor: Option<PathBuf>,
        #[arg(long)]
        ligand: Option<PathBuf>,
        #[arg(long, default_value = "./md_validation")]
        out_dir: PathBuf,
        /// Production MD length in nanoseconds (default 20.0 ns; use 0.05-0.5 ns for fast checks)
        #[arg(long, default_value_t = 20.0)]
        ns: f64,
        /// NVT equilibration duration in picoseconds
        #[arg(long, default_value_t = 50.0)]
        nvt_ps: f64,
        /// NPT equilibration duration in picoseconds
        #[arg(long, default_value_t = 100.0)]
        npt_ps: f64,
        #[arg(long, default_value_t = 300.0)]
        temp_k: f64,
        /// Explicit water box padding in nm (default 1.0 nm for 12 GB VRAM)
        #[arg(long, default_value_t = 1.0)]
        padding_nm: f64,
        /// Ligand partial charges (`nagl` | `mmff94` | `gasteiger` | `am1bcc`)
        #[arg(long, default_value = "nagl")]
        charges: String,
        #[arg(long, value_enum, default_value_t = CliMdPlatform::Auto)]
        platform: CliMdPlatform,
        /// Explicit Python interpreter inside an OpenMM venv (overrides CRAFT_DATA_DIR/envs/openmm)
        #[arg(long)]
        python: Option<PathBuf>,
        /// Stage runnable OpenMM script + config + ChimeraX .cxc without launching GPU MD
        #[arg(long)]
        dry_run: bool,
    },
    /// Run a fully reproducible 7-stage SBDD -> ADMET -> MD -> Retrosynthesis campaign with SHA-256 provenance
    Campaign {
        /// Path to a declarative JSON `CampaignSpec` file
        #[arg(long)]
        spec: Option<PathBuf>,
        /// Write a ready-to-edit CDK2 template `CampaignSpec` JSON file and exit
        #[arg(long)]
        init_template: Option<PathBuf>,
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

fn heavy_centroid(m: &struct_core::Molecule3D) -> [f64; 3] {
    let mut sum = [0.0; 3];
    let mut cnt = 0usize;
    for a in &m.atoms {
        if a.is_heavy() {
            sum[0] += a.coords[0];
            sum[1] += a.coords[1];
            sum[2] += a.coords[2];
            cnt += 1;
        }
    }
    if cnt == 0 {
        [0.0, 0.0, 0.0]
    } else {
        let k = cnt as f64;
        [sum[0] / k, sum[1] / k, sum[2] / k]
    }
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
        Commands::ProtPrep {
            input,
            out_pdb,
            out_pdbqt,
            out_report,
            out_md,
            extract_ligand,
            chains,
            ph,
            no_cofactors,
            no_metals,
            keep_ligands,
            keep_bridging_waters,
            no_flips,
            center,
            radius,
        } => {
            let pocket_center = if center.len() == 3 {
                Some([center[0], center[1], center[2]])
            } else {
                None
            };
            let cfg = PrepConfig {
                ph,
                chains,
                keep_cofactors: !no_cofactors,
                keep_metals: !no_metals,
                strip_ligands: !keep_ligands,
                keep_bridging_waters,
                optimize_hbonds: !no_flips,
                pocket_center,
                pocket_radius: radius,
                ..Default::default()
            };
            let prep = prepare_protein_file(&input, &cfg)?;
            if let Some(p) = &out_pdb {
                std::fs::write(p, write_pdb(&prep.receptor))?;
            }
            if let Some(p) = &out_pdbqt {
                std::fs::write(p, write_pdbqt_receptor(&prep.receptor))?;
            }
            if let Some(p) = &out_report {
                let summary_json = serde_json::json!({
                    "audit": prep.audit,
                    "protonation": prep.protonation,
                    "hbond_opt": prep.hbond_opt,
                    "bridging_waters": prep.bridging_waters,
                    "extracted_ligand_count": prep.extracted_ligands.len(),
                    "final_atom_count": prep.receptor.atoms.len(),
                    "elapsed_ms": prep.elapsed_ms,
                });
                std::fs::write(p, serde_json::to_string_pretty(&summary_json)?)?;
            }
            let md = prep.render_markdown();
            if let Some(p) = &out_md {
                std::fs::write(p, &md)?;
            }
            if let Some(p) = &extract_ligand
                && let Some(primary_lig) = prep
                    .extracted_ligands
                    .iter()
                    .max_by_key(|l| l.heavy_atom_count())
            {
                let ext = p
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("sdf")
                    .to_ascii_lowercase();
                if ext == "pdb" {
                    std::fs::write(p, write_pdb(primary_lig))?;
                } else {
                    std::fs::write(p, write_sdf_3d(&[primary_lig]))?;
                }
            }
            println!("{md}");
        }
        Commands::EnsembleDock {
            receptors,
            ligand,
            library,
            center,
            size,
            scoring,
            aggregation,
            exhaustiveness,
            no_hyde_consensus,
            max_ligands,
            out_csv,
            out_sdf,
            out_md,
        } => {
            if size.len() != 3 {
                return Err("--size must provide 3 comma-separated floats".into());
            }
            let sz = [size[0], size[1], size[2]];
            let parsed_centers: Vec<[f64; 3]> = if let Some(c_str) = &center {
                let mut out = Vec::new();
                for group in c_str.split(';') {
                    let vals: Vec<f64> = group
                        .split(',')
                        .map(|s| s.trim().parse::<f64>())
                        .collect::<Result<_, _>>()?;
                    if vals.len() != 3 {
                        return Err(
                            format!("invalid --center entry '{group}' (expected x,y,z)").into()
                        );
                    }
                    out.push([vals[0], vals[1], vals[2]]);
                }
                out
            } else {
                Vec::new()
            };
            let mut ens_recs = Vec::with_capacity(receptors.len());
            for (ri, rp) in receptors.iter().enumerate() {
                let rec = read_receptor_file(rp, true)?;
                let label = rp
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("receptor")
                    .to_string();
                let ctr =
                    if let Some(&c) = parsed_centers.get(ri).or_else(|| parsed_centers.first()) {
                        c
                    } else {
                        let pockets = detect_pockets(&rec, &PocketDetectConfig::default())?;
                        pockets
                            .first()
                            .map(|p| p.center)
                            .ok_or_else(|| format!("no pockets detected in '{}'", rp.display()))?
                    };
                let gbox = GridBox::from_center_size(ctr, sz, 0.375)?;
                ens_recs.push(EnsembleReceptor {
                    label,
                    receptor: rec,
                    grid_box: gbox,
                });
            }
            let lig_records = if let Some(lib_path) = &library {
                let bytes = std::fs::read(lib_path)?;
                let ext = lib_path
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("sdf");
                load_ligand_library_bytes(ext, &bytes, max_ligands)?
            } else if let Some(lig_path) = &ligand {
                let raw = std::fs::read_to_string(lig_path)?;
                let ext = lig_path
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("sdf")
                    .to_ascii_lowercase();
                let mut mol = match ext.as_str() {
                    "pdbqt" => parse_pdbqt(&raw)?,
                    "mol2" => parse_mol2(&raw)?,
                    "pdb" => parse_pdb(&raw, true)?,
                    _ => {
                        let mut v = parse_sdf_3d_with_limit(&raw, 1);
                        v.pop().ok_or("empty ligand SDF")??
                    }
                };
                struct_core::perceive_all(&mut mol);
                let nm = if mol.name.trim().is_empty() {
                    lig_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("ligand")
                        .to_string()
                } else {
                    mol.name.clone()
                };
                vec![dock_pipeline::LigandInputRecord {
                    name: nm,
                    molecule: mol,
                    is_active: None,
                }]
            } else {
                return Err("provide either --ligand or --library for ensemble-dock".into());
            };
            let sf = match scoring {
                CliScoring::Vina => ScoringFunction::Vina,
                CliScoring::Vinardo => ScoringFunction::Vinardo,
                CliScoring::Hybrid => ScoringFunction::DockCraftHybrid,
            };
            let agg = match aggregation {
                CliAggregation::Boltzmann => EnsembleAggregation::Boltzmann,
                CliAggregation::Best => EnsembleAggregation::BestConformation,
                CliAggregation::Mean => EnsembleAggregation::Mean,
                CliAggregation::Rrf => EnsembleAggregation::ReciprocalRank,
            };
            let cfg = DockingConfig {
                scoring: sf,
                exhaustiveness,
                ..Default::default()
            };
            let rep = run_ensemble_docking(&ens_recs, &lig_records, &cfg, agg, !no_hyde_consensus)?;
            if let Some(p) = &out_csv {
                std::fs::write(p, rep.to_csv())?;
            }
            if let Some(p) = &out_sdf {
                let refs: Vec<&struct_core::Molecule3D> = rep
                    .candidates
                    .iter()
                    .map(|c| &c.best_pose.molecule)
                    .collect();
                std::fs::write(p, write_sdf_3d(&refs))?;
            }
            let md = rep.to_markdown();
            if let Some(p) = &out_md {
                std::fs::write(p, &md)?;
            }
            println!("{md}");
        }
        Commands::Screen {
            receptor,
            library,
            out,
            center,
            size,
            box_from_chimerax,
            box_padding,
            scoring,
            exhaustiveness,
            max_ligands,
            pharmacophore_file,
            pharm_weight,
            sidecar,
            sidecar_dir,
        } => {
            let rec = read_receptor_file(&receptor, true)?;
            let blueprint_opt = if let Some(pf) = &pharmacophore_file {
                let raw_bp = std::fs::read_to_string(pf)?;
                Some(SiteBlueprint::from_json_str(&raw_bp)?)
            } else {
                None
            };
            let gbox = if box_from_chimerax {
                let root = default_chimerax_export_dir();
                std::fs::create_dir_all(&root)?;
                let client = ChimeraXRest::with_default_port(root.clone());
                let selection = grid_box_from_chimerax_selection(&client, &root, box_padding)?;
                eprintln!(
                    "Box from the ChimeraX selection: {} residues ({}), {} heavy atoms, centre [{:.2}, {:.2}, {:.2}], size [{:.1}, {:.1}, {:.1}]",
                    selection.residues.len(),
                    selection.residues.join(", "),
                    selection.heavy_atoms,
                    selection.grid_box.center[0],
                    selection.grid_box.center[1],
                    selection.grid_box.center[2],
                    selection.grid_box.size[0],
                    selection.grid_box.size[1],
                    selection.grid_box.size[2],
                );
                selection.grid_box
            } else if center.len() == 3 && size.len() == 3 {
                GridBox::from_center_size(
                    [center[0], center[1], center[2]],
                    [size[0], size[1], size[2]],
                    0.375,
                )?
            } else if let Some(bp) = &blueprint_opt {
                bp.grid_box.clone()
            } else {
                return Err(
                    "--center and --size must be comma-separated 3-vectors x,y,z, or pass --box-from-chimerax / --pharmacophore-file"
                        .into(),
                );
            };
            let bytes = std::fs::read(&library)?;
            let ext = library
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("smi");
            let lib = load_ligand_library_bytes(ext, &bytes, max_ligands)?;
            let constraints = blueprint_opt
                .as_ref()
                .map_or_else(Vec::new, |bp| bp.to_docking_constraints(3, pharm_weight));
            let cfg = DockingConfig {
                scoring: match scoring {
                    CliScoring::Vina => ScoringFunction::Vina,
                    CliScoring::Vinardo => ScoringFunction::Vinardo,
                    CliScoring::Hybrid => ScoringFunction::DockCraftHybrid,
                },
                exhaustiveness,
                constraints,
                ..Default::default()
            };
            let report = run_screening_campaign(&rec, &gbox, &lib, &cfg)?;
            std::fs::write(&out, campaign_report_to_csv(&report))?;

            if sidecar {
                let dir = sidecar_dir.unwrap_or_else(default_chemcraft_results_dir);
                std::fs::create_dir_all(&dir)?;
                let stem = library
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("screening");
                let receptor_name = receptor
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("receptor");
                let scoring_name = match scoring {
                    CliScoring::Vina => "vina",
                    CliScoring::Vinardo => "vinardo",
                    CliScoring::Hybrid => "dockcraft-hybrid",
                };
                let path = write_results_sidecar(
                    &report,
                    receptor_name,
                    &gbox,
                    scoring_name,
                    exhaustiveness,
                    &dir,
                    stem,
                )?;
                eprintln!("Results sidecar for ChemCraft: {}", path.display());
            }
            println!(
                "Screened {} compounds in {} ms -> {}",
                report.results.len(),
                report.elapsed_ms,
                out.display()
            );
        }
        Commands::AnalyzeSite {
            receptor,
            ligand,
            center,
            size,
            auto_pocket,
            keep_waters,
            spacing,
            max_features,
            out_json,
            out_md,
        } => {
            let mut rec = read_receptor_file(&receptor, !keep_waters)?;
            if let Some(stem) = receptor.file_stem().and_then(|s| s.to_str()) {
                rec.name = stem.to_string();
            }
            let mut holo_owned = Vec::new();
            for lp in &ligand {
                let raw = std::fs::read_to_string(lp)?;
                let ext = lp
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("sdf")
                    .to_ascii_lowercase();
                let mut mol = match ext.as_str() {
                    "mol2" => parse_mol2(&raw)?,
                    "pdbqt" => parse_pdbqt(&raw)?,
                    "pdb" => parse_pdb(&raw, true)?,
                    _ => parse_sdf_3d_with_limit(&raw, 1)
                        .pop()
                        .ok_or("empty holo ligand SDF")??,
                };
                struct_core::perceive_all(&mut mol);
                holo_owned.push(mol);
            }
            let holo_refs: Vec<&struct_core::Molecule3D> = holo_owned.iter().collect();

            let gbox = if center.len() == 3 && size.len() == 3 {
                GridBox::from_center_size(
                    [center[0], center[1], center[2]],
                    [size[0], size[1], size[2]],
                    0.375,
                )?
            } else if let Some(first_holo) = holo_owned.first()
                && !auto_pocket
            {
                GridBox::from_ligand(first_holo, 5.5)?
            } else {
                let pockets = detect_pockets(&rec, &PocketDetectConfig::default())?;
                let top = pockets
                    .first()
                    .ok_or("no druggable pocket detected; pass --center and --size")?;
                top.grid_box.clone()
            };

            let cfg = SiteAnalysisConfig {
                spacing_a: spacing,
                max_features,
                ..Default::default()
            };
            let blueprint = analyze_binding_site(&rec, &gbox, &holo_refs, &cfg)?;
            let js = blueprint.to_json_pretty()?;
            if let Some(jp) = &out_json {
                std::fs::write(jp, &js)?;
            }
            if let Some(mp) = &out_md {
                std::fs::write(mp, blueprint.render_markdown())?;
            }
            if out_json.is_some() || out_md.is_some() {
                eprintln!(
                    "Site blueprint '{}': {:.0} A3, LxWxD = {:.1}x{:.1}x{:.1} A, {} features, {} distance pairs, {} excl spheres ({} ms) [CRAFT_DATA_DIR={}]",
                    blueprint.target_name,
                    blueprint.geometry.accessible_volume_a3,
                    blueprint.geometry.principal_dims_a[0],
                    blueprint.geometry.principal_dims_a[1],
                    blueprint.geometry.principal_dims_a[2],
                    blueprint.features.len(),
                    blueprint.distance_matrix.len(),
                    blueprint.exclusion_spheres.len(),
                    blueprint.elapsed_ms,
                    craft_data_dir().display()
                );
            }
            println!("{js}");
        }
        Commands::PharmScreen {
            blueprint,
            library,
            out_csv,
            out_sdf,
            min_match,
            max_rmsd,
            max_clashes,
            conformers,
            top,
            max_ligands,
        } => {
            let bp_raw = std::fs::read_to_string(&blueprint)?;
            let bp = SiteBlueprint::from_json_str(&bp_raw)?;
            let cfg = PharmScreenConfig {
                min_matched_features: min_match,
                max_rmsd_a: max_rmsd,
                max_exclusion_clashes: max_clashes,
                max_conformers: conformers.max(1),
                max_hits: top,
                ..Default::default()
            };

            let fname_lower = library
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let is_smi_stream = fname_lower.ends_with(".smi")
                || fname_lower.ends_with(".ism")
                || fname_lower.ends_with(".smi.gz")
                || fname_lower.ends_with(".ism.gz");

            let report = if is_smi_stream {
                let file = std::fs::File::open(&library)?;
                let reader: Box<dyn BufRead> = if fname_lower.ends_with(".gz") {
                    Box::new(BufReader::new(GzDecoder::new(file)))
                } else {
                    Box::new(BufReader::new(file))
                };
                let mut idx = 0usize;
                let iter = reader
                    .lines()
                    .map_while(Result::ok)
                    .filter_map(move |line| {
                        let trimmed = line.trim();
                        if trimmed.is_empty() || trimmed.starts_with('#') {
                            return None;
                        }
                        let mut parts = trimmed.split_whitespace();
                        let smi = parts.next()?.to_string();
                        if smi.eq_ignore_ascii_case("smiles") {
                            return None;
                        }
                        idx += 1;
                        if max_ligands > 0 && idx > max_ligands {
                            return None;
                        }
                        let name = parts
                            .next()
                            .map_or_else(|| format!("mol_{idx}"), ToString::to_string);
                        Some((smi, name))
                    });
                screen_smiles_stream(iter, &bp, &cfg, 2048)
            } else {
                let raw_bytes = if fname_lower.ends_with(".gz") {
                    let file = std::fs::File::open(&library)?;
                    let mut dec = GzDecoder::new(file);
                    let mut buf = Vec::new();
                    std::io::Read::read_to_end(&mut dec, &mut buf)?;
                    buf
                } else {
                    std::fs::read(&library)?
                };
                let ext = if fname_lower.ends_with(".sdf.gz") || fname_lower.ends_with(".sdf") {
                    "sdf"
                } else {
                    library
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("sdf")
                };
                let lim = if max_ligands == 0 {
                    100_000
                } else {
                    max_ligands
                };
                let recs = load_ligand_library_bytes(ext, &raw_bytes, lim)?;
                let mols: Vec<struct_core::Molecule3D> =
                    recs.into_iter().map(|r| r.molecule).collect();
                screen_molecules_parallel(&mols, &bp, &cfg)
            };

            if let Some(csv_path) = &out_csv {
                let mut csv_out = String::from(
                    "rank,name,smiles,matched_count,total_features,matched_ids,pharm_rmsd_a,exclusion_clashes,pharm_score\n",
                );
                for (i, h) in report.hits.iter().enumerate() {
                    csv_out.push_str(&format!(
                        "{},{},{},{},{},{},{:.3},{},{:.3}\n",
                        i + 1,
                        h.name.replace(',', "_"),
                        h.smiles.as_deref().unwrap_or("").replace(',', "_"),
                        h.matched_count,
                        h.total_active_features,
                        h.matched_feature_ids.join(";"),
                        h.pharm_rmsd_a,
                        h.exclusion_clashes,
                        h.pharm_score
                    ));
                }
                std::fs::write(csv_path, csv_out)?;
            }
            if let Some(sdf_path) = &out_sdf {
                let refs: Vec<&struct_core::Molecule3D> =
                    report.hits.iter().map(|h| &h.aligned_molecule).collect();
                std::fs::write(sdf_path, write_sdf_3d(&refs))?;
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "target_name": report.target_name,
                    "total_evaluated": report.total_evaluated,
                    "passed_count": report.passed_count,
                    "returned_hits": report.hits.len(),
                    "elapsed_ms": report.elapsed_ms,
                    "ms_per_ligand": (report.elapsed_ms as f64) / (report.total_evaluated.max(1) as f64),
                }))?
            );
        }
        Commands::ComparePockets {
            target,
            off_targets,
            ligand,
            target_center,
            off_target_centers,
            size,
            spacing,
            pre_aligned,
            in_blueprint,
            out_blueprint,
            out_json,
            out_md,
            out_cxc,
            out_aligned_dir,
        } => {
            let tgt_rec = read_receptor_file(&target, true)?;
            let tgt_name = target
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Target")
                .to_string();
            let holo_lig = if let Some(lp) = &ligand {
                let raw = std::fs::read_to_string(lp)?;
                let ext = lp
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let mut m = if ext == "pdb" {
                    parse_pdb(&raw, true)?
                } else if ext == "mol2" {
                    parse_mol2(&raw)?
                } else {
                    let mut list = parse_sdf_3d_with_limit(&raw, 1);
                    list.pop().ok_or("empty holo ligand file")??
                };
                struct_core::perceive_all(&mut m);
                Some(m)
            } else {
                None
            };
            let t_center = if target_center.len() == 3 {
                [target_center[0], target_center[1], target_center[2]]
            } else if let Some(hl) = &holo_lig {
                heavy_centroid(hl)
            } else {
                let pk = detect_pockets(&tgt_rec, &PocketDetectConfig::default())?;
                pk.first()
                    .ok_or("no pocket detected in target; pass --target-center")?
                    .center
            };
            let box_size = if size.len() == 3 {
                [size[0], size[1], size[2]]
            } else {
                [22.0, 22.0, 22.0]
            };
            let gbox = GridBox::from_center_size(t_center, box_size, 0.375)?;

            let explicit_ot_centers: Vec<[f64; 3]> = if let Some(raw_c) = &off_target_centers {
                raw_c
                    .split(';')
                    .filter(|s| !s.trim().is_empty())
                    .map(|part| -> Result<[f64; 3], Box<dyn std::error::Error>> {
                        let vals: Vec<f64> = part
                            .split(',')
                            .map(|x| x.trim().parse::<f64>())
                            .collect::<Result<_, _>>()?;
                        if vals.len() != 3 {
                            return Err(
                                "each center in --off-target-centers must have 3 floats".into()
                            );
                        }
                        Ok([vals[0], vals[1], vals[2]])
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                Vec::new()
            };

            let mut loaded_ots: Vec<(String, struct_core::Molecule3D, [f64; 3])> =
                Vec::with_capacity(off_targets.len());
            for (idx, otp) in off_targets.iter().enumerate() {
                let orec = read_receptor_file(otp, true)?;
                let oname = otp
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("OffTarget")
                    .to_string();
                let oc = if let Some(&c) = explicit_ot_centers.get(idx) {
                    c
                } else if pre_aligned {
                    t_center
                } else {
                    let pk = detect_pockets(&orec, &PocketDetectConfig::default())?;
                    pk.first().map_or(t_center, |p| p.center)
                };
                loaded_ots.push((oname, orec, oc));
            }
            let ot_inputs: Vec<OffTargetPocketInput<'_>> = loaded_ots
                .iter()
                .map(|(n, r, c)| OffTargetPocketInput {
                    name: n.clone(),
                    receptor: r,
                    pocket_center: *c,
                })
                .collect();
            let holo_refs: Vec<&struct_core::Molecule3D> = holo_lig.iter().collect();
            let cfg = PocketSelectivityConfig {
                spacing_a: spacing,
                pre_aligned,
                ..Default::default()
            };
            let out = analyze_pocket_selectivity(
                &tgt_name, &tgt_rec, &gbox, &holo_refs, &ot_inputs, &cfg,
            )?;

            let mut aligned_refs: Vec<(String, String)> = Vec::new();
            let aligned_dir = out_aligned_dir
                .or_else(|| out_cxc.as_ref().and_then(|p| p.parent().map(PathBuf::from)));
            if let Some(adir) = &aligned_dir {
                std::fs::create_dir_all(adir)?;
                for (oname, amol) in &out.aligned_offtargets {
                    let p = adir.join(format!("{oname}_aligned.pdb"));
                    std::fs::write(&p, write_pdb(amol))?;
                    aligned_refs.push((oname.clone(), p.display().to_string()));
                }
            }
            if let Some(jp) = &out_json {
                if let Some(par) = jp.parent() {
                    std::fs::create_dir_all(par)?;
                }
                std::fs::write(jp, serde_json::to_string_pretty(&out.report)?)?;
            }
            let md = out.report.render_markdown();
            if let Some(mp) = &out_md {
                if let Some(par) = mp.parent() {
                    std::fs::create_dir_all(par)?;
                }
                std::fs::write(mp, &md)?;
            }
            if let Some(cp) = &out_cxc {
                if let Some(par) = cp.parent() {
                    std::fs::create_dir_all(par)?;
                }
                let refs_str: Vec<(&str, &str)> = aligned_refs
                    .iter()
                    .map(|(a, b)| (a.as_str(), b.as_str()))
                    .collect();
                let cxc = out
                    .report
                    .chimerax_overlay_script(&target.display().to_string(), &refs_str);
                std::fs::write(cp, cxc)?;
            }
            if let (Some(in_bp), Some(out_bp)) = (&in_blueprint, &out_blueprint) {
                let raw_bp = std::fs::read_to_string(in_bp)?;
                let base_bp = SiteBlueprint::from_json_str(&raw_bp)?;
                let sel_bp = out.report.augment_blueprint(&base_bp);
                if let Some(par) = out_bp.parent() {
                    std::fs::create_dir_all(par)?;
                }
                std::fs::write(out_bp, sel_bp.to_json_pretty()?)?;
            }
            println!("{md}");
        }
        Commands::SafetyPanel {
            prepare_raw_dir,
            manifest,
            ligands,
            target_dg,
            min_margin,
            exhaustiveness,
            out_json,
            out_md,
            out_poses_dir,
        } => {
            let manifest_path = manifest.unwrap_or_else(default_safety_panel_manifest_path);
            let panel_dir = manifest_path
                .parent()
                .map_or_else(default_safety_panel_dir, PathBuf::from);

            if let Some(raw_dir) = prepare_raw_dir {
                let mut catalog = default_bowes_tier1_catalog(&panel_dir);
                std::fs::create_dir_all(panel_dir.join("prepared"))?;
                let mut prepared_ok = 0usize;
                for entry in &mut catalog.targets {
                    let raw_pdb = raw_dir.join(format!("{}.pdb", entry.pdb_id));
                    if !raw_pdb.is_file() {
                        entry.enabled = false;
                        continue;
                    }
                    let cfg = PrepConfig {
                        chains: entry.chains.clone(),
                        ph: 7.4,
                        keep_cofactors: true,
                        keep_metals: true,
                        strip_ligands: true,
                        optimize_hbonds: true,
                        ..Default::default()
                    };
                    match prepare_protein_file(&raw_pdb, &cfg) {
                        Ok(prep) => {
                            std::fs::write(&entry.prepared_pdb, write_pdb(&prep.receptor))?;
                            // Determine pocket center:
                            // 1) Primary extracted organic ligand (>= 8 heavy atoms)
                            // 2) Anchor residue CA centroid (for apo/cryo-EM pores like hERG 5VA1 TYR652/PHE656)
                            // 3) Auto-detected top pocket fallback
                            let lig_center = prep
                                .extracted_ligands
                                .iter()
                                .filter(|l| l.heavy_atom_count() >= 8)
                                .max_by_key(|l| l.heavy_atom_count())
                                .map(heavy_centroid);
                            let anchor_center = if !entry.anchor_res_seqs.is_empty() {
                                let pts: Vec<[f64; 3]> = prep
                                    .receptor
                                    .atoms
                                    .iter()
                                    .filter(|a| {
                                        !a.is_hetatm
                                            && a.name.trim().eq_ignore_ascii_case("CA")
                                            && entry.anchor_res_seqs.contains(&a.res_seq)
                                    })
                                    .map(|a| a.coords)
                                    .collect();
                                if pts.is_empty() {
                                    None
                                } else {
                                    let k = pts.len() as f64;
                                    let mut s = [0.0; 3];
                                    for p in &pts {
                                        s[0] += p[0] / k;
                                        s[1] += p[1] / k;
                                        s[2] += p[2] / k;
                                    }
                                    Some(s)
                                }
                            } else {
                                None
                            };
                            entry.box_center = if entry.id == "hERG" {
                                anchor_center.or(lig_center).unwrap_or([0.0, 0.0, 0.0])
                            } else {
                                lig_center.or(anchor_center).unwrap_or_else(|| {
                                    detect_pockets(&prep.receptor, &PocketDetectConfig::default())
                                        .ok()
                                        .and_then(|pk| pk.first().map(|p| p.center))
                                        .unwrap_or([0.0, 0.0, 0.0])
                                })
                            };
                            entry.enabled = true;
                            prepared_ok += 1;
                            eprintln!(
                                "[SafetyPanel] Prepared {:6} ({}) -> center ({:.1}, {:.1}, {:.1}) [Grade {}]",
                                entry.id,
                                entry.pdb_id,
                                entry.box_center[0],
                                entry.box_center[1],
                                entry.box_center[2],
                                prep.audit.quality_grade
                            );
                        }
                        Err(err) => {
                            eprintln!(
                                "[SafetyPanel] Skipped {} ({}): {err}",
                                entry.id, entry.pdb_id
                            );
                            entry.enabled = false;
                        }
                    }
                }
                catalog.save_to_path(&manifest_path)?;
                println!(
                    "Prepared {prepared_ok}/{} Bowes Tier-1 anti-targets -> {}",
                    catalog.targets.len(),
                    manifest_path.display()
                );
                if ligands.is_none() {
                    return Ok(());
                }
            }

            let lig_path =
                ligands.ok_or("provide --ligands <SMI/SDF> (or --prepare-raw-dir <DIR>)")?;
            let panel_manifest = SafetyPanelManifest::load_from_path(&manifest_path)?;
            let bytes = std::fs::read(&lig_path)?;
            let ext = lig_path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("smi");
            let lig_records = load_ligand_library_bytes(ext, &bytes, 100)?;
            let sp_cfg = SafetyPanelConfig {
                exhaustiveness,
                target_affinity_kcal: target_dg,
                min_margin_kcal: min_margin,
                out_poses_dir,
                ..Default::default()
            };
            let report = run_safety_panel(&lig_records, &panel_manifest, &sp_cfg)?;
            if let Some(jp) = &out_json {
                if let Some(par) = jp.parent() {
                    std::fs::create_dir_all(par)?;
                }
                std::fs::write(jp, serde_json::to_string_pretty(&report)?)?;
            }
            let md = report.render_markdown();
            if let Some(mp) = &out_md {
                if let Some(par) = mp.parent() {
                    std::fs::create_dir_all(par)?;
                }
                std::fs::write(mp, &md)?;
            }
            println!("{md}");
        }
        Commands::DetectPockets {
            receptor,
            max_pockets,
        } => {
            let rec = read_receptor_file(&receptor, true)?;
            let cfg = PocketDetectConfig::default();
            let mut pockets = detect_pockets(&rec, &cfg)?;
            pockets.truncate(max_pockets);
            println!("{}", serde_json::to_string_pretty(&pockets)?);
        }
        Commands::DetectWaters {
            receptor,
            center,
            radius,
            min_hbonds,
        } => {
            if center.len() != 3 {
                return Err("--center must be a comma-separated 3-vector x,y,z".into());
            }
            let raw = std::fs::read_to_string(&receptor)?;
            let rec_raw = parse_pdb(&raw, false)?;
            let waters = detect_bridging_waters(
                &rec_raw,
                [center[0], center[1], center[2]],
                radius,
                min_hbonds,
            );
            println!("{}", serde_json::to_string_pretty(&waters)?);
        }
        Commands::ValidatePose {
            receptor,
            ligand,
            keep_waters,
        } => {
            let mut rec = read_receptor_file(&receptor, !keep_waters)?;
            if let Some(stem) = receptor.file_stem().and_then(|s| s.to_str()) {
                rec.name = stem.to_string();
            }
            let lig_raw = std::fs::read_to_string(&ligand)?;
            let mut ligs = parse_sdf_3d_with_limit(&lig_raw, 1);
            let mut lig = ligs.pop().ok_or("empty ligand SDF")??;
            struct_core::perceive_all(&mut lig);
            let pb = validate_pose_geometry(&rec, &lig);
            let torsions = analyze_torsions(&lig);
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "posebusters": pb,
                    "torsion_report": torsions,
                }))?
            );
        }
        Commands::Dock {
            receptor,
            ligand,
            center,
            size,
            scoring,
            exhaustiveness,
            num_modes,
            seed,
            keep_waters,
            auto_waters,
            no_metal_coord,
            rerank_hyde,
            pharmacophore,
            pharmacophore_file,
            pharm_weight,
            flex_residues,
            covalent,
            out_sdf,
        } => {
            let blueprint_opt = if let Some(pf) = &pharmacophore_file {
                let raw_bp = std::fs::read_to_string(pf)?;
                Some(SiteBlueprint::from_json_str(&raw_bp)?)
            } else {
                None
            };
            let (ctr, sz) = if center.len() == 3 && size.len() == 3 {
                (
                    [center[0], center[1], center[2]],
                    [size[0], size[1], size[2]],
                )
            } else if let Some(bp) = &blueprint_opt {
                (bp.grid_box.center, bp.grid_box.size)
            } else {
                return Err(
                    "--center and --size must be comma-separated 3-vectors x,y,z (or pass --pharmacophore-file)"
                        .into(),
                );
            };
            let rec = if auto_waters && !keep_waters {
                let raw = std::fs::read_to_string(&receptor)?;
                let mut rec_raw = parse_pdb(&raw, false)?;
                let box_r = 0.5 * sz[0].max(sz[1]).max(sz[2]);
                let bw = detect_bridging_waters(&rec_raw, ctr, box_r, 2);
                let keep_ids: Vec<i32> = bw.iter().map(|w| w.res_seq).collect();
                prepare_receptor_keeping_waters(&mut rec_raw, &keep_ids);
                rec_raw
            } else {
                read_receptor_file(&receptor, !keep_waters)?
            };
            let lig_raw = std::fs::read_to_string(&ligand)?;
            let ext = ligand
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("sdf")
                .to_ascii_lowercase();
            let mut lig = match ext.as_str() {
                "pdbqt" => parse_pdbqt(&lig_raw)?,
                "mol2" => parse_mol2(&lig_raw)?,
                "pdb" => parse_pdb(&lig_raw, true)?,
                _ => {
                    let mut ligs = parse_sdf_3d_with_limit(&lig_raw, 1);
                    ligs.pop().ok_or("empty ligand SDF")??
                }
            };
            struct_core::perceive_all(&mut lig);
            let gbox = GridBox::from_center_size(ctr, sz, 0.375)?;
            let sf = match scoring {
                CliScoring::Vina => ScoringFunction::Vina,
                CliScoring::Vinardo => ScoringFunction::Vinardo,
                CliScoring::Hybrid => ScoringFunction::DockCraftHybrid,
            };
            let mut constraints: Vec<DockingConstraint> = blueprint_opt
                .as_ref()
                .map_or_else(Vec::new, |bp| bp.to_docking_constraints(5, pharm_weight));
            for spec in &pharmacophore {
                let parts: Vec<&str> = spec.split(',').collect();
                if parts.len() < 4 {
                    return Err(format!(
                        "invalid --pharmacophore '{spec}' (expected x,y,z,r[,kind])"
                    )
                    .into());
                }
                let px: f64 = parts[0].trim().parse()?;
                let py: f64 = parts[1].trim().parse()?;
                let pz: f64 = parts[2].trim().parse()?;
                let pr: f64 = parts[3].trim().parse()?;
                let kind = match parts
                    .get(4)
                    .map(|s| s.trim().to_ascii_lowercase())
                    .as_deref()
                {
                    Some("hdonor" | "donor") => PharmacophoreKind::HDonor,
                    Some("hacceptor" | "acceptor") => PharmacophoreKind::HAcceptor,
                    Some("aromatic" | "arom") => PharmacophoreKind::Aromatic,
                    Some("hydrophobic" | "hyd") => PharmacophoreKind::Hydrophobic,
                    Some("pos" | "positive" | "cation") => PharmacophoreKind::PositiveIonizable,
                    Some("neg" | "negative" | "anion") => PharmacophoreKind::NegativeIonizable,
                    Some("metal" | "metalchelate") => PharmacophoreKind::MetalChelate,
                    _ => PharmacophoreKind::AnyHeavy,
                };
                constraints.push(DockingConstraint::PharmacophoreSphere {
                    center: [px, py, pz],
                    radius: pr,
                    kind,
                    weight: 4.0,
                });
            }
            if let Some(cov) = &covalent {
                let parts: Vec<&str> = cov.split(',').collect();
                if parts.len() != 3 {
                    return Err("--covalent expects rec_atom_idx,lig_atom_idx,bond_len_a".into());
                }
                constraints.push(DockingConstraint::CovalentAttachment {
                    receptor_atom_idx: parts[0].trim().parse()?,
                    ligand_atom_idx: parts[1].trim().parse()?,
                    target_bond_len: parts[2].trim().parse()?,
                    force_k: 45.0,
                });
            }
            let cfg = DockingConfig {
                scoring: sf,
                exhaustiveness,
                num_modes,
                seed,
                metal_coordination: !no_metal_coord,
                constraints,
                flex_residues,
                ..Default::default()
            };
            let mut report = dock_ligand(&rec, &lig, &gbox, &cfg)?;
            if rerank_hyde && report.poses.len() > 1 {
                let eval_rec = report.flex_receptor.as_ref().unwrap_or(&rec);
                let mut scored: Vec<(f64, dock_engine::DockingPose)> = report
                    .poses
                    .into_iter()
                    .map(|p| {
                        let hyde_dg = score_pose_hyde(eval_rec, &p.molecule)
                            .map_or(p.affinity_kcal_mol, |h| h.total_delta_g_kcal);
                        let cons = consensus_affinity_kcal(
                            p.affinity_kcal_mol,
                            p.affinity_kcal_mol,
                            hyde_dg,
                        );
                        (cons, p)
                    })
                    .collect();
                scored.sort_by(|a, b| a.0.total_cmp(&b.0));
                report.poses = scored
                    .into_iter()
                    .enumerate()
                    .map(|(i, (_, mut p))| {
                        p.mode = i + 1;
                        p
                    })
                    .collect();
            }
            if let Some(sdf_path) = &out_sdf {
                let pose_refs: Vec<&struct_core::Molecule3D> =
                    report.poses.iter().map(|p| &p.molecule).collect();
                std::fs::write(sdf_path, write_sdf_3d(&pose_refs))?;
            }
            let modes: Vec<serde_json::Value> = report
                .poses
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "mode": p.mode,
                        "affinity_kcal": p.affinity_kcal_mol,
                        "terms": p.terms,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "scoring": format!("{:?}", sf),
                    "elapsed_ms": report.elapsed_ms,
                    "evaluations": report.total_evaluations,
                    "modes": modes,
                }))?
            );
        }
        Commands::ScorePose {
            receptor,
            ligand,
            full,
            keep_waters,
        } => {
            let mut rec = read_receptor_file(&receptor, !keep_waters)?;
            if let Some(stem) = receptor.file_stem().and_then(|s| s.to_str()) {
                rec.name = stem.to_string();
            }
            let lig_raw = std::fs::read_to_string(&ligand)?;
            let mut ligs = parse_sdf_3d_with_limit(&lig_raw, 1);
            let mut lig = ligs.pop().ok_or("empty ligand SDF")??;
            struct_core::perceive_all(&mut lig);
            let hyde = score_pose_hyde(&rec, &lig)?;
            if full {
                let vina = score_complex(&rec, &lig, ScoringFunction::Vina);
                let vinardo = score_complex(&rec, &lig, ScoringFunction::Vinardo);
                let hybrid = score_complex(&rec, &lig, ScoringFunction::DockCraftHybrid);
                let consensus = consensus_affinity_kcal(
                    vina.total_affinity_kcal,
                    vinardo.total_affinity_kcal,
                    hyde.total_delta_g_kcal,
                );
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "vina": vina,
                        "vinardo": vinardo,
                        "hybrid": hybrid,
                        "hyde": hyde,
                        "consensus_dg_kcal": consensus,
                    }))?
                );
            } else {
                println!("{}", serde_json::to_string_pretty(&hyde)?);
            }
        }
        Commands::Chimerax {
            receptor,
            ligand,
            name,
            out_dir,
            launch,
        } => {
            let rec = read_receptor_file(&receptor, true)?;
            let lig_raw = std::fs::read_to_string(&ligand)?;
            let mut ligs = parse_sdf_3d_with_limit(&lig_raw, 1);
            let mut lig = ligs.pop().ok_or("empty ligand SDF")??;
            struct_core::perceive_all(&mut lig);
            let hyde = score_pose_hyde(&rec, &lig)?;
            let complex_name = name.unwrap_or_else(|| {
                if lig.name.trim().is_empty() {
                    "DockCraftComplex".to_string()
                } else {
                    lig.name.clone()
                }
            });
            let bundle = if launch {
                launch_chimerax_for_scene(&complex_name, &rec, &lig, &hyde, out_dir.as_deref())?
            } else {
                write_chimerax_session_bundle(&complex_name, &rec, &lig, &hyde, out_dir.as_deref())?
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "scene_dir": bundle.scene_dir.display().to_string(),
                    "script": bundle.script_path.display().to_string(),
                    "receptor": bundle.receptor_path.display().to_string(),
                    "ligand": bundle.ligand_path.display().to_string(),
                    "bundle": bundle.bundle_path.display().to_string(),
                    "chimerax_command": bundle.open_command,
                    "chimerax_on_path": is_chimerax_available(),
                    "launched": launch,
                }))?
            );
        }
        Commands::MdValidate {
            check_env,
            receptor,
            ligand,
            out_dir,
            ns,
            nvt_ps,
            npt_ps,
            temp_k,
            padding_nm,
            charges,
            platform,
            python,
            dry_run,
        } => {
            if check_env {
                let status = check_openmm_env(python.as_deref());
                println!("{}", serde_json::to_string_pretty(&status)?);
                return Ok(());
            }
            let rec_p = receptor.ok_or("--receptor is required (or pass --check-env)")?;
            let lig_p = ligand.ok_or("--ligand is required (or pass --check-env)")?;
            let plat = match platform {
                CliMdPlatform::Auto => MdPlatform::Auto,
                CliMdPlatform::Cuda => MdPlatform::Cuda,
                CliMdPlatform::Opencl => MdPlatform::OpenCl,
                CliMdPlatform::Cpu => MdPlatform::Cpu,
            };
            let cfg = MdValidateConfig {
                receptor_path: rec_p,
                ligand_path: lig_p,
                out_dir,
                production_ns: ns,
                nvt_ps,
                npt_ps,
                temperature_k: temp_k,
                padding_nm,
                charge_method: charges,
                platform: plat,
                python_override: python,
                dry_run,
                ..Default::default()
            };
            let rep = run_md_validation(&cfg)?;
            println!("{}", rep.render_markdown());
        }
        Commands::Campaign {
            spec,
            init_template,
        } => {
            if let Some(tpl_path) = init_template {
                let tpl = default_cdk2_campaign_template();
                std::fs::write(&tpl_path, serde_json::to_string_pretty(&tpl)?)?;
                println!(
                    "Wrote reproducible CampaignSpec template -> {}",
                    tpl_path.display()
                );
                return Ok(());
            }
            let sp = spec.ok_or("provide --spec <campaign.json> (or --init-template <path>)")?;
            let raw = std::fs::read_to_string(&sp)?;
            let parsed: CampaignSpec = serde_json::from_str(&raw)?;
            let manifest = run_campaign(&parsed)?;
            println!("{}", manifest.render_markdown());
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

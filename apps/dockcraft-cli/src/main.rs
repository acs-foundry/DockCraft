#![forbid(unsafe_code)]

mod mcp;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use dock_engine::{
    DockingConfig, DockingConstraint, PharmacophoreKind, ScoringFunction, dock_ligand,
    score_complex,
};
use dock_pipeline::{
    ChimeraXRest, DEFAULT_MAX_FILE_BYTES, PathSandbox, campaign_report_to_csv,
    default_chemcraft_results_dir, default_chimerax_export_dir, grid_box_from_chimerax_selection,
    is_chimerax_available, launch_chimerax_for_scene, load_ligand_library_bytes,
    run_screening_campaign, write_chimerax_session_bundle, write_results_sidecar,
};
use hyde_scorer::{
    analyze_torsions, consensus_affinity_kcal, score_pose_hyde, validate_pose_geometry,
};
use pocket_detect::{GridBox, PocketDetectConfig, detect_pockets};
use struct_core::{
    detect_bridging_waters, parse_mmcif, parse_mol2, parse_pdb, parse_pdbqt,
    parse_sdf_3d_with_limit, prepare_receptor, prepare_receptor_keeping_waters,
    write_pdbqt_receptor, write_sdf_3d,
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
        /// Also write a `*.dockcraft.json` results sidecar plus the docked poses, for ChemCraft
        /// to fold back into its SAR table. Default directory:
        /// $XDG_DATA_HOME/ChemCraft/dockcraft_results
        #[arg(long)]
        sidecar: bool,
        /// Directory for the sidecar, overriding the default
        #[arg(long)]
        sidecar_dir: Option<PathBuf>,
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
        /// Pharmacophore sphere constraint `x,y,z,radius,kind` (kind: hdonor|hacceptor|aromatic|hydrophobic|metal|any)
        #[arg(long, allow_hyphen_values = true)]
        pharmacophore: Vec<String>,
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
            box_from_chimerax,
            box_padding,
            scoring,
            exhaustiveness,
            max_ligands,
            sidecar,
            sidecar_dir,
        } => {
            let rec = read_receptor_file(&receptor, true)?;
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
            } else {
                if center.len() != 3 || size.len() != 3 {
                    return Err(
                        "--center and --size must be comma-separated 3-vectors x,y,z,                          or pass --box-from-chimerax"
                            .into(),
                    );
                }
                GridBox::from_center_size(
                    [center[0], center[1], center[2]],
                    [size[0], size[1], size[2]],
                    0.375,
                )?
            };
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
            let rec = read_receptor_file(&receptor, !keep_waters)?;
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
            flex_residues,
            covalent,
            out_sdf,
        } => {
            if center.len() != 3 || size.len() != 3 {
                return Err("--center and --size must be comma-separated 3-vectors x,y,z".into());
            }
            let rec = if auto_waters && !keep_waters {
                let raw = std::fs::read_to_string(&receptor)?;
                let mut rec_raw = parse_pdb(&raw, false)?;
                let box_r = 0.5 * size[0].max(size[1]).max(size[2]);
                let bw =
                    detect_bridging_waters(&rec_raw, [center[0], center[1], center[2]], box_r, 2);
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
            let gbox = GridBox::from_center_size(
                [center[0], center[1], center[2]],
                [size[0], size[1], size[2]],
                0.375,
            )?;
            let sf = match scoring {
                CliScoring::Vina => ScoringFunction::Vina,
                CliScoring::Vinardo => ScoringFunction::Vinardo,
                CliScoring::Hybrid => ScoringFunction::DockCraftHybrid,
            };
            let mut constraints: Vec<DockingConstraint> = Vec::new();
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
            let rec = read_receptor_file(&receptor, !keep_waters)?;
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

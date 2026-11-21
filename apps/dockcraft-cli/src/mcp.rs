#![forbid(unsafe_code)]

use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use dock_engine::{DockingConfig, ScoringFunction, dock_ligand};
use dock_pipeline::{
    AntiTargetEntry, DEFAULT_MAX_FILE_BYTES, GninaRescoreConfig, LeadOptConfig, LeadOptMode,
    LoadedAntiTarget, PathSandbox, SafetyPanelConfig, SceneExportRequest, build_chimerax_md_script,
    campaign_report_to_csv, canonical_export_target, check_fep_env, check_gnina_env,
    check_openmm_env, load_ligand_library_bytes, plan_rbfe_network, run_gnina_rescore,
    run_lead_optimization, run_safety_panel_loaded, run_screening_campaign,
    write_chimerax_session_bundle, write_sandboxed_scene_package,
};
use hyde_scorer::{SubstituentPreset, mutate_and_minimize_in_pocket, score_pose_hyde};
use pharm_core::{
    OffTargetPocketInput, PharmScreenConfig, PocketSelectivityConfig, SiteAnalysisConfig,
    SiteBlueprint, analyze_binding_site, analyze_pocket_selectivity, screen_molecules_parallel,
};
use pocket_detect::{GridBox, PocketDetectConfig, detect_pockets};
use prot_prep::{
    EnsembleAggregation, EnsembleReceptor, PrepConfig, prepare_protein_from_str,
    run_ensemble_docking,
};
use serde_json::{Value, json};
use struct_core::{
    MAX_LIGAND_ATOMS, Molecule3D, parse_mmcif, parse_mol2, parse_pdb, parse_pdbqt,
    parse_sdf_3d_with_limit, parse_smiles_to_3d, prepare_receptor, write_pdb, write_sdf_3d,
};

pub const MAX_MCP_MESSAGE_BYTES: usize = 1_048_576;
pub const MAX_TCP_CONCURRENT: usize = 32;
pub const MAX_TCP_SOCKETS: usize = 128;
pub const MAX_MCP_LIBRARY_RECORDS: usize = 1000;

#[derive(Default)]
pub struct SessionState {
    pub initialized: bool,
}

fn json_err(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message.into()
        }
    })
}

fn json_ok(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn parse_receptor_by_path(
    path: &Path,
    content: &str,
    strip_waters: bool,
) -> Result<Molecule3D, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("pdb")
        .to_ascii_lowercase();
    let mut mol = match ext.as_str() {
        "cif" | "mmcif" => parse_mmcif(content, strip_waters),
        "pdbqt" => parse_pdbqt(content),
        "mol2" => parse_mol2(content),
        _ => parse_pdb(content, strip_waters),
    }
    .map_err(|e| format!("failed to parse receptor '{}': {e}", path.display()))?;
    prepare_receptor(&mut mol, strip_waters);
    Ok(mol)
}

fn parse_single_ligand(sandbox: &PathSandbox, val: &Value) -> Result<Molecule3D, String> {
    if let Some(smi) = val.get("ligand_smiles").and_then(|v| v.as_str()) {
        return parse_smiles_to_3d(smi, Some("ligand")).map_err(|e| e.to_string());
    }
    let path_str = val
        .get("ligand_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing 'ligand_path' or 'ligand_smiles'".to_string())?;
    let path = Path::new(path_str);
    let text = sandbox.read_string(path).map_err(|e| e.to_string())?;
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("sdf")
        .to_ascii_lowercase();
    let mut mol = match ext.as_str() {
        "mol2" => parse_mol2(&text).map_err(|e| e.to_string())?,
        "pdbqt" => parse_pdbqt(&text).map_err(|e| e.to_string())?,
        "pdb" => parse_pdb(&text, false).map_err(|e| e.to_string())?,
        _ => {
            let mut list = parse_sdf_3d_with_limit(&text, 1);
            list.pop()
                .ok_or_else(|| "empty SDF".to_string())?
                .map_err(|e| e.to_string())?
        }
    };
    if mol.atoms.len() > MAX_LIGAND_ATOMS {
        return Err(format!(
            "ligand atom count {} exceeds limit {}",
            mol.atoms.len(),
            MAX_LIGAND_ATOMS
        ));
    }
    struct_core::perceive_all(&mut mol);
    Ok(mol)
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "detect_pockets",
            "description": "Detects druggable binding pockets in a receptor inside read_root and ranks them by druggability score, volume, and SASA.",
            "inputSchema": {
                "type": "object",
                "required": ["receptor_path"],
                "properties": {
                    "receptor_path": { "type": "string" },
                    "strip_waters": { "type": "boolean" },
                    "max_pockets": { "type": "integer", "minimum": 1, "maximum": 20 }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["pocket_count", "pockets"],
                "properties": {
                    "pocket_count": { "type": "integer" },
                    "pockets": { "type": "array" }
                }
            }
        },
        {
            "name": "run_docking",
            "description": "Runs single-ligand docking or multi-ligand PyRx-style virtual screening inside the sandboxed GridBox.",
            "inputSchema": {
                "type": "object",
                "required": ["receptor_path", "box_center", "box_size"],
                "properties": {
                    "receptor_path": { "type": "string" },
                    "ligand_path": { "type": "string" },
                    "ligand_smiles": { "type": "string" },
                    "library_path": { "type": "string" },
                    "box_center": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "box_size": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "scoring": { "type": "string", "enum": ["vina", "vinardo", "hybrid"] },
                    "exhaustiveness": { "type": "integer", "minimum": 1, "maximum": 128 },
                    "out_sdf_path": { "type": "string" },
                    "out_csv_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["mode", "results"],
                "properties": {
                    "mode": { "type": "string" },
                    "results": { "type": "array" }
                }
            }
        },
        {
            "name": "optimize_ligand_in_pocket",
            "description": "Scores a complex with HYDE per-atom coronas and optionally applies an in-pocket substituent mutation (-H,-F,-Cl,-OH,-NH2,-CH3,-CF3,-CN,-OCH3).",
            "inputSchema": {
                "type": "object",
                "required": ["receptor_path"],
                "properties": {
                    "receptor_path": { "type": "string" },
                    "ligand_path": { "type": "string" },
                    "anchor_atom_index": { "type": "integer", "minimum": 0 },
                    "substituent": { "type": "string", "enum": ["H", "F", "Cl", "OH", "NH2", "CH3", "CF3", "CN", "OCH3"] },
                    "out_sdf_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["total_delta_g_kcal", "estimated_ki_nm", "atom_scores"],
                "properties": {
                    "total_delta_g_kcal": { "type": "number" },
                    "estimated_ki_nm": { "type": "number" },
                    "atom_scores": { "type": "array" }
                }
            }
        },
        {
            "name": "export_complex",
            "description": "Exports the docked complex and HYDE annotations as a ChimeraX scene (.cxc) or a PyMOL script (.pml) inside write_root. With chimerax_session=true the scene goes instead to the ChimeraX export directory, which is the only place ChimeraX's Flatpak sandbox can read it; out_script_path is then ignored. The legacy target names 'chymera' and 'molcraft' are still accepted.",
            "inputSchema": {
                "type": "object",
                "required": ["receptor_path", "ligand_path", "target"],
                "properties": {
                    "receptor_path": { "type": "string" },
                    "ligand_path": { "type": "string" },
                    "target": { "type": "string", "enum": ["chimerax", "pymol", "chymera", "molcraft", "pdb", "sdf"] },
                    "out_script_path": { "type": "string" },
                    "out_bundle_path": { "type": "string" },
                    "chimerax_session": { "type": "boolean" },
                    "complex_name": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["target", "written_files"],
                "properties": {
                    "target": { "type": "string" },
                    "written_files": { "type": "array" }
                }
            }
        },
        {
            "name": "analyze_binding_site",
            "description": "Profiles an active site with 7-probe 3D Molecular Interaction Fields (MIFs), PCA physical dimensions, optimal inter-feature distance matrix, exclusion spheres, and water assessment; optionally writes .pharm.json and Markdown blueprint inside write_root.",
            "inputSchema": {
                "type": "object",
                "required": ["receptor_path"],
                "properties": {
                    "receptor_path": { "type": "string" },
                    "ligand_path": { "type": "string" },
                    "box_center": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "box_size": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "keep_waters": { "type": "boolean" },
                    "max_features": { "type": "integer", "minimum": 2, "maximum": 20 },
                    "out_blueprint_path": { "type": "string" },
                    "out_markdown_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["geometry", "features", "distance_matrix"],
                "properties": {
                    "geometry": { "type": "object" },
                    "features": { "type": "array" },
                    "distance_matrix": { "type": "array" }
                }
            }
        },
        {
            "name": "pharmacophore_screen",
            "description": "Runs ultra-fast 3D pharmacophore + exclusion-volume screening of a ligand library against a .pharm.json blueprint inside read_root/write_root.",
            "inputSchema": {
                "type": "object",
                "required": ["blueprint_path", "library_path"],
                "properties": {
                    "blueprint_path": { "type": "string" },
                    "library_path": { "type": "string" },
                    "min_match": { "type": "integer", "minimum": 1, "maximum": 12 },
                    "max_rmsd": { "type": "number" },
                    "max_clashes": { "type": "integer", "minimum": 0, "maximum": 20 },
                    "max_hits": { "type": "integer", "minimum": 1, "maximum": 1000 },
                    "out_sdf_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["total_evaluated", "passed_count", "hits"],
                "properties": {
                    "total_evaluated": { "type": "integer" },
                    "passed_count": { "type": "integer" },
                    "hits": { "type": "array" }
                }
            }
        },
        {
            "name": "prepare_protein",
            "description": "Audits and prepares a PDB/mmCIF/AlphaFold receptor inside read_root: resolves highest-occupancy altlocs, audits missing loops/atoms/clashes/B-factors, assigns PROPKA-inspired local pKa protonation states (HID/HIE/HIP/ASH/GLH/LYN/CYM/CYX), optimizes 180° ASN/GLN/HIS H-bond flips & polar-H torsions, and writes prepared PDB/Markdown inside write_root.",
            "inputSchema": {
                "type": "object",
                "required": ["receptor_path"],
                "properties": {
                    "receptor_path": { "type": "string" },
                    "ph": { "type": "number" },
                    "chains": { "type": "array", "items": { "type": "string" } },
                    "keep_cofactors": { "type": "boolean" },
                    "keep_metals": { "type": "boolean" },
                    "keep_bridging_waters": { "type": "boolean" },
                    "optimize_hbonds": { "type": "boolean" },
                    "out_pdb_path": { "type": "string" },
                    "out_markdown_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["audit", "protonation", "hbond_opt"],
                "properties": {
                    "audit": { "type": "object" },
                    "protonation": { "type": "object" },
                    "hbond_opt": { "type": "object" }
                }
            }
        },
        {
            "name": "ensemble_dock",
            "description": "Runs parallel multi-conformation ensemble docking across 2..8 receptor conformations inside read_root with Boltzmann / Best / Mean / RRF consensus aggregation.",
            "inputSchema": {
                "type": "object",
                "required": ["receptor_paths", "box_center", "box_size"],
                "properties": {
                    "receptor_paths": { "type": "array", "items": { "type": "string" }, "minItems": 1, "maxItems": 8 },
                    "ligand_smiles": { "type": "string" },
                    "ligand_path": { "type": "string" },
                    "library_path": { "type": "string" },
                    "box_center": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "box_size": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "aggregation": { "type": "string", "enum": ["boltzmann", "best", "mean", "rrf"] },
                    "exhaustiveness": { "type": "integer", "minimum": 1, "maximum": 32 },
                    "out_csv_path": { "type": "string" },
                    "out_sdf_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["receptor_labels", "candidates"],
                "properties": {
                    "receptor_labels": { "type": "array" },
                    "candidates": { "type": "array" }
                }
            }
        },
        {
            "name": "prepare_md_validation",
            "description": "Probes the isolated OpenMM GPU environment and stages a sandboxed explicit-solvent OpenMM MD validation bundle (config JSON + ChimeraX trajectory script) inside write_root without executing arbitrary host scripts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "receptor_path": { "type": "string" },
                    "ligand_path": { "type": "string" },
                    "production_ns": { "type": "number" },
                    "out_config_path": { "type": "string" },
                    "out_chimerax_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["env_status"],
                "properties": {
                    "env_status": { "type": "object" }
                }
            }
        },
        {
            "name": "compare_pockets",
            "description": "Compare target binding pocket vs 1..4 off-target/paralog receptors via rigid pocket alignment and 7-probe ΔMIF + steric-exclusivity selectivity analysis.",
            "inputSchema": {
                "type": "object",
                "required": ["target_path", "offtarget_paths", "box_center", "box_size"],
                "properties": {
                    "target_path": { "type": "string" },
                    "offtarget_paths": { "type": "array", "items": { "type": "string" } },
                    "box_center": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "box_size": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "pre_aligned": { "type": "boolean" },
                    "out_markdown_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["selectivity_feasibility_score", "offtargets", "selectivity_hotspots"],
                "properties": {
                    "selectivity_feasibility_score": { "type": "number" },
                    "offtargets": { "type": "array" },
                    "selectivity_hotspots": { "type": "array" }
                }
            }
        },
        {
            "name": "run_safety_panel",
            "description": "Run Inverse Docking Safety Pharmacology screening of a ligand SMILES against 1..15 sandboxed anti-target receptor PDBs.",
            "inputSchema": {
                "type": "object",
                "required": ["anti_target_paths", "ligand_smiles", "box_center", "box_size"],
                "properties": {
                    "anti_target_paths": { "type": "array", "items": { "type": "string" } },
                    "ligand_smiles": { "type": "string" },
                    "box_center": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "box_size": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
                    "target_dg_kcal": { "type": "number" },
                    "exhaustiveness": { "type": "integer", "minimum": 1, "maximum": 16 },
                    "out_markdown_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["targets_evaluated", "summaries"],
                "properties": {
                    "targets_evaluated": { "type": "integer" },
                    "summaries": { "type": "array" }
                }
            }
        },
        {
            "name": "run_lead_optimization",
            "description": "Runs closed-loop 3D in-pocket lead optimization (substituent growth + optional ΔMIF selectivity + PoseBusters + Pareto ranking) inside read_root/write_root and optionally writes aligned congeneric series SDF & Markdown dossier.",
            "inputSchema": {
                "type": "object",
                "required": ["receptor_path", "ligand_path"],
                "properties": {
                    "receptor_path": { "type": "string" },
                    "ligand_path": { "type": "string" },
                    "selectivity_json_path": { "type": "string" },
                    "mode": { "type": "string", "enum": ["combined", "pocket-growth", "som-blocking"] },
                    "tpp_profile": { "type": "string", "enum": ["oral", "cns", "peripheral", "oncology"] },
                    "max_vectors": { "type": "integer", "minimum": 1, "maximum": 8 },
                    "max_analogs": { "type": "integer", "minimum": 2, "maximum": 40 },
                    "use_external_clis": { "type": "boolean" },
                    "out_series_sdf_path": { "type": "string" },
                    "out_markdown_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["parent_name", "valid_unique_analogs", "analogs"],
                "properties": {
                    "parent_name": { "type": "string" },
                    "valid_unique_analogs": { "type": "integer" },
                    "analogs": { "type": "array" }
                }
            }
        },
        {
            "name": "prepare_fep_network",
            "description": "Probes OpenMM/pymbar/OpenFE RBFE availability, plans a star + cycle-closure 3D perturbation network from 2..12 aligned congeneric ligands in read_root, and optionally stages config JSON & report inside write_root.",
            "inputSchema": {
                "type": "object",
                "required": ["ligands_sdf_path"],
                "properties": {
                    "receptor_path": { "type": "string" },
                    "offtarget_receptor_path": { "type": "string" },
                    "ligands_sdf_path": { "type": "string" },
                    "n_windows": { "type": "integer", "minimum": 5, "maximum": 21 },
                    "ns_per_window": { "type": "number" },
                    "max_edges": { "type": "integer", "minimum": 1, "maximum": 24 },
                    "out_config_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["env_status", "planned_edges"],
                "properties": {
                    "env_status": { "type": "object" },
                    "planned_edges": { "type": "array" }
                }
            }
        },
        {
            "name": "rescore_poses_cnn",
            "description": "Probes opt-in GNINA CNN binary availability and optionally runs GNINA --score_only 3-way consensus rescoring (Vina + HYDE + CNN) on sandboxed receptor & poses SDF inside read_root/write_root.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "receptor_path": { "type": "string" },
                    "poses_sdf_path": { "type": "string" },
                    "force_cpu": { "type": "boolean" },
                    "out_csv_path": { "type": "string" }
                }
            },
            "outputSchema": {
                "type": "object",
                "required": ["env_status"],
                "properties": {
                    "env_status": { "type": "object" }
                }
            }
        }
    ])
}

fn parse_vec3(val: Option<&Value>, field: &str) -> Result<[f64; 3], String> {
    let arr = val
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("missing array parameter '{field}'"))?;
    if arr.len() != 3 {
        return Err(format!("'{field}' must have length 3"));
    }
    let x = arr[0]
        .as_f64()
        .ok_or_else(|| format!("invalid number in '{field}[0]'"))?;
    let y = arr[1]
        .as_f64()
        .ok_or_else(|| format!("invalid number in '{field}[1]'"))?;
    let z = arr[2]
        .as_f64()
        .ok_or_else(|| format!("invalid number in '{field}[2]'"))?;
    Ok([x, y, z])
}

fn handle_tool_call(
    sandbox: &PathSandbox,
    name: &str,
    args: &Value,
) -> Result<Value, (i64, String)> {
    match name {
        "detect_pockets" => {
            let rec_path = args
                .get("receptor_path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'receptor_path'".to_string()))?;
            let strip = args
                .get("strip_waters")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let max_p = args
                .get("max_pockets")
                .and_then(|v| v.as_u64())
                .unwrap_or(8)
                .clamp(1, 20) as usize;
            let raw = sandbox
                .read_string(Path::new(rec_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let rec = parse_receptor_by_path(Path::new(rec_path), &raw, strip)
                .map_err(|e| (-32602, e))?;
            let mut pockets = detect_pockets(&rec, &PocketDetectConfig::default())
                .map_err(|e| (-32000, e.to_string()))?;
            pockets.truncate(max_p);
            let compact: Vec<Value> = pockets
                .iter()
                .map(|p| {
                    json!({
                        "rank": p.rank,
                        "center": p.center,
                        "grid_box": p.grid_box,
                        "volume_a3": p.volume_a3,
                        "sasa_a2": p.sasa_a2,
                        "druggability_score": p.druggability_score,
                        "hydrophobicity": p.hydrophobicity,
                        "lining_residues_count": p.lining_residues.len()
                    })
                })
                .collect();
            Ok(json!({
                "pocket_count": compact.len(),
                "pockets": compact
            }))
        }
        "run_docking" => {
            let rec_path = args
                .get("receptor_path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'receptor_path'".to_string()))?;
            let center =
                parse_vec3(args.get("box_center"), "box_center").map_err(|e| (-32602, e))?;
            let size = parse_vec3(args.get("box_size"), "box_size").map_err(|e| (-32602, e))?;
            let gbox = GridBox::from_center_size(center, size, 0.375)
                .map_err(|e| (-32602, e.to_string()))?;

            let rec_raw = sandbox
                .read_string(Path::new(rec_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let rec = parse_receptor_by_path(Path::new(rec_path), &rec_raw, true)
                .map_err(|e| (-32602, e))?;

            let scoring = match args.get("scoring").and_then(|v| v.as_str()) {
                Some("vinardo") => ScoringFunction::Vinardo,
                Some("hybrid") => ScoringFunction::DockCraftHybrid,
                _ => ScoringFunction::Vina,
            };
            // The engine already raises the walker count on its own for flexible ligands, so the
            // default stays modest. The ceiling used to be 32, which sat just below the regime
            // where a 13-rotatable-bond ligand converges; cost grows linearly in this number and
            // the per-request caps bound the rest, so 128 is affordable and no longer truncates
            // a deliberate request for a deeper search.
            let exhaustiveness = args
                .get("exhaustiveness")
                .and_then(|v| v.as_u64())
                .unwrap_or(6)
                .clamp(1, 128) as usize;
            let cfg = DockingConfig {
                scoring,
                exhaustiveness,
                ..Default::default()
            };

            if let Some(lib_path) = args.get("library_path").and_then(|v| v.as_str()) {
                let bytes = sandbox
                    .read_bytes(Path::new(lib_path))
                    .map_err(|e| (-32602, e.to_string()))?;
                let ext = Path::new(lib_path)
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("smi");
                let lib = load_ligand_library_bytes(ext, &bytes, MAX_MCP_LIBRARY_RECORDS)
                    .map_err(|e| (-32602, e.to_string()))?;
                let report = run_screening_campaign(&rec, &gbox, &lib, &cfg)
                    .map_err(|e| (-32000, e.to_string()))?;
                if let Some(out_csv) = args.get("out_csv_path").and_then(|v| v.as_str()) {
                    sandbox
                        .write_string(Path::new(out_csv), &campaign_report_to_csv(&report))
                        .map_err(|e| (-32602, e.to_string()))?;
                }
                let rows: Vec<Value> = report
                    .results
                    .iter()
                    .map(|c| {
                        json!({
                            "rank": c.rank,
                            "name": c.name,
                            "affinity_kcal_mol": c.best_affinity_kcal,
                            "hyde_delta_g_kcal": c.hyde_delta_g_kcal,
                            "mpo_score": c.mpo_score,
                            "admet_score": c.admet.admet_score
                        })
                    })
                    .collect();
                Ok(json!({
                    "mode": "screening",
                    "enrichment": report.enrichment,
                    "results": rows
                }))
            } else {
                let lig = parse_single_ligand(sandbox, args).map_err(|e| (-32602, e))?;
                let report =
                    dock_ligand(&rec, &lig, &gbox, &cfg).map_err(|e| (-32000, e.to_string()))?;
                if let Some(out_sdf) = args.get("out_sdf_path").and_then(|v| v.as_str()) {
                    let refs: Vec<&Molecule3D> = report.poses.iter().map(|p| &p.molecule).collect();
                    sandbox
                        .write_string(Path::new(out_sdf), &write_sdf_3d(&refs))
                        .map_err(|e| (-32602, e.to_string()))?;
                }
                let poses: Vec<Value> = report
                    .poses
                    .iter()
                    .map(|p| {
                        json!({
                            "mode": p.mode,
                            "affinity_kcal_mol": p.affinity_kcal_mol,
                            "rmsd_to_best": p.terms.rmsd_to_best
                        })
                    })
                    .collect();
                Ok(json!({
                    "mode": "single_ligand",
                    "results": poses
                }))
            }
        }
        "optimize_ligand_in_pocket" => {
            let rec_path = args
                .get("receptor_path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'receptor_path'".to_string()))?;
            let rec_raw = sandbox
                .read_string(Path::new(rec_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let rec = parse_receptor_by_path(Path::new(rec_path), &rec_raw, true)
                .map_err(|e| (-32602, e))?;
            let lig = parse_single_ligand(sandbox, args).map_err(|e| (-32602, e))?;

            if let Some(sub_str) = args.get("substituent").and_then(|v| v.as_str()) {
                let anchor = args
                    .get("anchor_atom_index")
                    .and_then(|v| v.as_u64())
                    .ok_or((
                        -32602,
                        "missing 'anchor_atom_index' for substituent mutation".to_string(),
                    ))? as usize;
                let preset = match sub_str {
                    "H" => SubstituentPreset::H,
                    "F" => SubstituentPreset::F,
                    "Cl" => SubstituentPreset::Cl,
                    "OH" => SubstituentPreset::OH,
                    "NH2" => SubstituentPreset::NH2,
                    "CH3" => SubstituentPreset::CH3,
                    "CF3" => SubstituentPreset::CF3,
                    "CN" => SubstituentPreset::CN,
                    "OCH3" => SubstituentPreset::OCH3,
                    other => return Err((-32602, format!("unknown substituent '{other}'"))),
                };
                let outcome = mutate_and_minimize_in_pocket(&rec, &lig, anchor, preset)
                    .map_err(|e| (-32602, e.to_string()))?;
                if let Some(out_sdf) = args.get("out_sdf_path").and_then(|v| v.as_str()) {
                    sandbox
                        .write_string(
                            Path::new(out_sdf),
                            &write_sdf_3d(&[&outcome.mutated_ligand]),
                        )
                        .map_err(|e| (-32602, e.to_string()))?;
                }
                Ok(json!({
                    "before_kcal": outcome.before_kcal,
                    "after_kcal": outcome.after_kcal,
                    "delta_delta_g_kcal": outcome.delta_delta_g_kcal,
                    "total_delta_g_kcal": outcome.report.total_delta_g_kcal,
                    "estimated_ki_nm": outcome.report.estimated_ki_nm,
                    "atom_scores": outcome.report.atom_scores,
                    "interactions": outcome.report.interactions
                }))
            } else {
                let report = score_pose_hyde(&rec, &lig).map_err(|e| (-32000, e.to_string()))?;
                Ok(serde_json::to_value(&report).map_err(|e| (-32000, e.to_string()))?)
            }
        }
        "export_complex" => {
            let rec_path = args
                .get("receptor_path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'receptor_path'".to_string()))?;
            let target = args
                .get("target")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'target'".to_string()))?;
            let chimerax_session = args
                .get("chimerax_session")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            // The ChimeraX session path owns its own directory, so it does not need a caller path
            // and must not take one: the whole point is that the scene lands inside the export
            // root that ChimeraX's sandbox allows.
            let out_script = if chimerax_session {
                ""
            } else {
                args.get("out_script_path")
                    .and_then(|v| v.as_str())
                    .ok_or((-32602, "missing 'out_script_path'".to_string()))?
            };

            let rec_raw = sandbox
                .read_string(Path::new(rec_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let rec = parse_receptor_by_path(Path::new(rec_path), &rec_raw, true)
                .map_err(|e| (-32602, e))?;
            let lig = parse_single_ligand(sandbox, args).map_err(|e| (-32602, e))?;
            let hyde = score_pose_hyde(&rec, &lig).map_err(|e| (-32000, e.to_string()))?;

            if chimerax_session {
                if canonical_export_target(target) != Some("chimerax") {
                    return Err((
                        -32602,
                        format!("chimerax_session requires target 'chimerax', got '{target}'"),
                    ));
                }
                let complex_name = args
                    .get("complex_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("DockCraftComplex");
                let bundle = write_chimerax_session_bundle(complex_name, &rec, &lig, &hyde, None)
                    .map_err(|e| (-32000, e.to_string()))?;
                return Ok(json!({
                    "target": "chimerax",
                    "written_files": [
                        bundle.script_path.display().to_string(),
                        bundle.receptor_path.display().to_string(),
                        bundle.ligand_path.display().to_string(),
                        bundle.bundle_path.display().to_string(),
                    ],
                    "scene_dir": bundle.scene_dir.display().to_string(),
                    "chimerax_command": bundle.open_command,
                }));
            }

            let written = match canonical_export_target(target).unwrap_or(target) {
                "chimerax" | "pymol" => write_sandboxed_scene_package(
                    sandbox,
                    &SceneExportRequest {
                        target,
                        complex_name: "DockCraftComplex",
                        out_script_path: Path::new(out_script),
                        out_bundle_path: args
                            .get("out_bundle_path")
                            .and_then(|v| v.as_str())
                            .map(Path::new),
                    },
                    &rec,
                    &lig,
                    &hyde,
                )
                .map_err(|e| (-32602, e.to_string()))?,
                "pdb" => {
                    sandbox
                        .write_string(Path::new(out_script), &write_pdb(&rec))
                        .map_err(|e| (-32602, e.to_string()))?;
                    vec![out_script.to_string()]
                }
                "sdf" => {
                    sandbox
                        .write_string(Path::new(out_script), &write_sdf_3d(&[&lig]))
                        .map_err(|e| (-32602, e.to_string()))?;
                    vec![out_script.to_string()]
                }
                other => return Err((-32602, format!("unsupported export target '{other}'"))),
            };
            Ok(json!({
                "target": target,
                "written_files": written
            }))
        }
        "analyze_binding_site" => {
            let rec_path = args
                .get("receptor_path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'receptor_path'".to_string()))?;
            let keep_waters = args
                .get("keep_waters")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let rec_raw = sandbox
                .read_string(Path::new(rec_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let rec = parse_receptor_by_path(Path::new(rec_path), &rec_raw, !keep_waters)
                .map_err(|e| (-32602, e))?;
            let holo_opt =
                if args.get("ligand_path").is_some() || args.get("ligand_smiles").is_some() {
                    Some(parse_single_ligand(sandbox, args).map_err(|e| (-32602, e))?)
                } else {
                    None
                };
            let gbox = if args.get("box_center").is_some() && args.get("box_size").is_some() {
                let c =
                    parse_vec3(args.get("box_center"), "box_center").map_err(|e| (-32602, e))?;
                let s = parse_vec3(args.get("box_size"), "box_size").map_err(|e| (-32602, e))?;
                GridBox::from_center_size(c, s, 0.375).map_err(|e| (-32602, e.to_string()))?
            } else if let Some(hl) = &holo_opt {
                GridBox::from_ligand(hl, 5.5).map_err(|e| (-32602, e.to_string()))?
            } else {
                let pockets = detect_pockets(&rec, &PocketDetectConfig::default())
                    .map_err(|e| (-32000, e.to_string()))?;
                pockets
                    .first()
                    .ok_or((-32000, "no druggable pocket found".to_string()))?
                    .grid_box
                    .clone()
            };
            let max_features = args
                .get("max_features")
                .and_then(|v| v.as_u64())
                .unwrap_or(10)
                .clamp(2, 20) as usize;
            let holo_slice: Vec<&Molecule3D> = holo_opt.iter().collect();
            let bp = analyze_binding_site(
                &rec,
                &gbox,
                &holo_slice,
                &SiteAnalysisConfig {
                    max_features,
                    ..Default::default()
                },
            )
            .map_err(|e| (-32000, e.to_string()))?;
            if let Some(out_bp) = args.get("out_blueprint_path").and_then(|v| v.as_str()) {
                let js = bp.to_json_pretty().map_err(|e| (-32000, e.to_string()))?;
                sandbox
                    .write_string(Path::new(out_bp), &js)
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            if let Some(out_md) = args.get("out_markdown_path").and_then(|v| v.as_str()) {
                sandbox
                    .write_string(Path::new(out_md), &bp.render_markdown())
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            Ok(json!({
                "target_name": bp.target_name,
                "geometry": bp.geometry,
                "features": bp.features,
                "distance_matrix": bp.distance_matrix,
                "waters": bp.waters,
                "exclusion_spheres_count": bp.exclusion_spheres.len(),
                "elapsed_ms": bp.elapsed_ms,
            }))
        }
        "pharmacophore_screen" => {
            let bp_path = args
                .get("blueprint_path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'blueprint_path'".to_string()))?;
            let lib_path = args
                .get("library_path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'library_path'".to_string()))?;
            let bp_raw = sandbox
                .read_string(Path::new(bp_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let bp = SiteBlueprint::from_json_str(&bp_raw).map_err(|e| (-32602, e.to_string()))?;
            let bytes = sandbox
                .read_bytes(Path::new(lib_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let ext = Path::new(lib_path)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("smi");
            let records = load_ligand_library_bytes(ext, &bytes, MAX_MCP_LIBRARY_RECORDS)
                .map_err(|e| (-32602, e.to_string()))?;
            let mols: Vec<Molecule3D> = records.into_iter().map(|r| r.molecule).collect();
            let cfg = PharmScreenConfig {
                min_matched_features: args
                    .get("min_match")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3)
                    .clamp(1, 12) as usize,
                max_rmsd_a: args
                    .get("max_rmsd")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.65),
                max_exclusion_clashes: args
                    .get("max_clashes")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3)
                    .clamp(0, 20) as usize,
                max_hits: args
                    .get("max_hits")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(200)
                    .clamp(1, 1000) as usize,
                ..Default::default()
            };
            let rep = screen_molecules_parallel(&mols, &bp, &cfg);
            if let Some(out_sdf) = args.get("out_sdf_path").and_then(|v| v.as_str()) {
                let refs: Vec<&Molecule3D> = rep.hits.iter().map(|h| &h.aligned_molecule).collect();
                sandbox
                    .write_string(Path::new(out_sdf), &write_sdf_3d(&refs))
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            let compact: Vec<Value> = rep
                .hits
                .iter()
                .take(50)
                .map(|h| {
                    json!({
                        "name": h.name,
                        "smiles": h.smiles,
                        "matched_count": h.matched_count,
                        "matched_feature_ids": h.matched_feature_ids,
                        "pharm_rmsd_a": h.pharm_rmsd_a,
                        "exclusion_clashes": h.exclusion_clashes,
                        "pharm_score": h.pharm_score,
                    })
                })
                .collect();
            Ok(json!({
                "total_evaluated": rep.total_evaluated,
                "passed_count": rep.passed_count,
                "elapsed_ms": rep.elapsed_ms,
                "hits": compact,
            }))
        }
        "prepare_protein" => {
            let rec_path = args
                .get("receptor_path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'receptor_path'".to_string()))?;
            let raw = sandbox
                .read_string(Path::new(rec_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let ext = Path::new(rec_path)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("pdb");
            let chains: Vec<char> = args
                .get("chains")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().and_then(|s| s.chars().next()))
                        .collect()
                })
                .unwrap_or_default();
            let cfg = PrepConfig {
                ph: args.get("ph").and_then(|v| v.as_f64()).unwrap_or(7.4),
                chains,
                keep_cofactors: args
                    .get("keep_cofactors")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
                keep_metals: args
                    .get("keep_metals")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
                keep_bridging_waters: args
                    .get("keep_bridging_waters")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                optimize_hbonds: args
                    .get("optimize_hbonds")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
                ..Default::default()
            };
            let prep =
                prepare_protein_from_str(&raw, ext, &cfg).map_err(|e| (-32000, e.to_string()))?;
            if let Some(out_pdb) = args.get("out_pdb_path").and_then(|v| v.as_str()) {
                sandbox
                    .write_string(Path::new(out_pdb), &write_pdb(&prep.receptor))
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            if let Some(out_md) = args.get("out_markdown_path").and_then(|v| v.as_str()) {
                sandbox
                    .write_string(Path::new(out_md), &prep.render_markdown())
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            Ok(json!({
                "audit": prep.audit,
                "protonation": prep.protonation,
                "hbond_opt": prep.hbond_opt,
                "bridging_waters": prep.bridging_waters,
                "extracted_ligand_count": prep.extracted_ligands.len(),
                "final_atom_count": prep.receptor.atoms.len(),
                "elapsed_ms": prep.elapsed_ms,
            }))
        }
        "ensemble_dock" => {
            let rec_arr = args
                .get("receptor_paths")
                .and_then(|v| v.as_array())
                .filter(|a| !a.is_empty() && a.len() <= 8)
                .ok_or((
                    -32602,
                    "'receptor_paths' must be an array of 1..8 paths".to_string(),
                ))?;
            let center =
                parse_vec3(args.get("box_center"), "box_center").map_err(|e| (-32602, e))?;
            let size = parse_vec3(args.get("box_size"), "box_size").map_err(|e| (-32602, e))?;
            let gbox = GridBox::from_center_size(center, size, 0.375)
                .map_err(|e| (-32602, e.to_string()))?;
            let mut ens_recs = Vec::with_capacity(rec_arr.len());
            for rp_val in rec_arr {
                let rp_str = rp_val
                    .as_str()
                    .ok_or((-32602, "each receptor path must be a string".to_string()))?;
                let raw = sandbox
                    .read_string(Path::new(rp_str))
                    .map_err(|e| (-32602, e.to_string()))?;
                let rec = parse_receptor_by_path(Path::new(rp_str), &raw, true)
                    .map_err(|e| (-32602, e))?;
                let label = Path::new(rp_str)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("rec")
                    .to_string();
                ens_recs.push(EnsembleReceptor {
                    label,
                    receptor: rec,
                    grid_box: gbox.clone(),
                });
            }
            let lig_records =
                if let Some(lib_str) = args.get("library_path").and_then(|v| v.as_str()) {
                    let bytes = sandbox
                        .read_bytes(Path::new(lib_str))
                        .map_err(|e| (-32602, e.to_string()))?;
                    let ext = Path::new(lib_str)
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("smi");
                    load_ligand_library_bytes(ext, &bytes, MAX_MCP_LIBRARY_RECORDS)
                        .map_err(|e| (-32602, e.to_string()))?
                } else {
                    let lig = parse_single_ligand(sandbox, args).map_err(|e| (-32602, e))?;
                    vec![dock_pipeline::LigandInputRecord {
                        name: lig.name.clone(),
                        molecule: lig,
                        is_active: None,
                    }]
                };
            let agg = match args.get("aggregation").and_then(|v| v.as_str()) {
                Some("best") => EnsembleAggregation::BestConformation,
                Some("mean") => EnsembleAggregation::Mean,
                Some("rrf") => EnsembleAggregation::ReciprocalRank,
                _ => EnsembleAggregation::Boltzmann,
            };
            let exhaustiveness = args
                .get("exhaustiveness")
                .and_then(|v| v.as_u64())
                .unwrap_or(4)
                .clamp(1, 32) as usize;
            let cfg = DockingConfig {
                exhaustiveness,
                ..Default::default()
            };
            let rep = run_ensemble_docking(&ens_recs, &lig_records, &cfg, agg, true)
                .map_err(|e| (-32000, e))?;
            if let Some(out_csv) = args.get("out_csv_path").and_then(|v| v.as_str()) {
                sandbox
                    .write_string(Path::new(out_csv), &rep.to_csv())
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            if let Some(out_sdf) = args.get("out_sdf_path").and_then(|v| v.as_str()) {
                let refs: Vec<&Molecule3D> = rep
                    .candidates
                    .iter()
                    .map(|c| &c.best_pose.molecule)
                    .collect();
                sandbox
                    .write_string(Path::new(out_sdf), &write_sdf_3d(&refs))
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            let compact: Vec<Value> = rep
                .candidates
                .iter()
                .take(50)
                .map(|c| {
                    json!({
                        "rank": c.rank,
                        "name": c.name,
                        "ensemble_score_kcal": c.ensemble_score_kcal,
                        "best_single_kcal": c.best_single_kcal,
                        "mean_kcal": c.mean_kcal,
                        "spread_std_kcal": c.spread_std_kcal,
                        "best_receptor": c.best_receptor,
                        "per_conformation": c.per_conformation,
                    })
                })
                .collect();
            Ok(json!({
                "receptor_labels": rep.receptor_labels,
                "aggregation": rep.aggregation.as_str(),
                "elapsed_ms": rep.elapsed_ms,
                "candidates": compact,
            }))
        }
        "prepare_md_validation" => {
            let env_status = check_openmm_env(None);
            let ns = args
                .get("production_ns")
                .and_then(|v| v.as_f64())
                .unwrap_or(20.0)
                .clamp(0.01, 500.0);
            if let Some(cfg_out) = args.get("out_config_path").and_then(|v| v.as_str()) {
                let rec = args
                    .get("receptor_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("receptor.pdb");
                let lig = args
                    .get("ligand_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("ligand.sdf");
                let cfg_json = serde_json::to_string_pretty(&json!({
                    "receptor_path": rec,
                    "ligand_path": lig,
                    "production_ns": ns,
                    "nvt_ps": 50.0,
                    "npt_ps": 100.0,
                    "temperature_k": 300.0,
                    "padding_nm": 1.0,
                    "ionic_strength_m": 0.15,
                    "platform": "auto"
                }))
                .map_err(|e| (-32000, e.to_string()))?;
                sandbox
                    .write_string(Path::new(cfg_out), &cfg_json)
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            if let Some(cxc_out) = args.get("out_chimerax_path").and_then(|v| v.as_str()) {
                sandbox
                    .write_string(
                        Path::new(cxc_out),
                        &build_chimerax_md_script("complex_equil.pdb", "production.dcd"),
                    )
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            Ok(json!({
                "env_status": env_status,
                "production_ns": ns,
            }))
        }
        "compare_pockets" => {
            let tgt_path = args
                .get("target_path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'target_path'".to_string()))?;
            let ot_arr = args
                .get("offtarget_paths")
                .and_then(|v| v.as_array())
                .filter(|a| !a.is_empty() && a.len() <= 4)
                .ok_or((
                    -32602,
                    "'offtarget_paths' must be an array of 1..4 paths".to_string(),
                ))?;
            let center =
                parse_vec3(args.get("box_center"), "box_center").map_err(|e| (-32602, e))?;
            let size = parse_vec3(args.get("box_size"), "box_size").map_err(|e| (-32602, e))?;
            let gbox = GridBox::from_center_size(center, size, 0.375)
                .map_err(|e| (-32602, e.to_string()))?;
            let pre_aligned = args
                .get("pre_aligned")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let tgt_raw = sandbox
                .read_string(Path::new(tgt_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let tgt_rec = parse_receptor_by_path(Path::new(tgt_path), &tgt_raw, true)
                .map_err(|e| (-32602, e))?;
            let tgt_name = Path::new(tgt_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Target");

            let mut loaded_ots = Vec::with_capacity(ot_arr.len());
            for v in ot_arr {
                let p = v
                    .as_str()
                    .ok_or((-32602, "each offtarget path must be a string".to_string()))?;
                let raw = sandbox
                    .read_string(Path::new(p))
                    .map_err(|e| (-32602, e.to_string()))?;
                let rec =
                    parse_receptor_by_path(Path::new(p), &raw, true).map_err(|e| (-32602, e))?;
                let nm = Path::new(p)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("OffTarget")
                    .to_string();
                loaded_ots.push((nm, rec));
            }
            let inputs: Vec<OffTargetPocketInput<'_>> = loaded_ots
                .iter()
                .map(|(nm, r)| OffTargetPocketInput {
                    name: nm.clone(),
                    receptor: r,
                    pocket_center: center,
                })
                .collect();
            let cfg = PocketSelectivityConfig {
                pre_aligned,
                ..Default::default()
            };
            let out = analyze_pocket_selectivity(tgt_name, &tgt_rec, &gbox, &[], &inputs, &cfg)
                .map_err(|e| (-32000, e.to_string()))?;
            if let Some(out_md) = args.get("out_markdown_path").and_then(|v| v.as_str()) {
                sandbox
                    .write_string(Path::new(out_md), &out.report.render_markdown())
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            serde_json::to_value(&out.report).map_err(|e| (-32000, e.to_string()))
        }
        "run_safety_panel" => {
            let at_arr = args
                .get("anti_target_paths")
                .and_then(|v| v.as_array())
                .filter(|a| !a.is_empty() && a.len() <= 15)
                .ok_or((
                    -32602,
                    "'anti_target_paths' must be an array of 1..15 paths".to_string(),
                ))?;
            let center =
                parse_vec3(args.get("box_center"), "box_center").map_err(|e| (-32602, e))?;
            let size = parse_vec3(args.get("box_size"), "box_size").map_err(|e| (-32602, e))?;
            let gbox = GridBox::from_center_size(center, size, 0.375)
                .map_err(|e| (-32602, e.to_string()))?;
            let lig = parse_single_ligand(sandbox, args).map_err(|e| (-32602, e))?;
            let target_dg = args.get("target_dg_kcal").and_then(|v| v.as_f64());
            let exhaustiveness = args
                .get("exhaustiveness")
                .and_then(|v| v.as_u64())
                .unwrap_or(2)
                .clamp(1, 16) as usize;

            let mut loaded = Vec::with_capacity(at_arr.len());
            for v in at_arr {
                let p = v
                    .as_str()
                    .ok_or((-32602, "each anti-target path must be a string".to_string()))?;
                let raw = sandbox
                    .read_string(Path::new(p))
                    .map_err(|e| (-32602, e.to_string()))?;
                let rec =
                    parse_receptor_by_path(Path::new(p), &raw, true).map_err(|e| (-32602, e))?;
                let nm = Path::new(p)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("AntiTarget")
                    .to_string();
                loaded.push(LoadedAntiTarget {
                    entry: AntiTargetEntry {
                        id: nm.clone(),
                        name: nm,
                        pdb_id: "MCP".to_string(),
                        chains: vec!['A'],
                        prepared_pdb: Path::new(p).to_path_buf(),
                        box_center: center,
                        box_size: size,
                        organ_system: "Safety Panel".to_string(),
                        clinical_liability: "Sandboxed MCP anti-target".to_string(),
                        alert_dg_kcal: -8.5,
                        anchor_res_seqs: Vec::new(),
                        enabled: true,
                    },
                    receptor: rec,
                    grid_box: gbox.clone(),
                });
            }
            let lig_rec = dock_pipeline::LigandInputRecord {
                name: lig.name.clone(),
                molecule: lig,
                is_active: None,
            };
            let cfg = SafetyPanelConfig {
                exhaustiveness,
                target_affinity_kcal: target_dg,
                ..Default::default()
            };
            let report = run_safety_panel_loaded(&[lig_rec], &loaded, "mcp-sandbox", &cfg)
                .map_err(|e| (-32000, e.to_string()))?;
            if let Some(out_md) = args.get("out_markdown_path").and_then(|v| v.as_str()) {
                sandbox
                    .write_string(Path::new(out_md), &report.render_markdown())
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            serde_json::to_value(&report).map_err(|e| (-32000, e.to_string()))
        }
        "run_lead_optimization" => {
            let rec_path = args
                .get("receptor_path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'receptor_path'".to_string()))?;
            let rec_raw = sandbox
                .read_string(Path::new(rec_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let mut rec = parse_receptor_by_path(Path::new(rec_path), &rec_raw, true)
                .map_err(|e| (-32602, e))?;
            if let Some(stem) = Path::new(rec_path).file_stem().and_then(|s| s.to_str()) {
                rec.name = stem.to_string();
            }
            let lig = parse_single_ligand(sandbox, args).map_err(|e| (-32602, e))?;

            let mode = match args.get("mode").and_then(|v| v.as_str()) {
                Some("pocket-growth") => LeadOptMode::PocketGrowth,
                Some("som-blocking") => LeadOptMode::SomBlocking,
                _ => LeadOptMode::Combined,
            };
            let tpp_profile = args
                .get("tpp_profile")
                .and_then(|v| v.as_str())
                .unwrap_or("oncology")
                .to_string();
            let max_vectors = args
                .get("max_vectors")
                .and_then(|v| v.as_u64())
                .unwrap_or(4)
                .clamp(1, 8) as usize;
            let max_analogs = args
                .get("max_analogs")
                .and_then(|v| v.as_u64())
                .unwrap_or(20)
                .clamp(2, 40) as usize;
            let use_external_craft_clis = args
                .get("use_external_clis")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let stage_dir = std::env::temp_dir().join(format!(
                "dockcraft_mcp_leadopt_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos())
            ));
            let _ = std::fs::create_dir_all(&stage_dir);

            let selectivity_json =
                if let Some(sel_p) = args.get("selectivity_json_path").and_then(|v| v.as_str()) {
                    let sel_raw = sandbox
                        .read_string(Path::new(sel_p))
                        .map_err(|e| (-32602, e.to_string()))?;
                    let tmp_sel = stage_dir.join("selectivity_staged.json");
                    std::fs::write(&tmp_sel, sel_raw).map_err(|e| (-32000, e.to_string()))?;
                    Some(tmp_sel)
                } else {
                    None
                };

            let cfg = LeadOptConfig {
                mode,
                tpp_profile,
                selectivity_json,
                max_vectors,
                max_analogs,
                top_retro: 4,
                use_external_craft_clis,
                out_dir: stage_dir.clone(),
            };
            let report =
                run_lead_optimization(&rec, &lig, &cfg).map_err(|e| (-32000, e.to_string()))?;

            if let Some(out_sdf) = args.get("out_series_sdf_path").and_then(|v| v.as_str()) {
                let series_raw = std::fs::read_to_string(&report.series_sdf_path)
                    .map_err(|e| (-32000, e.to_string()))?;
                sandbox
                    .write_string(Path::new(out_sdf), &series_raw)
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            if let Some(out_md) = args.get("out_markdown_path").and_then(|v| v.as_str()) {
                sandbox
                    .write_string(Path::new(out_md), &report.render_markdown())
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            let _ = std::fs::remove_dir_all(&stage_dir);
            serde_json::to_value(&report).map_err(|e| (-32000, e.to_string()))
        }
        "prepare_fep_network" => {
            let env_status = check_fep_env(None);
            let ligs_path = args
                .get("ligands_sdf_path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "missing 'ligands_sdf_path'".to_string()))?;
            let sdf_raw = sandbox
                .read_string(Path::new(ligs_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let ligs: Vec<Molecule3D> = parse_sdf_3d_with_limit(&sdf_raw, 12)
                .into_iter()
                .filter_map(Result::ok)
                .collect();
            if ligs.len() < 2 {
                return Err((
                    -32602,
                    format!("need at least 2 valid 3D ligands in '{ligs_path}'"),
                ));
            }
            let n_windows = args
                .get("n_windows")
                .and_then(|v| v.as_u64())
                .unwrap_or(11)
                .clamp(5, 21) as usize;
            let ns_per_window = args
                .get("ns_per_window")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5)
                .clamp(0.01, 20.0);
            let max_edges = args
                .get("max_edges")
                .and_then(|v| v.as_u64())
                .unwrap_or(12)
                .clamp(1, 24) as usize;
            let planned_edges = plan_rbfe_network(&ligs, max_edges);
            if let Some(out_cfg) = args.get("out_config_path").and_then(|v| v.as_str()) {
                let rec_p = args
                    .get("receptor_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("receptor.pdb");
                let ot_p = args.get("offtarget_receptor_path").and_then(|v| v.as_str());
                let cfg_str = serde_json::to_string_pretty(&json!({
                    "receptor_path": rec_p,
                    "offtarget_receptor_path": ot_p,
                    "ligands_sdf_path": ligs_path,
                    "n_windows": n_windows,
                    "ns_per_window": ns_per_window,
                    "planned_edges": planned_edges,
                }))
                .map_err(|e| (-32000, e.to_string()))?;
                sandbox
                    .write_string(Path::new(out_cfg), &cfg_str)
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            Ok(json!({
                "env_status": env_status,
                "ligand_count": ligs.len(),
                "n_windows": n_windows,
                "ns_per_window": ns_per_window,
                "planned_edges": planned_edges,
            }))
        }
        "rescore_poses_cnn" => {
            let env_status = check_gnina_env(None);
            let Some(rec_path) = args.get("receptor_path").and_then(|v| v.as_str()) else {
                return Ok(json!({ "env_status": env_status }));
            };
            let Some(poses_path) = args.get("poses_sdf_path").and_then(|v| v.as_str()) else {
                return Ok(json!({ "env_status": env_status }));
            };
            // Always enforce sandbox read validation before probing/executing GNINA
            let rec_raw = sandbox
                .read_string(Path::new(rec_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let poses_raw = sandbox
                .read_string(Path::new(poses_path))
                .map_err(|e| (-32602, e.to_string()))?;
            let rec = parse_receptor_by_path(Path::new(rec_path), &rec_raw, true)
                .map_err(|e| (-32602, e))?;
            if !env_status.available {
                return Ok(json!({
                    "env_status": env_status,
                    "executed": false,
                    "note": "GNINA binary is not installed; install at CRAFT_DATA_DIR/envs/gnina/gnina to execute live CNN rescoring."
                }));
            }
            let stage_dir = std::env::temp_dir().join(format!(
                "dockcraft_mcp_cnn_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos())
            ));
            let _ = std::fs::create_dir_all(&stage_dir);
            let tmp_rec = stage_dir.join("rec.pdb");
            let tmp_poses = stage_dir.join("poses.sdf");
            std::fs::write(&tmp_rec, &rec_raw).map_err(|e| (-32000, e.to_string()))?;
            std::fs::write(&tmp_poses, &poses_raw).map_err(|e| (-32000, e.to_string()))?;
            let force_cpu = args
                .get("force_cpu")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let cfg = GninaRescoreConfig {
                receptor_path: tmp_rec,
                poses_sdf_path: tmp_poses,
                out_sdf_path: None,
                out_csv_path: None,
                force_cpu,
                gnina_override: None,
            };
            let report =
                run_gnina_rescore(&cfg, Some(&rec)).map_err(|e| (-32000, e.to_string()))?;
            if let Some(out_csv) = args.get("out_csv_path").and_then(|v| v.as_str()) {
                sandbox
                    .write_string(Path::new(out_csv), &report.to_csv())
                    .map_err(|e| (-32602, e.to_string()))?;
            }
            let _ = std::fs::remove_dir_all(&stage_dir);
            Ok(json!({
                "env_status": env_status,
                "executed": true,
                "report": report,
            }))
        }
        other => Err((-32601, format!("unknown tool '{other}'"))),
    }
}

pub fn handle_mcp_line(
    sandbox: &PathSandbox,
    state: &mut SessionState,
    raw_line: &str,
) -> Option<Value> {
    if raw_line.len() > MAX_MCP_MESSAGE_BYTES {
        return Some(json_err(Value::Null, -32600, "message exceeds 1 MiB limit"));
    }
    let req: Value = match serde_json::from_str(raw_line) {
        Ok(v) => v,
        Err(e) => return Some(json_err(Value::Null, -32700, format!("parse error: {e}"))),
    };
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = match req.get("method").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return Some(json_err(id, -32600, "missing 'method'")),
    };

    let sandbox_ref = sandbox;
    let params = req.get("params").cloned().unwrap_or(Value::Null);

    if method == "initialize" {
        state.initialized = true;
        return Some(json_ok(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "dockcraft-mcp",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        ));
    }

    if method == "ping" {
        return Some(json_ok(id, json!({})));
    }

    if !state.initialized {
        return Some(json_err(
            id,
            -32002,
            "session not initialized: call 'initialize' first",
        ));
    }

    if method == "notifications/initialized" {
        return None;
    }

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match method.as_str() {
        "tools/list" => json_ok(id.clone(), json!({ "tools": tool_definitions() })),
        "tools/call" => {
            let Some(tname) = params.get("name").and_then(|v| v.as_str()) else {
                return json_err(id.clone(), -32602, "missing tool 'name'");
            };
            let targs = params.get("arguments").cloned().unwrap_or(json!({}));
            match handle_tool_call(sandbox_ref, tname, &targs) {
                Ok(structured) => json_ok(
                    id.clone(),
                    json!({
                        "content": [{ "type": "text", "text": structured.to_string() }],
                        "structuredContent": structured
                    }),
                ),
                Err((code, msg)) => json_err(id.clone(), code, msg),
            }
        }
        other => json_err(id.clone(), -32601, format!("method '{other}' not found")),
    }));

    Some(match res {
        Ok(v) => v,
        Err(_) => json_err(id, -32603, "internal tool panic isolated"),
    })
}

fn read_bounded_line<R: Read>(
    reader: &mut BufReader<R>,
    cap: usize,
) -> std::io::Result<Option<String>> {
    let mut buf = Vec::new();
    let n = reader
        .by_ref()
        .take((cap + 1) as u64)
        .read_until(b'\n', &mut buf)?;
    if n == 0 {
        return Ok(None);
    }
    if buf.len() > cap {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "message exceeds 1 MiB limit",
        ));
    }
    let s = String::from_utf8(buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(Some(s.trim_end_matches(['\r', '\n']).to_string()))
}

pub fn serve_stdio(sandbox: &PathSandbox) -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut out = stdout.lock();
    let mut state = SessionState::default();

    loop {
        match read_bounded_line(&mut reader, MAX_MCP_MESSAGE_BYTES) {
            Ok(Some(line)) => {
                if line.trim().is_empty() {
                    continue;
                }
                if let Some(resp) = handle_mcp_line(sandbox, &mut state, &line) {
                    writeln!(out, "{resp}")?;
                    out.flush()?;
                }
            }
            Ok(None) => break,
            Err(e) => {
                let err = json_err(Value::Null, -32600, e.to_string());
                writeln!(out, "{err}")?;
                out.flush()?;
                break;
            }
        }
    }
    Ok(())
}

struct TcpConn {
    id: u64,
    stream: TcpStream,
    read_buf: Vec<u8>,
    authed: bool,
    state: SessionState,
    in_flight: bool,
    last_active: Instant,
}

struct WorkerDone {
    conn_id: u64,
    state: SessionState,
    response: Option<Value>,
}

pub fn serve_tcp(sandbox: PathSandbox, bind_addr: &str, auth_token: &str) -> std::io::Result<()> {
    if auth_token.trim().is_empty() {
        return Err(std::io::Error::new(
            ErrorKind::InvalidInput,
            "--auth-token must be non-empty for TCP transport",
        ));
    }
    let addr: SocketAddr = bind_addr.parse().map_err(|e| {
        std::io::Error::new(
            ErrorKind::InvalidInput,
            format!("invalid --bind '{bind_addr}': {e}"),
        )
    })?;
    if !addr.ip().is_loopback() {
        return Err(std::io::Error::new(
            ErrorKind::PermissionDenied,
            format!("refusing non-loopback TCP bind '{addr}'; use 127.0.0.1 or ::1"),
        ));
    }
    let listener = TcpListener::bind(addr)?;
    serve_tcp_with_listener(listener, sandbox, auth_token, None)
}

pub fn serve_tcp_with_listener(
    listener: TcpListener,
    sandbox: PathSandbox,
    auth_token: &str,
    stop: Option<&std::sync::atomic::AtomicBool>,
) -> std::io::Result<()> {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    listener.set_nonblocking(true)?;
    let sandbox = Arc::new(sandbox);
    let expected_tok = auth_token.to_string();
    let mut conns: Vec<TcpConn> = Vec::new();
    let mut next_id = 1u64;
    let (tx, rx) = mpsc::channel::<WorkerDone>();
    let mut active_workers = 0usize;
    let mut tmp = [0u8; 8192];

    loop {
        if let Some(flag) = stop
            && flag.load(Ordering::Relaxed)
        {
            break;
        }

        // 1. Accept any waiting TCP sockets non-blockingly
        while conns.len() < MAX_TCP_SOCKETS {
            match listener.accept() {
                Ok((stream, _)) => {
                    if stream.set_nonblocking(true).is_ok() {
                        conns.push(TcpConn {
                            id: next_id,
                            stream,
                            read_buf: Vec::new(),
                            authed: false,
                            state: SessionState::default(),
                            in_flight: false,
                            last_active: Instant::now(),
                        });
                        next_id += 1;
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }

        // 2. Deliver finished worker responses
        while let Ok(done) = rx.try_recv() {
            active_workers = active_workers.saturating_sub(1);
            if let Some(conn) = conns.iter_mut().find(|c| c.id == done.conn_id) {
                conn.state = done.state;
                conn.in_flight = false;
                conn.last_active = Instant::now();
                if let Some(resp) = done.response {
                    let _ = conn.stream.set_nonblocking(false);
                    let _ = writeln!(conn.stream, "{resp}");
                    let _ = conn.stream.flush();
                    let _ = conn.stream.set_nonblocking(true);
                }
            }
        }

        // 3. Evict silent unauthenticated sockets (> 2s) and stale idle sockets (> 30s)
        let now = Instant::now();
        conns.retain(|c| {
            let idle = now.duration_since(c.last_active);
            if !c.authed && idle > Duration::from_secs(2) {
                return false;
            }
            if idle > Duration::from_secs(30) && !c.in_flight {
                return false;
            }
            true
        });

        // 4. Non-blocking read & dispatch ready lines
        let mut to_drop = Vec::new();
        for (idx, conn) in conns.iter_mut().enumerate() {
            if conn.in_flight {
                continue;
            }
            loop {
                match conn.stream.read(&mut tmp) {
                    Ok(0) => {
                        to_drop.push(conn.id);
                        break;
                    }
                    Ok(n) => {
                        conn.last_active = Instant::now();
                        if conn.read_buf.len() + n > MAX_MCP_MESSAGE_BYTES {
                            let _ = conn.stream.set_nonblocking(false);
                            let _ = writeln!(
                                conn.stream,
                                "{}",
                                json_err(Value::Null, -32600, "message exceeds 1 MiB limit")
                            );
                            to_drop.push(conn.id);
                            break;
                        }
                        conn.read_buf.extend_from_slice(&tmp[..n]);
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(_) => {
                        to_drop.push(conn.id);
                        break;
                    }
                }
            }

            if to_drop.contains(&conn.id) {
                continue;
            }

            if let Some(nl_pos) = conn.read_buf.iter().position(|&b| b == b'\n') {
                let raw_bytes: Vec<u8> = conn.read_buf.drain(..=nl_pos).collect();
                let Ok(line) = String::from_utf8(raw_bytes) else {
                    to_drop.push(conn.id);
                    continue;
                };
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if !conn.authed {
                    let auth_ok = serde_json::from_str::<Value>(trimmed)
                        .ok()
                        .and_then(|v| {
                            v.get("auth_token")
                                .and_then(|t| t.as_str())
                                .map(|s| constant_time_eq(s.as_bytes(), expected_tok.as_bytes()))
                        })
                        .unwrap_or(false);
                    let _ = conn.stream.set_nonblocking(false);
                    if !auth_ok {
                        let _ = writeln!(
                            conn.stream,
                            "{}",
                            json_err(Value::Null, -32001, "unauthorized")
                        );
                        let _ = conn.stream.flush();
                        to_drop.push(conn.id);
                    } else {
                        conn.authed = true;
                        let _ = writeln!(
                            conn.stream,
                            "{}",
                            json_ok(Value::Null, json!({ "authenticated": true }))
                        );
                        let _ = conn.stream.flush();
                        let _ = conn.stream.set_nonblocking(true);
                    }
                } else if active_workers < MAX_TCP_CONCURRENT {
                    conn.in_flight = true;
                    active_workers += 1;
                    let sb_c = Arc::clone(&sandbox);
                    let tx_c = tx.clone();
                    let cid = conn.id;
                    let mut st_c = SessionState {
                        initialized: conn.state.initialized,
                    };
                    let line_owned = trimmed.to_string();
                    std::thread::spawn(move || {
                        let resp = handle_mcp_line(&sb_c, &mut st_c, &line_owned);
                        let _ = tx_c.send(WorkerDone {
                            conn_id: cid,
                            state: st_c,
                            response: resp,
                        });
                    });
                }
            }
            let _ = idx;
        }

        if !to_drop.is_empty() {
            conns.retain(|c| !to_drop.contains(&c.id));
        }

        std::thread::sleep(Duration::from_millis(4));
    }

    Ok(())
}

#[allow(dead_code)]
fn _default_cap_marker() -> u64 {
    DEFAULT_MAX_FILE_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn mcp_enforces_init_gate_size_cap_sandbox_and_runs_all_tools()
    -> Result<(), Box<dyn std::error::Error>> {
        let base = std::env::temp_dir().join(format!("dockcraft_mcp_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let rdir = base.join("read");
        let wdir = base.join("write");
        let secret = base.join("secret.pdb");
        std::fs::create_dir_all(&rdir)?;
        std::fs::create_dir_all(&wdir)?;
        std::fs::write(
            &secret,
            b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n",
        )?;
        symlink(&secret, rdir.join("escape.pdb"))?;

        // Create minimal pocket PDB inside read root
        let rec_pdb = concat!(
            "ATOM      1  N   LYS A   1      -2.850   0.000   0.000  1.00 20.00           N\n",
            "ATOM      2  O   ASP A   2       3.200   0.000   0.000  1.00 20.00           O\n",
            "ATOM      3  CA  VAL A   3       0.000   3.600   0.000  1.00 20.00           C\n",
            "ATOM      4  CA  VAL A   4       0.000  -3.600   0.000  1.00 20.00           C\n",
            "ATOM      5  CA  VAL A   5       0.000   0.000   3.600  1.00 20.00           C\n",
            "ATOM      6  CA  VAL A   6       0.000   0.000  -3.600  1.00 20.00           C\n",
            "END\n"
        );
        std::fs::write(rdir.join("pocket.pdb"), rec_pdb)?;

        let sandbox = PathSandbox::new(
            Some(rdir.clone()),
            Some(wdir.clone()),
            DEFAULT_MAX_FILE_BYTES,
        )?;
        let mut state = SessionState::default();

        // 1. Pre-initialize calls (including notifications/initialized) rejected with -32002
        let notif_bypass = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        )
        .ok_or("expected error for pre-init notification")?;
        assert_eq!(notif_bypass["error"]["code"], -32002);
        assert!(!state.initialized);

        let pre = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
        )
        .ok_or("expected response")?;
        assert_eq!(pre["error"]["code"], -32002);

        // 2. Oversized > 1 MiB payload rejected
        let big = format!(
            r#"{{"jsonrpc":"2.0","id":2,"method":"ping","pad":"{}"}}"#,
            "A".repeat(MAX_MCP_MESSAGE_BYTES)
        );
        let over = handle_mcp_line(&sandbox, &mut state, &big).ok_or("expected response")?;
        assert_eq!(over["error"]["code"], -32600);

        // 3. Initialize succeeds
        let init = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":3,"method":"initialize"}"#,
        )
        .ok_or("expected response")?;
        assert!(init["result"]["capabilities"].is_object());

        // 4. Symlink escape inside read_root rejected with -32602
        let esc = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"detect_pockets","arguments":{"receptor_path":"escape.pdb"}}}"#,
        )
        .ok_or("expected response")?;
        assert_eq!(esc["error"]["code"], -32602);

        // 5. Run docking via SMILES -> write out.sdf inside write_root
        let dock_resp = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"run_docking","arguments":{"receptor_path":"pocket.pdb","ligand_smiles":"CCO","box_center":[0,0,0],"box_size":[14,14,14],"exhaustiveness":2,"out_sdf_path":"docked.sdf"}}}"#,
        )
        .ok_or("expected response")?;
        assert_eq!(
            dock_resp["result"]["structuredContent"]["mode"],
            "single_ligand"
        );
        assert!(wdir.join("docked.sdf").exists());

        // 6. Export complex to MolCraft & Chymera writes self-contained receptor.pdb & ligand.sdf
        std::fs::copy(wdir.join("docked.sdf"), rdir.join("docked.sdf"))?;
        let exp_resp = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"export_complex","arguments":{"receptor_path":"pocket.pdb","ligand_path":"docked.sdf","target":"chymera","out_script_path":"scene.cxc","out_bundle_path":"scene.chymera.json"}}}"#,
        )
        .ok_or("expected response")?;
        assert_eq!(exp_resp["result"]["structuredContent"]["target"], "chymera");
        assert!(wdir.join("receptor.pdb").exists());
        assert!(wdir.join("ligand.sdf").exists());
        assert!(wdir.join("scene.cxc").exists());
        assert!(wdir.join("scene.chymera.json").exists());

        // 7. Analyze binding site -> write blueprint -> run pharmacophore_screen
        let site_resp = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":70,"method":"tools/call","params":{"name":"analyze_binding_site","arguments":{"receptor_path":"pocket.pdb","ligand_path":"docked.sdf","box_center":[0,0,0],"box_size":[12,12,12],"out_blueprint_path":"site.pharm.json","out_markdown_path":"site.md"}}}"#,
        )
        .ok_or("expected response")?;
        assert!(site_resp["result"]["structuredContent"]["features"].is_array());
        assert!(wdir.join("site.pharm.json").exists());
        assert!(wdir.join("site.md").exists());
        std::fs::copy(wdir.join("site.pharm.json"), rdir.join("site.pharm.json"))?;
        std::fs::write(rdir.join("mini.smi"), "CCO ethanol\nc1ccccc1O phenol\n")?;
        let ps_resp = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":71,"method":"tools/call","params":{"name":"pharmacophore_screen","arguments":{"blueprint_path":"site.pharm.json","library_path":"mini.smi","min_match":1,"out_sdf_path":"pharm_hits.sdf"}}}"#,
        )
        .ok_or("expected response")?;
        assert_eq!(ps_resp["result"]["structuredContent"]["total_evaluated"], 2);
        assert!(wdir.join("pharm_hits.sdf").exists());

        // 7b. Prepare protein + ensemble dock inside sandbox
        let prep_resp = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":72,"method":"tools/call","params":{"name":"prepare_protein","arguments":{"receptor_path":"pocket.pdb","ph":7.4,"out_pdb_path":"prepared.pdb","out_markdown_path":"audit.md"}}}"#,
        )
        .ok_or("expected response")?;
        assert!(prep_resp["result"]["structuredContent"]["audit"]["quality_grade"].is_string());
        assert!(wdir.join("prepared.pdb").exists());
        assert!(wdir.join("audit.md").exists());

        let ens_resp = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":73,"method":"tools/call","params":{"name":"ensemble_dock","arguments":{"receptor_paths":["pocket.pdb","pocket.pdb"],"ligand_smiles":"CCO","box_center":[0,0,0],"box_size":[12,12,12],"aggregation":"boltzmann","exhaustiveness":2,"out_csv_path":"ensemble.csv"}}}"#,
        )
        .ok_or("expected response")?;
        assert_eq!(
            ens_resp["result"]["structuredContent"]["candidates"]
                .as_array()
                .map(Vec::len),
            Some(1)
        );
        assert!(wdir.join("ensemble.csv").exists());

        let md_resp = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":74,"method":"tools/call","params":{"name":"prepare_md_validation","arguments":{"receptor_path":"pocket.pdb","ligand_path":"docked.sdf","production_ns":5.0,"out_config_path":"md_cfg.json","out_chimerax_path":"md_view.cxc"}}}"#,
        )
        .ok_or("expected response")?;
        assert!(md_resp["result"]["structuredContent"]["env_status"].is_object());
        assert!(wdir.join("md_cfg.json").exists());
        assert!(wdir.join("md_view.cxc").exists());

        // 7c. Ola 1 MCP tools: compare_pockets & run_safety_panel (plus sandbox escape rejection)
        let sel_resp = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":75,"method":"tools/call","params":{"name":"compare_pockets","arguments":{"target_path":"pocket.pdb","offtarget_paths":["pocket.pdb"],"box_center":[0,0,0],"box_size":[12,12,12],"pre_aligned":true,"out_markdown_path":"selectivity.md"}}}"#,
        )
        .ok_or("expected response")?;
        assert!(
            sel_resp["result"]["structuredContent"]["selectivity_feasibility_score"].is_number()
        );
        assert!(wdir.join("selectivity.md").exists());

        let sp_resp = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":76,"method":"tools/call","params":{"name":"run_safety_panel","arguments":{"anti_target_paths":["pocket.pdb"],"ligand_smiles":"CCO","box_center":[0,0,0],"box_size":[12,12,12],"target_dg_kcal":-9.0,"exhaustiveness":1,"out_markdown_path":"safety.md"}}}"#,
        )
        .ok_or("expected response")?;
        assert_eq!(
            sp_resp["result"]["structuredContent"]["targets_evaluated"],
            1
        );
        assert!(wdir.join("safety.md").exists());

        let sp_esc = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":77,"method":"tools/call","params":{"name":"run_safety_panel","arguments":{"anti_target_paths":["escape.pdb"],"ligand_smiles":"CCO","box_center":[0,0,0],"box_size":[12,12,12]}}}"#,
        )
        .ok_or("expected response")?;
        assert_eq!(sp_esc["error"]["code"], -32602);

        // 7d. Ola 2 MCP tools: run_lead_optimization, prepare_fep_network, rescore_poses_cnn + sandbox escape checks
        let lo_resp = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":78,"method":"tools/call","params":{"name":"run_lead_optimization","arguments":{"receptor_path":"pocket.pdb","ligand_path":"docked.sdf","mode":"pocket-growth","max_vectors":2,"max_analogs":8,"use_external_clis":false,"out_series_sdf_path":"lead_series.sdf","out_markdown_path":"lead_opt.md"}}}"#,
        )
        .ok_or("expected response")?;
        assert!(
            lo_resp["result"]["structuredContent"]["valid_unique_analogs"]
                .as_u64()
                .unwrap_or(0)
                >= 1
        );
        assert!(wdir.join("lead_series.sdf").exists());
        assert!(wdir.join("lead_opt.md").exists());

        std::fs::copy(wdir.join("lead_series.sdf"), rdir.join("lead_series.sdf"))?;
        let fep_resp = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":79,"method":"tools/call","params":{"name":"prepare_fep_network","arguments":{"receptor_path":"pocket.pdb","ligands_sdf_path":"lead_series.sdf","n_windows":7,"ns_per_window":0.1,"out_config_path":"fep_cfg.json"}}}"#,
        )
        .ok_or("expected response")?;
        assert!(fep_resp["result"]["structuredContent"]["env_status"].is_object());
        assert!(
            fep_resp["result"]["structuredContent"]["planned_edges"]
                .as_array()
                .map_or(0, Vec::len)
                >= 1
        );
        assert!(wdir.join("fep_cfg.json").exists());

        let cnn_probe = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":80,"method":"tools/call","params":{"name":"rescore_poses_cnn","arguments":{}}}"#,
        )
        .ok_or("expected response")?;
        assert!(cnn_probe["result"]["structuredContent"]["env_status"].is_object());

        let cnn_esc = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":81,"method":"tools/call","params":{"name":"rescore_poses_cnn","arguments":{"receptor_path":"escape.pdb","poses_sdf_path":"docked.sdf"}}}"#,
        )
        .ok_or("expected response")?;
        assert_eq!(cnn_esc["error"]["code"], -32602);

        // 8. Reject oversized 200^3 A box DoS immediately
        let huge_box = handle_mcp_line(
            &sandbox,
            &mut state,
            r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"run_docking","arguments":{"receptor_path":"pocket.pdb","ligand_smiles":"CCO","box_center":[0,0,0],"box_size":[200,200,200]}}}"#,
        )
        .ok_or("expected response")?;
        assert_eq!(huge_box["error"]["code"], -32602);

        // 8. TCP refuses non-loopback 0.0.0.0 and empty token
        assert!(serve_tcp(sandbox.clone(), "0.0.0.0:7879", "secret-token").is_err());
        assert!(serve_tcp(sandbox.clone(), "127.0.0.1:7879", "").is_err());

        // 9. Silent-connection immunity: 40 silent TCP sockets must NOT block an authenticated client
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_c = std::sync::Arc::clone(&stop);
        let sb_c = sandbox.clone();
        let srv = std::thread::spawn(move || {
            let _ = serve_tcp_with_listener(listener, sb_c, "tok-super-secret", Some(&stop_c));
        });

        let mut silent_clients = Vec::new();
        for _ in 0..40 {
            if let Ok(s) = TcpStream::connect(("127.0.0.1", port)) {
                silent_clients.push(s);
            }
        }
        let mut good = TcpStream::connect(("127.0.0.1", port))?;
        good.set_read_timeout(Some(Duration::from_secs(3)))?;
        writeln!(good, r#"{{"auth_token":"tok-super-secret"}}"#)?;
        good.flush()?;
        let mut rdr = BufReader::new(good.try_clone()?);
        let mut auth_line = String::new();
        rdr.read_line(&mut auth_line)?;
        assert!(auth_line.contains("authenticated"));

        writeln!(good, r#"{{"jsonrpc":"2.0","id":10,"method":"initialize"}}"#)?;
        good.flush()?;
        let mut init_line = String::new();
        rdr.read_line(&mut init_line)?;
        assert!(init_line.contains("dockcraft-mcp"));

        drop(silent_clients);
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = srv.join();

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}

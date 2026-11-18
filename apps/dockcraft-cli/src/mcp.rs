#![forbid(unsafe_code)]

use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use dock_engine::{DockingConfig, ScoringFunction, dock_ligand};
use dock_pipeline::{
    DEFAULT_MAX_FILE_BYTES, PathSandbox, SceneExportRequest, campaign_report_to_csv,
    canonical_export_target, load_ligand_library_bytes, run_screening_campaign,
    write_chimerax_session_bundle, write_sandboxed_scene_package,
};
use hyde_scorer::{SubstituentPreset, mutate_and_minimize_in_pocket, score_pose_hyde};
use pocket_detect::{GridBox, PocketDetectConfig, detect_pockets};
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

        // 7. Reject oversized 200^3 A box DoS immediately
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

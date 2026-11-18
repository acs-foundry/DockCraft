#![forbid(unsafe_code)]

use std::time::Instant;

use dock_engine::grid::AtomChannelKey;
use dock_engine::{AffinityGridSet, DockingConfig, DockingPose, dock_ligand_with_grids};
use hyde_scorer::{HydeReport, score_pose_hyde};
use pocket_detect::GridBox;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use struct_core::{Ad4Type, Molecule3D, parse_sdf_3d_with_limit, parse_smiles_to_3d, perceive_all};
use thiserror::Error;

use crate::admet::{AdmetProfile, evaluate_admet};
use crate::sandbox::{SandboxError, read_bounded_chemcraft_entries};

#[derive(Debug, Error)]
pub enum CampaignError {
    #[error("structure error: {0}")]
    Struct(#[from] struct_core::StructError),
    #[error("docking error: {0}")]
    Dock(#[from] dock_engine::DockError),
    #[error("hyde scoring error: {0}")]
    Hyde(#[from] hyde_scorer::HydeError),
    #[error("sandbox error: {0}")]
    Sandbox(#[from] SandboxError),
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("no valid ligands in input library")]
    EmptyLibrary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LigandInputRecord {
    pub name: String,
    pub molecule: Molecule3D,
    pub is_active: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenedCandidate {
    pub rank: usize,
    pub name: String,
    pub is_active: Option<bool>,
    /// Measured potency carried over from the input library (a ChemCraft SAR row, or an SDF
    /// `> <pIC50>` tag). The pipeline parsed these and then dropped them, so a run against a
    /// library with real affinities could not be checked against them.
    pub measured_pic50: Option<f64>,
    /// Everything else the input library carried for this compound, untouched.
    pub source_properties: std::collections::BTreeMap<String, String>,
    pub best_affinity_kcal: f64,
    pub hyde_delta_g_kcal: f64,
    pub estimated_ki_nm: f64,
    pub ligand_efficiency: f64,
    pub mpo_score: f64,
    pub admet: AdmetProfile,
    pub best_pose: DockingPose,
    pub hyde_report: HydeReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentMetrics {
    pub total_compounds: usize,
    pub total_actives: usize,
    pub roc_auc: f64,
    pub ef_1_percent: f64,
    pub ef_5_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignReport {
    pub results: Vec<ScreenedCandidate>,
    pub enrichment: Option<EnrichmentMetrics>,
    pub elapsed_ms: u128,
}

pub fn load_ligand_library_bytes(
    ext_hint: &str,
    raw: &[u8],
    max_records: usize,
) -> Result<Vec<LigandInputRecord>, CampaignError> {
    let ext = ext_hint.to_ascii_lowercase();
    if ext == "sdf" || ext == "sd" {
        let text = String::from_utf8_lossy(raw);
        let mol_results = parse_sdf_3d_with_limit(&text, max_records);
        return Ok(mol_results
            .into_iter()
            .enumerate()
            .filter_map(|(i, res)| {
                let mut m = res.ok()?;
                let is_active = m
                    .properties
                    .get("IS_ACTIVE")
                    .or_else(|| m.properties.get("active"))
                    .and_then(|v| parse_bool_flag(v));
                if m.name.trim().is_empty() {
                    m.name = format!("mol_{}", i + 1);
                }
                perceive_all(&mut m);
                Some(LigandInputRecord {
                    name: m.name.clone(),
                    molecule: m,
                    is_active,
                })
            })
            .collect());
    }

    if ext == "chemcraft" {
        let entries = read_bounded_chemcraft_entries(raw)?;
        let mut out = Vec::new();
        for (name, content) in entries {
            if name.ends_with(".csv") {
                out.extend(load_csv_smiles(
                    &String::from_utf8_lossy(&content),
                    max_records - out.len(),
                )?);
            } else if name.ends_with(".sdf") {
                out.extend(load_ligand_library_bytes(
                    "sdf",
                    &content,
                    max_records - out.len(),
                )?);
            } else if name.ends_with(".json") {
                let val: serde_json::Value = serde_json::from_slice(&content)?;
                extract_chemcraft_json_mols(&val, max_records - out.len(), &mut out);
            }
            if out.len() >= max_records {
                break;
            }
        }
        return if out.is_empty() {
            Err(CampaignError::EmptyLibrary)
        } else {
            Ok(out)
        };
    }

    if ext == "csv" {
        return load_csv_smiles(&String::from_utf8_lossy(raw), max_records);
    }

    // Default: .smi (whitespace separated `SMILES [NAME] [ACTIVE]`)
    let text = String::from_utf8_lossy(raw);
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if out.len() >= max_records {
            break;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        let smiles = parts[0];
        let name = parts
            .get(1)
            .copied()
            .unwrap_or(&format!("lig_{}", idx + 1))
            .to_string();
        let is_active = parts.get(2).and_then(|s| parse_bool_flag(s));
        let mol = parse_smiles_to_3d(smiles, Some(&name))?;
        out.push(LigandInputRecord {
            name,
            molecule: mol,
            is_active,
        });
    }
    if out.is_empty() {
        return Err(CampaignError::EmptyLibrary);
    }
    Ok(out)
}

fn parse_bool_flag(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "active" => Some(true),
        "0" | "false" | "no" | "decoy" | "inactive" => Some(false),
        _ => None,
    }
}

fn load_csv_smiles(
    text: &str,
    max_records: usize,
) -> Result<Vec<LigandInputRecord>, CampaignError> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(text.as_bytes());
    let headers = rdr.headers()?.clone();
    let smi_col = headers
        .iter()
        .position(|h| {
            matches!(
                h.to_ascii_lowercase().as_str(),
                "smiles" | "canonical_smiles" | "smi"
            )
        })
        .unwrap_or(0);
    let name_col = headers.iter().position(|h| {
        matches!(
            h.to_ascii_lowercase().as_str(),
            "name" | "id" | "compound_id" | "title"
        )
    });
    let act_col = headers.iter().position(|h| {
        matches!(
            h.to_ascii_lowercase().as_str(),
            "is_active" | "active" | "label"
        )
    });

    let mut out = Vec::new();
    for (idx, rec) in rdr.records().enumerate() {
        if out.len() >= max_records {
            break;
        }
        let rec = rec?;
        let Some(smi) = rec.get(smi_col).map(str::trim).filter(|s| !s.is_empty()) else {
            continue;
        };
        let name = name_col
            .and_then(|c| rec.get(c))
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("row_{}", idx + 1));
        let is_active = act_col.and_then(|c| rec.get(c)).and_then(parse_bool_flag);
        if let Ok(mol) = parse_smiles_to_3d(smi, Some(&name)) {
            out.push(LigandInputRecord {
                name,
                molecule: mol,
                is_active,
            });
        }
    }
    Ok(out)
}

/// Pulls one ligand out of a ChemCraft-shaped JSON row.
///
/// Accepts `canonical_smiles` (what a real `SarRow` carries) as well as the looser `smiles` used by
/// hand-written libraries, and carries the row's measured data through on the molecule's own
/// property map: `pIC50`, `Series`, and anything under `properties`. Those are the columns that
/// make a screening run checkable against measured affinity instead of against a guess.
fn push_chemcraft_row(item: &serde_json::Value, idx: usize, out: &mut Vec<LigandInputRecord>) {
    let Some(smi) = item
        .get("canonical_smiles")
        .or_else(|| item.get("smiles"))
        .and_then(|v| v.as_str())
    else {
        return;
    };
    let name = item
        .get("name")
        .or_else(|| item.get("id"))
        .and_then(|v| v.as_str())
        .map_or_else(|| format!("cc_{}", idx + 1), ToString::to_string);

    let Ok(mut mol) = parse_smiles_to_3d(smi, Some(&name)) else {
        return;
    };

    // Keys for matching the results back to the row they came from. ChemCraft owns the canonical
    // spelling of a structure, so its own string is the join key, with the row id as a tiebreaker.
    mol.properties
        .insert("ChemCraft_canonical_smiles".to_string(), smi.to_string());
    if let Some(id) = item.get("id").and_then(serde_json::Value::as_u64) {
        mol.properties
            .insert("ChemCraft_row_id".to_string(), id.to_string());
    }

    // Free-form property bag from the ChemCraft row.
    if let Some(props) = item.get("properties").and_then(|v| v.as_object()) {
        for (k, v) in props {
            let text = v
                .as_str()
                .map_or_else(|| v.to_string(), ToString::to_string);
            mol.properties.insert(k.clone(), text);
        }
    }
    // Typed columns of `SarRow`, which win over anything of the same name in the bag.
    if let Some(pic50) = item.get("pic50").and_then(serde_json::Value::as_f64) {
        mol.properties
            .insert("pIC50".to_string(), pic50.to_string());
    }
    if let Some(series) = item.get("series").and_then(|v| v.as_str()) {
        mol.properties
            .insert("Series".to_string(), series.to_string());
    }

    let is_active = item
        .get("is_active")
        .and_then(serde_json::Value::as_bool)
        .or_else(|| {
            item.get("active")
                .and_then(|v| v.as_str())
                .and_then(parse_bool_flag)
        })
        .or_else(|| {
            mol.properties
                .get("IS_ACTIVE")
                .or_else(|| mol.properties.get("active"))
                .and_then(|v| parse_bool_flag(v))
        });

    out.push(LigandInputRecord {
        name,
        molecule: mol,
        is_active,
    });
}

/// Extracts ligands from the JSON inside a `.chemcraft` archive.
///
/// A genuine ChemCraft project serialises `ChemCraftDocument` into `manifest.json` as
/// `{format, version, canvas_mol, reaction_arrows, sar_table: {rows: [...]}, sar_filter}`. This
/// function used to look only for a top-level `molecules` / `rows` / `compounds` key, none of which
/// a real document has, so feeding DockCraft an actual ChemCraft project failed with
/// "no valid ligands in input library" and the integration only ever worked with a hand-written
/// JSON that ChemCraft does not produce.
fn extract_chemcraft_json_mols(
    val: &serde_json::Value,
    limit: usize,
    out: &mut Vec<LigandInputRecord>,
) {
    // 1. A real ChemCraft project: the SAR table rows.
    if let Some(rows) = val
        .get("sar_table")
        .and_then(|t| t.get("rows"))
        .and_then(|v| v.as_array())
    {
        for (idx, item) in rows.iter().enumerate() {
            if out.len() >= limit {
                return;
            }
            push_chemcraft_row(item, idx, out);
        }
        if !out.is_empty() {
            return;
        }
    }

    // 2. Looser shapes: a bare list, or a document keyed by `molecules` / `rows` / `compounds`.
    let arr = val
        .get("molecules")
        .or_else(|| val.get("rows"))
        .or_else(|| val.get("compounds"))
        .and_then(|v| v.as_array())
        .or_else(|| val.as_array());
    let Some(items) = arr else {
        return;
    };
    for (idx, item) in items.iter().enumerate() {
        if out.len() >= limit {
            break;
        }
        push_chemcraft_row(item, idx, out);
    }
}

/// Reads a measured pIC50 off a molecule's property map, however the source spelled it.
#[must_use]
pub fn measured_pic50_of(mol: &Molecule3D) -> Option<f64> {
    for key in ["pIC50", "pic50", "PIC50", "pChEMBL", "pActivity"] {
        if let Some(v) = mol.properties.get(key)
            && let Ok(x) = v.trim().parse::<f64>()
            && x.is_finite()
        {
            return Some(x);
        }
    }
    // IC50 in nanomolar converts cleanly: pIC50 = 9 - log10(IC50_nM).
    for key in ["IC50_nM", "IC50_nm", "ic50_nm"] {
        if let Some(v) = mol.properties.get(key)
            && let Ok(nm) = v.trim().parse::<f64>()
            && nm > 0.0
        {
            return Some(9.0 - nm.log10());
        }
    }
    None
}

#[must_use]
pub fn compute_mpo_score(affinity_kcal: f64, hyde_dg: f64, admet: &AdmetProfile) -> f64 {
    // Continuous ramp over [-11, 0] kcal/mol so MPO separates weak vs medium vs tight binders
    let aff_norm = (-affinity_kcal / 11.0).clamp(0.0, 1.0);
    let hyde_norm = ((-hyde_dg + 1.0) / 11.0).clamp(0.0, 1.0);
    let admet_norm = (admet.admet_score / 100.0).clamp(0.0, 1.0);
    let pains_mult = if admet.pains_alerts.is_empty() {
        1.0
    } else {
        0.55
    };
    ((0.45 * aff_norm + 0.30 * hyde_norm + 0.25 * admet_norm) * pains_mult).clamp(0.0, 1.0)
}

#[must_use]
pub fn compute_enrichment_metrics(results: &[ScreenedCandidate]) -> Option<EnrichmentMetrics> {
    let labeled: Vec<(usize, bool)> = results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.is_active.map(|a| (i, a)))
        .collect();
    let total_n = labeled.len();
    let total_act = labeled.iter().filter(|&&(_, a)| a).count();
    if total_n < 4 || total_act == 0 || total_act == total_n {
        return None;
    }

    // Exact ROC AUC via Mann-Whitney rank / pairwise comparison
    let mut pairs_correct = 0.0_f64;
    let total_pairs = (total_act * (total_n - total_act)) as f64;
    for &(i_act, is_a) in &labeled {
        if !is_a {
            continue;
        }
        let s_a = results[i_act].best_affinity_kcal;
        for &(i_dec, is_d) in &labeled {
            if is_d {
                continue;
            }
            let s_d = results[i_dec].best_affinity_kcal;
            if s_a < s_d {
                pairs_correct += 1.0;
            } else if (s_a - s_d).abs() < 1e-9 {
                pairs_correct += 0.5;
            }
        }
    }
    let roc_auc = pairs_correct / total_pairs;

    let calc_ef = |frac: f64| -> f64 {
        let top_k = ((total_n as f64 * frac).ceil() as usize)
            .max(1)
            .min(total_n);
        let hits = labeled.iter().take(top_k).filter(|&&(_, a)| a).count() as f64;
        let expected = (total_act as f64) * (top_k as f64 / total_n as f64);
        if expected > 0.0 { hits / expected } else { 0.0 }
    };

    Some(EnrichmentMetrics {
        total_compounds: total_n,
        total_actives: total_act,
        roc_auc,
        ef_1_percent: calc_ef(0.01),
        ef_5_percent: calc_ef(0.05),
    })
}

pub fn run_screening_campaign(
    receptor: &Molecule3D,
    grid_box: &GridBox,
    library: &[LigandInputRecord],
    config: &DockingConfig,
) -> Result<CampaignReport, CampaignError> {
    if library.is_empty() {
        return Err(CampaignError::EmptyLibrary);
    }
    let start = Instant::now();

    // Collect union of all atom channel keys across the whole library to precompute shared affinity grids ONCE.
    let mut all_channels = Vec::new();
    for item in library {
        for a in &item.molecule.atoms {
            if a.is_heavy() {
                all_channels.push(AtomChannelKey::from_atom(a));
            }
        }
    }
    if all_channels.is_empty() {
        all_channels.push(AtomChannelKey {
            ad4: Ad4Type::C,
            h_donor: false,
            h_acceptor: false,
            hydrophobic: true,
        });
    }
    let grids = AffinityGridSet::build(receptor, grid_box, &all_channels, config.scoring)?;

    let mut screened: Vec<ScreenedCandidate> = library
        .par_iter()
        .enumerate()
        .filter_map(|(idx, item)| {
            let mut local_cfg = config.clone();
            local_cfg.seed = config.seed.wrapping_add((idx as u64 + 1) * 7919);
            let report =
                dock_ligand_with_grids(receptor, &item.molecule, &grids, &local_cfg).ok()?;
            let best_pose = report.poses.into_iter().next()?;
            let hyde = score_pose_hyde(receptor, &best_pose.molecule).ok()?;
            let admet = evaluate_admet(&item.molecule);
            let mpo =
                compute_mpo_score(best_pose.affinity_kcal_mol, hyde.total_delta_g_kcal, &admet);
            Some(ScreenedCandidate {
                rank: 0,
                name: item.name.clone(),
                is_active: item.is_active,
                measured_pic50: measured_pic50_of(&item.molecule),
                source_properties: item.molecule.properties.clone(),
                best_affinity_kcal: best_pose.affinity_kcal_mol,
                hyde_delta_g_kcal: hyde.total_delta_g_kcal,
                estimated_ki_nm: hyde.estimated_ki_nm,
                ligand_efficiency: hyde.ligand_efficiency,
                mpo_score: mpo,
                admet,
                best_pose,
                hyde_report: hyde,
            })
        })
        .collect();

    if screened.is_empty() {
        return Err(CampaignError::EmptyLibrary);
    }

    screened.sort_by(|a, b| a.best_affinity_kcal.total_cmp(&b.best_affinity_kcal));
    for (r, item) in screened.iter_mut().enumerate() {
        item.rank = r + 1;
    }

    let enrichment = compute_enrichment_metrics(&screened);

    Ok(CampaignReport {
        results: screened,
        enrichment,
        elapsed_ms: start.elapsed().as_millis(),
    })
}

#[must_use]
pub fn campaign_report_to_csv(report: &CampaignReport) -> String {
    let mut out = String::from(
        "rank,name,is_active,measured_pic50,affinity_kcal,hyde_delta_g_kcal,ki_nm,ligand_efficiency,mpo_score,mw,clogp,tpsa,lipinski_violations,admet_score,pains_alerts\n",
    );
    for c in &report.results {
        let act = match c.is_active {
            Some(true) => "true",
            Some(false) => "false",
            None => "",
        };
        let pains = c.admet.pains_alerts.join("|");
        let pic50 = c
            .measured_pic50
            .map_or_else(String::new, |v| format!("{v:.3}"));
        out.push_str(&format!(
            "{},{},{},{},{:.3},{:.3},{:.2},{:.3},{:.3},{:.2},{:.2},{:.1},{},{:.1},{}\n",
            c.rank,
            c.name.replace(',', "_"),
            act,
            pic50,
            c.best_affinity_kcal,
            c.hyde_delta_g_kcal,
            c.estimated_ki_nm,
            c.ligand_efficiency,
            c.mpo_score,
            c.admet.mw,
            c.admet.clogp,
            c.admet.tpsa,
            c.admet.lipinski_violations,
            c.admet.admet_score,
            pains
        ));
    }
    out
}

#![forbid(unsafe_code)]

use std::time::Instant;

use dock_engine::grid::AtomChannelKey;
use dock_engine::{
    AffinityGridSet, DockingConfig, DockingPose, ScoringFunction, dock_ligand_with_grids,
};
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

fn extract_chemcraft_json_mols(
    val: &serde_json::Value,
    limit: usize,
    out: &mut Vec<LigandInputRecord>,
) {
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
        let smi = item
            .get("smiles")
            .or_else(|| item.get("canonical_smiles"))
            .and_then(|v| v.as_str());
        let Some(smi) = smi else {
            continue;
        };
        let name = item
            .get("name")
            .or_else(|| item.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or(&format!("cc_{}", idx + 1))
            .to_string();
        let is_active = item.get("is_active").and_then(|v| v.as_bool()).or_else(|| {
            item.get("active")
                .and_then(|v| v.as_str())
                .and_then(parse_bool_flag)
        });
        if let Ok(mol) = parse_smiles_to_3d(smi, Some(&name)) {
            out.push(LigandInputRecord {
                name,
                molecule: mol,
                is_active,
            });
        }
    }
}

#[must_use]
pub fn compute_mpo_score(affinity_kcal: f64, hyde_dg: f64, admet: &AdmetProfile) -> f64 {
    // Normalize affinity from [-12, -2] kcal/mol into [0, 1]
    let aff_norm = ((-affinity_kcal - 2.0) / 10.0).clamp(0.0, 1.0);
    let hyde_norm = ((-hyde_dg - 1.5) / 10.0).clamp(0.0, 1.0);
    let admet_norm = (admet.admet_score / 100.0).clamp(0.0, 1.0);
    let pains_mult = if admet.pains_alerts.is_empty() {
        1.0
    } else {
        0.65
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
    let _ = ScoringFunction::Vina;

    Ok(CampaignReport {
        results: screened,
        enrichment,
        elapsed_ms: start.elapsed().as_millis(),
    })
}

#[must_use]
pub fn campaign_report_to_csv(report: &CampaignReport) -> String {
    let mut out = String::from(
        "rank,name,is_active,affinity_kcal,hyde_delta_g_kcal,ki_nm,ligand_efficiency,mpo_score,mw,clogp,tpsa,lipinski_violations,admet_score,pains_alerts\n",
    );
    for c in &report.results {
        let act = match c.is_active {
            Some(true) => "true",
            Some(false) => "false",
            None => "",
        };
        let pains = c.admet.pains_alerts.join("|");
        out.push_str(&format!(
            "{},{},{},{:.3},{:.3},{:.2},{:.3},{:.3},{:.2},{:.2},{:.1},{},{:.1},{}\n",
            c.rank,
            c.name.replace(',', "_"),
            act,
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

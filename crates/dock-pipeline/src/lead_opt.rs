#![forbid(unsafe_code)]

//! Closed-Loop Multi-Parameter Lead Optimization Engine (`lead-opt`):
//! unites `DockCraft` (`hyde-scorer::mutate_and_minimize_in_pocket`, Vina+HYDE consensus,
//! PoseBusters & torsion audit, optional Ola 1 ΔMIF selectivity guidance), `AdmetCraft v1.1.0`
//! (32 TDC+Tox21 QSAR, SEA Bowes Safety-20, Phase I/II Metabolite Guard, TPP MPO), and
//! `ChemCraft v0.3.0` (`mol-synth` Buy-Before-Make + Retro*) to generate, filter, and Pareto-rank
//! synthesizable, safe, selective 3D congeneric series ready for RBFE/FEP.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use dock_engine::{ScoringFunction, score_complex};
use hyde_scorer::{
    SubstituentPreset, analyze_torsions, consensus_affinity_kcal, mutate_and_minimize_in_pocket,
    score_pose_hyde, validate_pose_geometry,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use struct_core::{Molecule3D, write_sdf_3d};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LeadOptError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("lead optimization error: {0}")]
    Opt(String),
}

/// Strategy controlling which parent atoms and MedChem substituent libraries are explored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LeadOptMode {
    /// Explore sterically open 3D pocket & ΔMIF selectivity growth vectors.
    PocketGrowth,
    /// Target metabolic Sites of Metabolism (SoM) from AdmetCraft with blocking substituents.
    SomBlocking,
    /// Union of top 3D pocket/selectivity growth vectors and metabolic SoM blocking sites.
    #[default]
    Combined,
}

impl LeadOptMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PocketGrowth => "pocket-growth",
            Self::SomBlocking => "som-blocking",
            Self::Combined => "combined",
        }
    }
}

/// Configuration for a closed-loop multi-parameter lead optimization round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadOptConfig {
    pub mode: LeadOptMode,
    /// AdmetCraft Target Product Profile (`oral` | `cns` | `peripheral` | `oncology`).
    pub tpp_profile: String,
    /// Optional path to an Ola 1 `selectivity_report.json` to bias growth vectors toward `SEL_*` hotspots.
    pub selectivity_json: Option<PathBuf>,
    /// Maximum number of parent attachment sites explored per round (default `5`).
    pub max_vectors: usize,
    /// Maximum deduplicated analogs retained in report & aligned `lead_opt_series.sdf` (default `30`).
    pub max_analogs: usize,
    /// Number of top Pareto analogs audited with `chemcraft-cli retro` (default `6`).
    pub top_retro: usize,
    /// Invoke sibling `admetcraft-cli` and `chemcraft-cli` when installed on PATH (`true` by default).
    pub use_external_craft_clis: bool,
    pub out_dir: PathBuf,
}

impl Default for LeadOptConfig {
    fn default() -> Self {
        Self {
            mode: LeadOptMode::Combined,
            tpp_profile: "oncology".to_string(),
            selectivity_json: None,
            max_vectors: 5,
            max_analogs: 30,
            top_retro: 6,
            use_external_craft_clis: true,
            out_dir: PathBuf::from("./lead_opt"),
        }
    }
}

#[derive(Debug, Clone)]
struct MutationVector {
    atom_idx: usize,
    atom_label: String,
    rationale: String,
    clearance_a: f64,
    selectivity_bonus: f64,
    is_som_target: bool,
}

/// Single evaluated analog in the closed-loop Lead Optimization series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadOptAnalog {
    pub rank: usize,
    pub pareto_rank: usize,
    pub name: String,
    pub is_parent: bool,
    pub transformation: String,
    pub rationale: String,
    pub smiles: String,
    pub vina_dg_kcal: f64,
    pub hyde_dg_kcal: f64,
    pub consensus_dg_kcal: f64,
    /// `consensus_dg - parent_consensus_dg` (negative = potency improvement).
    pub delta_dg_kcal: f64,
    pub posebusters_valid: bool,
    pub torsion_red_alerts: usize,
    pub tpp_mpo: Option<f64>,
    pub delta_tpp_mpo: Option<f64>,
    pub tpp_decision: Option<String>,
    pub tox21_alert_count: Option<usize>,
    pub sea_high_risk_count: Option<usize>,
    pub metabolite_bioact_count: Option<usize>,
    pub synth_steps: Option<usize>,
    pub synth_stock_pct: Option<f64>,
    pub composite_score: f64,
}

/// Complete Closed-Loop Lead Optimization dossier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadOptReport {
    pub receptor_name: String,
    pub parent_name: String,
    pub mode: LeadOptMode,
    pub tpp_profile: String,
    pub vectors_explored: usize,
    pub total_mutants_generated: usize,
    pub valid_unique_analogs: usize,
    pub pareto_front_count: usize,
    pub elapsed_ms: u128,
    pub analogs: Vec<LeadOptAnalog>,
    pub series_sdf_path: String,
    pub fep_pair_sdf_path: Option<String>,
}

const GROWTH_PRESETS: &[SubstituentPreset] = &[
    SubstituentPreset::F,
    SubstituentPreset::Cl,
    SubstituentPreset::CH3,
    SubstituentPreset::CF3,
    SubstituentPreset::OCF3,
    SubstituentPreset::CPr,
    SubstituentPreset::OH,
    SubstituentPreset::NH2,
    SubstituentPreset::CN,
    SubstituentPreset::OCH3,
    SubstituentPreset::CONH2,
    SubstituentPreset::SO2Me,
];

const SOM_BLOCK_PRESETS: &[SubstituentPreset] = &[
    SubstituentPreset::F,
    SubstituentPreset::CF3,
    SubstituentPreset::OCF3,
    SubstituentPreset::CPr,
    SubstituentPreset::Cl,
    SubstituentPreset::CN,
];

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn norm3(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-8 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn load_selectivity_hotspots(path: Option<&Path>) -> Vec<([f64; 3], String)> {
    let Some(p) = path else {
        return Vec::new();
    };
    let Ok(raw) = std::fs::read_to_string(p) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(arr) = v.get("selectivity_hotspots").and_then(|x| x.as_array()) {
        for hs in arr {
            if let Some(c) = hs.get("center").and_then(|x| x.as_array())
                && c.len() == 3
                && let (Some(x), Some(y), Some(z)) = (c[0].as_f64(), c[1].as_f64(), c[2].as_f64())
            {
                let id = hs
                    .get("id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("SEL")
                    .to_string();
                out.push(([x, y, z], id));
            }
        }
    }
    out
}

fn find_sibling_cli(bin: &str) -> Option<PathBuf> {
    if let Ok(home) = std::env::var("HOME") {
        let local = PathBuf::from(home).join(".local/bin").join(bin);
        if local.exists() {
            return Some(local);
        }
    }
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let cand = dir.join(bin);
            if cand.exists() {
                return Some(cand);
            }
        }
    }
    None
}

/// Identifies candidate growth vectors on `ligand` pointing into sterically accessible pocket
/// space (bonus if aligned toward any ΔMIF `SEL_*` selectivity hotspot).
fn select_mutation_vectors(
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    sel_hotspots: &[([f64; 3], String)],
    som_atom_elements: &[(String, String)],
    cfg: &LeadOptConfig,
) -> Vec<MutationVector> {
    let rec_heavy: Vec<[f64; 3]> = receptor
        .atoms
        .iter()
        .filter(|a| a.is_heavy())
        .map(|a| a.coords)
        .collect();

    let mut vectors: Vec<MutationVector> = Vec::new();
    let mut used_parents: HashSet<usize> = HashSet::new();

    for (idx, atom) in ligand.atoms.iter().enumerate() {
        // Primary growth vectors: explicit H atoms attached to C/N/O (or terminal F/Cl/CH3 atoms)
        let is_terminal = ligand.adj.get(idx).is_some_and(|e| e.len() == 1);
        let is_h = atom.element == "H";
        let is_subst_heavy =
            atom.element == "C" && ligand.adj.get(idx).is_some_and(|e| e.len() <= 2);
        if !is_h
            && !(is_terminal && matches!(atom.element.as_str(), "F" | "Cl" | "Br"))
            && !is_subst_heavy
        {
            continue;
        }
        let parent_idx = if is_h || is_terminal {
            let Some(&(p, _)) = ligand.adj.get(idx).and_then(|e| e.first()) else {
                continue;
            };
            // Keep at most one H vector per parent heavy atom (best clearance wins)
            p
        } else {
            idx
        };
        let parent_el = ligand
            .atoms
            .get(parent_idx)
            .map_or("", |a| a.element.as_str());
        if parent_el != "C" && !is_terminal {
            continue;
        }

        // Compute outward growth probe position 2.4 Å along the bond vector
        let p_pos = ligand.atoms[parent_idx].coords;
        let dir = if parent_idx != idx {
            norm3([
                atom.coords[0] - p_pos[0],
                atom.coords[1] - p_pos[1],
                atom.coords[2] - p_pos[2],
            ])
        } else {
            let mut sum = [0.0; 3];
            for &(nb, _) in &ligand.adj[idx] {
                let np = ligand.atoms[nb].coords;
                sum[0] += p_pos[0] - np[0];
                sum[1] += p_pos[1] - np[1];
                sum[2] += p_pos[2] - np[2];
            }
            norm3(sum)
        };
        if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 0.25 {
            continue;
        }
        let probe = [
            p_pos[0] + 2.5 * dir[0],
            p_pos[1] + 2.5 * dir[1],
            p_pos[2] + 2.5 * dir[2],
        ];
        let min_rec_d = rec_heavy
            .iter()
            .map(|&rp| dist3(probe, rp))
            .fold(99.0_f64, f64::min);
        if min_rec_d < 2.25 {
            continue; // Directly buried into receptor wall
        }

        let mut sel_bonus = 0.0_f64;
        let mut sel_tag = String::new();
        for (hpos, hid) in sel_hotspots {
            let d_hs = dist3(probe, *hpos);
            if d_hs <= 4.0 {
                let b = (4.0 - d_hs) * 1.25;
                if b > sel_bonus {
                    sel_bonus = b;
                    sel_tag = hid.clone();
                }
            }
        }

        // Check if parent carbon is an aliphatic CH2/CH3 or electron-rich aromatic CH matching SoM hints
        let is_som = som_atom_elements.iter().any(|(el, rxn)| {
            el == parent_el
                && ((rxn.contains("O-Dealkylation")
                    && ligand.adj[parent_idx]
                        .iter()
                        .any(|&(nb, _)| ligand.atoms[nb].element == "O"))
                    || (rxn.contains("N-Dealkylation")
                        && ligand.adj[parent_idx]
                            .iter()
                            .any(|&(nb, _)| ligand.atoms[nb].element == "N"))
                    || (rxn.contains("Benzylic")
                        && !ligand.atoms[parent_idx].aromatic
                        && ligand.adj[parent_idx]
                            .iter()
                            .any(|&(nb, _)| ligand.atoms[nb].aromatic))
                    || (rxn.contains("Aromatic") && ligand.atoms[parent_idx].aromatic))
        });

        if cfg.mode == LeadOptMode::SomBlocking && !is_som && !som_atom_elements.is_empty() {
            continue;
        }

        let rationale = if is_som && !sel_tag.is_empty() {
            format!("SoM metabolic block + ΔMIF {sel_tag} vector (clr={min_rec_d:.1} Å)")
        } else if is_som {
            format!("SoM metabolic hotspot blocking (clr={min_rec_d:.1} Å)")
        } else if !sel_tag.is_empty() {
            format!("ΔMIF selectivity vector -> {sel_tag} (clr={min_rec_d:.1} Å)")
        } else {
            format!("Open 3D pocket growth vector (clr={min_rec_d:.1} Å)")
        };

        vectors.push(MutationVector {
            atom_idx: idx,
            atom_label: format!("{}{}", atom.element, idx + 1),
            rationale,
            clearance_a: min_rec_d,
            selectivity_bonus: sel_bonus,
            is_som_target: is_som,
        });
    }

    // Prioritize selectivity-aligned & SoM vectors first, then best steric clearance, deduplicating per parent atom
    vectors.sort_by(|a, b| {
        let sa = a.selectivity_bonus
            + if a.is_som_target { 2.0 } else { 0.0 }
            + a.clearance_a.min(5.0) * 0.4;
        let sb = b.selectivity_bonus
            + if b.is_som_target { 2.0 } else { 0.0 }
            + b.clearance_a.min(5.0) * 0.4;
        sb.total_cmp(&sa)
    });

    let mut chosen = Vec::new();
    for v in vectors {
        let parent_key = ligand
            .adj
            .get(v.atom_idx)
            .and_then(|e| if e.len() == 1 { Some(e[0].0) } else { None })
            .unwrap_or(v.atom_idx);
        if used_parents.insert(parent_key) {
            chosen.push(v);
            if chosen.len() >= cfg.max_vectors.clamp(1, 16) {
                break;
            }
        }
    }
    chosen
}

#[derive(Debug, Clone, Default)]
struct AdmetJsonRow {
    canonical_smiles: String,
    tpp_mpo: Option<f64>,
    tpp_decision: Option<String>,
    tox21_alerts: Option<usize>,
    sea_high: Option<usize>,
    met_bioact: Option<usize>,
    som_hints: Vec<(String, String)>,
}

fn evaluate_sdf_with_admetcraft(
    sdf_path: &Path,
    out_json: &Path,
    tpp: &str,
) -> HashMap<String, AdmetJsonRow> {
    let mut map = HashMap::new();
    let Some(ac) = find_sibling_cli("admetcraft-cli") else {
        return map;
    };
    let ok = Command::new(&ac)
        .arg("batch")
        .arg(sdf_path)
        .arg("--out")
        .arg(out_json)
        .args(["--format", "json", "--tpp", tpp])
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        return map;
    }
    let Ok(raw) = std::fs::read_to_string(out_json) else {
        return map;
    };
    let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(&raw) else {
        return map;
    };
    for item in arr {
        let name = item
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if name.is_empty() {
            continue;
        }
        let smiles = item
            .get("canonical_smiles")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let tpp_mpo = item
            .get("tpp_mpo")
            .and_then(|t| t.get("score"))
            .and_then(|v| v.as_f64());
        let tpp_decision = item
            .get("tpp_mpo")
            .and_then(|t| t.get("decision"))
            .and_then(|v| v.as_str())
            .map(ToString::to_string);
        let tox21_alerts = item.get("tox21_summary").map(|t| {
            let nr = t
                .get("endocrine_alerts")
                .and_then(|a| a.as_array())
                .map_or(0, Vec::len);
            let sr = t
                .get("stress_alerts")
                .and_then(|a| a.as_array())
                .map_or(0, Vec::len);
            nr + sr
        });
        let sea_high = item
            .get("off_targets")
            .and_then(|o| o.get("high_risk_count"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        let met_bioact = item
            .get("metabolite_guard")
            .and_then(|m| m.get("bioactivation_alerts"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        let mut som_hints = Vec::new();
        if let Some(soms) = item.get("metabolism_sites").and_then(|s| s.as_array()) {
            for s in soms.iter().take(5) {
                if let (Some(el), Some(rxn)) = (
                    s.get("atom_symbol").and_then(|v| v.as_str()),
                    s.get("reaction").and_then(|v| v.as_str()),
                ) {
                    som_hints.push((el.to_string(), rxn.to_string()));
                }
            }
        }
        map.insert(
            name,
            AdmetJsonRow {
                canonical_smiles: smiles,
                tpp_mpo,
                tpp_decision,
                tox21_alerts,
                sea_high,
                met_bioact,
                som_hints,
            },
        );
    }
    map
}

fn probe_chemcraft_retro(smiles: &str, name: &str) -> (Option<usize>, Option<f64>) {
    let Some(cc) = find_sibling_cli("chemcraft-cli") else {
        return (None, None);
    };
    let tmp_json = std::env::temp_dir().join(format!(
        "dockcraft_retro_{}_{}.json",
        std::process::id(),
        name.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>()
    ));
    let Ok(out) = Command::new(&cc)
        .args(["retro", smiles, "--name", name, "--top", "1"])
        .arg("--out-json")
        .arg(&tmp_json)
        .output()
    else {
        return (None, None);
    };
    let parsed = if out.status.success() {
        std::fs::read_to_string(&tmp_json)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    } else {
        None
    };
    let _ = std::fs::remove_file(&tmp_json);
    let Some(v) = parsed else {
        return (None, None);
    };
    if let Some(best) = v
        .get("routes")
        .and_then(|r| r.as_array())
        .and_then(|a| a.first())
    {
        let steps = best
            .get("total_steps")
            .or_else(|| best.get("longest_linear_seq"))
            .and_then(|x| x.as_u64())
            .map(|n| n as usize);
        let stock = best
            .get("stock_coverage_pct")
            .and_then(|x| x.as_f64())
            .map(|p| (p * 10.0).round() / 10.0);
        return (steps, stock);
    }
    (None, None)
}

/// Computes non-dominated Pareto front ranks (`1` = Pareto-optimal frontier).
fn assign_pareto_ranks(analogs: &mut [LeadOptAnalog]) {
    let n = analogs.len();
    let mut assigned = vec![0usize; n];
    let mut current_rank = 1usize;
    let mut remaining = n;

    while remaining > 0 {
        let mut front = Vec::new();
        for i in 0..n {
            if assigned[i] != 0 {
                continue;
            }
            let mut dominated = false;
            for j in 0..n {
                if i == j || assigned[j] != 0 {
                    continue;
                }
                if dominates(&analogs[j], &analogs[i]) {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                front.push(i);
            }
        }
        if front.is_empty() {
            for r in &mut assigned {
                if *r == 0 {
                    *r = current_rank;
                }
            }
            break;
        }
        for i in front {
            assigned[i] = current_rank;
            remaining -= 1;
        }
        current_rank += 1;
    }

    for (a, r) in analogs.iter_mut().zip(assigned) {
        a.pareto_rank = r;
    }
}

fn dominates(a: &LeadOptAnalog, b: &LeadOptAnalog) -> bool {
    // Objectives: lower consensus_dg_kcal, higher tpp_mpo, lower total safety flags
    let a_dg = a.consensus_dg_kcal;
    let b_dg = b.consensus_dg_kcal;
    let a_mpo = a.tpp_mpo.unwrap_or(0.5);
    let b_mpo = b.tpp_mpo.unwrap_or(0.5);
    let a_flags = a.tox21_alert_count.unwrap_or(0)
        + a.sea_high_risk_count.unwrap_or(0)
        + a.metabolite_bioact_count.unwrap_or(0)
        + a.torsion_red_alerts;
    let b_flags = b.tox21_alert_count.unwrap_or(0)
        + b.sea_high_risk_count.unwrap_or(0)
        + b.metabolite_bioact_count.unwrap_or(0)
        + b.torsion_red_alerts;

    let not_worse = a_dg <= b_dg + 1e-6 && a_mpo >= b_mpo - 1e-6 && a_flags <= b_flags;
    let strictly_better = a_dg < b_dg - 0.05 || a_mpo > b_mpo + 0.005 || a_flags < b_flags;
    not_worse && strictly_better
}

/// Runs a complete closed-loop 3D Lead Optimization round starting from `parent_ligand` docked
/// inside `receptor`.
pub fn run_lead_optimization(
    receptor: &Molecule3D,
    parent_ligand: &Molecule3D,
    cfg: &LeadOptConfig,
) -> Result<LeadOptReport, LeadOptError> {
    let t0 = Instant::now();
    std::fs::create_dir_all(&cfg.out_dir)?;

    let rec_name = if receptor.name.trim().is_empty() {
        "Receptor".to_string()
    } else {
        receptor.name.clone()
    };
    let mut parent = parent_ligand.clone();
    struct_core::perceive_all(&mut parent);
    if parent.name.trim().is_empty() {
        parent.name = "Parent".to_string();
    }
    let parent_name = parent.name.clone();

    // 1. Score parent baseline & optionally fetch SoM hints from AdmetCraft
    let parent_vina = score_complex(receptor, &parent, ScoringFunction::Vina).total_affinity_kcal;
    let parent_vinardo =
        score_complex(receptor, &parent, ScoringFunction::Vinardo).total_affinity_kcal;
    let parent_hyde =
        score_pose_hyde(receptor, &parent).map_or(parent_vina, |h| h.total_delta_g_kcal);
    let parent_cons = consensus_affinity_kcal(parent_vina, parent_vinardo, parent_hyde);
    let parent_pb = validate_pose_geometry(receptor, &parent).valid;
    let parent_tors_red = analyze_torsions(&parent).red_count;

    let mut som_hints = Vec::new();
    if cfg.use_external_craft_clis {
        let parent_sdf = cfg.out_dir.join("_parent.sdf");
        let parent_admet_json = cfg.out_dir.join("_parent_admet.json");
        std::fs::write(&parent_sdf, write_sdf_3d(&[&parent]))?;
        let pmap = evaluate_sdf_with_admetcraft(&parent_sdf, &parent_admet_json, &cfg.tpp_profile);
        if let Some(row) = pmap.values().next() {
            som_hints = row.som_hints.clone();
        }
    }

    // 2. Select mutation vectors (3D clearance + ΔMIF selectivity hotspots + SoM sites)
    let sel_hotspots = load_selectivity_hotspots(cfg.selectivity_json.as_deref());
    let vectors = select_mutation_vectors(receptor, &parent, &sel_hotspots, &som_hints, cfg);

    // 3. Build (vector, preset) mutation tasks and execute in parallel via rayon
    let mut tasks: Vec<(&MutationVector, SubstituentPreset)> = Vec::new();
    for v in &vectors {
        let presets = if cfg.mode == LeadOptMode::SomBlocking || v.is_som_target {
            SOM_BLOCK_PRESETS
        } else {
            GROWTH_PRESETS
        };
        for &p in presets {
            tasks.push((v, p));
        }
    }

    struct RawMutant {
        name: String,
        transformation: String,
        rationale: String,
        mol: Molecule3D,
        vina: f64,
        hyde: f64,
        cons: f64,
        pb_ok: bool,
        tors_red: usize,
    }

    let raw_mutants: Vec<RawMutant> = tasks
        .par_iter()
        .filter_map(|&(vec, preset)| {
            let outcome =
                mutate_and_minimize_in_pocket(receptor, &parent, vec.atom_idx, preset).ok()?;
            let mut m = outcome.mutated_ligand;
            let tag = format!("{}_{}_{:?}", parent_name, vec.atom_label, preset);
            m.name = tag.clone();
            let vina = score_complex(receptor, &m, ScoringFunction::Vina).total_affinity_kcal;
            let vinardo = score_complex(receptor, &m, ScoringFunction::Vinardo).total_affinity_kcal;
            let hyde = outcome.after_kcal;
            let cons = consensus_affinity_kcal(vina, vinardo, hyde);
            // Drop severely clashing poses (> +3.0 kcal/mol worse than parent)
            if cons > parent_cons + 3.0 {
                return None;
            }
            let pb_ok = validate_pose_geometry(receptor, &m).valid;
            let tors_red = analyze_torsions(&m).red_count;
            Some(RawMutant {
                name: tag,
                transformation: format!("{} -> {:?}", vec.atom_label, preset),
                rationale: vec.rationale.clone(),
                mol: m,
                vina,
                hyde,
                cons,
                pb_ok,
                tors_red,
            })
        })
        .collect();
    let total_mutants = raw_mutants.len();

    // 4. Batch-evaluate parent + all mutants in AdmetCraft v1.1.0 via SDF -> JSON
    let mut all_mols_for_admet: Vec<&Molecule3D> = Vec::with_capacity(raw_mutants.len() + 1);
    all_mols_for_admet.push(&parent);
    for rm in &raw_mutants {
        all_mols_for_admet.push(&rm.mol);
    }
    let raw_sdf = cfg.out_dir.join("all_candidates_raw.sdf");
    let admet_json = cfg.out_dir.join("admet_candidates.json");
    std::fs::write(&raw_sdf, write_sdf_3d(&all_mols_for_admet))?;

    let admet_map = if cfg.use_external_craft_clis {
        evaluate_sdf_with_admetcraft(&raw_sdf, &admet_json, &cfg.tpp_profile)
    } else {
        HashMap::new()
    };

    let parent_admet = admet_map.get(&parent_name).cloned().unwrap_or_default();
    let parent_mpo = parent_admet.tpp_mpo;

    // 5. Deduplicate by canonical SMILES (retaining best consensus ΔG per unique structure)
    let mut seen_smiles: HashSet<String> = HashSet::new();
    if !parent_admet.canonical_smiles.is_empty() {
        seen_smiles.insert(parent_admet.canonical_smiles.clone());
    }

    let mut analog_rows: Vec<(LeadOptAnalog, Molecule3D)> = Vec::new();
    let parent_row = LeadOptAnalog {
        rank: 0,
        pareto_rank: 1,
        name: parent_name.clone(),
        is_parent: true,
        transformation: "Parent (Reference)".to_string(),
        rationale: "Starting docked lead pose".to_string(),
        smiles: parent_admet
            .canonical_smiles
            .clone()
            .if_empty_then(parent.source_smiles.clone().unwrap_or_default()),
        vina_dg_kcal: round2(parent_vina),
        hyde_dg_kcal: round2(parent_hyde),
        consensus_dg_kcal: round2(parent_cons),
        delta_dg_kcal: 0.0,
        posebusters_valid: parent_pb,
        torsion_red_alerts: parent_tors_red,
        tpp_mpo: parent_mpo.map(round3),
        delta_tpp_mpo: Some(0.0),
        tpp_decision: parent_admet.tpp_decision.clone(),
        tox21_alert_count: parent_admet.tox21_alerts,
        sea_high_risk_count: parent_admet.sea_high,
        metabolite_bioact_count: parent_admet.met_bioact,
        synth_steps: None,
        synth_stock_pct: None,
        composite_score: 0.0,
    };
    analog_rows.push((parent_row, parent.clone()));

    let mut sorted_muts = raw_mutants;
    sorted_muts.sort_by(|a, b| a.cons.total_cmp(&b.cons));

    for rm in sorted_muts {
        let adm = admet_map.get(&rm.name).cloned().unwrap_or_default();
        if !adm.canonical_smiles.is_empty() && !seen_smiles.insert(adm.canonical_smiles.clone()) {
            continue;
        }
        let ddg = rm.cons - parent_cons;
        let dmpo = match (adm.tpp_mpo, parent_mpo) {
            (Some(m), Some(pm)) => Some(round3(m - pm)),
            _ => None,
        };
        let flags = adm.tox21_alerts.unwrap_or(0)
            + adm.sea_high.unwrap_or(0)
            + adm.met_bioact.unwrap_or(0)
            + rm.tors_red;
        let comp_score =
            (-ddg * 0.45) + dmpo.unwrap_or(0.0) * 3.0 + if rm.pb_ok { 0.4 } else { -1.0 }
                - (flags as f64) * 0.35;

        analog_rows.push((
            LeadOptAnalog {
                rank: 0,
                pareto_rank: 0,
                name: rm.name,
                is_parent: false,
                transformation: rm.transformation,
                rationale: rm.rationale,
                smiles: adm.canonical_smiles,
                vina_dg_kcal: round2(rm.vina),
                hyde_dg_kcal: round2(rm.hyde),
                consensus_dg_kcal: round2(rm.cons),
                delta_dg_kcal: round2(ddg),
                posebusters_valid: rm.pb_ok,
                torsion_red_alerts: rm.tors_red,
                tpp_mpo: adm.tpp_mpo.map(round3),
                delta_tpp_mpo: dmpo,
                tpp_decision: adm.tpp_decision,
                tox21_alert_count: adm.tox21_alerts,
                sea_high_risk_count: adm.sea_high,
                metabolite_bioact_count: adm.met_bioact,
                synth_steps: None,
                synth_stock_pct: None,
                composite_score: round3(comp_score),
            },
            rm.mol,
        ));
    }

    // 6. Assign Pareto frontier ranks, sort, and annotate top analogs with ChemCraft Retro*
    let mut analogs_only: Vec<LeadOptAnalog> = analog_rows.iter().map(|(a, _)| a.clone()).collect();
    assign_pareto_ranks(&mut analogs_only);
    for (row, a) in analog_rows.iter_mut().zip(analogs_only) {
        row.0 = a;
    }

    analog_rows.sort_by(|a, b| {
        a.0.pareto_rank
            .cmp(&b.0.pareto_rank)
            .then_with(|| b.0.composite_score.total_cmp(&a.0.composite_score))
            .then_with(|| a.0.consensus_dg_kcal.total_cmp(&b.0.consensus_dg_kcal))
    });
    analog_rows.truncate(cfg.max_analogs.clamp(2, 100));

    if cfg.use_external_craft_clis && cfg.top_retro > 0 {
        for (a, _) in analog_rows.iter_mut().take(cfg.top_retro) {
            if !a.smiles.trim().is_empty() {
                let (steps, stock) = probe_chemcraft_retro(&a.smiles, &a.name);
                a.synth_steps = steps;
                a.synth_stock_pct = stock;
            }
        }
    }

    for (i, (a, _)) in analog_rows.iter_mut().enumerate() {
        a.rank = i + 1;
    }

    let pareto_front_count = analog_rows
        .iter()
        .filter(|(a, _)| a.pareto_rank == 1)
        .count();

    // 7. Write aligned series SDF (parent first + top analogs), top SMILES, and 2-ligand FEP pair SDF
    let series_sdf_path = cfg.out_dir.join("lead_opt_series.sdf");
    let top_smi_path = cfg.out_dir.join("lead_opt_top.smi");
    let mut ordered_mols: Vec<&Molecule3D> = Vec::with_capacity(analog_rows.len());
    if let Some((_, pm)) = analog_rows.iter().find(|(a, _)| a.is_parent) {
        ordered_mols.push(pm);
    }
    for (a, m) in &analog_rows {
        if !a.is_parent {
            ordered_mols.push(m);
        }
    }
    std::fs::write(&series_sdf_path, write_sdf_3d(&ordered_mols))?;

    let mut smi_buf = String::new();
    for (a, _) in &analog_rows {
        if !a.smiles.trim().is_empty() {
            let _ = writeln!(smi_buf, "{} {}", a.smiles, a.name);
        }
    }
    std::fs::write(&top_smi_path, smi_buf)?;

    // Export Parent + #1 best non-parent analog into `fep_pair.sdf` for immediate RBFE validation
    let fep_pair_path =
        if let Some((_, best_mut_mol)) = analog_rows.iter().find(|(a, _)| !a.is_parent) {
            let p = cfg.out_dir.join("fep_pair.sdf");
            std::fs::write(&p, write_sdf_3d(&[&parent, best_mut_mol]))?;
            Some(p.display().to_string())
        } else {
            None
        };

    let analogs_final: Vec<LeadOptAnalog> = analog_rows.into_iter().map(|(a, _)| a).collect();
    let report = LeadOptReport {
        receptor_name: rec_name,
        parent_name,
        mode: cfg.mode,
        tpp_profile: cfg.tpp_profile.clone(),
        vectors_explored: vectors.len(),
        total_mutants_generated: total_mutants,
        valid_unique_analogs: analogs_final.len(),
        pareto_front_count,
        elapsed_ms: t0.elapsed().as_millis(),
        analogs: analogs_final,
        series_sdf_path: series_sdf_path.display().to_string(),
        fep_pair_sdf_path: fep_pair_path,
    };

    std::fs::write(
        cfg.out_dir.join("lead_opt_report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    std::fs::write(
        cfg.out_dir.join("lead_opt_report.md"),
        report.render_markdown(),
    )?;
    Ok(report)
}

trait StrFallback {
    fn if_empty_then(self, fallback: String) -> String;
}

impl StrFallback for String {
    fn if_empty_then(self, fallback: String) -> String {
        if self.trim().is_empty() {
            fallback
        } else {
            self
        }
    }
}

#[inline]
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[inline]
fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

impl LeadOptReport {
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();
        let _ = writeln!(
            md,
            "# Closed-Loop Multi-Parameter Lead Optimization Report: {} in {}\n",
            self.parent_name, self.receptor_name
        );
        let _ = writeln!(
            md,
            "- **Mode:** `{}` | **TPP:** `{}` | **Vectors Explored:** `{}` | **Generated:** `{}` -> **Unique Valid:** `{}` (**{}** Pareto-optimal) | **Runtime:** `{:.2} s`\n",
            self.mode.as_str(),
            self.tpp_profile,
            self.vectors_explored,
            self.total_mutants_generated,
            self.valid_unique_analogs,
            self.pareto_front_count,
            (self.elapsed_ms as f64) / 1000.0
        );
        md.push_str("## Ranked Matched Molecular Pair Series (Pareto Frontier First)\n\n");
        md.push_str("| Rank | Pareto | Analog | Transformation | Cons. ΔG | ΔΔG | PB | TPP MPO (Δ) | Tox21/SEA/Bio | Retro* | Rationale |\n");
        md.push_str("|---:|---:|---|---|---:|---:|:---:|---|---|---|---|\n");
        for a in self.analogs.iter().take(25) {
            let mpo = match (a.tpp_mpo, a.delta_tpp_mpo) {
                (Some(m), Some(dm)) => format!("{m:.3} ({dm:+.3})"),
                (Some(m), None) => format!("{m:.3}"),
                _ => "—".to_string(),
            };
            let guard = format!(
                "{}/{}/{}",
                a.tox21_alert_count
                    .map_or_else(|| "-".to_string(), |v| v.to_string()),
                a.sea_high_risk_count
                    .map_or_else(|| "-".to_string(), |v| v.to_string()),
                a.metabolite_bioact_count
                    .map_or_else(|| "-".to_string(), |v| v.to_string())
            );
            let retro = match (a.synth_steps, a.synth_stock_pct) {
                (Some(st), Some(sp)) => format!("{st} step ({sp:.0}%)"),
                _ => "—".to_string(),
            };
            let _ = writeln!(
                md,
                "| {} | **P{}** | `{}` | `{}` | `{:.2}` | **`{:+.2}`** | {} | {} | `{}` | {} | {} |",
                a.rank,
                a.pareto_rank,
                a.name,
                a.transformation,
                a.consensus_dg_kcal,
                a.delta_dg_kcal,
                if a.posebusters_valid { "OK" } else { "WARN" },
                mpo,
                guard,
                retro,
                a.rationale
            );
        }
        let _ = writeln!(
            md,
            "\n- **Aligned Congeneric 3D Series SDF (for FEP/MD):** `{}`",
            self.series_sdf_path
        );
        if let Some(fp) = &self.fep_pair_sdf_path {
            let _ = writeln!(md, "- **Top RBFE Perturbation Pair SDF:** `{fp}`");
        }
        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::{Ad4Type, Atom3D, parse_smiles_to_3d, prepare_receptor};

    #[test]
    fn lead_opt_generates_pareto_ranked_3d_analogs_and_fep_pair()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut rec = Molecule3D::new("SynthPocket");
        for (i, &p) in [
            [6.5, 0.0, 0.0],
            [-6.5, 0.0, 0.0],
            [0.0, 6.5, 0.0],
            [0.0, -6.5, 0.0],
            [0.0, 0.0, 6.5],
            [0.0, 0.0, -6.5],
        ]
        .iter()
        .enumerate()
        {
            let mut a = Atom3D::new(i + 1, "C", p);
            a.res_name = "VAL".to_string();
            a.res_seq = (i + 1) as i32;
            a.ad4_type = Ad4Type::C;
            a.hydrophobic = true;
            rec.add_atom(a);
        }
        prepare_receptor(&mut rec, true);

        let lig = parse_smiles_to_3d("c1ccccc1", Some("BenzeneParent"))?;
        let base =
            std::env::temp_dir().join(format!("dockcraft_lead_opt_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let cfg = LeadOptConfig {
            mode: LeadOptMode::PocketGrowth,
            max_vectors: 2,
            max_analogs: 10,
            use_external_craft_clis: false,
            out_dir: base.clone(),
            ..Default::default()
        };
        let rep = run_lead_optimization(&rec, &lig, &cfg)?;
        assert!(rep.vectors_explored >= 1);
        assert!(rep.total_mutants_generated >= 4);
        assert!(rep.pareto_front_count >= 1);
        assert!(Path::new(&rep.series_sdf_path).exists());
        assert!(rep.fep_pair_sdf_path.is_some());
        assert!(!rep.render_markdown().is_empty());
        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}

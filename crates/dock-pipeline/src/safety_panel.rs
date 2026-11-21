#![forbid(unsafe_code)]

//! Automated **Inverse Docking Safety Pharmacology Panel** (`safety-panel`):
//! screens prioritized lead molecules in parallel against a curated panel of 15 Tier-1 Bowes
//! critical human anti-targets (hERG, 5-HT2B, ADRB2, CYP3A4, CYP2D6, CYP2C9, PXR, DRD2, CHRM1,
//! AChE, PPARγ, NR3C1, AR, ERα, COX-2), computes selectivity margins
//! (`ΔΔG = ΔG_offtarget - ΔG_target`), promiscuity indices, and flags 3D poses for ChimeraX.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use dock_engine::{DockingConfig, ScoringFunction, dock_ligand, score_complex};
use hyde_scorer::{consensus_affinity_kcal, score_pose_hyde};
use pocket_detect::GridBox;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use struct_core::{Molecule3D, parse_pdb, parse_pdbqt, prepare_receptor, write_sdf_3d};
use thiserror::Error;

use crate::campaign::LigandInputRecord;
use crate::md_validate::craft_data_dir;

#[derive(Debug, Error)]
pub enum SafetyPanelError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safety panel error: {0}")]
    Panel(String),
}

/// Default directory for curated safety anti-target PDBs & manifest (`CRAFT_DATA_DIR/safety_panel`).
#[must_use]
pub fn default_safety_panel_dir() -> PathBuf {
    craft_data_dir().join("safety_panel")
}

/// Default manifest JSON path (`CRAFT_DATA_DIR/safety_panel/panel.json`).
#[must_use]
pub fn default_safety_panel_manifest_path() -> PathBuf {
    default_safety_panel_dir().join("panel.json")
}

/// Risk tier for a single `(ligand, anti-target)` inverse-docking pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairRisk {
    Clear,
    Caution,
    RedFlag,
}

impl PairRisk {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Clear => "CLEAR",
            Self::Caution => "CAUTION",
            Self::RedFlag => "RED-FLAG",
        }
    }
}

/// Global inverse-docking safety verdict for one candidate ligand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyVerdict {
    Pass,
    Caution,
    Fail,
}

impl SafetyVerdict {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Caution => "CAUTION",
            Self::Fail => "FAIL",
        }
    }
}

/// Single anti-target receptor entry inside a `SafetyPanelManifest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiTargetEntry {
    pub id: String,
    pub name: String,
    pub pdb_id: String,
    #[serde(default)]
    pub chains: Vec<char>,
    pub prepared_pdb: PathBuf,
    pub box_center: [f64; 3],
    #[serde(default = "default_panel_box_size")]
    pub box_size: [f64; 3],
    pub organ_system: String,
    pub clinical_liability: String,
    /// Absolute affinity alert threshold (kcal/mol; e.g. `-8.8` -> more negative triggers alert).
    #[serde(default = "default_alert_dg")]
    pub alert_dg_kcal: f64,
    /// Optional residue sequence numbers used to anchor apo/cryo-EM binding cavities (e.g. hERG TYR652/PHE656).
    #[serde(default)]
    pub anchor_res_seqs: Vec<i32>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_panel_box_size() -> [f64; 3] {
    [22.0, 22.0, 22.0]
}
fn default_alert_dg() -> f64 {
    -8.8
}
fn default_true() -> bool {
    true
}

/// Serializable manifest describing all anti-targets in a Safety Panel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyPanelManifest {
    pub format: String,
    pub version: String,
    pub description: String,
    pub targets: Vec<AntiTargetEntry>,
}

impl SafetyPanelManifest {
    pub fn load_from_path(path: &Path) -> Result<Self, SafetyPanelError> {
        let raw = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), SafetyPanelError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

/// Built-in Tier-1 Bowes curated catalog of 15 critical human safety anti-targets.
#[must_use]
pub fn default_bowes_tier1_catalog(panel_dir: &Path) -> SafetyPanelManifest {
    let prep = |id: &str, pdb: &str| panel_dir.join("prepared").join(format!("{id}_{pdb}.pdb"));
    let targets = vec![
        AntiTargetEntry {
            id: "hERG".to_string(),
            name: "hERG K+ Channel (KCNH2 Cryo-EM)".to_string(),
            pdb_id: "5VA1".to_string(),
            chains: vec!['A', 'B', 'C', 'D'],
            prepared_pdb: prep("hERG", "5VA1"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [24.0, 24.0, 24.0],
            organ_system: "Cardiovascular".to_string(),
            clinical_liability: "QT prolongation / Torsades de Pointes arrhythmia".to_string(),
            alert_dg_kcal: -8.2,
            anchor_res_seqs: vec![652, 656],
            enabled: true,
        },
        AntiTargetEntry {
            id: "5HT2B".to_string(),
            name: "Serotonin 5-HT2B Receptor".to_string(),
            pdb_id: "4IB4".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("5HT2B", "4IB4"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [22.0, 22.0, 22.0],
            organ_system: "Cardiovascular".to_string(),
            clinical_liability: "Cardiac valvulopathy (mitral/aortic fibrosis)".to_string(),
            alert_dg_kcal: -8.6,
            anchor_res_seqs: vec![135, 340],
            enabled: true,
        },
        AntiTargetEntry {
            id: "ADRB2".to_string(),
            name: "Beta-2 Adrenergic Receptor".to_string(),
            pdb_id: "2RH1".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("ADRB2", "2RH1"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [22.0, 22.0, 22.0],
            organ_system: "Cardiovascular".to_string(),
            clinical_liability: "Tachycardia / bronchospasm".to_string(),
            alert_dg_kcal: -8.6,
            anchor_res_seqs: vec![113, 203],
            enabled: true,
        },
        AntiTargetEntry {
            id: "CYP3A4".to_string(),
            name: "Cytochrome P450 3A4".to_string(),
            pdb_id: "1TQN".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("CYP3A4", "1TQN"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [24.0, 24.0, 24.0],
            organ_system: "Hepatic / DDI".to_string(),
            clinical_liability: "Major drug-drug interaction & MBI hepatotoxicity".to_string(),
            alert_dg_kcal: -8.8,
            anchor_res_seqs: vec![304, 305, 369],
            enabled: true,
        },
        AntiTargetEntry {
            id: "CYP2D6".to_string(),
            name: "Cytochrome P450 2D6".to_string(),
            pdb_id: "4WNU".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("CYP2D6", "4WNU"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [22.0, 22.0, 22.0],
            organ_system: "Hepatic / DDI".to_string(),
            clinical_liability: "Polymorphic CYP2D6 inhibition / CNS DDI".to_string(),
            alert_dg_kcal: -8.8,
            anchor_res_seqs: vec![216, 301],
            enabled: true,
        },
        AntiTargetEntry {
            id: "CYP2C9".to_string(),
            name: "Cytochrome P450 2C9".to_string(),
            pdb_id: "1R9O".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("CYP2C9", "1R9O"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [22.0, 22.0, 22.0],
            organ_system: "Hepatic / DDI".to_string(),
            clinical_liability: "Warfarin / sulfonylurea severe DDI".to_string(),
            alert_dg_kcal: -8.8,
            anchor_res_seqs: vec![108, 293],
            enabled: true,
        },
        AntiTargetEntry {
            id: "PXR".to_string(),
            name: "Pregnane X Receptor (NR1I2)".to_string(),
            pdb_id: "1M13".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("PXR", "1M13"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [24.0, 24.0, 24.0],
            organ_system: "Hepatic / DDI".to_string(),
            clinical_liability: "CYP3A4 induction / loss of co-medication efficacy".to_string(),
            alert_dg_kcal: -8.8,
            anchor_res_seqs: vec![247, 285, 299],
            enabled: true,
        },
        AntiTargetEntry {
            id: "DRD2".to_string(),
            name: "Dopamine D2 Receptor".to_string(),
            pdb_id: "6CM4".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("DRD2", "6CM4"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [22.0, 22.0, 22.0],
            organ_system: "CNS".to_string(),
            clinical_liability: "Extrapyramidal symptoms / hyperprolactinemia".to_string(),
            alert_dg_kcal: -8.6,
            anchor_res_seqs: vec![114, 389],
            enabled: true,
        },
        AntiTargetEntry {
            id: "CHRM1".to_string(),
            name: "Muscarinic M1 Acetylcholine Receptor".to_string(),
            pdb_id: "5CXV".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("CHRM1", "5CXV"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [22.0, 22.0, 22.0],
            organ_system: "CNS / Autonomic".to_string(),
            clinical_liability: "Anticholinergic delirium / dry mouth / tachycardia".to_string(),
            alert_dg_kcal: -8.6,
            anchor_res_seqs: vec![105, 381],
            enabled: true,
        },
        AntiTargetEntry {
            id: "ACHE".to_string(),
            name: "Human Acetylcholinesterase".to_string(),
            pdb_id: "4EY7".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("ACHE", "4EY7"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [22.0, 22.0, 22.0],
            organ_system: "CNS / Neuromuscular".to_string(),
            clinical_liability: "Cholinergic crisis / bradycardia".to_string(),
            alert_dg_kcal: -9.0,
            anchor_res_seqs: vec![86, 203, 337],
            enabled: true,
        },
        AntiTargetEntry {
            id: "PPARG".to_string(),
            name: "PPAR-gamma Nuclear Receptor".to_string(),
            pdb_id: "2PRG".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("PPARG", "2PRG"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [24.0, 24.0, 24.0],
            organ_system: "Endocrine / CV".to_string(),
            clinical_liability: "Fluid retention / congestive heart failure / edema".to_string(),
            alert_dg_kcal: -8.8,
            anchor_res_seqs: vec![285, 323, 473],
            enabled: true,
        },
        AntiTargetEntry {
            id: "NR3C1".to_string(),
            name: "Glucocorticoid Receptor".to_string(),
            pdb_id: "1M2Z".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("NR3C1", "1M2Z"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [22.0, 22.0, 22.0],
            organ_system: "Endocrine / Immune".to_string(),
            clinical_liability: "Immunosuppression / metabolic & HPA-axis toxicity".to_string(),
            alert_dg_kcal: -9.0,
            anchor_res_seqs: vec![564, 611, 739],
            enabled: true,
        },
        AntiTargetEntry {
            id: "AR".to_string(),
            name: "Androgen Receptor".to_string(),
            pdb_id: "2AM9".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("AR", "2AM9"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [22.0, 22.0, 22.0],
            organ_system: "Endocrine".to_string(),
            clinical_liability: "Anti-androgenic / reproductive endocrine disruption".to_string(),
            alert_dg_kcal: -9.0,
            anchor_res_seqs: vec![705, 752, 877],
            enabled: true,
        },
        AntiTargetEntry {
            id: "ESR1".to_string(),
            name: "Estrogen Receptor Alpha".to_string(),
            pdb_id: "3ERT".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("ESR1", "3ERT"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [22.0, 22.0, 22.0],
            organ_system: "Endocrine".to_string(),
            clinical_liability: "Estrogenic / proliferative endocrine disruption".to_string(),
            alert_dg_kcal: -9.0,
            anchor_res_seqs: vec![353, 394, 524],
            enabled: true,
        },
        AntiTargetEntry {
            id: "COX2".to_string(),
            name: "Cyclooxygenase-2 (PTGS2)".to_string(),
            pdb_id: "5IKR".to_string(),
            chains: vec!['A'],
            prepared_pdb: prep("COX2", "5IKR"),
            box_center: [0.0, 0.0, 0.0],
            box_size: [22.0, 22.0, 22.0],
            organ_system: "Cardiovascular / Renal".to_string(),
            clinical_liability: "Cardiovascular thrombotic risk / nephrotoxicity".to_string(),
            alert_dg_kcal: -8.8,
            anchor_res_seqs: vec![120, 385, 530],
            enabled: true,
        },
    ];

    SafetyPanelManifest {
        format: "dockcraft-safety-panel".to_string(),
        version: "bowes-tier1-15-v0.6".to_string(),
        description:
            "Curated Tier-1 Bowes human safety pharmacology inverse-docking panel (15 anti-targets)"
                .to_string(),
        targets,
    }
}

/// Configuration for executing an Inverse Docking Safety Panel run.
#[derive(Debug, Clone)]
pub struct SafetyPanelConfig {
    pub exhaustiveness: usize,
    pub scoring: ScoringFunction,
    /// Optional primary target affinity (`kcal/mol`, e.g. `-11.5`) used to compute `ΔΔG = ΔG_off - ΔG_target`.
    pub target_affinity_kcal: Option<f64>,
    /// Minimum required selectivity margin (`ΔG_off - ΔG_target >= min_margin_kcal` for Clear; default `1.8`).
    pub min_margin_kcal: f64,
    /// Optional directory where flagged off-target 3D poses (`.sdf`) are exported for ChimeraX inspection.
    pub out_poses_dir: Option<PathBuf>,
}

impl Default for SafetyPanelConfig {
    fn default() -> Self {
        Self {
            exhaustiveness: 4,
            scoring: ScoringFunction::Vina,
            target_affinity_kcal: None,
            min_margin_kcal: 1.8,
            out_poses_dir: None,
        }
    }
}

/// Single `(ligand, anti-target)` inverse-docking result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffTargetDockHit {
    pub target_id: String,
    pub target_name: String,
    pub pdb_id: String,
    pub organ_system: String,
    pub clinical_liability: String,
    pub vina_dg_kcal: f64,
    pub consensus_dg_kcal: f64,
    /// `consensus_dg_kcal - target_affinity_kcal` (positive = safer margin over primary target).
    pub selectivity_margin_kcal: Option<f64>,
    pub alert_threshold_kcal: f64,
    pub risk: PairRisk,
    pub pose_sdf_path: Option<String>,
}

/// Aggregated safety-panel scorecard for one lead ligand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LigandSafetySummary {
    pub ligand_name: String,
    pub target_affinity_kcal: Option<f64>,
    pub red_flag_count: usize,
    pub caution_count: usize,
    pub clear_count: usize,
    /// Fraction of evaluated anti-targets flagged (`red_flag_count / total_targets`).
    pub promiscuity_index: f64,
    pub worst_offtarget_id: Option<String>,
    pub worst_offtarget_dg_kcal: Option<f64>,
    pub min_selectivity_margin_kcal: Option<f64>,
    pub verdict: SafetyVerdict,
    pub hits: Vec<OffTargetDockHit>,
}

/// Complete Inverse Docking Safety Panel report across 1..N candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyPanelReport {
    pub panel_version: String,
    pub targets_evaluated: usize,
    pub ligands_evaluated: usize,
    pub elapsed_ms: u128,
    pub summaries: Vec<LigandSafetySummary>,
}

/// Pre-loaded anti-target receptor ready for in-memory parallel inverse docking.
pub struct LoadedAntiTarget {
    pub entry: AntiTargetEntry,
    pub receptor: Molecule3D,
    pub grid_box: GridBox,
}

/// Loads all enabled anti-target PDB/PDBQT structures referenced in `manifest`.
pub fn load_safety_panel_receptors(
    manifest: &SafetyPanelManifest,
) -> Result<Vec<LoadedAntiTarget>, SafetyPanelError> {
    let mut loaded = Vec::new();
    for entry in manifest.targets.iter().filter(|t| t.enabled) {
        if !entry.prepared_pdb.is_file() {
            continue;
        }
        let raw = std::fs::read_to_string(&entry.prepared_pdb)?;
        let ext = entry
            .prepared_pdb
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("pdb")
            .to_ascii_lowercase();
        let mut rec = if ext == "pdbqt" {
            parse_pdbqt(&raw).map_err(|e| SafetyPanelError::Panel(e.to_string()))?
        } else {
            parse_pdb(&raw, true).map_err(|e| SafetyPanelError::Panel(e.to_string()))?
        };
        prepare_receptor(&mut rec, true);
        let grid_box = GridBox::from_center_size(entry.box_center, entry.box_size, 0.375)
            .map_err(|e| SafetyPanelError::Panel(e.to_string()))?;
        loaded.push(LoadedAntiTarget {
            entry: entry.clone(),
            receptor: rec,
            grid_box,
        });
    }
    if loaded.is_empty() {
        return Err(SafetyPanelError::Panel(
            "no prepared anti-target PDB structures found on disk; run `dockcraft-cli safety-panel --prepare-raw-dir ...` first".to_string(),
        ));
    }
    Ok(loaded)
}

/// Runs parallel Inverse Docking of `ligands` against all `receptors` in memory.
pub fn run_safety_panel_loaded(
    ligands: &[LigandInputRecord],
    receptors: &[LoadedAntiTarget],
    panel_version: &str,
    config: &SafetyPanelConfig,
) -> Result<SafetyPanelReport, SafetyPanelError> {
    if ligands.is_empty() || receptors.is_empty() {
        return Err(SafetyPanelError::Panel(
            "ligands and safety-panel receptors must both be non-empty".to_string(),
        ));
    }
    if let Some(dir) = &config.out_poses_dir {
        std::fs::create_dir_all(dir)?;
    }
    let t0 = Instant::now();
    let dock_cfg = DockingConfig {
        scoring: config.scoring,
        exhaustiveness: config.exhaustiveness.clamp(1, 32),
        num_modes: 3,
        ..Default::default()
    };

    let mut summaries = Vec::with_capacity(ligands.len());
    for lig in ligands {
        let hits_and_poses: Vec<(OffTargetDockHit, Option<Molecule3D>)> = receptors
            .par_iter()
            .map(|tgt| {
                let (vina_dg, cons_dg, best_pose) =
                    match dock_ligand(&tgt.receptor, &lig.molecule, &tgt.grid_box, &dock_cfg) {
                        Ok(rep) => {
                            if let Some(p0) = rep.poses.first() {
                                let vinardo = score_complex(
                                    &tgt.receptor,
                                    &p0.molecule,
                                    ScoringFunction::Vinardo,
                                );
                                let hyde_dg = score_pose_hyde(&tgt.receptor, &p0.molecule)
                                    .map_or(p0.affinity_kcal_mol, |h| h.total_delta_g_kcal);
                                let cons = consensus_affinity_kcal(
                                    p0.affinity_kcal_mol,
                                    vinardo.total_affinity_kcal,
                                    hyde_dg,
                                );
                                (p0.affinity_kcal_mol, cons, Some(p0.molecule.clone()))
                            } else {
                                (0.0, 0.0, None)
                            }
                        }
                        Err(_) => (0.0, 0.0, None),
                    };
                let vina_r = (vina_dg * 100.0).round() / 100.0;
                let cons_r = (cons_dg * 100.0).round() / 100.0;
                let margin = config
                    .target_affinity_kcal
                    .map(|tdg| ((cons_r - tdg) * 100.0).round() / 100.0);

                let risk = classify_pair_risk(
                    cons_r,
                    tgt.entry.alert_dg_kcal,
                    margin,
                    config.min_margin_kcal,
                );
                let hit = OffTargetDockHit {
                    target_id: tgt.entry.id.clone(),
                    target_name: tgt.entry.name.clone(),
                    pdb_id: tgt.entry.pdb_id.clone(),
                    organ_system: tgt.entry.organ_system.clone(),
                    clinical_liability: tgt.entry.clinical_liability.clone(),
                    vina_dg_kcal: vina_r,
                    consensus_dg_kcal: cons_r,
                    selectivity_margin_kcal: margin,
                    alert_threshold_kcal: tgt.entry.alert_dg_kcal,
                    risk,
                    pose_sdf_path: None,
                };
                (hit, best_pose)
            })
            .collect();

        let mut hits = Vec::with_capacity(hits_and_poses.len());
        let mut red = 0usize;
        let mut caution = 0usize;
        let mut clear = 0usize;

        for (mut hit, pose_opt) in hits_and_poses {
            match hit.risk {
                PairRisk::RedFlag => red += 1,
                PairRisk::Caution => caution += 1,
                PairRisk::Clear => clear += 1,
            }
            if hit.risk != PairRisk::Clear
                && let (Some(dir), Some(pose)) = (&config.out_poses_dir, pose_opt)
            {
                let safe_lig: String = lig
                    .name
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '_' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                let p = dir.join(format!("{safe_lig}_vs_{}.sdf", hit.target_id));
                if std::fs::write(&p, write_sdf_3d(&[&pose])).is_ok() {
                    hit.pose_sdf_path = Some(p.display().to_string());
                }
            }
            hits.push(hit);
        }

        hits.sort_by(|a, b| a.consensus_dg_kcal.total_cmp(&b.consensus_dg_kcal));
        let worst = hits.first();
        let min_margin = hits
            .iter()
            .filter_map(|h| h.selectivity_margin_kcal)
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |cur| cur.min(v)))
            });
        let total = hits.len().max(1) as f64;
        let promiscuity = ((red as f64) / total * 1000.0).round() / 1000.0;
        let verdict = if red >= 2
            || (red == 1
                && hits
                    .iter()
                    .any(|h| h.target_id == "hERG" && h.risk == PairRisk::RedFlag))
        {
            SafetyVerdict::Fail
        } else if red >= 1 || caution >= 2 {
            SafetyVerdict::Caution
        } else {
            SafetyVerdict::Pass
        };

        summaries.push(LigandSafetySummary {
            ligand_name: lig.name.clone(),
            target_affinity_kcal: config.target_affinity_kcal,
            red_flag_count: red,
            caution_count: caution,
            clear_count: clear,
            promiscuity_index: promiscuity,
            worst_offtarget_id: worst.map(|w| w.target_id.clone()),
            worst_offtarget_dg_kcal: worst.map(|w| w.consensus_dg_kcal),
            min_selectivity_margin_kcal: min_margin,
            verdict,
            hits,
        });
    }

    Ok(SafetyPanelReport {
        panel_version: panel_version.to_string(),
        targets_evaluated: receptors.len(),
        ligands_evaluated: summaries.len(),
        elapsed_ms: t0.elapsed().as_millis(),
        summaries,
    })
}

/// Convenience wrapper loading structures from `manifest` and executing inverse docking.
pub fn run_safety_panel(
    ligands: &[LigandInputRecord],
    manifest: &SafetyPanelManifest,
    config: &SafetyPanelConfig,
) -> Result<SafetyPanelReport, SafetyPanelError> {
    let receptors = load_safety_panel_receptors(manifest)?;
    run_safety_panel_loaded(ligands, &receptors, &manifest.version, config)
}

fn classify_pair_risk(
    off_dg: f64,
    alert_thresh: f64,
    margin: Option<f64>,
    min_margin: f64,
) -> PairRisk {
    let abs_red = off_dg <= alert_thresh;
    let abs_caution = off_dg <= (alert_thresh + 0.8);
    match margin {
        Some(m) => {
            if abs_red || m < 0.5 {
                PairRisk::RedFlag
            } else if m < min_margin || abs_caution {
                PairRisk::Caution
            } else {
                PairRisk::Clear
            }
        }
        None => {
            if abs_red {
                PairRisk::RedFlag
            } else if abs_caution {
                PairRisk::Caution
            } else {
                PairRisk::Clear
            }
        }
    }
}

impl SafetyPanelReport {
    /// Renders a clean Markdown Inverse Docking Safety Pharmacology Dossier.
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();
        let _ = writeln!(
            md,
            "# Inverse Docking Safety Pharmacology Report ({})\n",
            self.panel_version
        );
        let _ = writeln!(
            md,
            "- **Anti-Targets Evaluated:** `{}` | **Ligands Screened:** `{}` | **Runtime:** `{:.2} s`\n",
            self.targets_evaluated,
            self.ligands_evaluated,
            (self.elapsed_ms as f64) / 1000.0
        );

        md.push_str("## 1. Candidate Safety Summary\n\n");
        md.push_str("| Candidate | Verdict | Red-Flags | Caution | Clear | Promiscuity | Worst Off-Target (ΔG) | Min Margin ΔΔG |\n");
        md.push_str("|---|---|---:|---:|---:|---:|---|---:|\n");
        for s in &self.summaries {
            let worst = match (&s.worst_offtarget_id, s.worst_offtarget_dg_kcal) {
                (Some(id), Some(dg)) => format!("`{id}` ({dg:.2} kcal/mol)"),
                _ => "—".to_string(),
            };
            let mm = s
                .min_selectivity_margin_kcal
                .map_or_else(|| "—".to_string(), |v| format!("{v:+.2} kcal/mol"));
            let _ = writeln!(
                md,
                "| `{}` | **{}** | {} | {} | {} | `{:.2}` | {} | {} |",
                s.ligand_name,
                s.verdict.as_str(),
                s.red_flag_count,
                s.caution_count,
                s.clear_count,
                s.promiscuity_index,
                worst,
                mm
            );
        }
        md.push('\n');

        for s in &self.summaries {
            let _ = writeln!(md, "## 2. Off-Target Matrix — `{}`\n", s.ligand_name);
            md.push_str("| Anti-Target | PDB | Organ System | Risk | Consensus ΔG | Vina ΔG | Margin ΔΔG | Clinical Liability |\n");
            md.push_str("|---|---|---|---|---:|---:|---:|---|\n");
            for h in &s.hits {
                let mg = h
                    .selectivity_margin_kcal
                    .map_or_else(|| "—".to_string(), |v| format!("{v:+.2}"));
                let _ = writeln!(
                    md,
                    "| `{}` ({}) | `{}` | {} | **{}** | `{:.2}` | `{:.2}` | `{}` | {} |",
                    h.target_id,
                    h.target_name,
                    h.pdb_id,
                    h.organ_system,
                    h.risk.as_str(),
                    h.consensus_dg_kcal,
                    h.vina_dg_kcal,
                    mg,
                    h.clinical_liability
                );
            }
            md.push('\n');
        }
        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::{Ad4Type, Atom3D, parse_smiles_to_3d};

    fn make_mini_target(id: &str, radius: f64) -> LoadedAntiTarget {
        let mut rec = Molecule3D::new(id);
        let pts = [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
            [radius * 0.7, radius * 0.7, 0.0],
            [-radius * 0.7, -radius * 0.7, 0.0],
        ];
        for (i, &p) in pts.iter().enumerate() {
            let mut a = Atom3D::new(i + 1, "C", p);
            a.res_name = "VAL".to_string();
            a.res_seq = (i + 1) as i32;
            a.ad4_type = Ad4Type::C;
            a.hydrophobic = true;
            rec.add_atom(a);
        }
        prepare_receptor(&mut rec, true);
        let gbox = GridBox::from_center_size([0.0, 0.0, 0.0], [14.0, 14.0, 14.0], 0.375).unwrap_or(
            GridBox {
                center: [0.0, 0.0, 0.0],
                size: [14.0, 14.0, 14.0],
                spacing: 0.375,
            },
        );
        LoadedAntiTarget {
            entry: AntiTargetEntry {
                id: id.to_string(),
                name: format!("Synthetic {id}"),
                pdb_id: "TEST".to_string(),
                chains: vec!['A'],
                prepared_pdb: PathBuf::from("test.pdb"),
                box_center: [0.0, 0.0, 0.0],
                box_size: [14.0, 14.0, 14.0],
                organ_system: "Test".to_string(),
                clinical_liability: "Synthetic test liability".to_string(),
                alert_dg_kcal: -4.0,
                anchor_res_seqs: Vec::new(),
                enabled: true,
            },
            receptor: rec,
            grid_box: gbox,
        }
    }

    #[test]
    fn safety_panel_docks_and_computes_margins_and_markdown()
    -> Result<(), Box<dyn std::error::Error>> {
        let lig = parse_smiles_to_3d("c1ccccc1", Some("Benzene"))?;
        let recs = vec![
            make_mini_target("OPEN_POCKET", 5.0),
            make_mini_target("WIDE_POCKET", 6.5),
        ];
        let report = run_safety_panel_loaded(
            &[LigandInputRecord {
                name: "Benzene".to_string(),
                molecule: lig,
                is_active: None,
            }],
            &recs,
            "test-v1",
            &SafetyPanelConfig {
                exhaustiveness: 2,
                target_affinity_kcal: Some(-10.0),
                ..Default::default()
            },
        )?;
        assert_eq!(report.targets_evaluated, 2);
        assert_eq!(report.ligands_evaluated, 1);
        assert_eq!(report.summaries[0].hits.len(), 2);
        assert!(report.summaries[0].min_selectivity_margin_kcal.is_some());
        assert!(!report.render_markdown().is_empty());
        Ok(())
    }
}

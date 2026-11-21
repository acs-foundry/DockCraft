#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::time::Instant;

use dock_engine::grid::AtomChannelKey;
use dock_engine::{AffinityGridSet, DockingConfig, DockingPose, dock_ligand_with_grids};
use dock_pipeline::{EnrichmentMetrics, LigandInputRecord};
use hyde_scorer::{consensus_affinity_kcal, score_pose_hyde};
use pocket_detect::GridBox;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use struct_core::{Ad4Type, Molecule3D};

/// Consensus aggregation strategy across a multi-conformation receptor ensemble.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EnsembleAggregation {
    /// Boltzmann thermodynamic average `-RT ln((1/N) Σ exp(-ΔG_i / RT))` at 298.15 K (`RT = 0.5922 kcal/mol`).
    #[default]
    Boltzmann,
    /// Induced-fit best conformation (`min_i ΔG_i`).
    BestConformation,
    /// Arithmetic mean across all conformations.
    Mean,
    /// Reciprocal Rank Fusion (`1 / Σ (1 / (60 + rank_i))`) normalized as pseudo-ΔG ordering.
    ReciprocalRank,
}

impl EnsembleAggregation {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Boltzmann => "Boltzmann",
            Self::BestConformation => "BestConformation",
            Self::Mean => "Mean",
            Self::ReciprocalRank => "ReciprocalRank",
        }
    }
}

/// One receptor conformation inside an ensemble docking run.
#[derive(Debug, Clone)]
pub struct EnsembleReceptor {
    pub label: String,
    pub receptor: Molecule3D,
    pub grid_box: GridBox,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformationScore {
    pub receptor_label: String,
    pub vina_kcal: f64,
    pub hyde_kcal: f64,
    pub consensus_kcal: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleCandidate {
    pub rank: usize,
    pub name: String,
    pub is_active: Option<bool>,
    pub ensemble_score_kcal: f64,
    pub best_single_kcal: f64,
    pub mean_kcal: f64,
    pub spread_std_kcal: f64,
    pub best_receptor: String,
    pub per_conformation: Vec<ConformationScore>,
    pub best_pose: DockingPose,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleReport {
    pub receptor_labels: Vec<String>,
    pub aggregation: EnsembleAggregation,
    pub use_hyde_consensus: bool,
    pub candidates: Vec<EnsembleCandidate>,
    pub enrichment: Option<EnrichmentMetrics>,
    pub elapsed_ms: u128,
}

impl EnsembleReport {
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "Rank,Compound,IsActive,Ensemble_dG_kcal,BestSingle_kcal,Mean_kcal,SpreadStd_kcal,BestReceptor",
        );
        for lbl in &self.receptor_labels {
            let _ = write!(out, ",{lbl}_kcal");
        }
        out.push('\n');
        for c in &self.candidates {
            let act = match c.is_active {
                Some(true) => "1",
                Some(false) => "0",
                None => "",
            };
            let _ = write!(
                out,
                "{},{},{},{:.3},{:.3},{:.3},{:.3},{}",
                c.rank,
                c.name,
                act,
                c.ensemble_score_kcal,
                c.best_single_kcal,
                c.mean_kcal,
                c.spread_std_kcal,
                c.best_receptor
            );
            for pc in &c.per_conformation {
                let _ = write!(out, ",{:.3}", pc.consensus_kcal);
            }
            out.push('\n');
        }
        out
    }

    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        let _ = writeln!(
            md,
            "# Ensemble Docking Report ({} conformations · {} aggregation)\n",
            self.receptor_labels.len(),
            self.aggregation.as_str()
        );
        let _ = writeln!(md, "- **Receptors:** {}", self.receptor_labels.join(", "));
        let _ = writeln!(
            md,
            "- **Ligands Docked:** {} | **Elapsed:** {} ms",
            self.candidates.len(),
            self.elapsed_ms
        );
        if let Some(enr) = &self.enrichment {
            let _ = writeln!(
                md,
                "- **Enrichment:** ROC AUC `{:.3}` | BEDROC(20) `{:.3}` | EF1% `{:.2}` | EF5% `{:.2}` ({} actives / {})",
                enr.roc_auc,
                enr.bedroc_20,
                enr.ef_1_percent,
                enr.ef_5_percent,
                enr.total_actives,
                enr.total_compounds
            );
        }
        md.push_str("\n| Rank | Compound | Ensemble ΔG | Best ΔG | Mean ± σ | Best Receptor |\n|---:|---|---:|---:|---:|---|\n");
        for c in self.candidates.iter().take(25) {
            let _ = writeln!(
                md,
                "| {} | {} | {:.2} | {:.2} | {:.2} ± {:.2} | `{}` |",
                c.rank,
                c.name,
                c.ensemble_score_kcal,
                c.best_single_kcal,
                c.mean_kcal,
                c.spread_std_kcal,
                c.best_receptor
            );
        }
        md
    }
}

/// Runs parallel multi-conformation ensemble docking over `receptors` and `ligands`, consolidating
/// poses and affinities via `aggregation` (default: Boltzmann thermodynamic consensus at 298.15 K).
pub fn run_ensemble_docking(
    receptors: &[EnsembleReceptor],
    ligands: &[LigandInputRecord],
    cfg: &DockingConfig,
    aggregation: EnsembleAggregation,
    use_hyde_consensus: bool,
) -> Result<EnsembleReport, String> {
    if receptors.is_empty() {
        return Err("ensemble docking requires at least 1 receptor conformation".to_string());
    }
    if ligands.is_empty() {
        return Err("ensemble docking requires at least 1 ligand".to_string());
    }
    let t0 = Instant::now();

    let enable_metal =
        cfg.metal_coordination || cfg.scoring == dock_engine::ScoringFunction::DockCraftHybrid;

    // Collect union of required AD4 atom channels across all ligands once
    let mut keys: Vec<AtomChannelKey> = Vec::new();
    for rec in ligands {
        for (i, a) in rec.molecule.atoms.iter().enumerate() {
            if !a.is_heavy() {
                continue;
            }
            let k = AtomChannelKey::from_mol_atom(&rec.molecule, i, enable_metal);
            if !keys.contains(&k) {
                keys.push(k);
            }
        }
    }
    if keys.is_empty() {
        keys.push(AtomChannelKey {
            ad4: Ad4Type::C,
            h_donor: false,
            h_acceptor: false,
            hydrophobic: true,
            chelator_tier: 0,
        });
    }

    // Precompute affinity grids for each receptor conformation in parallel
    let grids_res: Result<Vec<AffinityGridSet>, String> = receptors
        .par_iter()
        .map(|er| {
            AffinityGridSet::build_with_options(
                &er.receptor,
                &er.grid_box,
                &keys,
                cfg.scoring,
                enable_metal,
                None,
            )
            .map_err(|e| format!("grid build failed for '{}': {e}", er.label))
        })
        .collect();
    let grids = grids_res?;

    // Dock every ligand against every conformation in parallel
    let mut raw_candidates: Vec<EnsembleCandidate> = ligands
        .par_iter()
        .filter_map(|lig| {
            let mut per_conf = Vec::with_capacity(receptors.len());
            let mut best_pose: Option<DockingPose> = None;
            let mut best_single = f64::INFINITY;
            let mut best_rec_label = String::new();

            for (ri, er) in receptors.iter().enumerate() {
                let mut local_cfg = cfg.clone();
                local_cfg.num_modes = local_cfg.num_modes.max(3);
                let Ok(rep) =
                    dock_ligand_with_grids(&er.receptor, &lig.molecule, &grids[ri], &local_cfg)
                else {
                    continue;
                };
                let Some(top_pose) = rep.poses.into_iter().next() else {
                    continue;
                };
                let vina_dg = top_pose.affinity_kcal_mol;
                let hyde_dg = score_pose_hyde(&er.receptor, &top_pose.molecule)
                    .map_or(vina_dg, |h| h.total_delta_g_kcal);
                let eff_dg = if use_hyde_consensus {
                    consensus_affinity_kcal(vina_dg, vina_dg, hyde_dg)
                } else {
                    vina_dg
                };
                if eff_dg < best_single {
                    best_single = eff_dg;
                    best_rec_label = er.label.clone();
                    best_pose = Some(top_pose);
                }
                per_conf.push(ConformationScore {
                    receptor_label: er.label.clone(),
                    vina_kcal: round3(vina_dg),
                    hyde_kcal: round3(hyde_dg),
                    consensus_kcal: round3(eff_dg),
                });
            }

            let best_pose = best_pose?;
            let vals: Vec<f64> = per_conf.iter().map(|c| c.consensus_kcal).collect();
            let n = vals.len().max(1) as f64;
            let mean = vals.iter().sum::<f64>() / n;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt();
            let boltz = boltzmann_average_kcal(&vals);

            let ens_score = match aggregation {
                EnsembleAggregation::Boltzmann => boltz,
                EnsembleAggregation::BestConformation => best_single,
                EnsembleAggregation::Mean => mean,
                EnsembleAggregation::ReciprocalRank => boltz, // refined below after per-receptor ranking
            };

            Some(EnsembleCandidate {
                rank: 0,
                name: lig.name.clone(),
                is_active: lig.is_active,
                ensemble_score_kcal: round3(ens_score),
                best_single_kcal: round3(best_single),
                mean_kcal: round3(mean),
                spread_std_kcal: round3(std),
                best_receptor: best_rec_label,
                per_conformation: per_conf,
                best_pose,
            })
        })
        .collect();

    if aggregation == EnsembleAggregation::ReciprocalRank && !raw_candidates.is_empty() {
        let m = raw_candidates.len();
        let mut rrf = vec![0.0f64; m];
        for ri in 0..receptors.len() {
            let mut order: Vec<usize> = (0..m).collect();
            order.sort_by(|&a, &b| {
                let va = raw_candidates[a]
                    .per_conformation
                    .get(ri)
                    .map_or(0.0, |c| c.consensus_kcal);
                let vb = raw_candidates[b]
                    .per_conformation
                    .get(ri)
                    .map_or(0.0, |c| c.consensus_kcal);
                va.total_cmp(&vb)
            });
            for (rank0, &idx) in order.iter().enumerate() {
                rrf[idx] += 1.0 / (60.0 + (rank0 as f64) + 1.0);
            }
        }
        // Sort descending by RRF score, tie-break by Boltzmann ΔG
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| {
            rrf[b].total_cmp(&rrf[a]).then_with(|| {
                raw_candidates[a]
                    .ensemble_score_kcal
                    .total_cmp(&raw_candidates[b].ensemble_score_kcal)
            })
        });
        let mut reordered = Vec::with_capacity(m);
        for (new_r, idx) in indices.into_iter().enumerate() {
            let mut c = raw_candidates[idx].clone();
            c.rank = new_r + 1;
            reordered.push(c);
        }
        raw_candidates = reordered;
    } else {
        raw_candidates.sort_by(|a, b| a.ensemble_score_kcal.total_cmp(&b.ensemble_score_kcal));
        for (i, c) in raw_candidates.iter_mut().enumerate() {
            c.rank = i + 1;
        }
    }

    let enrichment = compute_enrichment(&raw_candidates);
    Ok(EnsembleReport {
        receptor_labels: receptors.iter().map(|r| r.label.clone()).collect(),
        aggregation,
        use_hyde_consensus,
        candidates: raw_candidates,
        enrichment,
        elapsed_ms: t0.elapsed().as_millis(),
    })
}

/// Thermodynamically rigorous Boltzmann-weighted ensemble free energy at 298.15 K
/// (`RT = 0.5922 kcal/mol`): `-RT * ln((1/N) Σ exp(-ΔG_i / RT))`.
#[must_use]
pub fn boltzmann_average_kcal(dgs: &[f64]) -> f64 {
    if dgs.is_empty() {
        return 0.0;
    }
    const RT: f64 = 0.5922;
    let min_dg = dgs.iter().copied().fold(f64::INFINITY, f64::min);
    let sum_exp: f64 = dgs.iter().map(|&g| (-(g - min_dg) / RT).exp()).sum();
    let mean_exp = (sum_exp / (dgs.len() as f64)).max(1e-30);
    min_dg - RT * mean_exp.ln()
}

fn compute_enrichment(cands: &[EnsembleCandidate]) -> Option<EnrichmentMetrics> {
    let labeled: Vec<bool> = cands.iter().filter_map(|c| c.is_active).collect();
    let total = labeled.len();
    let actives = labeled.iter().filter(|&&b| b).count();
    let decoys = total.saturating_sub(actives);
    if actives == 0 || decoys == 0 {
        return None;
    }
    let mut tp = 0.0;
    let mut auc_sum = 0.0;
    for &is_act in &labeled {
        if is_act {
            tp += 1.0;
        } else {
            auc_sum += tp;
        }
    }
    let roc_auc = auc_sum / ((actives as f64) * (decoys as f64));

    let ef = |frac: f64| -> f64 {
        let k = ((total as f64) * frac).ceil().max(1.0) as usize;
        let k = k.min(total);
        let hits = labeled.iter().take(k).filter(|&&b| b).count() as f64;
        (hits / (k as f64)) / ((actives as f64) / (total as f64))
    };

    // BEDROC(alpha=20.0)
    let alpha = 20.0f64;
    let n_f = total as f64;
    let na_f = actives as f64;
    let ra = na_f / n_f;
    let sum_rie: f64 = labeled
        .iter()
        .enumerate()
        .filter(|(_, b)| **b)
        .map(|(i, _)| (-alpha * ((i as f64) + 1.0) / n_f).exp())
        .sum();
    let denom =
        (ra / (1.0 - (-alpha).exp())) * ((1.0 - (-alpha).exp()) / ((alpha / n_f).exp() - 1.0));
    let rie = if denom.abs() > 1e-12 {
        (sum_rie / na_f) / (denom / na_f).max(1e-12)
    } else {
        0.0
    };
    let rie_min = (1.0 - (alpha * ra).exp()) / (ra * (1.0 - alpha.exp()));
    let rie_max = (1.0 - (-alpha * ra).exp()) / (ra * (1.0 - (-alpha).exp()));
    let bedroc_20 = if (rie_max - rie_min).abs() > 1e-12 {
        ((rie - rie_min) / (rie_max - rie_min)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Some(EnrichmentMetrics {
        total_compounds: total,
        total_actives: actives,
        roc_auc: round3(roc_auc),
        bedroc_20: round3(bedroc_20),
        ef_1_percent: round3(ef(0.01)),
        ef_5_percent: round3(ef(0.05)),
    })
}

#[inline]
fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

#![forbid(unsafe_code)]

//! Structural Pocket Selectivity Analysis (**ΔMIF**, ProBiS / PocketMatch-inspired):
//! aligns 1..N paralog/off-target binding pockets onto the primary target pocket via invariant
//! geometric triangle hashing + trimmed Kabsch ICP refinement, audits conserved vs differential
//! lining residues, evaluates voxel-matched 7-probe Differential Molecular Interaction Fields
//! (`ΔE = E_target - min_k E_offtarget_k`) and Steric-Exclusivity Subpockets, emits ChimeraX
//! `.cxc` overlays, and augments `.pharm.json` blueprints to enrich selective ligands.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::Instant;

use dock_engine::PharmacophoreKind;
use pocket_detect::GridBox;
use serde::{Deserialize, Serialize};
use struct_core::Molecule3D;

use crate::PharmError;
use crate::kabsch::{RigidAlignment, kabsch_align};
use crate::mif::{
    FeatureSource, PROBE_KINDS, SiteBlueprint, SiteFeature, evaluate_mif_at_points,
    extract_target_cavity_voxels,
};

/// Configuration for structural pocket selectivity comparison (`compare-pockets`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketSelectivityConfig {
    /// Voxel spacing for ΔMIF comparison in Å (default `0.75`).
    pub spacing_a: f64,
    /// Radius around each pocket center for Cα extraction during superposition (default `13.5` Å).
    pub pocket_radius_a: f64,
    /// Skip rigid alignment when off-targets are already superposed onto the target frame.
    pub pre_aligned: bool,
    /// Target probe energy threshold (kcal/mol) required to qualify as a hotspot (default `-1.25`).
    pub min_target_energy_kcal: f64,
    /// Differential ΔE threshold (`E_target - E_offtarget <= min_delta_energy_kcal`, default `-0.85`).
    pub min_delta_energy_kcal: f64,
    /// Maximum selectivity hotspots emitted (default `10`).
    pub max_hotspots: usize,
}

impl Default for PocketSelectivityConfig {
    fn default() -> Self {
        Self {
            spacing_a: 0.75,
            pocket_radius_a: 13.5,
            pre_aligned: false,
            min_target_energy_kcal: -1.25,
            min_delta_energy_kcal: -0.85,
            max_hotspots: 10,
        }
    }
}

/// Classification of a 3D pocket selectivity hotspot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectivityHotspotKind {
    /// Accessible in target pocket, sterically blocked by a bulky wall residue in off-target(s).
    StericExclusivitySubpocket,
    /// Pharmacophore probe is significantly more favorable in target than in off-target(s).
    DifferentialPharmacophore(PharmacophoreKind),
    /// Strongly favorable in both target and all off-targets (core potency anchor, not selective).
    ConservedAnchor(PharmacophoreKind),
}

impl SelectivityHotspotKind {
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::StericExclusivitySubpocket => "StericExclusivity".to_string(),
            Self::DifferentialPharmacophore(k) => format!("Diff-{k:?}"),
            Self::ConservedAnchor(k) => format!("Conserved-{k:?}"),
        }
    }
}

/// Single actionable 3D selectivity hotspot inside the target binding site.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectivityHotspot {
    pub id: String,
    pub kind: SelectivityHotspotKind,
    pub center: [f64; 3],
    pub radius_a: f64,
    pub target_energy_kcal: f64,
    pub worst_offtarget_energy_kcal: f64,
    /// `target_energy - worst_offtarget_energy` (more negative = stronger selectivity driver).
    pub delta_energy_kcal: f64,
    pub target_residue: String,
    pub offtarget_blocking_or_counter_residue: String,
    pub design_guidance: String,
}

/// Pairwise lining-residue correspondence between target and an aligned off-target pocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueSubstitution {
    pub target_residue: String,
    pub offtarget_residue: String,
    pub ca_distance_a: f64,
    pub conserved: bool,
    pub property_change: String,
}

/// Pairwise alignment & similarity summary between target and one off-target receptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffTargetComparisonSummary {
    pub offtarget_name: String,
    pub pocket_ca_rmsd_a: f64,
    pub matched_ca_count: usize,
    pub pocket_sequence_identity_pct: f64,
    pub cavity_jaccard: f64,
    pub mif_cosine_similarity: f64,
    pub steric_exclusive_voxel_count: usize,
    pub differential_residues: Vec<ResidueSubstitution>,
    pub conserved_residues_count: usize,
}

/// Complete structural pocket selectivity report comparing 1 target vs 1..N off-targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectivityReport {
    pub target_name: String,
    pub target_grid_box: GridBox,
    pub spacing_a: f64,
    pub target_cavity_voxels: usize,
    /// Global feasibility of achieving structural selectivity `[0.0, 1.0]` (higher = easier).
    pub selectivity_feasibility_score: f64,
    pub offtargets: Vec<OffTargetComparisonSummary>,
    pub selectivity_hotspots: Vec<SelectivityHotspot>,
    pub conserved_anchors: Vec<SelectivityHotspot>,
    pub elapsed_ms: u128,
}

/// Input specification for one off-target / paralog receptor.
pub struct OffTargetPocketInput<'a> {
    pub name: String,
    pub receptor: &'a Molecule3D,
    pub pocket_center: [f64; 3],
}

/// Output bundle containing both the quantitative `SelectivityReport` and the superposed off-target
/// structures (in the target coordinate frame, ready to export as PDB / ChimeraX scene).
pub struct SelectivityAnalysisOutput {
    pub report: SelectivityReport,
    pub aligned_offtargets: Vec<(String, Molecule3D)>,
}

#[derive(Clone)]
struct CaPoint {
    coords: [f64; 3],
    chain: char,
    res_name: String,
    res_seq: i32,
}

impl CaPoint {
    fn label(&self) -> String {
        format!("{}:{}{}", self.chain, self.res_name.trim(), self.res_seq)
    }
}

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn extract_pocket_cas(rec: &Molecule3D, center: [f64; 3], radius_a: f64) -> Vec<CaPoint> {
    let mut out = Vec::new();
    for a in &rec.atoms {
        if a.is_hetatm {
            continue;
        }
        if a.name.trim().eq_ignore_ascii_case("CA") && dist3(a.coords, center) <= radius_a {
            out.push(CaPoint {
                coords: a.coords,
                chain: a.chain_id,
                res_name: a.res_name.trim().to_ascii_uppercase(),
                res_seq: a.res_seq,
            });
        }
    }
    // Fallback for minimal synthetic structures without CA backbone: use heavy atoms near pocket
    if out.len() < 4 {
        out.clear();
        let mut seen = HashMap::new();
        for a in &rec.atoms {
            if a.is_heavy() && !a.is_hetatm && dist3(a.coords, center) <= radius_a {
                let key = (a.chain_id, a.res_seq);
                seen.entry(key).or_insert_with(|| CaPoint {
                    coords: a.coords,
                    chain: a.chain_id,
                    res_name: a.res_name.trim().to_ascii_uppercase(),
                    res_seq: a.res_seq,
                });
            }
        }
        out.extend(seen.into_values());
    }
    out.sort_by(|a, b| {
        a.chain
            .cmp(&b.chain)
            .then_with(|| a.res_seq.cmp(&b.res_seq))
    });
    out
}

fn residue_class(rn: &str) -> &'static str {
    match rn {
        "ASP" | "GLU" | "ASH" | "GLH" => "Acidic(-)",
        "LYS" | "ARG" | "LYN" => "Basic(+)",
        "HIS" | "HID" | "HIE" | "HIP" => "Histidine/Arom",
        "PHE" | "TYR" | "TRP" => "Bulky Aromatic",
        "LEU" | "ILE" | "VAL" | "MET" | "PRO" => "Hydrophobic",
        "ALA" | "GLY" => "Small Aliphatic",
        "SER" | "THR" | "ASN" | "GLN" | "CYS" | "CYX" | "CYM" => "Polar",
        _ => "Other",
    }
}

fn apply_alignment_to_mol(mol: &Molecule3D, aln: &RigidAlignment) -> Molecule3D {
    let mut out = mol.clone();
    for a in &mut out.atoms {
        a.coords = aln.apply(a.coords);
    }
    out
}

/// Superposes `off_rec` onto `tgt_rec` centered on their respective binding pockets using
/// sequence-numbered anchor seeding when available, falling back to invariant geometric triangle
/// hashing, followed by 8 iterations of trimmed mutual-nearest-neighbor Kabsch ICP refinement.
fn align_offtarget_to_target_pocket(
    tgt_cas: &[CaPoint],
    off_cas: &[CaPoint],
    pre_aligned: bool,
) -> (RigidAlignment, f64, usize) {
    let identity = RigidAlignment {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0, 0.0, 0.0],
        rmsd: 0.0,
    };
    if pre_aligned || tgt_cas.len() < 3 || off_cas.len() < 3 {
        let (rmsd, matched) = eval_ca_alignment(tgt_cas, off_cas, &identity, 2.5);
        return (identity, rmsd, matched);
    }

    // 1. Candidate seed A: exact (res_seq, res_name) pairs (fast & exact for homologous numbering e.g. CDK2 vs CDK1)
    let mut best_aln = identity;
    let mut best_score = -1.0_f64;

    let mut exact_mob = Vec::new();
    let mut exact_tgt = Vec::new();
    for tc in tgt_cas {
        if let Some(oc) = off_cas
            .iter()
            .find(|o| o.res_seq == tc.res_seq && o.res_name == tc.res_name)
        {
            exact_tgt.push(tc.coords);
            exact_mob.push(oc.coords);
        }
    }
    if exact_mob.len() >= 4 {
        let cand = kabsch_align(&exact_mob, &exact_tgt);
        let (rmsd, cnt) = eval_ca_alignment(tgt_cas, off_cas, &cand, 2.5);
        let sc = (cnt as f64) - 0.5 * rmsd;
        if sc > best_score {
            best_score = sc;
            best_aln = cand;
        }
    }

    // 2. Candidate seed B: rotation/translation-invariant geometric triangle hashing (handles shifted numbering e.g. CDK2 vs CDK4)
    let need_triangle_search = {
        let (_, cnt) = eval_ca_alignment(tgt_cas, off_cas, &best_aln, 2.5);
        cnt < (tgt_cas.len().min(off_cas.len()) / 2).max(4)
    };
    if need_triangle_search {
        let t_sub: Vec<&CaPoint> = tgt_cas.iter().take(28).collect();
        let o_sub: Vec<&CaPoint> = off_cas.iter().take(28).collect();

        // Build target triangles with well-conditioned edges in [3.5, 15.0] Å
        let mut tgt_tris: Vec<(usize, usize, usize, [f64; 3])> = Vec::new();
        for i in 0..t_sub.len() {
            for j in (i + 1)..t_sub.len() {
                let d01 = dist3(t_sub[i].coords, t_sub[j].coords);
                if !(3.5..=15.0).contains(&d01) {
                    continue;
                }
                for k in (j + 1)..t_sub.len() {
                    let d12 = dist3(t_sub[j].coords, t_sub[k].coords);
                    let d20 = dist3(t_sub[k].coords, t_sub[i].coords);
                    if (3.5..=15.0).contains(&d12) && (3.5..=15.0).contains(&d20) {
                        tgt_tris.push((i, j, k, [d01, d12, d20]));
                        if tgt_tris.len() >= 450 {
                            break;
                        }
                    }
                }
                if tgt_tris.len() >= 450 {
                    break;
                }
            }
            if tgt_tris.len() >= 450 {
                break;
            }
        }

        let mut o_tris: Vec<(usize, usize, usize, [f64; 3])> = Vec::new();
        for i in 0..o_sub.len() {
            for j in (i + 1)..o_sub.len() {
                let d01 = dist3(o_sub[i].coords, o_sub[j].coords);
                if !(3.5..=15.0).contains(&d01) {
                    continue;
                }
                for k in (j + 1)..o_sub.len() {
                    let d12 = dist3(o_sub[j].coords, o_sub[k].coords);
                    let d20 = dist3(o_sub[k].coords, o_sub[i].coords);
                    if (3.5..=15.0).contains(&d12) && (3.5..=15.0).contains(&d20) {
                        o_tris.push((i, j, k, [d01, d12, d20]));
                        if o_tris.len() >= 450 {
                            break;
                        }
                    }
                }
                if o_tris.len() >= 450 {
                    break;
                }
            }
            if o_tris.len() >= 450 {
                break;
            }
        }

        let perms: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut tested = 0usize;
        for &(ti, tj, tk, td) in &tgt_tris {
            let t_pts = [t_sub[ti].coords, t_sub[tj].coords, t_sub[tk].coords];
            let t_names = [
                &t_sub[ti].res_name,
                &t_sub[tj].res_name,
                &t_sub[tk].res_name,
            ];
            for &(oi, oj, ok, _) in &o_tris {
                let o_idx = [oi, oj, ok];
                for p in &perms {
                    let oa = o_sub[o_idx[p[0]]];
                    let ob = o_sub[o_idx[p[1]]];
                    let oc = o_sub[o_idx[p[2]]];
                    let d01 = dist3(oa.coords, ob.coords);
                    if (d01 - td[0]).abs() > 0.65 {
                        continue;
                    }
                    let d12 = dist3(ob.coords, oc.coords);
                    if (d12 - td[1]).abs() > 0.65 {
                        continue;
                    }
                    let d20 = dist3(oc.coords, oa.coords);
                    if (d20 - td[2]).abs() > 0.65 {
                        continue;
                    }
                    // Prefer triangles with at least 1 conserved residue identity when pockets are large
                    let id_matches = usize::from(&oa.res_name == t_names[0])
                        + usize::from(&ob.res_name == t_names[1])
                        + usize::from(&oc.res_name == t_names[2]);
                    if tgt_cas.len() >= 12 && id_matches == 0 {
                        continue;
                    }
                    let cand = kabsch_align(&[oa.coords, ob.coords, oc.coords], &t_pts);
                    let (rmsd, cnt) = eval_ca_alignment(tgt_cas, off_cas, &cand, 2.5);
                    let sc = (cnt as f64) + 0.15 * (id_matches as f64) - 0.5 * rmsd;
                    if sc > best_score {
                        best_score = sc;
                        best_aln = cand;
                    }
                    tested += 1;
                    if tested >= 3_000 {
                        break;
                    }
                }
                if tested >= 3_000 {
                    break;
                }
            }
            if tested >= 3_000 {
                break;
            }
        }
    }

    // 3. Trimmed mutual-nearest-neighbor ICP Kabsch refinement (8 iterations)
    let mut cur = best_aln;
    for iter in 0..8 {
        let cutoff = if iter < 4 { 3.2 } else { 2.3 };
        let mut mob = Vec::new();
        let mut tgt = Vec::new();
        for tc in tgt_cas {
            let mut best_o: Option<(usize, f64)> = None;
            for (oi, oc) in off_cas.iter().enumerate() {
                let d = dist3(cur.apply(oc.coords), tc.coords);
                if d <= cutoff && best_o.is_none_or(|(_, bd)| d < bd) {
                    best_o = Some((oi, d));
                }
            }
            if let Some((oi, _)) = best_o {
                mob.push(off_cas[oi].coords);
                tgt.push(tc.coords);
            }
        }
        if mob.len() < 3 {
            break;
        }
        cur = kabsch_align(&mob, &tgt);
    }

    let (final_rmsd, final_cnt) = eval_ca_alignment(tgt_cas, off_cas, &cur, 2.5);
    (cur, final_rmsd, final_cnt)
}

fn eval_ca_alignment(
    tgt_cas: &[CaPoint],
    off_cas: &[CaPoint],
    aln: &RigidAlignment,
    cutoff: f64,
) -> (f64, usize) {
    let mut sum_sq = 0.0_f64;
    let mut cnt = 0usize;
    for tc in tgt_cas {
        let mut best_d = f64::INFINITY;
        for oc in off_cas {
            let d = dist3(aln.apply(oc.coords), tc.coords);
            if d < best_d {
                best_d = d;
            }
        }
        if best_d <= cutoff {
            sum_sq += best_d * best_d;
            cnt += 1;
        }
    }
    if cnt == 0 {
        (99.0, 0)
    } else {
        ((sum_sq / (cnt as f64)).sqrt(), cnt)
    }
}

/// Runs full Structural Pocket Selectivity (ΔMIF) analysis comparing `target_receptor` against
/// 1..N `offtargets` on the exact target active-site voxel grid.
pub fn analyze_pocket_selectivity(
    target_name: &str,
    target_receptor: &Molecule3D,
    target_grid_box: &GridBox,
    holo_ligands: &[&Molecule3D],
    offtargets: &[OffTargetPocketInput<'_>],
    config: &PocketSelectivityConfig,
) -> Result<SelectivityAnalysisOutput, PharmError> {
    let t0 = Instant::now();
    if target_receptor.heavy_atom_count() == 0 || offtargets.is_empty() {
        return Err(PharmError::EmptyStructure);
    }

    let (eff_spacing, cavity_coords) = extract_target_cavity_voxels(
        target_receptor,
        target_grid_box,
        holo_ligands,
        config.spacing_a,
    )?;
    if cavity_coords.is_empty() {
        return Err(PharmError::NoPocketAtoms);
    }

    let tgt_samples = evaluate_mif_at_points(target_receptor, target_grid_box, &cavity_coords);
    let tgt_cas = extract_pocket_cas(
        target_receptor,
        target_grid_box.center,
        config.pocket_radius_a,
    );

    let mut aligned_offtargets = Vec::with_capacity(offtargets.len());
    let mut all_off_samples = Vec::with_capacity(offtargets.len());
    let mut off_summaries = Vec::with_capacity(offtargets.len());

    for ot in offtargets {
        let off_cas = extract_pocket_cas(ot.receptor, ot.pocket_center, config.pocket_radius_a);
        let (aln, ca_rmsd, matched_ca) =
            align_offtarget_to_target_pocket(&tgt_cas, &off_cas, config.pre_aligned);
        let aligned_mol = apply_alignment_to_mol(ot.receptor, &aln);
        let off_samples = evaluate_mif_at_points(&aligned_mol, target_grid_box, &cavity_coords);

        // Residue substitution audit within 10.5 Å of pocket center
        let mut diff_res = Vec::new();
        let mut conserved_cnt = 0usize;
        let mut total_paired = 0usize;
        for tc in &tgt_cas {
            if dist3(tc.coords, target_grid_box.center) > 10.5 {
                continue;
            }
            let mut nearest: Option<(&CaPoint, f64)> = None;
            for oc in &off_cas {
                let d = dist3(aln.apply(oc.coords), tc.coords);
                if d <= 2.6 && nearest.is_none_or(|(_, bd)| d < bd) {
                    nearest = Some((oc, d));
                }
            }
            if let Some((oc, d)) = nearest {
                total_paired += 1;
                let conserved = tc.res_name == oc.res_name;
                if conserved {
                    conserved_cnt += 1;
                } else {
                    let c1 = residue_class(&tc.res_name);
                    let c2 = residue_class(&oc.res_name);
                    let change = if c1 == c2 {
                        format!("Conservative ({c1})")
                    } else {
                        format!("{c1} -> {c2}")
                    };
                    diff_res.push(ResidueSubstitution {
                        target_residue: tc.label(),
                        offtarget_residue: oc.label(),
                        ca_distance_a: (d * 100.0).round() / 100.0,
                        conserved: false,
                        property_change: change,
                    });
                }
            }
        }
        diff_res.sort_by(|a, b| a.target_residue.cmp(&b.target_residue));
        let seq_id = if total_paired > 0 {
            ((conserved_cnt as f64) * 100.0 / (total_paired as f64) * 10.0).round() / 10.0
        } else {
            0.0
        };

        // Cavity Jaccard & 7-probe MIF cosine similarity
        let mut shared_open = 0usize;
        let mut steric_excl = 0usize;
        let mut dot_mif = 0.0_f64;
        let mut norm_t = 0.0_f64;
        let mut norm_o = 0.0_f64;
        for (ts, os) in tgt_samples.iter().zip(off_samples.iter()) {
            if !ts.steric_clash && !os.steric_clash && os.min_wall_dist_a <= 5.2 {
                shared_open += 1;
            }
            if !ts.steric_clash
                && ts.min_wall_dist_a >= 2.50
                && (os.steric_clash || os.min_wall_dist_a < 2.25)
            {
                steric_excl += 1;
            }
            for p in 0..7 {
                let et = ts.energies[p];
                let eo = os.energies[p];
                dot_mif += et * eo;
                norm_t += et * et;
                norm_o += eo * eo;
            }
        }
        let jaccard = (shared_open as f64) / (tgt_samples.len().max(1) as f64);
        let mif_cos = if norm_t > 1e-9 && norm_o > 1e-9 {
            (dot_mif / (norm_t.sqrt() * norm_o.sqrt())).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        off_summaries.push(OffTargetComparisonSummary {
            offtarget_name: ot.name.clone(),
            pocket_ca_rmsd_a: (ca_rmsd * 100.0).round() / 100.0,
            matched_ca_count: matched_ca,
            pocket_sequence_identity_pct: seq_id,
            cavity_jaccard: (jaccard * 1000.0).round() / 1000.0,
            mif_cosine_similarity: (mif_cos * 1000.0).round() / 1000.0,
            steric_exclusive_voxel_count: steric_excl,
            differential_residues: diff_res,
            conserved_residues_count: conserved_cnt,
        });
        aligned_offtargets.push((ot.name.clone(), aligned_mol));
        all_off_samples.push(off_samples);
    }

    // Aggregate ΔMIF across all off-targets (worst-case off-target = most negative energy)
    type ProbeCandTuple = (f64, PharmacophoreKind, [f64; 3], f64, f64, String, String);
    let mut steric_cands: Vec<(f64, [f64; 3], String, String)> = Vec::new();
    let mut diff_cands: Vec<ProbeCandTuple> = Vec::new();
    let mut cons_cands: Vec<ProbeCandTuple> = Vec::new();

    for vi in 0..tgt_samples.len() {
        let ts = &tgt_samples[vi];
        if ts.steric_clash {
            continue;
        }
        // Check steric exclusivity subpocket (open in target, blocked in >= 1 off-target)
        let mut worst_block_dist = 99.0_f64;
        let mut blocking_res = String::new();
        for (oi, off_vec) in all_off_samples.iter().enumerate() {
            let os = &off_vec[vi];
            if (os.steric_clash || os.min_wall_dist_a < 2.25)
                && os.min_wall_dist_a < worst_block_dist
            {
                worst_block_dist = os.min_wall_dist_a;
                blocking_res = format!("{}@{}", os.nearest_wall_residue, offtargets[oi].name);
            }
        }
        if ts.min_wall_dist_a >= 2.55 && worst_block_dist < 2.25 {
            // Clearance advantage in Å (bigger = stronger steric gate)
            let clearance_gain = ts.min_wall_dist_a - worst_block_dist;
            steric_cands.push((
                -clearance_gain,
                ts.coords,
                ts.nearest_wall_residue.clone(),
                blocking_res,
            ));
        }

        // Check each of the 7 pharmacophore probes
        for (pi, &kind) in PROBE_KINDS.iter().enumerate() {
            let et = ts.energies[pi];
            if et > config.min_target_energy_kcal {
                continue;
            }
            let mut best_off_e = 0.0_f64;
            let mut best_off_res = String::new();
            for (oi, off_vec) in all_off_samples.iter().enumerate() {
                let os = &off_vec[vi];
                let eo = if os.steric_clash {
                    0.0
                } else {
                    os.energies[pi]
                };
                if eo < best_off_e {
                    best_off_e = eo;
                    best_off_res = format!("{}@{}", os.key_residues[pi], offtargets[oi].name);
                }
            }
            let delta = et - best_off_e;
            if delta <= config.min_delta_energy_kcal {
                let counter = if best_off_res.is_empty() {
                    "absent/blocked in off-target".to_string()
                } else {
                    best_off_res
                };
                diff_cands.push((
                    delta,
                    kind,
                    ts.coords,
                    et,
                    best_off_e,
                    ts.key_residues[pi].clone(),
                    counter,
                ));
            } else if best_off_e <= config.min_target_energy_kcal && delta.abs() <= 0.55 {
                cons_cands.push((
                    et,
                    kind,
                    ts.coords,
                    et,
                    best_off_e,
                    ts.key_residues[pi].clone(),
                    best_off_res,
                ));
            }
        }
    }

    // Non-maximum suppression (NMS) to emit spatially clean hotspots
    steric_cands.sort_by(|a, b| a.0.total_cmp(&b.0));
    diff_cands.sort_by(|a, b| a.0.total_cmp(&b.0));
    cons_cands.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut selectivity_hotspots: Vec<SelectivityHotspot> = Vec::new();
    let max_steric = (config.max_hotspots / 2).max(2);
    for (neg_gain, pos, tres, bres) in steric_cands {
        if selectivity_hotspots.len() >= max_steric {
            break;
        }
        if selectivity_hotspots
            .iter()
            .all(|h| dist3(h.center, pos) >= 2.6)
        {
            let gain = -neg_gain;
            let id = format!("SEL_S{}", selectivity_hotspots.len() + 1);
            selectivity_hotspots.push(SelectivityHotspot {
                id,
                kind: SelectivityHotspotKind::StericExclusivitySubpocket,
                center: pos,
                radius_a: 1.5,
                target_energy_kcal: 0.0,
                worst_offtarget_energy_kcal: (gain * 2.5 * 100.0).round() / 100.0,
                delta_energy_kcal: (-gain * 2.5 * 100.0).round() / 100.0,
                target_residue: tres,
                offtarget_blocking_or_counter_residue: bres.clone(),
                design_guidance: format!(
                    "Grow rigid/bulky substituent into subpocket (+{gain:.2} Å steric clearance vs {bres})"
                ),
            });
        }
    }

    for (delta, kind, pos, et, eo, tres, ores) in diff_cands {
        if selectivity_hotspots.len() >= config.max_hotspots {
            break;
        }
        let sep = match kind {
            PharmacophoreKind::Hydrophobic | PharmacophoreKind::Aromatic => 2.8,
            _ => 2.3,
        };
        if selectivity_hotspots
            .iter()
            .filter(|h| h.kind == SelectivityHotspotKind::DifferentialPharmacophore(kind))
            .all(|h| dist3(h.center, pos) >= sep)
            && selectivity_hotspots
                .iter()
                .all(|h| dist3(h.center, pos) >= 1.8)
        {
            let id = format!("SEL_P{}", selectivity_hotspots.len() + 1);
            selectivity_hotspots.push(SelectivityHotspot {
                id,
                kind: SelectivityHotspotKind::DifferentialPharmacophore(kind),
                center: pos,
                radius_a: 1.4,
                target_energy_kcal: (et * 100.0).round() / 100.0,
                worst_offtarget_energy_kcal: (eo * 100.0).round() / 100.0,
                delta_energy_kcal: (delta * 100.0).round() / 100.0,
                target_residue: tres.clone(),
                offtarget_blocking_or_counter_residue: ores,
                design_guidance: format!(
                    "Anchor {kind:?} group toward {tres} (ΔMIF = {delta:.2} kcal/mol selectivity gain)"
                ),
            });
        }
    }

    let mut conserved_anchors: Vec<SelectivityHotspot> = Vec::new();
    for (_, kind, pos, et, eo, tres, ores) in cons_cands {
        if conserved_anchors.len() >= 6 {
            break;
        }
        if conserved_anchors
            .iter()
            .all(|h| dist3(h.center, pos) >= 2.4)
        {
            let id = format!("CONS_{}", conserved_anchors.len() + 1);
            conserved_anchors.push(SelectivityHotspot {
                id,
                kind: SelectivityHotspotKind::ConservedAnchor(kind),
                center: pos,
                radius_a: 1.35,
                target_energy_kcal: (et * 100.0).round() / 100.0,
                worst_offtarget_energy_kcal: (eo * 100.0).round() / 100.0,
                delta_energy_kcal: ((et - eo) * 100.0).round() / 100.0,
                target_residue: tres.clone(),
                offtarget_blocking_or_counter_residue: ores,
                design_guidance: format!(
                    "Conserved {kind:?} potency anchor ({tres}); maintain for affinity but pair with SEL_* hotspots"
                ),
            });
        }
    }

    // Feasibility score in [0, 1]: combines MIF dissimilarity, differential residue count, and hotspot count
    let mean_mif_sim = off_summaries
        .iter()
        .map(|s| s.mif_cosine_similarity)
        .sum::<f64>()
        / (off_summaries.len().max(1) as f64);
    let hotspot_term = ((selectivity_hotspots.len() as f64) / 6.0).clamp(0.0, 1.0);
    let mif_div_term = (1.0 - mean_mif_sim).clamp(0.0, 1.0);
    let feasibility = ((0.55 * hotspot_term + 0.45 * mif_div_term) * 1000.0).round() / 1000.0;

    let report = SelectivityReport {
        target_name: target_name.to_string(),
        target_grid_box: target_grid_box.clone(),
        spacing_a: eff_spacing,
        target_cavity_voxels: cavity_coords.len(),
        selectivity_feasibility_score: feasibility,
        offtargets: off_summaries,
        selectivity_hotspots,
        conserved_anchors,
        elapsed_ms: t0.elapsed().as_millis(),
    };

    Ok(SelectivityAnalysisOutput {
        report,
        aligned_offtargets,
    })
}

impl SelectivityReport {
    /// Augments a target `SiteBlueprint` with boosted differential selectivity features so
    /// downstream `pharm-screen` and pharmacophore-guided docking prioritize selective ligands.
    #[must_use]
    pub fn augment_blueprint(&self, base: &SiteBlueprint) -> SiteBlueprint {
        let mut bp = base.clone();
        for hs in &self.selectivity_hotspots {
            let kind = match hs.kind {
                SelectivityHotspotKind::DifferentialPharmacophore(k) => k,
                SelectivityHotspotKind::StericExclusivitySubpocket => {
                    PharmacophoreKind::Hydrophobic
                }
                SelectivityHotspotKind::ConservedAnchor(k) => k,
            };
            // Boost existing matching feature within 2.4 Å, otherwise append new selective feature
            if let Some(existing) = bp
                .features
                .iter_mut()
                .find(|f| f.kind == kind && dist3(f.center, hs.center) <= 2.4)
            {
                existing.weight = (existing.weight * 1.45).min(3.5);
                existing.enabled = true;
            } else if bp.features.len() < 14 {
                bp.features.push(SiteFeature {
                    id: hs.id.clone(),
                    kind,
                    center: hs.center,
                    radius_a: hs.radius_a,
                    direction: None,
                    energy_kcal: hs.target_energy_kcal.min(-1.5),
                    key_residue: hs.target_residue.clone(),
                    weight: 1.65,
                    enabled: true,
                    required: false,
                    source: FeatureSource::UserDefined,
                });
            }
        }
        bp.rebuild_distance_matrix();
        bp
    }

    /// Renders a human-readable Markdown Structural Pocket Selectivity (ΔMIF) report.
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();
        let _ = writeln!(
            md,
            "# Structural Pocket Selectivity (ΔMIF) Report: {}\n",
            self.target_name
        );
        let _ = writeln!(
            md,
            "- **Target Cavity Voxels:** `{}` (`{:.2} Å` spacing) | **Selectivity Feasibility Score:** **`{:.3}`** | **Runtime:** `{} ms`\n",
            self.target_cavity_voxels,
            self.spacing_a,
            self.selectivity_feasibility_score,
            self.elapsed_ms
        );

        md.push_str("## 1. Off-Target Pocket Superposition & Similarity\n\n");
        md.push_str("| Off-Target | Pocket Cα RMSD | Matched Cα | Pocket Seq. ID | Cavity Jaccard | 7-Probe MIF Cosine | Steric-Excl. Voxels | Diff. Residues |\n");
        md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
        for ot in &self.offtargets {
            let _ = writeln!(
                md,
                "| `{}` | `{:.2} Å` | {} | `{:.1}%` | `{:.3}` | `{:.3}` | {} | {} |",
                ot.offtarget_name,
                ot.pocket_ca_rmsd_a,
                ot.matched_ca_count,
                ot.pocket_sequence_identity_pct,
                ot.cavity_jaccard,
                ot.mif_cosine_similarity,
                ot.steric_exclusive_voxel_count,
                ot.differential_residues.len()
            );
        }
        md.push('\n');

        md.push_str("## 2. Actionable Selectivity Hotspots (Target-Unique / ΔMIF)\n\n");
        if self.selectivity_hotspots.is_empty() {
            md.push_str("_No strong differential hotspots exceeded the ΔMIF threshold (pockets are highly conserved)._\n\n");
        } else {
            md.push_str("| ID | Type | Center (Å) | Target E | Off-Target E | ΔE | Target Res | Off-Target Counter | Design Guidance |\n");
            md.push_str("|---|---|---|---:|---:|---:|---|---|---|\n");
            for hs in &self.selectivity_hotspots {
                let _ = writeln!(
                    md,
                    "| `{}` | `{}` | `({:.2}, {:.2}, {:.2})` | `{:.2}` | `{:.2}` | **`{:.2}`** | `{}` | `{}` | {} |",
                    hs.id,
                    hs.kind.label(),
                    hs.center[0],
                    hs.center[1],
                    hs.center[2],
                    hs.target_energy_kcal,
                    hs.worst_offtarget_energy_kcal,
                    hs.delta_energy_kcal,
                    hs.target_residue,
                    hs.offtarget_blocking_or_counter_residue,
                    hs.design_guidance
                );
            }
            md.push('\n');
        }

        if !self.conserved_anchors.is_empty() {
            md.push_str(
                "## 3. Conserved Core Anchors (Potency Drivers Shared with Off-Targets)\n\n",
            );
            md.push_str("| ID | Type | Center (Å) | Target E | Off-Target E | Key Residue |\n|---|---|---|---:|---:|---|\n");
            for ca in &self.conserved_anchors {
                let _ = writeln!(
                    md,
                    "| `{}` | `{}` | `({:.2}, {:.2}, {:.2})` | `{:.2}` | `{:.2}` | `{}` |",
                    ca.id,
                    ca.kind.label(),
                    ca.center[0],
                    ca.center[1],
                    ca.center[2],
                    ca.target_energy_kcal,
                    ca.worst_offtarget_energy_kcal,
                    ca.target_residue
                );
            }
            md.push('\n');
        }

        for ot in &self.offtargets {
            if !ot.differential_residues.is_empty() {
                let _ = writeln!(
                    md,
                    "## 4. Differential Pocket Lining Residues ({} vs {})\n",
                    self.target_name, ot.offtarget_name
                );
                md.push_str("| Target Residue | Off-Target Residue | Cα Dist (Å) | Chemical Shift |\n|---|---|---:|---|\n");
                for r in &ot.differential_residues {
                    let _ = writeln!(
                        md,
                        "| `{}` | `{}` | `{:.2}` | {} |",
                        r.target_residue, r.offtarget_residue, r.ca_distance_a, r.property_change
                    );
                }
                md.push('\n');
            }
        }
        md
    }

    /// Generates a UCSF ChimeraX `.cxc` script visualizing Target vs aligned Off-Target(s),
    /// emerald-green spheres for selectivity hotspots, and amber spheres for conserved anchors.
    #[must_use]
    pub fn chimerax_overlay_script(
        &self,
        target_pdb_ref: &str,
        aligned_offtarget_pdb_refs: &[(&str, &str)],
    ) -> String {
        let mut s = String::from("# DockCraft ΔMIF Structural Pocket Selectivity Overlay\n");
        let _ = writeln!(s, "open \"{target_pdb_ref}\"");
        let _ = writeln!(s, "color #1 cornflowerblue cartoons");
        for (idx, (_, path_ref)) in aligned_offtarget_pdb_refs.iter().enumerate() {
            let mid = idx + 2;
            let _ = writeln!(s, "open \"{path_ref}\"");
            let _ = writeln!(s, "color #{mid} salmon cartoons");
            let _ = writeln!(s, "transparency #{mid} 45 cartoons");
        }
        for hs in &self.selectivity_hotspots {
            let color = match hs.kind {
                SelectivityHotspotKind::StericExclusivitySubpocket => "limegreen",
                SelectivityHotspotKind::DifferentialPharmacophore(_) => "mediumseagreen",
                SelectivityHotspotKind::ConservedAnchor(_) => "gold",
            };
            let _ = writeln!(
                s,
                "shape sphere center {:.3},{:.3},{:.3} radius {:.2} color {} transparency 40 name {}",
                hs.center[0], hs.center[1], hs.center[2], hs.radius_a, color, hs.id
            );
        }
        for ca in &self.conserved_anchors {
            let _ = writeln!(
                s,
                "shape sphere center {:.3},{:.3},{:.3} radius {:.2} color goldenrod transparency 65 name {}",
                ca.center[0], ca.center[1], ca.center[2], ca.radius_a, ca.id
            );
        }
        let c = self.target_grid_box.center;
        let _ = writeln!(s, "cofr {:.3},{:.3},{:.3}", c[0], c[1], c[2]);
        s.push_str("lighting soft\n");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::{Ad4Type, Atom3D};

    fn make_pocket_receptor(
        name: &str,
        include_asp_selectivity: bool,
        block_subpocket: bool,
    ) -> Molecule3D {
        let mut m = Molecule3D::new(name);
        let add = |m: &mut Molecule3D,
                   el: &str,
                   aname: &str,
                   rn: &str,
                   seq: i32,
                   pos: [f64; 3],
                   hd: bool,
                   ha: bool,
                   hyd: bool,
                   fc: i8| {
            let mut a = Atom3D::new(m.atoms.len() + 1, el, pos);
            a.chain_id = 'A';
            a.is_hetatm = false;
            a.name = aname.to_string();
            a.res_name = rn.to_string();
            a.res_seq = seq;
            a.ad4_type = match el {
                "O" => Ad4Type::OA,
                "N" => Ad4Type::N,
                _ => Ad4Type::C,
            };
            a.h_donor = hd;
            a.h_acceptor = ha;
            a.hydrophobic = hyd;
            a.formal_charge = fc;
            m.add_atom(a);
        };
        // Conserved pocket backbone scaffold (6 residues with CA atoms)
        let scaffold = [
            ("VAL", 10, [-4.5, 0.0, 0.0]),
            ("ALA", 20, [4.5, 0.0, 0.0]),
            ("LEU", 30, [0.0, 4.5, 0.0]),
            ("ILE", 40, [0.0, -4.5, 0.0]),
            ("PHE", 50, [0.0, 0.0, 4.5]),
            ("MET", 60, [0.0, 0.0, -4.5]),
            ("LEU", 83, [-4.2, 2.0, 0.0]),
        ];
        for &(rn, seq, p) in &scaffold {
            add(&mut m, "C", "CA", rn, seq, p, false, false, true, 0);
        }
        // Conserved hinge O acceptor
        add(
            &mut m,
            "O",
            "O",
            "LEU",
            83,
            [-3.6, 1.5, 0.0],
            false,
            true,
            false,
            0,
        );

        // Differential residue #1: Target has ASP90 (negative charge at [2.8, 2.5, 0.0]), off-target has ALA90
        if include_asp_selectivity {
            add(
                &mut m,
                "C",
                "CA",
                "ASP",
                90,
                [4.2, 3.2, 0.0],
                false,
                false,
                false,
                0,
            );
            add(
                &mut m,
                "O",
                "OD1",
                "ASP",
                90,
                [3.0, 2.4, 0.0],
                false,
                true,
                false,
                -1,
            );
            add(
                &mut m,
                "O",
                "OD2",
                "ASP",
                90,
                [3.4, 3.6, 0.0],
                false,
                true,
                false,
                -1,
            );
        } else {
            add(
                &mut m,
                "C",
                "CA",
                "ALA",
                90,
                [4.2, 3.2, 0.0],
                false,
                false,
                true,
                0,
            );
            add(
                &mut m,
                "C",
                "CB",
                "ALA",
                90,
                [3.6, 2.8, 0.0],
                false,
                false,
                true,
                0,
            );
        }

        // Differential residue #2: Off-target has bulky TRP95 protruding into subpocket near [1.5, -1.8, 0.0]
        if block_subpocket {
            add(
                &mut m,
                "C",
                "CA",
                "TRP",
                95,
                [3.8, -3.4, 0.0],
                false,
                false,
                true,
                0,
            );
            add(
                &mut m,
                "C",
                "CG",
                "TRP",
                95,
                [2.2, -2.0, 0.0],
                false,
                false,
                true,
                0,
            );
            add(
                &mut m,
                "C",
                "CD1",
                "TRP",
                95,
                [1.3, -1.4, 0.0],
                false,
                false,
                true,
                0,
            );
            add(
                &mut m,
                "C",
                "CE2",
                "TRP",
                95,
                [1.5, -2.2, 0.6],
                false,
                false,
                true,
                0,
            );
        } else {
            add(
                &mut m,
                "C",
                "CA",
                "GLY",
                95,
                [3.8, -3.4, 0.0],
                false,
                false,
                true,
                0,
            );
        }
        m
    }

    #[test]
    fn selectivity_aligns_rotated_offtarget_and_detects_steric_and_electrostatic_hotspots()
    -> Result<(), Box<dyn std::error::Error>> {
        let tgt = make_pocket_receptor("TargetKinase", true, false);
        let mut off = make_pocket_receptor("OffTargetParalog", false, true);
        // Apply arbitrary 90-deg Z rotation + translation (15, -8, 12) to off-target
        for a in &mut off.atoms {
            let [x, y, z] = a.coords;
            a.coords = [-y + 15.0, x - 8.0, z + 12.0];
        }
        let gbox = GridBox::from_center_size([0.0, 0.0, 0.0], [11.0, 11.0, 11.0], 0.375)?;
        let ot_in = OffTargetPocketInput {
            name: "OffTargetParalog".to_string(),
            receptor: &off,
            pocket_center: [15.0, -8.0, 12.0],
        };
        let out = analyze_pocket_selectivity(
            "TargetKinase",
            &tgt,
            &gbox,
            &[],
            &[ot_in],
            &PocketSelectivityConfig {
                spacing_a: 0.75,
                ..Default::default()
            },
        )?;
        let rep = &out.report;
        assert_eq!(rep.offtargets.len(), 1);
        assert!(
            rep.offtargets[0].pocket_ca_rmsd_a < 0.05,
            "expected exact rigid recovery (<0.05 Å), got {:.3} Å",
            rep.offtargets[0].pocket_ca_rmsd_a
        );
        assert!(
            rep.offtargets[0]
                .differential_residues
                .iter()
                .any(
                    |r| r.target_residue.contains("ASP90") && r.offtarget_residue.contains("ALA90")
                ),
            "expected ASP90 -> ALA90 differential residue, got {:?}",
            rep.offtargets[0].differential_residues
        );
        assert!(
            rep.selectivity_hotspots
                .iter()
                .any(|h| h.kind == SelectivityHotspotKind::StericExclusivitySubpocket),
            "expected StericExclusivitySubpocket from TRP95 clash, got {:?}",
            rep.selectivity_hotspots
        );
        assert!(
            rep.selectivity_hotspots
                .iter()
                .any(|h| matches!(h.kind, SelectivityHotspotKind::DifferentialPharmacophore(_))),
            "expected DifferentialPharmacophore from ASP90, got {:?}",
            rep.selectivity_hotspots
        );
        assert!(!rep.render_markdown().is_empty());
        assert!(
            !rep.chimerax_overlay_script("target.pdb", &[("off", "off.pdb")])
                .is_empty()
        );
        Ok(())
    }
}

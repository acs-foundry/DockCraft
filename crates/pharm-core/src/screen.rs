#![forbid(unsafe_code)]

use std::time::Instant;

use dock_engine::PharmacophoreKind;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use struct_core::{Molecule3D, cross3, metal_chelation_tier};

use crate::kabsch::{RigidAlignment, kabsch_align};
use crate::mif::{SiteBlueprint, SiteFeature};

/// Chemical pharmacophore feature perceived on a candidate ligand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LigandFeature {
    pub kind: PharmacophoreKind,
    pub center: [f64; 3],
    pub direction: Option<[f64; 3]>,
    pub atom_indices: Vec<usize>,
}

/// Configuration for ultra-fast 3D pharmacophore & exclusion-volume screening (`pharm-screen`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmScreenConfig {
    /// Minimum number of enabled blueprint features that must match simultaneously (default `3`).
    pub min_matched_features: usize,
    /// Extra distance slack (Å) beyond feature sphere radii for conformational flexibility.
    pub dist_slack_a: f64,
    /// Maximum feature superposition RMSD (Å) after Horn quaternion Kabsch alignment.
    pub max_rmsd_a: f64,
    /// Maximum allowed heavy-atom clashes against receptor exclusion spheres.
    pub max_exclusion_clashes: usize,
    /// Number of torsional conformers sampled per flexible ligand (`1` = fastest, `3–5` = higher recall).
    pub max_conformers: usize,
    /// Whether an aromatic ring on the ligand can also satisfy a `Hydrophobic` site hotspot.
    pub aromatic_matches_hydrophobic: bool,
    /// Maximum number of top hits to keep sorted in output (`0` = keep all matches).
    pub max_hits: usize,
}

impl Default for PharmScreenConfig {
    fn default() -> Self {
        Self {
            min_matched_features: 3,
            dist_slack_a: 0.65,
            max_rmsd_a: 1.65,
            max_exclusion_clashes: 3,
            max_conformers: 3,
            aromatic_matches_hydrophobic: true,
            max_hits: 5_000,
        }
    }
}

/// Result for a single ligand passing the 3D pharmacophore + exclusion-volume funnel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmScreenHit {
    pub name: String,
    pub smiles: Option<String>,
    pub matched_count: usize,
    pub total_active_features: usize,
    pub matched_feature_ids: Vec<String>,
    pub pharm_rmsd_a: f64,
    pub exclusion_clashes: usize,
    /// Higher is better: weighted sum of matched hotspot affinities minus RMSD/clash penalties.
    pub pharm_score: f64,
    pub aligned_molecule: Molecule3D,
}

/// Summary report for a `pharm-screen` batch/streaming run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmScreenReport {
    pub target_name: String,
    pub total_evaluated: usize,
    pub passed_count: usize,
    pub elapsed_ms: u128,
    pub hits: Vec<PharmScreenHit>,
}

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
fn norm3(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-8 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn rotate_point_around_axis(p: [f64; 3], origin: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let r = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    let c = angle.cos();
    let s = angle.sin();
    let k_dot_r = dot3(axis, r);
    let k_cross_r = cross3(axis, r);
    [
        origin[0] + r[0] * c + k_cross_r[0] * s + axis[0] * k_dot_r * (1.0 - c),
        origin[1] + r[1] * c + k_cross_r[1] * s + axis[1] * k_dot_r * (1.0 - c),
        origin[2] + r[2] * c + k_cross_r[2] * s + axis[2] * k_dot_r * (1.0 - c),
    ]
}

/// Extracts all 7 chemical pharmacophore feature classes from a prepared 3D ligand.
#[must_use]
pub fn extract_ligand_features(mol: &Molecule3D) -> Vec<LigandFeature> {
    let mut out = Vec::new();
    let mut ring_hydrophobic_used = vec![false; mol.atoms.len()];

    // 1. Aromatic ring centroids + normals (5/6-membered aromatic rings)
    for ring in &mol.rings {
        if !(5..=6).contains(&ring.len()) {
            continue;
        }
        if !ring
            .iter()
            .all(|&i| mol.atoms.get(i).is_some_and(|a| a.aromatic))
        {
            continue;
        }
        let k = ring.len() as f64;
        let mut cen = [0.0; 3];
        for &idx in ring {
            let p = mol.atoms[idx].coords;
            cen[0] += p[0] / k;
            cen[1] += p[1] / k;
            cen[2] += p[2] / k;
            if mol.atoms[idx].hydrophobic {
                ring_hydrophobic_used[idx] = true;
            }
        }
        let mut nrm = [0.0; 3];
        for i in 0..ring.len() {
            let pa = mol.atoms[ring[i]].coords;
            let pb = mol.atoms[ring[(i + 1) % ring.len()]].coords;
            let cr = cross3(
                [pa[0] - cen[0], pa[1] - cen[1], pa[2] - cen[2]],
                [pb[0] - cen[0], pb[1] - cen[1], pb[2] - cen[2]],
            );
            nrm[0] += cr[0];
            nrm[1] += cr[1];
            nrm[2] += cr[2];
        }
        let n_unit = norm3(nrm);
        out.push(LigandFeature {
            kind: PharmacophoreKind::Aromatic,
            center: cen,
            direction: if dot3(n_unit, n_unit) > 0.5 {
                Some(n_unit)
            } else {
                None
            },
            atom_indices: ring.clone(),
        });
        // An aromatic carbon ring also provides a lipophilic center at its centroid
        if ring.iter().filter(|&&i| mol.atoms[i].hydrophobic).count() >= 3 {
            out.push(LigandFeature {
                kind: PharmacophoreKind::Hydrophobic,
                center: cen,
                direction: None,
                atom_indices: ring.clone(),
            });
        }
    }

    // 2. Polar, Ionizable, and Metal-chelating heavy atoms
    let mut aliph_hyd: Vec<usize> = Vec::new();
    for (idx, a) in mol.atoms.iter().enumerate() {
        if !a.is_heavy() {
            continue;
        }
        if a.h_donor {
            let dir = mol.adj.get(idx).and_then(|adj| {
                adj.iter()
                    .find(|&&(nb, _)| mol.atoms.get(nb).is_some_and(|h| h.element == "H"))
                    .map(|&(nb, _)| {
                        let hc = mol.atoms[nb].coords;
                        norm3([
                            hc[0] - a.coords[0],
                            hc[1] - a.coords[1],
                            hc[2] - a.coords[2],
                        ])
                    })
            });
            out.push(LigandFeature {
                kind: PharmacophoreKind::HDonor,
                center: a.coords,
                direction: dir,
                atom_indices: vec![idx],
            });
        }
        if a.h_acceptor {
            out.push(LigandFeature {
                kind: PharmacophoreKind::HAcceptor,
                center: a.coords,
                direction: None,
                atom_indices: vec![idx],
            });
        }
        if a.formal_charge > 0 {
            out.push(LigandFeature {
                kind: PharmacophoreKind::PositiveIonizable,
                center: a.coords,
                direction: None,
                atom_indices: vec![idx],
            });
        }
        if a.formal_charge < 0 {
            out.push(LigandFeature {
                kind: PharmacophoreKind::NegativeIonizable,
                center: a.coords,
                direction: None,
                atom_indices: vec![idx],
            });
        }
        if metal_chelation_tier(mol, idx) >= 2 {
            out.push(LigandFeature {
                kind: PharmacophoreKind::MetalChelate,
                center: a.coords,
                direction: None,
                atom_indices: vec![idx],
            });
        }
        if a.hydrophobic && !ring_hydrophobic_used[idx] {
            aliph_hyd.push(idx);
        }
    }

    // 3. Group remaining aliphatic/halogen hydrophobic atoms into compact lipophilic centers (<= 2.6 A)
    let mut used_aliph = vec![false; aliph_hyd.len()];
    for i in 0..aliph_hyd.len() {
        if used_aliph[i] {
            continue;
        }
        used_aliph[i] = true;
        let seed_idx = aliph_hyd[i];
        let mut group = vec![seed_idx];
        let p0 = mol.atoms[seed_idx].coords;
        for j in (i + 1)..aliph_hyd.len() {
            if !used_aliph[j] && dist3(p0, mol.atoms[aliph_hyd[j]].coords) <= 2.6 {
                used_aliph[j] = true;
                group.push(aliph_hyd[j]);
            }
        }
        let k = group.len() as f64;
        let mut cen = [0.0; 3];
        for &ai in &group {
            let p = mol.atoms[ai].coords;
            cen[0] += p[0] / k;
            cen[1] += p[1] / k;
            cen[2] += p[2] / k;
        }
        out.push(LigandFeature {
            kind: PharmacophoreKind::Hydrophobic,
            center: cen,
            direction: None,
            atom_indices: group,
        });
    }

    out
}

#[inline]
fn kinds_compatible(
    site_kind: PharmacophoreKind,
    lig_kind: PharmacophoreKind,
    arom_matches_hyd: bool,
) -> bool {
    site_kind == lig_kind
        || site_kind == PharmacophoreKind::AnyHeavy
        || (arom_matches_hyd
            && site_kind == PharmacophoreKind::Hydrophobic
            && lig_kind == PharmacophoreKind::Aromatic)
}

/// Generates up to `max_conf` deterministic low-strain conformers by twisting rotatable bonds.
fn sample_quick_conformers(mol: &Molecule3D, max_conf: usize) -> Vec<Molecule3D> {
    let mut out = Vec::with_capacity(max_conf.max(1));
    out.push(mol.clone());
    if max_conf <= 1 || mol.rotatable_bonds.is_empty() {
        return out;
    }
    const PRESET_DELTAS: [f64; 4] = [
        std::f64::consts::FRAC_PI_3 * 2.0,  // +120 deg
        -std::f64::consts::FRAC_PI_3 * 2.0, // -120 deg
        std::f64::consts::PI,               // 180 deg
        std::f64::consts::FRAC_PI_2,        // +90 deg
    ];
    for ci in 1..max_conf {
        let mut copy = mol.clone();
        for (ri, rb) in mol.rotatable_bonds.iter().enumerate() {
            let angle = PRESET_DELTAS[(ci + ri) % PRESET_DELTAS.len()];
            let pu = copy.atoms[rb.u].coords;
            let pv = copy.atoms[rb.v].coords;
            let axis = norm3([pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]]);
            if dot3(axis, axis) < 0.5 {
                continue;
            }
            for &mi in &rb.moving_atoms {
                if let Some(atom) = copy.atoms.get_mut(mi) {
                    atom.coords = rotate_point_around_axis(atom.coords, pu, axis, angle);
                }
            }
        }
        out.push(copy);
    }
    out
}

struct CliqueSearchCtx<'a> {
    site_feats: &'a [&'a SiteFeature],
    lig_feats: &'a [LigandFeature],
    compat: &'a [Vec<usize>],
    site_dists: &'a [Vec<f64>],
    site_tols: &'a [Vec<f64>],
    min_match: usize,
    best_mappings: Vec<Vec<(usize, usize)>>,
    nodes_visited: usize,
    node_budget: usize,
}

impl<'a> CliqueSearchCtx<'a> {
    fn dfs(&mut self, s_idx: usize, current: &mut Vec<(usize, usize)>, used_lig: &mut [bool]) {
        self.nodes_visited += 1;
        if self.nodes_visited > self.node_budget {
            return;
        }
        let remaining = self.site_feats.len() - s_idx;
        if current.len() + remaining < self.min_match {
            return;
        }
        if s_idx == self.site_feats.len() {
            if current.len() >= self.min_match {
                // Keep top diverse maximal mappings (prioritize larger cardinality)
                if self.best_mappings.len() < 16 {
                    self.best_mappings.push(current.clone());
                } else if let Some((min_pos, min_len)) = self
                    .best_mappings
                    .iter()
                    .enumerate()
                    .map(|(i, m)| (i, m.len()))
                    .min_by_key(|&(_, l)| l)
                    && current.len() > min_len
                {
                    self.best_mappings[min_pos] = current.clone();
                }
            }
            return;
        }

        // Branch 1: try matching site feature `s_idx` with each distance-consistent ligand feature
        for &l_idx in &self.compat[s_idx] {
            if used_lig[l_idx] {
                continue;
            }
            let l_pos = self.lig_feats[l_idx].center;
            let consistent = current.iter().all(|&(prev_s, prev_l)| {
                let d_lig = dist3(l_pos, self.lig_feats[prev_l].center);
                let d_site = self.site_dists[s_idx][prev_s];
                let tol = self.site_tols[s_idx][prev_s];
                (d_lig - d_site).abs() <= tol
            });
            if consistent {
                used_lig[l_idx] = true;
                current.push((s_idx, l_idx));
                self.dfs(s_idx + 1, current, used_lig);
                current.pop();
                used_lig[l_idx] = false;
            }
        }

        // Branch 2: skip site feature `s_idx` unless it is marked strictly required
        if !self.site_feats[s_idx].required {
            self.dfs(s_idx + 1, current, used_lig);
        }
    }
}

/// Evaluates a single candidate ligand against an active-site `SiteBlueprint`, returning its best
/// 3D Kabsch-aligned pose and pharmacophore score if it passes all geometric/exclusion gates.
#[must_use]
pub fn screen_ligand_against_blueprint(
    mol: &Molecule3D,
    blueprint: &SiteBlueprint,
    cfg: &PharmScreenConfig,
) -> Option<PharmScreenHit> {
    let active: Vec<&SiteFeature> = blueprint.features.iter().filter(|f| f.enabled).collect();
    if active.is_empty() {
        return None;
    }
    let min_match = cfg.min_matched_features.min(active.len()).max(1);

    // Precompute site pairwise distances and tolerances
    let n_s = active.len();
    let mut site_dists = vec![vec![0.0; n_s]; n_s];
    let mut site_tols = vec![vec![0.0; n_s]; n_s];
    for i in 0..n_s {
        for j in (i + 1)..n_s {
            let d = dist3(active[i].center, active[j].center);
            let tol = (active[i].radius_a + active[j].radius_a) * 0.55 + cfg.dist_slack_a;
            site_dists[i][j] = d;
            site_dists[j][i] = d;
            site_tols[i][j] = tol;
            site_tols[j][i] = tol;
        }
    }

    // Fast type-count rejection on initial conformer before generating conformers
    let init_feats = extract_ligand_features(mol);
    let mut possible_site_matches = 0usize;
    for sf in &active {
        let has_any = init_feats
            .iter()
            .any(|lf| kinds_compatible(sf.kind, lf.kind, cfg.aromatic_matches_hydrophobic));
        if has_any {
            possible_site_matches += 1;
        } else if sf.required {
            return None;
        }
    }
    if possible_site_matches < min_match {
        return None;
    }

    let conformers = sample_quick_conformers(mol, cfg.max_conformers);
    let mut best_hit: Option<PharmScreenHit> = None;

    for conf in &conformers {
        let lig_feats = extract_ligand_features(conf);
        let compat: Vec<Vec<usize>> = active
            .iter()
            .map(|sf| {
                lig_feats
                    .iter()
                    .enumerate()
                    .filter(|(_, lf)| {
                        kinds_compatible(sf.kind, lf.kind, cfg.aromatic_matches_hydrophobic)
                    })
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();

        let mut ctx = CliqueSearchCtx {
            site_feats: &active,
            lig_feats: &lig_feats,
            compat: &compat,
            site_dists: &site_dists,
            site_tols: &site_tols,
            min_match,
            best_mappings: Vec::new(),
            nodes_visited: 0,
            node_budget: 2_500,
        };
        let mut cur = Vec::with_capacity(n_s);
        let mut used = vec![false; lig_feats.len()];
        ctx.dfs(0, &mut cur, &mut used);

        for mapping in ctx.best_mappings {
            if mapping.len() < min_match {
                continue;
            }
            let mob: Vec<[f64; 3]> = mapping.iter().map(|&(_, l)| lig_feats[l].center).collect();
            let tgt: Vec<[f64; 3]> = mapping.iter().map(|&(s, _)| active[s].center).collect();
            let aln: RigidAlignment = kabsch_align(&mob, &tgt);
            if aln.rmsd > cfg.max_rmsd_a {
                continue;
            }

            // Count receptor exclusion-sphere clashes, continuous penetration depth, and pocket shape fit
            let mut clashes = 0usize;
            let mut clash_depth = 0.0_f64;
            let mut n_heavy = 0usize;
            let mut n_inside_envelope = 0usize;
            let g_ctr = blueprint.geometry.center;
            let g_axes = blueprint.geometry.principal_axes;
            let g_half = [
                (0.55 * blueprint.geometry.principal_dims_a[0]).max(4.5),
                (0.55 * blueprint.geometry.principal_dims_a[1]).max(4.0),
                (0.55 * blueprint.geometry.principal_dims_a[2]).max(3.5),
            ];
            for a in &conf.atoms {
                if !a.is_heavy() {
                    continue;
                }
                n_heavy += 1;
                let p = aln.apply(a.coords);
                // Pocket PCA envelope containment
                let dp = [p[0] - g_ctr[0], p[1] - g_ctr[1], p[2] - g_ctr[2]];
                let u0 = dot3(dp, g_axes[0]) / g_half[0];
                let u1 = dot3(dp, g_axes[1]) / g_half[1];
                let u2 = dot3(dp, g_axes[2]) / g_half[2];
                if u0 * u0 + u1 * u1 + u2 * u2 <= 1.45 {
                    n_inside_envelope += 1;
                }
                let mut atom_pen = 0.0_f64;
                for ex in &blueprint.exclusion_spheres {
                    let dx = p[0] - ex.center[0];
                    if dx.abs() >= ex.radius_a {
                        continue;
                    }
                    let dy = p[1] - ex.center[1];
                    if dy.abs() >= ex.radius_a {
                        continue;
                    }
                    let dz = p[2] - ex.center[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < ex.radius_a * ex.radius_a {
                        let pen = ex.radius_a - d2.sqrt();
                        if pen > atom_pen {
                            atom_pen = pen;
                        }
                    }
                }
                if atom_pen > 0.0 {
                    clashes += 1;
                    clash_depth += atom_pen;
                    if clashes > cfg.max_exclusion_clashes {
                        break;
                    }
                }
            }
            if clashes > cfg.max_exclusion_clashes {
                continue;
            }

            // Score: sum of matched hotspot strength * weight + directional vector alignment
            // + pocket shape containment - RMSD & continuous steric clash penalties
            let mut raw_score = 0.0_f64;
            let mut dir_bonus = 0.0_f64;
            let mut ids = Vec::with_capacity(mapping.len());
            for &(s, l) in &mapping {
                raw_score += (-active[s].energy_kcal).max(0.5) * active[s].weight;
                if let (Some(sd), Some(ld)) = (active[s].direction, lig_feats[l].direction) {
                    let rld = aln.apply_dir(ld);
                    let c = dot3(rld, sd).clamp(-1.0, 1.0);
                    match active[s].kind {
                        PharmacophoreKind::HDonor => {
                            dir_bonus += 0.65 * c;
                        }
                        PharmacophoreKind::Aromatic => {
                            dir_bonus += 0.35 * c.abs();
                        }
                        _ => {}
                    }
                }
                ids.push(active[s].id.clone());
            }
            let shape_frac = if n_heavy > 0 {
                (n_inside_envelope as f64) / (n_heavy as f64)
            } else {
                0.0
            };
            let pharm_score = raw_score + dir_bonus + 1.80 * shape_frac
                - 1.35 * aln.rmsd
                - 0.70 * (clashes as f64)
                - 1.40 * clash_depth;

            if best_hit
                .as_ref()
                .is_none_or(|prev| pharm_score > prev.pharm_score)
            {
                let mut aligned = conf.clone();
                for a in &mut aligned.atoms {
                    a.coords = aln.apply(a.coords);
                }
                aligned
                    .properties
                    .insert("PharmCraft_Score".to_string(), format!("{pharm_score:.3}"));
                aligned.properties.insert(
                    "PharmCraft_Matched".to_string(),
                    format!("{}/{}", mapping.len(), active.len()),
                );
                aligned
                    .properties
                    .insert("PharmCraft_RMSD".to_string(), format!("{:.3}", aln.rmsd));
                best_hit = Some(PharmScreenHit {
                    name: mol.name.clone(),
                    smiles: mol.source_smiles.clone(),
                    matched_count: mapping.len(),
                    total_active_features: active.len(),
                    matched_feature_ids: ids,
                    pharm_rmsd_a: (aln.rmsd * 1000.0).round() / 1000.0,
                    exclusion_clashes: clashes,
                    pharm_score: (pharm_score * 1000.0).round() / 1000.0,
                    aligned_molecule: aligned,
                });
            }
        }
    }

    best_hit
}

/// Screens a slice of prepared 3D ligands in parallel with Rayon against `blueprint`.
#[must_use]
pub fn screen_molecules_parallel(
    ligands: &[Molecule3D],
    blueprint: &SiteBlueprint,
    cfg: &PharmScreenConfig,
) -> PharmScreenReport {
    let t0 = Instant::now();
    let mut hits: Vec<PharmScreenHit> = ligands
        .par_iter()
        .filter_map(|lig| screen_ligand_against_blueprint(lig, blueprint, cfg))
        .collect();
    hits.sort_by(|a, b| {
        b.pharm_score
            .total_cmp(&a.pharm_score)
            .then_with(|| b.matched_count.cmp(&a.matched_count))
            .then_with(|| a.pharm_rmsd_a.total_cmp(&b.pharm_rmsd_a))
    });
    let passed_count = hits.len();
    if cfg.max_hits > 0 && hits.len() > cfg.max_hits {
        hits.truncate(cfg.max_hits);
    }
    PharmScreenReport {
        target_name: blueprint.target_name.clone(),
        total_evaluated: ligands.len(),
        passed_count,
        elapsed_ms: t0.elapsed().as_millis(),
        hits,
    }
}

/// Streams a large SMILES library (`.smi` lines `(smiles, name)`) in fixed-size chunks across all
/// CPU cores, keeping peak memory bounded even on 1M+ compound libraries in `/Datos/CraftBio/libraries`.
#[must_use]
pub fn screen_smiles_stream<I>(
    iter: I,
    blueprint: &SiteBlueprint,
    cfg: &PharmScreenConfig,
    chunk_size: usize,
) -> PharmScreenReport
where
    I: Iterator<Item = (String, String)>,
{
    let t0 = Instant::now();
    let batch_cap = chunk_size.clamp(128, 8192);
    let mut total_evaluated = 0usize;
    let mut passed_count = 0usize;
    let mut top_hits: Vec<PharmScreenHit> = Vec::new();
    let mut batch: Vec<(String, String)> = Vec::with_capacity(batch_cap);

    let process_batch =
        |items: &[(String, String)], top: &mut Vec<PharmScreenHit>, passed: &mut usize| {
            let local_hits: Vec<PharmScreenHit> = items
                .par_iter()
                .filter_map(|(smi, name)| {
                    let lig = struct_core::parse_smiles_to_3d(smi, Some(name)).ok()?;
                    screen_ligand_against_blueprint(&lig, blueprint, cfg)
                })
                .collect();
            *passed += local_hits.len();
            top.extend(local_hits);
            top.sort_by(|a, b| {
                b.pharm_score
                    .total_cmp(&a.pharm_score)
                    .then_with(|| b.matched_count.cmp(&a.matched_count))
            });
            if cfg.max_hits > 0 && top.len() > cfg.max_hits {
                top.truncate(cfg.max_hits);
            }
        };

    for item in iter {
        total_evaluated += 1;
        batch.push(item);
        if batch.len() >= batch_cap {
            process_batch(&batch, &mut top_hits, &mut passed_count);
            batch.clear();
        }
    }
    if !batch.is_empty() {
        process_batch(&batch, &mut top_hits, &mut passed_count);
    }

    PharmScreenReport {
        target_name: blueprint.target_name.clone(),
        total_evaluated,
        passed_count,
        elapsed_ms: t0.elapsed().as_millis(),
        hits: top_hits,
    }
}

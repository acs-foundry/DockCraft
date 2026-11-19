#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use struct_core::{Atom3D, Molecule3D, metal_chelation_tier};

use crate::vina::{ScoringFunction, pairwise_energy_and_deriv_full};

/// Chemical feature class for pharmacophore-constrained docking (Glide / SeeSAR / GOLD style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PharmacophoreKind {
    HDonor,
    HAcceptor,
    Aromatic,
    Hydrophobic,
    PositiveIonizable,
    NegativeIonizable,
    MetalChelate,
    AnyHeavy,
}

/// Analytical geometric / chemical constraint enforced during global search and L-BFGS minimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DockingConstraint {
    /// Requires at least one matching ligand heavy atom to sit within `radius` Å of `center`.
    /// Uses a smooth soft-min / closest-atom quadratic potential outside `radius` (`0.0` inside).
    PharmacophoreSphere {
        center: [f64; 3],
        radius: f64,
        kind: PharmacophoreKind,
        weight: f64,
    },
    /// Template / core-tether constraint anchoring specified ligand atom indices near target 3D
    /// reference coordinates (with a flat-bottom tolerance `tolerance_a`).
    CoreTether {
        atom_targets: Vec<(usize, [f64; 3])>,
        tolerance_a: f64,
        force_k: f64,
    },
    /// Covalent warhead attachment constraint tethering `ligand_atom_idx` to `receptor_atom_idx`
    /// at `target_bond_len` (e.g. `1.82 Å` for Cys SG–C, `1.43 Å` for Ser OG–C, `1.47 Å` for Lys NZ–C).
    CovalentAttachment {
        receptor_atom_idx: usize,
        ligand_atom_idx: usize,
        target_bond_len: f64,
        force_k: f64,
    },
}

impl DockingConstraint {
    /// Returns `(rec_atom_idx, lig_atom_idx)` if this constraint is a covalent attachment.
    #[must_use]
    pub fn covalent_pair(&self) -> Option<(usize, usize)> {
        match self {
            Self::CovalentAttachment {
                receptor_atom_idx,
                ligand_atom_idx,
                ..
            } => Some((*receptor_atom_idx, *ligand_atom_idx)),
            _ => None,
        }
    }
}

/// Pre-resolved constraint data bound to a specific ligand & receptor for zero-allocation evaluation.
#[derive(Debug, Clone)]
pub(crate) enum ResolvedConstraint {
    Pharmacophore {
        center: [f64; 3],
        radius: f64,
        weight: f64,
        matching_atoms: Vec<usize>,
    },
    CoreTether {
        atom_targets: Vec<(usize, [f64; 3])>,
        tolerance_a: f64,
        force_k: f64,
    },
    Covalent {
        rec_coords: [f64; 3],
        lig_atom: usize,
        target_len: f64,
        force_k: f64,
    },
}

pub(crate) fn resolve_constraints(
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    constraints: &[DockingConstraint],
) -> Vec<ResolvedConstraint> {
    let mut out = Vec::with_capacity(constraints.len());
    for c in constraints {
        match c {
            DockingConstraint::PharmacophoreSphere {
                center,
                radius,
                kind,
                weight,
            } => {
                let mut matching = Vec::new();
                for (idx, a) in ligand.atoms.iter().enumerate() {
                    if !a.is_heavy() {
                        continue;
                    }
                    let ok = match kind {
                        PharmacophoreKind::HDonor => a.h_donor,
                        PharmacophoreKind::HAcceptor => a.h_acceptor,
                        PharmacophoreKind::Aromatic => a.aromatic,
                        PharmacophoreKind::Hydrophobic => a.hydrophobic,
                        PharmacophoreKind::PositiveIonizable => a.formal_charge > 0,
                        PharmacophoreKind::NegativeIonizable => a.formal_charge < 0,
                        PharmacophoreKind::MetalChelate => metal_chelation_tier(ligand, idx) >= 1,
                        PharmacophoreKind::AnyHeavy => true,
                    };
                    if ok {
                        matching.push(idx);
                    }
                }
                if !matching.is_empty() {
                    out.push(ResolvedConstraint::Pharmacophore {
                        center: *center,
                        radius: radius.max(0.25),
                        weight: weight.max(0.1),
                        matching_atoms: matching,
                    });
                }
            }
            DockingConstraint::CoreTether {
                atom_targets,
                tolerance_a,
                force_k,
            } => {
                let valid: Vec<(usize, [f64; 3])> = atom_targets
                    .iter()
                    .copied()
                    .filter(|&(idx, _)| idx < ligand.atoms.len())
                    .collect();
                if !valid.is_empty() {
                    out.push(ResolvedConstraint::CoreTether {
                        atom_targets: valid,
                        tolerance_a: tolerance_a.max(0.0),
                        force_k: force_k.max(0.1),
                    });
                }
            }
            DockingConstraint::CovalentAttachment {
                receptor_atom_idx,
                ligand_atom_idx,
                target_bond_len,
                force_k,
            } => {
                if let (Some(ra), Some(_)) = (
                    receptor.atoms.get(*receptor_atom_idx),
                    ligand.atoms.get(*ligand_atom_idx),
                ) {
                    out.push(ResolvedConstraint::Covalent {
                        rec_coords: ra.coords,
                        lig_atom: *ligand_atom_idx,
                        target_len: target_bond_len.clamp(1.2, 2.5),
                        force_k: force_k.max(1.0),
                    });
                }
            }
        }
    }
    out
}

/// Evaluates constraint penalty energy and accumulates exact analytical gradients into `atom_grads`.
pub(crate) fn eval_constraints_and_grad(
    resolved: &[ResolvedConstraint],
    world_coords: &[[f64; 3]],
    atom_grads: Option<&mut [[f64; 3]]>,
) -> f64 {
    if resolved.is_empty() {
        return 0.0;
    }
    let mut e_tot = 0.0_f64;
    let mut grads_opt = atom_grads;

    for rc in resolved {
        match rc {
            ResolvedConstraint::Pharmacophore {
                center,
                radius,
                weight,
                matching_atoms,
            } => {
                // Find the closest matching ligand heavy atom to the pharmacophore sphere
                let mut best_idx = matching_atoms[0];
                let mut best_d2 = f64::INFINITY;
                for &ai in matching_atoms {
                    let p = world_coords[ai];
                    let d2 = (p[0] - center[0]).powi(2)
                        + (p[1] - center[1]).powi(2)
                        + (p[2] - center[2]).powi(2);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_idx = ai;
                    }
                }
                let dist = best_d2.sqrt();
                if dist > *radius {
                    let excess = dist - *radius;
                    e_tot += weight * excess * excess;
                    if let Some(grads) = grads_opt.as_deref_mut()
                        && dist > 1e-8
                    {
                        let coeff = 2.0 * weight * excess / dist;
                        let p = world_coords[best_idx];
                        grads[best_idx][0] += coeff * (p[0] - center[0]);
                        grads[best_idx][1] += coeff * (p[1] - center[1]);
                        grads[best_idx][2] += coeff * (p[2] - center[2]);
                    }
                }
            }
            ResolvedConstraint::CoreTether {
                atom_targets,
                tolerance_a,
                force_k,
            } => {
                for &(ai, tgt) in atom_targets {
                    let p = world_coords[ai];
                    let dx = p[0] - tgt[0];
                    let dy = p[1] - tgt[1];
                    let dz = p[2] - tgt[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist > *tolerance_a {
                        let excess = dist - *tolerance_a;
                        e_tot += force_k * excess * excess;
                        if let Some(grads) = grads_opt.as_deref_mut()
                            && dist > 1e-8
                        {
                            let coeff = 2.0 * force_k * excess / dist;
                            grads[ai][0] += coeff * dx;
                            grads[ai][1] += coeff * dy;
                            grads[ai][2] += coeff * dz;
                        }
                    }
                }
            }
            ResolvedConstraint::Covalent {
                rec_coords,
                lig_atom,
                target_len,
                force_k,
            } => {
                let p = world_coords[*lig_atom];
                let dx = p[0] - rec_coords[0];
                let dy = p[1] - rec_coords[1];
                let dz = p[2] - rec_coords[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let diff = dist - *target_len;
                e_tot += 0.5 * force_k * diff * diff;
                if let Some(grads) = grads_opt.as_deref_mut()
                    && dist > 1e-8
                {
                    let coeff = force_k * diff / dist;
                    grads[*lig_atom][0] += coeff * dx;
                    grads[*lig_atom][1] += coeff * dy;
                    grads[*lig_atom][2] += coeff * dz;
                }
            }
        }
    }
    e_tot
}

/// Rotatable sidechain model for flexible receptor docking (`flex_residues`, e.g. `"A:ASP189"` or `"ASP189"`).
#[derive(Debug, Clone)]
pub(crate) struct FlexSidechain {
    pub rec_atom_indices: Vec<usize>,
    pub base_atoms: Vec<Atom3D>,
    /// Sidechain chi torsions `(pivot_u_coords, axis_v_coords, local_moving_indices)`
    pub chi_torsions: Vec<(usize, usize, Vec<usize>)>,
}

fn matches_residue_spec(atom: &Atom3D, spec: &str) -> bool {
    let s = spec.trim().to_ascii_uppercase();
    if s.is_empty() {
        return false;
    }
    let (chain_filter, rest) = if let Some((ch, r)) = s.split_once(':') {
        (ch.chars().next(), r.trim())
    } else {
        (None, s.as_str())
    };
    if let Some(cf) = chain_filter
        && atom.chain_id.to_ascii_uppercase() != cf
    {
        return false;
    }
    let rn = atom.res_name.trim().to_ascii_uppercase();
    let full = format!("{rn}{}", atom.res_seq);
    rest == full || rest == atom.res_seq.to_string()
}

/// Extracts flexible sidechains (all heavy atoms distal to `CA-CB`) and returns
/// `(flex_sidechains, excluded_receptor_mask)`.
pub(crate) fn extract_flexible_sidechains(
    receptor: &Molecule3D,
    specs: &[String],
) -> (Vec<FlexSidechain>, Vec<bool>) {
    let mut excluded = vec![false; receptor.atoms.len()];
    if specs.is_empty() {
        return (Vec::new(), excluded);
    }

    // Group matching heavy atoms by (chain_id, res_seq)
    let mut res_groups: std::collections::BTreeMap<(char, i32), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, a) in receptor.atoms.iter().enumerate() {
        if !a.is_heavy() || a.is_hetatm {
            continue;
        }
        if specs.iter().any(|sp| matches_residue_spec(a, sp)) {
            res_groups
                .entry((a.chain_id, a.res_seq))
                .or_default()
                .push(idx);
        }
    }

    let mut sidechains = Vec::new();
    for (_key, atom_ids) in res_groups {
        let find_by_name = |nm: &str| -> Option<usize> {
            atom_ids
                .iter()
                .copied()
                .find(|&i| receptor.atoms[i].name.trim().eq_ignore_ascii_case(nm))
        };
        let (Some(ca), Some(cb)) = (find_by_name("CA"), find_by_name("CB")) else {
            continue;
        };
        // Distal heavy atoms beyond CB (exclude backbone N, CA, C, O, OXT and CB itself)
        let distal: Vec<usize> = atom_ids
            .iter()
            .copied()
            .filter(|&i| {
                !matches!(
                    receptor.atoms[i].name.trim().to_ascii_uppercase().as_str(),
                    "N" | "CA" | "C" | "O" | "OXT" | "CB"
                )
            })
            .collect();
        if distal.is_empty() {
            continue;
        }
        for &idx in &distal {
            excluded[idx] = true;
        }
        // Build local atom list: [0]=CA, [1]=CB, [2..]=distal atoms
        let mut all_local = Vec::with_capacity(distal.len() + 2);
        all_local.push(ca);
        all_local.push(cb);
        all_local.extend_from_slice(&distal);

        let base_atoms: Vec<Atom3D> = all_local
            .iter()
            .map(|&i| receptor.atoms[i].clone())
            .collect();
        let mut chi_torsions = Vec::new();
        // Chi1: CA(0) -> CB(1), moving all distal atoms 2..len
        chi_torsions.push((0, 1, (2..all_local.len()).collect()));
        // Chi2: if CG / CG1 / OG / SG exists at local index >= 2 and there are atoms beyond it
        if let Some(cg_local) = (2..all_local.len()).find(|&li| {
            let nm = base_atoms[li].name.trim().to_ascii_uppercase();
            matches!(nm.as_str(), "CG" | "CG1" | "SG" | "OG")
        }) {
            let beyond_cg: Vec<usize> = (2..all_local.len()).filter(|&li| li != cg_local).collect();
            if !beyond_cg.is_empty() {
                chi_torsions.push((1, cg_local, beyond_cg));
            }
        }

        sidechains.push(FlexSidechain {
            rec_atom_indices: all_local,
            base_atoms,
            chi_torsions,
        });
    }
    (sidechains, excluded)
}

fn rotate_pt(p: [f64; 3], origin: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let v = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    let ca = angle.cos();
    let sa = angle.sin();
    let dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
    let cx = axis[1] * v[2] - axis[2] * v[1];
    let cy = axis[2] * v[0] - axis[0] * v[2];
    let cz = axis[0] * v[1] - axis[1] * v[0];
    [
        origin[0] + v[0] * ca + cx * sa + axis[0] * dot * (1.0 - ca),
        origin[1] + v[1] * ca + cy * sa + axis[1] * dot * (1.0 - ca),
        origin[2] + v[2] * ca + cz * sa + axis[2] * dot * (1.0 - ca),
    ]
}

impl FlexSidechain {
    /// Applies `[chi1, chi2, ...]` angles to produce moved sidechain `Atom3D`s (only distal atoms 2..).
    pub(crate) fn posed_distal_atoms(&self, angles: &[f64]) -> Vec<Atom3D> {
        let mut atoms = self.base_atoms.clone();
        for (k, (u, v, movers)) in self.chi_torsions.iter().enumerate() {
            let ang = angles.get(k).copied().unwrap_or(0.0);
            if ang.abs() < 1e-9 {
                continue;
            }
            let pu = atoms[*u].coords;
            let pv = atoms[*v].coords;
            let d = [pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if len < 1e-6 {
                continue;
            }
            let ax = [d[0] / len, d[1] / len, d[2] / len];
            for &m in movers {
                atoms[m].coords = rotate_pt(atoms[m].coords, pu, ax, ang);
            }
        }
        atoms.split_off(2)
    }

    /// Evaluates pairwise interaction + ligand atom gradients between a posed flexible sidechain
    /// and ligand heavy atoms.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eval_against_ligand(
        &self,
        angles: &[f64],
        ligand: &Molecule3D,
        heavy_indices: &[usize],
        heavy_tiers: &[u8],
        world_coords: &[[f64; 3]],
        sf: ScoringFunction,
        enable_metal: bool,
        atom_grads: Option<&mut [[f64; 3]]>,
    ) -> f64 {
        let distal = self.posed_distal_atoms(angles);
        let mut e_sum = 0.0_f64;
        let mut grads_opt = atom_grads;
        for (hp, &li) in heavy_indices.iter().enumerate() {
            let mut la = ligand.atoms[li].clone();
            la.coords = world_coords[li];
            let tier = heavy_tiers[hp];
            for ra in &distal {
                let dx = la.coords[0] - ra.coords[0];
                let dy = la.coords[1] - ra.coords[1];
                let dz = la.coords[2] - ra.coords[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= 64.0 || r2 < 1e-8 {
                    continue;
                }
                let r = r2.sqrt();
                let (e, de, _) = pairwise_energy_and_deriv_full(ra, &la, tier, r, sf, enable_metal);
                e_sum += e;
                if let Some(grads) = grads_opt.as_deref_mut() {
                    let f = de / r;
                    grads[li][0] += f * dx;
                    grads[li][1] += f * dy;
                    grads[li][2] += f * dz;
                }
            }
        }
        e_sum
    }

    /// Cyclic rotamer/chi-angle optimization of a flexible sidechain against a fixed ligand pose
    /// (with soft harmonic restraint toward crystal rotamer so sidechains don't clash with protein).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn optimize_rotamer(
        &self,
        receptor: &Molecule3D,
        ligand: &Molecule3D,
        heavy_indices: &[usize],
        heavy_tiers: &[u8],
        world_coords: &[[f64; 3]],
        sf: ScoringFunction,
        enable_metal: bool,
    ) -> Vec<f64> {
        let mut best_angles = vec![0.0_f64; self.chi_torsions.len()];
        let score_angles = |angs: &[f64]| -> f64 {
            let e_lig = self.eval_against_ligand(
                angs,
                ligand,
                heavy_indices,
                heavy_tiers,
                world_coords,
                sf,
                enable_metal,
                None,
            );
            // Check steric clash of moved sidechain against rest of rigid receptor
            let distal = self.posed_distal_atoms(angs);
            let mut e_clash = 0.0_f64;
            for da in &distal {
                for (ri, ra) in receptor.atoms.iter().enumerate() {
                    if !ra.is_heavy()
                        || (ra.chain_id == da.chain_id && ra.res_seq == da.res_seq)
                        || self.rec_atom_indices.contains(&ri)
                    {
                        continue;
                    }
                    let d2 = (da.coords[0] - ra.coords[0]).powi(2)
                        + (da.coords[1] - ra.coords[1]).powi(2)
                        + (da.coords[2] - ra.coords[2]).powi(2);
                    if d2 < 2.6 * 2.6 {
                        let d = d2.sqrt();
                        e_clash += 4.0 * (2.6 - d) * (2.6 - d);
                    }
                }
            }
            // Mild preference for rotameric minima near crystal state
            let e_prior: f64 = angs.iter().map(|a| 0.08 * (1.0 - a.cos())).sum();
            e_lig + e_clash + e_prior
        };

        let mut best_e = score_angles(&best_angles);
        // Coarse 12-step rotamer sweep per chi + fine ±15° polish
        for chi in 0..best_angles.len() {
            for step in 0..12 {
                let mut cand = best_angles.clone();
                cand[chi] = (step as f64) * (std::f64::consts::TAU / 12.0) - std::f64::consts::PI;
                let e = score_angles(&cand);
                if e < best_e - 1e-4 {
                    best_e = e;
                    best_angles = cand;
                }
            }
            for delta_deg in [-20.0_f64, -10.0, -5.0, 5.0, 10.0, 20.0] {
                let mut cand = best_angles.clone();
                cand[chi] += delta_deg.to_radians();
                let e = score_angles(&cand);
                if e < best_e - 1e-4 {
                    best_e = e;
                    best_angles = cand;
                }
            }
        }
        best_angles
    }
}

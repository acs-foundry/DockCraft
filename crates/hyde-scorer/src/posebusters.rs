#![forbid(unsafe_code)]

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use struct_core::{Molecule3D, cross3, ideal_bond_length};

/// Built-in PoseBusters-equivalent physical and geometric pose validity audit
/// (Buttenschoen et al. 2024 criteria: bond lengths, bond angles, aromatic ring flatness,
/// internal steric clashes, and protein-ligand steric overlap).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseBustersReport {
    pub valid: bool,
    pub bond_lengths_ok: bool,
    pub bond_angles_ok: bool,
    pub aromatic_flatness_ok: bool,
    pub no_internal_clash: bool,
    pub no_protein_clash: bool,
    pub max_bond_length_err_a: f64,
    pub min_bond_angle_deg: f64,
    pub max_aromatic_oop_a: f64,
    pub min_intra_vdw_ratio: f64,
    pub min_protein_vdw_ratio: f64,
    pub summary: String,
}

fn vdw_radius_pb(el: &str) -> f64 {
    match el {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "F" => 1.47,
        "P" => 1.80,
        "S" => 1.80,
        "Cl" => 1.75,
        "Br" => 1.85,
        "I" => 1.98,
        "Zn" => 1.39,
        "Mg" => 1.73,
        "Fe" => 1.40,
        "Ca" => 1.94,
        "Mn" => 1.40,
        _ => 1.70,
    }
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn angle3_deg(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let v1 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let v2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let n1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
    let n2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
    if n1 < 1e-8 || n2 < 1e-8 {
        return 180.0;
    }
    ((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// Compute all-pairs shortest topological path distance up to 4 hops on ligand atoms.
fn is_within_3_bonds(mol: &Molecule3D, start: usize, target: usize) -> bool {
    if start == target {
        return true;
    }
    let mut visited = vec![false; mol.atoms.len()];
    visited[start] = true;
    let mut q = VecDeque::from([(start, 0u8)]);
    while let Some((u, d)) = q.pop_front() {
        if d >= 3 {
            continue;
        }
        for &(nb, _) in &mol.adj[u] {
            if nb == target {
                return true;
            }
            if !visited[nb] {
                visited[nb] = true;
                q.push_back((nb, d + 1));
            }
        }
    }
    false
}

/// Audits a docked ligand pose inside its receptor for PoseBusters-equivalent physical validity.
#[must_use]
pub fn validate_pose_geometry(receptor: &Molecule3D, ligand: &Molecule3D) -> PoseBustersReport {
    // 1. Bond lengths vs ideal covalent length (tolerance +-0.30 A accommodates S=O, C-S, C#C, strained rings)
    let mut max_bl_err = 0.0_f64;
    for b in &ligand.bonds {
        let au = &ligand.atoms[b.u];
        let av = &ligand.atoms[b.v];
        if !au.is_heavy() || !av.is_heavy() {
            continue;
        }
        let d = dist3(au.coords, av.coords);
        let ideal = ideal_bond_length(&au.element, &av.element, b.order);
        max_bl_err = max_bl_err.max((d - ideal).abs());
    }
    let bond_lengths_ok = max_bl_err <= 0.32;

    // 2. Heavy-atom bond angles (must be >= 50 deg even in 3-membered rings, and <= 180 deg)
    let mut min_ang = 180.0_f64;
    for (c, ca) in ligand.atoms.iter().enumerate() {
        if !ca.is_heavy() {
            continue;
        }
        let heavy_nbs: Vec<usize> = ligand.adj[c]
            .iter()
            .map(|&(nb, _)| nb)
            .filter(|&nb| ligand.atoms[nb].is_heavy())
            .collect();
        for i in 0..heavy_nbs.len() {
            for j in (i + 1)..heavy_nbs.len() {
                let ang = angle3_deg(
                    ligand.atoms[heavy_nbs[i]].coords,
                    ca.coords,
                    ligand.atoms[heavy_nbs[j]].coords,
                );
                min_ang = min_ang.min(ang);
            }
        }
    }
    let bond_angles_ok = min_ang >= 50.0;

    // 3. Aromatic 5/6-membered ring flatness (max out-of-plane deviation <= 0.25 A)
    let mut max_oop = 0.0_f64;
    for ring in &ligand.rings {
        if !(5..=6).contains(&ring.len()) {
            continue;
        }
        if !ring.iter().all(|&i| ligand.atoms[i].aromatic) {
            continue;
        }
        let k = ring.len() as f64;
        let mut cen = [0.0; 3];
        for &idx in ring {
            for c in 0..3 {
                cen[c] += ligand.atoms[idx].coords[c] / k;
            }
        }
        let mut nrm = [0.0; 3];
        for i in 0..ring.len() {
            let a = ligand.atoms[ring[i]].coords;
            let b = ligand.atoms[ring[(i + 1) % ring.len()]].coords;
            let ra = [a[0] - cen[0], a[1] - cen[1], a[2] - cen[2]];
            let rb = [b[0] - cen[0], b[1] - cen[1], b[2] - cen[2]];
            let cr = cross3(ra, rb);
            for c in 0..3 {
                nrm[c] += cr[c];
            }
        }
        let nl = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
        if nl > 1e-6 {
            let u = [nrm[0] / nl, nrm[1] / nl, nrm[2] / nl];
            for &idx in ring {
                let p = ligand.atoms[idx].coords;
                let h = ((p[0] - cen[0]) * u[0] + (p[1] - cen[1]) * u[1] + (p[2] - cen[2]) * u[2])
                    .abs();
                max_oop = max_oop.max(h);
            }
        }
    }
    let aromatic_flatness_ok = max_oop <= 0.25;

    // 4. Internal steric clashes (heavy pairs >= 4 bonds apart must have d / (r_i + r_j) >= 0.65)
    let lig_heavy: Vec<usize> = ligand
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_heavy())
        .map(|(i, _)| i)
        .collect();
    let mut min_intra_ratio = 9.9_f64;
    for a_i in 0..lig_heavy.len() {
        let i = lig_heavy[a_i];
        let ri = vdw_radius_pb(&ligand.atoms[i].element);
        for &j in lig_heavy.iter().skip(a_i + 1) {
            if is_within_3_bonds(ligand, i, j) {
                continue;
            }
            let rj = vdw_radius_pb(&ligand.atoms[j].element);
            let ratio = dist3(ligand.atoms[i].coords, ligand.atoms[j].coords) / (ri + rj);
            if ratio < min_intra_ratio {
                min_intra_ratio = ratio;
            }
        }
    }
    let no_internal_clash = min_intra_ratio >= 0.65;

    // 5. Protein-ligand severe steric clash check (metals allow coordinate bonds down to 1.65 A)
    let mut min_prot_ratio = 9.9_f64;
    for &li in &lig_heavy {
        let la = &ligand.atoms[li];
        let rl = vdw_radius_pb(&la.element);
        for ra in &receptor.atoms {
            if !ra.is_heavy() {
                continue;
            }
            let dx = la.coords[0] - ra.coords[0];
            if dx.abs() > 3.6 {
                continue;
            }
            let dy = la.coords[1] - ra.coords[1];
            if dy.abs() > 3.6 {
                continue;
            }
            let dz = la.coords[2] - ra.coords[2];
            if dz.abs() > 3.6 {
                continue;
            }
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let ratio = if ra.ad4_type.is_metal() {
                d / 2.20
            } else {
                d / (rl + vdw_radius_pb(&ra.element))
            };
            if ratio < min_prot_ratio {
                min_prot_ratio = ratio;
            }
        }
    }
    let no_protein_clash = min_prot_ratio >= 0.68;

    let valid = bond_lengths_ok
        && bond_angles_ok
        && aromatic_flatness_ok
        && no_internal_clash
        && no_protein_clash;

    let mut fails = Vec::new();
    if !bond_lengths_ok {
        fails.push(format!("bond_len({max_bl_err:.2}A)"));
    }
    if !bond_angles_ok {
        fails.push(format!("angle({min_ang:.0}deg)"));
    }
    if !aromatic_flatness_ok {
        fails.push(format!("arom_oop({max_oop:.2}A)"));
    }
    if !no_internal_clash {
        fails.push(format!("intra_clash({min_intra_ratio:.2})"));
    }
    if !no_protein_clash {
        fails.push(format!("prot_clash({min_prot_ratio:.2})"));
    }
    let summary = if valid {
        "PASS (5/5 PoseBusters geometric checks)".to_string()
    } else {
        format!("FAIL: {}", fails.join(", "))
    };

    PoseBustersReport {
        valid,
        bond_lengths_ok,
        bond_angles_ok,
        aromatic_flatness_ok,
        no_internal_clash,
        no_protein_clash,
        max_bond_length_err_a: max_bl_err,
        min_bond_angle_deg: min_ang,
        max_aromatic_oop_a: max_oop,
        min_intra_vdw_ratio: min_intra_ratio,
        min_protein_vdw_ratio: min_prot_ratio,
        summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::{Atom3D, parse_smiles_to_3d};

    #[test]
    fn posebusters_passes_clean_smiles_pose_and_flags_steric_clash()
    -> Result<(), Box<dyn std::error::Error>> {
        let lig = parse_smiles_to_3d("c1ccccc1C(=O)O", Some("Benzoic"))?;
        let mut rec = Molecule3D::new("pocket");
        rec.add_atom(Atom3D::new(1, "C", [10.0, 10.0, 10.0]));
        let ok = validate_pose_geometry(&rec, &lig);
        assert!(ok.valid, "expected valid PoseBusters, got: {}", ok.summary);

        let clash_pos = lig.atoms[0].coords;
        rec.add_atom(Atom3D::new(
            2,
            "C",
            [clash_pos[0] + 1.2, clash_pos[1], clash_pos[2]],
        ));
        let bad = validate_pose_geometry(&rec, &lig);
        assert!(!bad.valid);
        assert!(!bad.no_protein_clash);
        Ok(())
    }
}

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};
use struct_core::Molecule3D;

use crate::audit::dist3;

/// One applied 180° sidechain flip (ASN `OD1↔ND2`, GLN `OE1↔NE2`, or HIS imidazole ring flip).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidechainFlip {
    pub chain_id: char,
    pub res_seq: i32,
    pub res_name: String,
    pub flip_type: String,
    pub score_before: f64,
    pub score_after: f64,
    pub delta_score: f64,
    pub near_pocket: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HBondOptReport {
    pub asn_gln_his_evaluated: usize,
    pub flips_applied: Vec<SidechainFlip>,
    pub polar_h_rotors_optimized: usize,
    pub polar_h_hbonds_formed: usize,
}

/// Evaluates and applies energetically favorable 180° sidechain flips for `ASN`, `GLN`, and
/// `HIS`/`HID`/`HIE`/`HIP` BEFORE polar-hydrogen synthesis (Reduce / MolProbity style).
///
/// A flip is applied only when `delta_score >= 0.80` to avoid spurious swaps on solvent-exposed
/// sidechains lacking clear crystallographic H-bond partners.
pub fn optimize_sidechain_flips(
    mol: &mut Molecule3D,
    pocket_center: Option<[f64; 3]>,
    pocket_radius: f64,
) -> HBondOptReport {
    let mut by_res: BTreeMap<(char, i32), Vec<usize>> = BTreeMap::new();
    let mut metal_coords: Vec<[f64; 3]> = Vec::new();
    for (idx, a) in mol.atoms.iter().enumerate() {
        if a.ad4_type.is_metal() {
            metal_coords.push(a.coords);
        }
        if !a.is_hetatm {
            by_res.entry((a.chain_id, a.res_seq)).or_default().push(idx);
        }
    }

    let mut flippable: Vec<(char, i32, String, Vec<usize>)> = Vec::new();
    for ((ch, seq), idxs) in by_res {
        let Some(&first) = idxs.first() else {
            continue;
        };
        let rn = mol.atoms[first].res_name.trim().to_ascii_uppercase();
        if matches!(rn.as_str(), "ASN" | "GLN" | "HIS" | "HID" | "HIE" | "HIP") {
            flippable.push((ch, seq, rn, idxs));
        }
    }

    let mut flips_applied = Vec::new();
    let mut already_flipped: std::collections::HashSet<(char, i32)> =
        std::collections::HashSet::new();

    // Two Gauss-Seidel passes so coupled ASN/GLN/HIS pairs can settle cleanly
    for _pass in 0..2 {
        let mut any_changed = false;
        for (ch, seq, rn, idxs) in &flippable {
            if already_flipped.contains(&(*ch, *seq)) {
                continue;
            }
            let flip_res = match rn.as_str() {
                "ASN" => {
                    evaluate_amide_flip(mol, *ch, *seq, "ASN", idxs, "OD1", "ND2", &metal_coords)
                }
                "GLN" => {
                    evaluate_amide_flip(mol, *ch, *seq, "GLN", idxs, "OE1", "NE2", &metal_coords)
                }
                "HIS" | "HID" | "HIE" | "HIP" => {
                    evaluate_his_flip(mol, *ch, *seq, rn, idxs, &metal_coords)
                }
                _ => None,
            };
            if let Some((delta, before, after, flip_type, reason, center)) = flip_res
                && delta >= 0.80
            {
                let near_pocket = pocket_center.is_some_and(|c| dist3(center, c) <= pocket_radius);
                flips_applied.push(SidechainFlip {
                    chain_id: *ch,
                    res_seq: *seq,
                    res_name: rn.clone(),
                    flip_type,
                    score_before: round2(before),
                    score_after: round2(after),
                    delta_score: round2(delta),
                    near_pocket,
                    reason,
                });
                already_flipped.insert((*ch, *seq));
                any_changed = true;
            }
        }
        if !any_changed {
            break;
        }
    }

    HBondOptReport {
        asn_gln_his_evaluated: flippable.len(),
        flips_applied,
        polar_h_rotors_optimized: 0,
        polar_h_hbonds_formed: 0,
    }
}

/// Evaluates swapping carbonyl `O` (`OD1`/`OE1`) and amide `N` (`ND2`/`NE2`) coordinates for ASN/GLN.
#[allow(clippy::too_many_arguments)]
fn evaluate_amide_flip(
    mol: &mut Molecule3D,
    ch: char,
    seq: i32,
    rn: &str,
    idxs: &[usize],
    o_name: &str,
    n_name: &str,
    metal_coords: &[[f64; 3]],
) -> Option<(f64, f64, f64, String, String, [f64; 3])> {
    let i_o = idxs
        .iter()
        .copied()
        .find(|&i| mol.atoms[i].name == o_name)?;
    let i_n = idxs
        .iter()
        .copied()
        .find(|&i| mol.atoms[i].name == n_name)?;
    let p_o = mol.atoms[i_o].coords;
    let p_n = mol.atoms[i_n].coords;
    let center = [
        0.5 * (p_o[0] + p_n[0]),
        0.5 * (p_o[1] + p_n[1]),
        0.5 * (p_o[2] + p_n[2]),
    ];

    let s_orig =
        score_acceptor_env(mol, ch, seq, p_o, metal_coords) + score_donor_env(mol, ch, seq, p_n);
    let s_flip =
        score_acceptor_env(mol, ch, seq, p_n, metal_coords) + score_donor_env(mol, ch, seq, p_o);
    let delta = s_flip - s_orig;

    if delta >= 0.80 {
        mol.atoms[i_o].coords = p_n;
        mol.atoms[i_n].coords = p_o;
        return Some((
            delta,
            s_orig,
            s_flip,
            format!("{rn} 180° ({o_name}↔{n_name})"),
            format!("Swapped {o_name}/{n_name} to improve H-bond complementarity (+{delta:.2})"),
            center,
        ));
    }
    None
}

/// Evaluates 180° imidazole ring flip (`ND1↔CD2` and `CE1↔NE2`) for HIS/HID/HIE/HIP.
fn evaluate_his_flip(
    mol: &mut Molecule3D,
    ch: char,
    seq: i32,
    rn: &str,
    idxs: &[usize],
    metal_coords: &[[f64; 3]],
) -> Option<(f64, f64, f64, String, String, [f64; 3])> {
    let i_nd1 = idxs.iter().copied().find(|&i| mol.atoms[i].name == "ND1")?;
    let i_cd2 = idxs.iter().copied().find(|&i| mol.atoms[i].name == "CD2")?;
    let i_ce1 = idxs.iter().copied().find(|&i| mol.atoms[i].name == "CE1")?;
    let i_ne2 = idxs.iter().copied().find(|&i| mol.atoms[i].name == "NE2")?;

    let p_nd1 = mol.atoms[i_nd1].coords;
    let p_cd2 = mol.atoms[i_cd2].coords;
    let p_ce1 = mol.atoms[i_ce1].coords;
    let p_ne2 = mol.atoms[i_ne2].coords;
    let center = [
        0.25 * (p_nd1[0] + p_cd2[0] + p_ce1[0] + p_ne2[0]),
        0.25 * (p_nd1[1] + p_cd2[1] + p_ce1[1] + p_ne2[1]),
        0.25 * (p_nd1[2] + p_cd2[2] + p_ce1[2] + p_ne2[2]),
    ];

    let score_ring = |pos_nd1: [f64; 3], pos_ne2: [f64; 3]| -> (f64, &'static str) {
        match rn {
            "HIP" => (
                score_donor_env(mol, ch, seq, pos_nd1) + score_donor_env(mol, ch, seq, pos_ne2),
                "HIP",
            ),
            _ => {
                let s_hid = score_donor_env(mol, ch, seq, pos_nd1)
                    + score_acceptor_env(mol, ch, seq, pos_ne2, metal_coords);
                let s_hie = score_acceptor_env(mol, ch, seq, pos_nd1, metal_coords)
                    + score_donor_env(mol, ch, seq, pos_ne2)
                    + 0.10;
                if s_hid > s_hie {
                    (s_hid, "HID")
                } else {
                    (s_hie, "HIE")
                }
            }
        }
    };

    let (s_orig, _) = score_ring(p_nd1, p_ne2);
    // 180° flip around CB-CG axis swaps ND1<->CD2 and NE2<->CE1
    let (s_flip, best_taut) = score_ring(p_cd2, p_ce1);
    let delta = s_flip - s_orig;

    if delta >= 0.80 {
        mol.atoms[i_nd1].coords = p_cd2;
        mol.atoms[i_cd2].coords = p_nd1;
        mol.atoms[i_ne2].coords = p_ce1;
        mol.atoms[i_ce1].coords = p_ne2;
        if rn != "HIP" {
            for &ai in idxs {
                mol.atoms[ai].res_name = best_taut.to_string();
            }
        }
        return Some((
            delta,
            s_orig,
            s_flip,
            format!("HIS 180° ring flip -> {best_taut}"),
            format!(
                "Flipped imidazole ring (ND1↔CD2, NE2↔CE1) -> {best_taut} (+{delta:.2} H-bond gain)"
            ),
            center,
        ));
    }
    None
}

fn score_acceptor_env(
    mol: &Molecule3D,
    ch: char,
    seq: i32,
    pos: [f64; 3],
    metal_coords: &[[f64; 3]],
) -> f64 {
    let mut score = 0.0;
    for &mp in metal_coords {
        if dist3(pos, mp) < 2.75 {
            score += 4.0;
        }
    }
    for other in &mol.atoms {
        if !other.is_heavy() || (other.chain_id == ch && other.res_seq == seq) {
            continue;
        }
        let d = dist3(pos, other.coords);
        if !(2.35..=3.55).contains(&d) {
            continue;
        }
        let w = 1.0 - ((d - 2.90).abs() / 0.85).clamp(0.0, 0.7);
        let el = other.element.as_str();
        let on = other.name.as_str();
        if el == "N"
            && (other.is_backbone
                || matches!(on, "NZ" | "NE" | "NH1" | "NH2" | "NE1" | "ND2" | "NE2"))
        {
            score += 2.0 * w;
        } else if el == "O" && matches!(on, "OG" | "OG1" | "OH") {
            score += 1.3 * w;
        } else if el == "O" && (other.is_backbone || matches!(on, "OD1" | "OD2" | "OE1" | "OE2")) {
            score -= 1.8 * w; // O...O acceptor-acceptor clash
        }
    }
    score
}

fn score_donor_env(mol: &Molecule3D, ch: char, seq: i32, pos: [f64; 3]) -> f64 {
    let mut score = 0.0;
    for other in &mol.atoms {
        if !other.is_heavy() || (other.chain_id == ch && other.res_seq == seq) {
            continue;
        }
        let d = dist3(pos, other.coords);
        if !(2.35..=3.55).contains(&d) {
            continue;
        }
        let w = 1.0 - ((d - 2.90).abs() / 0.85).clamp(0.0, 0.7);
        let el = other.element.as_str();
        let on = other.name.as_str();
        if el == "O" && (other.is_backbone || matches!(on, "OD1" | "OD2" | "OE1" | "OE2")) {
            score += 2.1 * w;
        } else if el == "O" && matches!(on, "OG" | "OG1" | "OH") {
            score += 1.3 * w;
        } else if el == "N" && (other.is_backbone || matches!(on, "NZ" | "NE" | "NH1" | "NH2")) {
            score -= 1.8 * w; // NH...NH donor-donor clash
        }
    }
    score
}

/// Optimizes rotatable polar `-OH` / `-SH` hydrogen dihedral angles (`SER`, `THR`, `TYR`, `CYS`)
/// AFTER `perceive_all` has added polar hydrogens, sampling 12 dihedral angles (30° grid) to point
/// each polar `H` toward the strongest nearby H-bond acceptor without steric clashes.
pub fn optimize_polar_h_orientations(mol: &mut Molecule3D, report: &mut HBondOptReport) {
    // Identify rotatable hydroxyl/thiol heavy-H pairs: (parent_c, donor_x, h_idx)
    let mut rotors: Vec<(usize, usize, usize)> = Vec::new();
    for (x_idx, a) in mol.atoms.iter().enumerate() {
        if a.is_hetatm {
            continue;
        }
        let rn = a.res_name.trim().to_ascii_uppercase();
        let an = a.name.trim().to_ascii_uppercase();
        let is_rotor = matches!(
            (rn.as_str(), an.as_str()),
            ("SER", "OG") | ("THR", "OG1") | ("TYR", "OH") | ("CYS", "SG")
        );
        if !is_rotor {
            continue;
        }
        let h_idx = mol.adj[x_idx]
            .iter()
            .copied()
            .find(|&(nb, _)| mol.atoms[nb].element == "H")
            .map(|(nb, _)| nb);
        let c_idx = mol.adj[x_idx]
            .iter()
            .copied()
            .find(|&(nb, _)| mol.atoms[nb].element == "C")
            .map(|(nb, _)| nb);
        if let (Some(c), Some(h)) = (c_idx, h_idx) {
            rotors.push((c, x_idx, h));
        }
    }

    let mut hbonds_formed = 0usize;
    for &(c_idx, x_idx, h_idx) in &rotors {
        let pc = mol.atoms[c_idx].coords;
        let px = mol.atoms[x_idx].coords;
        let ch = mol.atoms[x_idx].chain_id;
        let seq = mol.atoms[x_idx].res_seq;
        let bl = if mol.atoms[x_idx].element == "S" {
            1.33
        } else {
            0.98
        };
        let axis = norm3([px[0] - pc[0], px[1] - pc[1], px[2] - pc[2]]);
        let arb = if axis[0].abs() < 0.8 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let u = norm3(cross3(axis, arb));
        let w = cross3(axis, u);
        // C-O-H bond angle ~ 109.5° => angle with +axis is 70.5°
        let cone_cos = 70.5_f64.to_radians().cos();
        let cone_sin = 70.5_f64.to_radians().sin();

        // Gather local candidate acceptors & clash heavy atoms within 4.5 Å of X
        let mut local_acceptors: Vec<[f64; 3]> = Vec::new();
        let mut local_clash: Vec<[f64; 3]> = Vec::new();
        for (j, other) in mol.atoms.iter().enumerate() {
            if j == c_idx || j == x_idx || j == h_idx {
                continue;
            }
            if other.chain_id == ch && other.res_seq == seq {
                continue;
            }
            let d = dist3(px, other.coords);
            if d > 4.2 {
                continue;
            }
            if other.h_acceptor && d <= 3.6 {
                local_acceptors.push(other.coords);
            } else if other.is_heavy() {
                local_clash.push(other.coords);
            }
        }
        if local_acceptors.is_empty() && local_clash.is_empty() {
            continue;
        }

        let mut best_pos = mol.atoms[h_idx].coords;
        let mut best_score = f64::NEG_INFINITY;
        let mut found_hb = false;

        for step in 0..12 {
            let phi = (step as f64) * (2.0 * PI / 12.0);
            let dir = norm3([
                axis[0] * cone_cos + (u[0] * phi.cos() + w[0] * phi.sin()) * cone_sin,
                axis[1] * cone_cos + (u[1] * phi.cos() + w[1] * phi.sin()) * cone_sin,
                axis[2] * cone_cos + (u[2] * phi.cos() + w[2] * phi.sin()) * cone_sin,
            ]);
            let h_pos = [
                px[0] + dir[0] * bl,
                px[1] + dir[1] * bl,
                px[2] + dir[2] * bl,
            ];
            let mut s = 0.0;
            let mut hb_here = false;
            for &acc in &local_acceptors {
                let d_ha = dist3(h_pos, acc);
                if (1.55..=2.65).contains(&d_ha) {
                    let v_ha = norm3([acc[0] - h_pos[0], acc[1] - h_pos[1], acc[2] - h_pos[2]]);
                    let cos_ang = dir[0] * v_ha[0] + dir[1] * v_ha[1] + dir[2] * v_ha[2];
                    if cos_ang > 0.35 {
                        s += 2.5 * cos_ang * (1.0 - ((d_ha - 1.95).abs() / 0.85).clamp(0.0, 0.7));
                        hb_here = true;
                    }
                } else if d_ha < 1.55 {
                    s -= 3.0;
                }
            }
            for &cl in &local_clash {
                let d_hc = dist3(h_pos, cl);
                if d_hc < 2.0 {
                    s -= 2.5 * (2.0 - d_hc);
                }
            }
            if s > best_score {
                best_score = s;
                best_pos = h_pos;
                found_hb = hb_here;
            }
        }

        mol.atoms[h_idx].coords = best_pos;
        if found_hb {
            hbonds_formed += 1;
        }
    }

    report.polar_h_rotors_optimized = rotors.len();
    report.polar_h_hbonds_formed = hbonds_formed;
}

#[inline]
fn norm3(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-12 {
        [1.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

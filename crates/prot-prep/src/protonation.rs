#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};
use struct_core::Molecule3D;

use crate::audit::dist3;

/// Per-residue local pKa prediction & assigned protonation state (PROPKA 3-inspired physical model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtonationAssignment {
    pub chain_id: char,
    pub res_seq: i32,
    pub orig_res_name: String,
    pub assigned_state: String,
    pub formal_charge: i8,
    pub model_pka: f64,
    pub predicted_pka: f64,
    pub delta_desolv: f64,
    pub delta_hbond: f64,
    pub delta_coulomb: f64,
    pub burial_pct: f64,
    pub metal_coordinated: bool,
    pub near_pocket: bool,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtonationReport {
    pub target_ph: f64,
    pub titratable_count: usize,
    pub non_canonical_count: usize,
    pub total_protein_charge: i32,
    pub assignments: Vec<ProtonationAssignment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TitratableKind {
    Acid, // ASP, GLU, CYS, TYR (neutral when protonated, -1 when deprotonated)
    Base, // HIS, LYS, ARG (neutral when deprotonated, +1 when protonated)
}

struct TitratableSite {
    chain_id: char,
    res_seq: i32,
    res_name: String,
    kind: TitratableKind,
    model_pka: f64,
    center: [f64; 3],
    key_atoms: Vec<usize>,
    all_atoms: Vec<usize>,
}

fn site_spec(res_name: &str) -> Option<(TitratableKind, f64, &'static [&'static str])> {
    match res_name.trim().to_ascii_uppercase().as_str() {
        "ASP" | "ASH" => Some((TitratableKind::Acid, 3.80, &["OD1", "OD2"])),
        "GLU" | "GLH" => Some((TitratableKind::Acid, 4.50, &["OE1", "OE2"])),
        "HIS" | "HID" | "HIE" | "HIP" => Some((TitratableKind::Base, 6.50, &["ND1", "NE2"])),
        "CYS" | "CYM" => Some((TitratableKind::Acid, 9.00, &["SG"])),
        "TYR" => Some((TitratableKind::Acid, 10.00, &["OH"])),
        "LYS" | "LYN" => Some((TitratableKind::Base, 10.50, &["NZ"])),
        "ARG" => Some((TitratableKind::Base, 12.50, &["NH1", "NH2", "NE"])),
        _ => None,
    }
}

/// Estimates local microenvironment pKa shifts (desolvation/burial + H-bonding + Coulombic +
/// catalytic metal coordination + disulfide bridges) and rewrites receptor residue names cleanly
/// to Amber/PROPKA states (`HID`/`HIE`/`HIP`, `ASH`, `GLH`, `LYN`, `CYM`, `CYX`).
pub fn assign_local_protonation(
    mol: &mut Molecule3D,
    target_ph: f64,
    pocket_center: Option<[f64; 3]>,
    pocket_radius: f64,
) -> ProtonationReport {
    // 1. Group protein atoms by (chain_id, res_seq)
    let mut by_res: BTreeMap<(char, i32), Vec<usize>> = BTreeMap::new();
    let mut metal_coords: Vec<[f64; 3]> = Vec::new();
    let mut heavy_coords: Vec<(usize, [f64; 3], char, i32)> = Vec::new();

    for (idx, a) in mol.atoms.iter().enumerate() {
        if a.is_heavy() {
            heavy_coords.push((idx, a.coords, a.chain_id, a.res_seq));
        }
        if a.ad4_type.is_metal() {
            metal_coords.push(a.coords);
        }
        if !a.is_hetatm {
            by_res.entry((a.chain_id, a.res_seq)).or_default().push(idx);
        }
    }

    // Detect disulfide cysteines first (`SG-SG < 2.35 Å`)
    let mut cys_sg: Vec<(char, i32, [f64; 3])> = Vec::new();
    for (&(ch, seq), idxs) in &by_res {
        if let Some(&sg_i) = idxs.iter().find(|&&i| {
            matches!(mol.atoms[i].res_name.as_str(), "CYS" | "CYX") && mol.atoms[i].name == "SG"
        }) {
            cys_sg.push((ch, seq, mol.atoms[sg_i].coords));
        }
    }
    let mut disulfide_set: HashSet<(char, i32)> = HashSet::new();
    for i in 0..cys_sg.len() {
        for j in (i + 1)..cys_sg.len() {
            if dist3(cys_sg[i].2, cys_sg[j].2) < 2.35 {
                disulfide_set.insert((cys_sg[i].0, cys_sg[i].1));
                disulfide_set.insert((cys_sg[j].0, cys_sg[j].1));
            }
        }
    }

    // 2. Collect titratable sites
    let mut sites: Vec<TitratableSite> = Vec::new();
    for (&(ch, seq), idxs) in &by_res {
        let Some(&first_i) = idxs.first() else {
            continue;
        };
        let rn = mol.atoms[first_i].res_name.trim().to_ascii_uppercase();
        if disulfide_set.contains(&(ch, seq)) {
            for &i in idxs {
                mol.atoms[i].res_name = "CYX".to_string();
            }
            continue;
        }
        let Some((kind, model_pka, key_names)) = site_spec(&rn) else {
            continue;
        };
        let key_atoms: Vec<usize> = idxs
            .iter()
            .copied()
            .filter(|&i| key_names.contains(&mol.atoms[i].name.as_str()))
            .collect();
        if key_atoms.is_empty() {
            continue;
        }
        let n = key_atoms.len() as f64;
        let center = [
            key_atoms
                .iter()
                .map(|&i| mol.atoms[i].coords[0])
                .sum::<f64>()
                / n,
            key_atoms
                .iter()
                .map(|&i| mol.atoms[i].coords[1])
                .sum::<f64>()
                / n,
            key_atoms
                .iter()
                .map(|&i| mol.atoms[i].coords[2])
                .sum::<f64>()
                / n,
        ];
        sites.push(TitratableSite {
            chain_id: ch,
            res_seq: seq,
            res_name: rn,
            kind,
            model_pka,
            center,
            key_atoms,
            all_atoms: idxs.clone(),
        });
    }

    // 3. Evaluate PROPKA-inspired environmental shifts for each titratable site
    let mut assignments = Vec::with_capacity(sites.len());
    let mut non_canonical_count = 0usize;
    let mut total_protein_charge = 0i32;

    for (si, site) in sites.iter().enumerate() {
        // A. Burial / Desolvation shift (PROPKA smooth contact count within 15.5 Å)
        let mut n_buried = 0.0f64;
        let mut local_contacts = 0usize;
        for &(_, pos, ch, seq) in &heavy_coords {
            if ch == site.chain_id && seq == site.res_seq {
                continue;
            }
            let d = dist3(site.center, pos);
            if d < 15.5 {
                let w = if d <= 4.0 {
                    1.0
                } else {
                    ((15.5 - d) / 11.5).clamp(0.0, 1.0)
                };
                n_buried += w;
                if d < 6.5 {
                    local_contacts += 1;
                }
            }
        }
        // PROPKA empirical burial scale: N_max ~ 260 weighted heavy neighbors inside 15.5 Å
        let burial_frac = ((n_buried - 75.0) / 175.0).clamp(0.0, 1.0);
        let local_scale = ((local_contacts as f64) / 28.0).clamp(0.0, 1.0);
        let effective_burial = (0.65 * burial_frac + 0.35 * local_scale).clamp(0.0, 1.0);
        // Desolvation destabilizes the charged form: +ΔpKa for acids, -ΔpKa for bases (up to ~3.2 units)
        let max_desolv = match site.res_name.as_str() {
            "ASP" | "ASH" | "GLU" | "GLH" => 3.4,
            "HIS" | "HID" | "HIE" | "HIP" => 2.2,
            "LYS" | "LYN" => 2.8,
            "CYS" | "CYM" => 2.4,
            _ => 1.8,
        };
        let delta_desolv = match site.kind {
            TitratableKind::Acid => max_desolv * effective_burial,
            TitratableKind::Base => -max_desolv * effective_burial,
        };

        // B. Local H-bond shift (polar backbone/sidechain neighbors 2.4–3.5 Å)
        let mut hb_donors = 0usize;
        let mut hb_acceptors = 0usize;
        for &ka in &site.key_atoms {
            let p = mol.atoms[ka].coords;
            for other in &mol.atoms {
                if !other.is_heavy()
                    || (other.chain_id == site.chain_id && other.res_seq == site.res_seq)
                {
                    continue;
                }
                let d = dist3(p, other.coords);
                if !(2.35..=3.55).contains(&d) {
                    continue;
                }
                let el = other.element.as_str();
                let on = other.name.as_str();
                if el == "N"
                    && (other.is_backbone
                        || matches!(on, "NE" | "NH1" | "NH2" | "NZ" | "ND2" | "NE2" | "NE1"))
                {
                    hb_donors += 1;
                } else if el == "O"
                    && (other.is_backbone || matches!(on, "OD1" | "OD2" | "OE1" | "OE2"))
                {
                    hb_acceptors += 1;
                } else if el == "O" && matches!(on, "OG" | "OG1" | "OH") {
                    hb_donors += 1;
                    hb_acceptors += 1;
                }
            }
        }
        // Donors stabilize anionic deprotonated acid (-ΔpKa) and deprotonated base (-ΔpKa);
        // Acceptors stabilize protonated donor (+ΔpKa for both acids & bases!).
        let delta_hbond = match site.kind {
            TitratableKind::Acid => (-0.55 * (hb_donors.min(3) as f64)
                + 0.45 * (hb_acceptors.min(2) as f64))
                .clamp(-1.8, 1.2),
            TitratableKind::Base => (0.55 * (hb_acceptors.min(3) as f64)
                - 0.40 * (hb_donors.min(2) as f64))
                .clamp(-1.0, 1.8),
        };

        // C. Coulombic interaction with neighboring titratable residues (< 7.5 Å)
        let mut delta_coulomb = 0.0f64;
        for (sj, other) in sites.iter().enumerate() {
            if si == sj {
                continue;
            }
            let d = dist3(site.center, other.center);
            if d >= 7.5 {
                continue;
            }
            // Linear distance decay from 4.0 Å (1.0) to 7.5 Å (0.0), amplified when buried
            let w = ((7.5 - d) / 3.5).clamp(0.0, 1.0) * (0.45 + 0.55 * effective_burial);
            let other_likely_charged = match other.kind {
                TitratableKind::Acid => other.model_pka < target_ph,
                TitratableKind::Base => other.model_pka > target_ph,
            };
            if !other_likely_charged {
                continue;
            }
            match (site.kind, other.kind) {
                // Acid near Base (salt bridge, e.g. GLU51-LYS33 in CDK2) -> lowers acid pKa
                (TitratableKind::Acid, TitratableKind::Base) => delta_coulomb -= 1.8 * w,
                // Acid near Acid (catalytic dyad, e.g. ASP-ASP) -> raises acid pKa (one protonates!)
                (TitratableKind::Acid, TitratableKind::Acid) => {
                    // Deterministic symmetry breaking: higher burial or higher res_seq absorbs larger shift
                    let asym = if site.res_seq > other.res_seq {
                        1.35
                    } else {
                        0.65
                    };
                    delta_coulomb += 2.2 * w * asym;
                }
                // Base near Acid (salt bridge) -> raises base pKa (stabilizes cation!)
                (TitratableKind::Base, TitratableKind::Acid) => delta_coulomb += 1.8 * w,
                // Base near Base -> lowers base pKa
                (TitratableKind::Base, TitratableKind::Base) => delta_coulomb -= 1.5 * w,
            }
        }
        delta_coulomb = delta_coulomb.clamp(-3.0, 3.5);

        // D. Direct catalytic metal coordination (< 2.75 Å to Zn/Mg/Fe/Mn/Ca/Cu)
        let metal_coordinated = site.key_atoms.iter().any(|&ka| {
            let p = mol.atoms[ka].coords;
            metal_coords.iter().any(|&mp| dist3(p, mp) < 2.75)
        });
        let delta_metal = if metal_coordinated {
            match site.res_name.as_str() {
                "CYS" | "CYM" => -4.0, // Metal-coordinated cysteine becomes thiolate CYM (-1)
                "HIS" | "HID" | "HIE" | "HIP" => -2.5, // Neutral metal-coordinating His
                "ASP" | "ASH" | "GLU" | "GLH" => -2.0, // Strongly anionic carboxylate
                _ => 0.0,
            }
        } else {
            0.0
        };

        let predicted_pka =
            (site.model_pka + delta_desolv + delta_hbond + delta_coulomb + delta_metal)
                .clamp(1.0, 14.5);

        // E. Determine final Amber/PROPKA residue state at `target_ph`
        let (assigned_state, formal_charge, rationale) = match site.res_name.as_str() {
            "ASP" | "ASH" => {
                if predicted_pka > target_ph {
                    non_canonical_count += 1;
                    (
                        "ASH".to_string(),
                        0i8,
                        format!(
                            "Protonated neutral Asp (pKa {predicted_pka:.2} > pH {target_ph:.1})"
                        ),
                    )
                } else {
                    (
                        "ASP".to_string(),
                        -1i8,
                        "Deprotonated aspartate (-1)".to_string(),
                    )
                }
            }
            "GLU" | "GLH" => {
                if predicted_pka > target_ph {
                    non_canonical_count += 1;
                    (
                        "GLH".to_string(),
                        0i8,
                        format!(
                            "Protonated neutral Glu (pKa {predicted_pka:.2} > pH {target_ph:.1})"
                        ),
                    )
                } else {
                    (
                        "GLU".to_string(),
                        -1i8,
                        "Deprotonated glutamate (-1)".to_string(),
                    )
                }
            }
            "LYS" | "LYN" => {
                if predicted_pka < target_ph {
                    non_canonical_count += 1;
                    (
                        "LYN".to_string(),
                        0i8,
                        format!("Neutral buried Lys (pKa {predicted_pka:.2} < pH {target_ph:.1})"),
                    )
                } else {
                    ("LYS".to_string(), 1i8, "Protonated lysine (+1)".to_string())
                }
            }
            "CYS" | "CYM" => {
                if predicted_pka < target_ph || metal_coordinated {
                    non_canonical_count += 1;
                    (
                        "CYM".to_string(),
                        -1i8,
                        if metal_coordinated {
                            "Metal-coordinated anionic thiolate (CYM -1)".to_string()
                        } else {
                            format!(
                                "Deprotonated thiolate (pKa {predicted_pka:.2} < pH {target_ph:.1})"
                            )
                        },
                    )
                } else {
                    (
                        "CYS".to_string(),
                        0i8,
                        "Neutral thiol cysteine (0)".to_string(),
                    )
                }
            }
            "HIS" | "HID" | "HIE" | "HIP" => {
                if !metal_coordinated && predicted_pka > target_ph {
                    non_canonical_count += 1;
                    (
                        "HIP".to_string(),
                        1i8,
                        format!(
                            "Doubly protonated His+ (pKa {predicted_pka:.2} > pH {target_ph:.1})"
                        ),
                    )
                } else {
                    let (taut, reason) = choose_his_tautomer(mol, site, &metal_coords);
                    (taut, 0i8, reason)
                }
            }
            "ARG" => (
                "ARG".to_string(),
                1i8,
                "Protonated guanidinium (+1)".to_string(),
            ),
            "TYR" => ("TYR".to_string(), 0i8, "Neutral phenol (0)".to_string()),
            other => (other.to_string(), 0i8, "Standard state".to_string()),
        };

        // Write assigned state onto every atom in this residue
        for &ai in &site.all_atoms {
            mol.atoms[ai].res_name = assigned_state.clone();
        }
        total_protein_charge += i32::from(formal_charge);

        let near_pocket = pocket_center.is_some_and(|c| dist3(site.center, c) <= pocket_radius);
        assignments.push(ProtonationAssignment {
            chain_id: site.chain_id,
            res_seq: site.res_seq,
            orig_res_name: site.res_name.clone(),
            assigned_state,
            formal_charge,
            model_pka: round2(site.model_pka),
            predicted_pka: round2(predicted_pka),
            delta_desolv: round2(delta_desolv),
            delta_hbond: round2(delta_hbond),
            delta_coulomb: round2(delta_coulomb),
            burial_pct: round1(effective_burial * 100.0),
            metal_coordinated,
            near_pocket,
            rationale,
        });
    }

    ProtonationReport {
        target_ph,
        titratable_count: assignments.len(),
        non_canonical_count,
        total_protein_charge,
        assignments,
    }
}

/// Scores `HID` (`ND1-H`, `NE2` acceptor) vs `HIE` (`NE2-H`, `ND1` acceptor) against local
/// H-bond partners and metal ions.
fn choose_his_tautomer(
    mol: &Molecule3D,
    site: &TitratableSite,
    metal_coords: &[[f64; 3]],
) -> (String, String) {
    let nd1 = site
        .key_atoms
        .iter()
        .copied()
        .find(|&i| mol.atoms[i].name == "ND1");
    let ne2 = site
        .key_atoms
        .iter()
        .copied()
        .find(|&i| mol.atoms[i].name == "NE2");
    let (Some(i_nd1), Some(i_ne2)) = (nd1, ne2) else {
        return (
            "HIE".to_string(),
            "Default epsilon tautomer (HIE)".to_string(),
        );
    };
    let p_nd1 = mol.atoms[i_nd1].coords;
    let p_ne2 = mol.atoms[i_ne2].coords;

    // Metal coordination overrides: the coordinating N MUST be unprotonated acceptor!
    if metal_coords.iter().any(|&m| dist3(p_nd1, m) < 2.75) {
        return (
            "HIE".to_string(),
            "ND1 coordinates catalytic metal -> protonated at NE2 (HIE)".to_string(),
        );
    }
    if metal_coords.iter().any(|&m| dist3(p_ne2, m) < 2.75) {
        return (
            "HID".to_string(),
            "NE2 coordinates catalytic metal -> protonated at ND1 (HID)".to_string(),
        );
    }

    let score_donor = |pos: [f64; 3]| -> f64 {
        let mut s = 0.0;
        for other in &mol.atoms {
            if !other.is_heavy()
                || (other.chain_id == site.chain_id && other.res_seq == site.res_seq)
            {
                continue;
            }
            let d = dist3(pos, other.coords);
            if !(2.40..=3.55).contains(&d) {
                continue;
            }
            let w = 1.0 - ((d - 2.85).abs() / 0.85).clamp(0.0, 0.7);
            if other.element == "O"
                && (other.is_backbone
                    || matches!(other.name.as_str(), "OD1" | "OD2" | "OE1" | "OE2"))
            {
                s += 2.0 * w; // Carbonyl/carboxylate acceptor loves H-donor!
            } else if other.element == "N"
                && (other.is_backbone || matches!(other.name.as_str(), "NZ" | "NH1" | "NH2" | "NE"))
            {
                s -= 1.5 * w; // Donor-donor clash
            }
        }
        s
    };

    let score_acceptor = |pos: [f64; 3]| -> f64 {
        let mut s = 0.0;
        for other in &mol.atoms {
            if !other.is_heavy()
                || (other.chain_id == site.chain_id && other.res_seq == site.res_seq)
            {
                continue;
            }
            let d = dist3(pos, other.coords);
            if !(2.40..=3.55).contains(&d) {
                continue;
            }
            let w = 1.0 - ((d - 2.85).abs() / 0.85).clamp(0.0, 0.7);
            if other.element == "N"
                && (other.is_backbone
                    || matches!(
                        other.name.as_str(),
                        "NZ" | "NH1" | "NH2" | "NE" | "ND2" | "NE2"
                    ))
            {
                s += 1.8 * w; // Backbone/sidechain NH donor loves N: acceptor!
            } else if other.element == "O" && matches!(other.name.as_str(), "OG" | "OG1" | "OH") {
                s += 1.2 * w;
            } else if other.element == "O" && other.is_backbone {
                s -= 1.0 * w; // Acceptor-acceptor repulsion
            }
        }
        s
    };

    let hid_score = score_donor(p_nd1) + score_acceptor(p_ne2);
    let hie_score = score_donor(p_ne2) + score_acceptor(p_nd1) + 0.15; // slight intrinsic HIE preference in water

    if hid_score > hie_score {
        (
            "HID".to_string(),
            format!(
                "Delta tautomer (HID) favored by H-bond network ({hid_score:.2} vs {hie_score:.2})"
            ),
        )
    } else {
        (
            "HIE".to_string(),
            format!(
                "Epsilon tautomer (HIE) favored by H-bond network ({hie_score:.2} vs {hid_score:.2})"
            ),
        )
    }
}

#[inline]
fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

#[inline]
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

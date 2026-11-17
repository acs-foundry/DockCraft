#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use struct_core::{Ad4Type, Atom3D, Molecule3D};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ScoringFunction {
    #[default]
    Vina,
    Vinardo,
    /// Vina terms combined with an explicit screened electrostatic term (`3.2 * q1*q2 / r^2`).
    DockCraftHybrid,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub raw_inter_kcal: f64,
    pub gauss1: f64,
    pub gauss2: f64,
    pub repulsion: f64,
    pub hydrophobic: f64,
    pub hbond: f64,
    pub electrostatic: f64,
    pub intra_kcal: f64,
    pub total_affinity_kcal: f64,
}

#[must_use]
pub fn vina_vdw_radius(ad4: Ad4Type, sf: ScoringFunction) -> f64 {
    match sf {
        ScoringFunction::Vina | ScoringFunction::DockCraftHybrid => ad4.vina_radius(),
        ScoringFunction::Vinardo => match ad4 {
            Ad4Type::C | Ad4Type::A => 1.90,
            Ad4Type::N | Ad4Type::NA | Ad4Type::NS => 1.75,
            Ad4Type::OA | Ad4Type::OS => 1.65,
            Ad4Type::SA | Ad4Type::S => 2.00,
            Ad4Type::P => 2.10,
            Ad4Type::F => 1.54,
            Ad4Type::Cl => 1.95,
            Ad4Type::Br => 2.15,
            Ad4Type::I => 2.35,
            _ => ad4.vina_radius(),
        },
    }
}

/// Smooth C1 ramp returning `(value, d_value / d_t)` that equals `1.0` for `x <= a`, `0.0` for `x >= b`.
#[inline]
pub(crate) fn smooth_down(x: f64, a: f64, b: f64) -> (f64, f64) {
    if x <= a {
        (1.0, 0.0)
    } else if x >= b {
        (0.0, 0.0)
    } else {
        let span = b - a;
        let t = (x - a) / span;
        let val = 1.0 - (3.0 * t * t - 2.0 * t * t * t);
        let deriv = -(6.0 * t * (1.0 - t)) / span;
        (val, deriv)
    }
}

#[inline]
pub fn is_hbond_pair(a: &Atom3D, b: &Atom3D) -> bool {
    (a.h_donor && b.h_acceptor) || (a.h_acceptor && b.h_donor)
}

#[inline]
pub fn is_hydrophobic_pair(a: &Atom3D, b: &Atom3D) -> bool {
    a.hydrophobic && b.hydrophobic
}

/// Evaluates pairwise interaction `(energy_kcal, dE / dr)` between a receptor atom `rec` and ligand atom `lig` separated by distance `r`.
#[inline]
pub fn pairwise_energy_and_deriv(
    rec: &Atom3D,
    lig: &Atom3D,
    r: f64,
    sf: ScoringFunction,
) -> (f64, f64, [f64; 6]) {
    if r >= 8.0 || !rec.is_heavy() || !lig.is_heavy() {
        return (0.0, 0.0, [0.0; 6]);
    }
    let r_opt = vina_vdw_radius(rec.ad4_type, sf) + vina_vdw_radius(lig.ad4_type, sf);
    let d = r - r_opt;

    // Smooth cutoff envelope in [7.2, 8.0] so forces vanish continuously.
    let (env, d_env) = if r > 7.2 {
        smooth_down(r, 7.2, 8.0)
    } else {
        (1.0, 0.0)
    };

    let (e_g1, de_g1, e_g2, de_g2, e_rep, de_rep, e_hyd, de_hyd, e_hb, de_hb) = match sf {
        ScoringFunction::Vina | ScoringFunction::DockCraftHybrid => {
            let w_g1 = -0.035_579;
            let w_g2 = -0.005_156;
            let w_rep = 0.840_245;
            let w_hyd = -0.035_069;
            let w_hb = -0.587_439;

            let t1 = d / 0.5;
            let g1 = (-t1 * t1).exp();
            let dg1 = g1 * (-2.0 * d / 0.25);

            let t2 = (d - 3.0) / 2.0;
            let g2 = (-t2 * t2).exp();
            let dg2 = g2 * (-2.0 * (d - 3.0) / 4.0);

            let (rep, drep) = if d < 0.0 {
                (d * d, 2.0 * d)
            } else {
                (0.0, 0.0)
            };

            let (hyd, dhyd) = if is_hydrophobic_pair(rec, lig) {
                smooth_down(d, 0.5, 1.5)
            } else {
                (0.0, 0.0)
            };

            let (hb, dhb) = if is_hbond_pair(rec, lig) {
                smooth_down(d, -0.7, 0.0)
            } else {
                (0.0, 0.0)
            };

            (
                w_g1 * g1,
                w_g1 * dg1,
                w_g2 * g2,
                w_g2 * dg2,
                w_rep * rep,
                w_rep * drep,
                w_hyd * hyd,
                w_hyd * dhyd,
                w_hb * hb,
                w_hb * dhb,
            )
        }
        ScoringFunction::Vinardo => {
            let w_g1 = -0.045;
            let w_rep = 0.80;
            let w_hyd = -0.035;
            let w_hb = -0.60;

            let t1 = d / 0.8;
            let g1 = (-t1 * t1).exp();
            let dg1 = g1 * (-2.0 * d / 0.64);

            let (rep, drep) = if d < 0.0 {
                (d * d, 2.0 * d)
            } else {
                (0.0, 0.0)
            };

            let (hyd, dhyd) = if is_hydrophobic_pair(rec, lig) {
                smooth_down(d, 0.0, 2.5)
            } else {
                (0.0, 0.0)
            };

            let (hb, dhb) = if is_hbond_pair(rec, lig) {
                smooth_down(d, -0.6, 0.0)
            } else {
                (0.0, 0.0)
            };

            (
                w_g1 * g1,
                w_g1 * dg1,
                0.0,
                0.0,
                w_rep * rep,
                w_rep * drep,
                w_hyd * hyd,
                w_hyd * dhyd,
                w_hb * hb,
                w_hb * dhb,
            )
        }
    };

    // Electrostatics apply ONLY in explicit `DockCraftHybrid`; `Vina` and `Vinardo` are pure literature forms.
    let q_prod = rec.charge * lig.charge;
    let r_eff = r.max(2.2);
    let (elec, delec) = if sf == ScoringFunction::DockCraftHybrid && q_prod.abs() > 1e-4 {
        let scale = 3.2;
        let e = scale * q_prod / (r_eff * r_eff);
        let de = if r >= 2.2 {
            -2.0 * scale * q_prod / (r_eff * r_eff * r_eff)
        } else {
            0.0
        };
        (e, de)
    } else {
        (0.0, 0.0)
    };

    let raw_e = e_g1 + e_g2 + e_rep + e_hyd + e_hb + elec;
    let raw_de = de_g1 + de_g2 + de_rep + de_hyd + de_hb + delec;

    let total_e = raw_e * env;
    let total_de = raw_de * env + raw_e * d_env;
    (
        total_e,
        total_de,
        [
            e_g1 * env,
            e_g2 * env,
            e_rep * env,
            e_hyd * env,
            e_hb * env,
            elec * env,
        ],
    )
}

#[must_use]
pub fn rotatable_penalty_divisor(n_rotatable: usize) -> f64 {
    1.0 + 0.05846 * (n_rotatable as f64)
}

/// Evaluates full exact complex affinity (no grid approximation), plus intramolecular clash check.
#[must_use]
pub fn score_complex(
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    sf: ScoringFunction,
) -> ScoreBreakdown {
    let mut bd = ScoreBreakdown::default();
    let rec_heavy: Vec<&Atom3D> = receptor.atoms.iter().filter(|a| a.is_heavy()).collect();
    let lig_heavy: Vec<(usize, &Atom3D)> = ligand
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_heavy())
        .collect();

    for &(_, la) in &lig_heavy {
        for &ra in &rec_heavy {
            let dx = la.coords[0] - ra.coords[0];
            if dx.abs() >= 8.0 {
                continue;
            }
            let dy = la.coords[1] - ra.coords[1];
            if dy.abs() >= 8.0 {
                continue;
            }
            let dz = la.coords[2] - ra.coords[2];
            if dz.abs() >= 8.0 {
                continue;
            }
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 >= 64.0 || r2 < 1e-8 {
                continue;
            }
            let r = r2.sqrt();
            let (e, _de, terms) = pairwise_energy_and_deriv(ra, la, r, sf);
            bd.raw_inter_kcal += e;
            bd.gauss1 += terms[0];
            bd.gauss2 += terms[1];
            bd.repulsion += terms[2];
            bd.hydrophobic += terms[3];
            bd.hbond += terms[4];
            bd.electrostatic += terms[5];
        }
    }

    // Intramolecular nonbonded steric check for 1-4+ ligand heavy pairs
    for i in 0..lig_heavy.len() {
        let (idx_i, ai) = lig_heavy[i];
        for &(idx_j, aj) in lig_heavy.iter().skip(i + 1) {
            if are_topologically_close(ligand, idx_i, idx_j) {
                continue;
            }
            let dx = ai.coords[0] - aj.coords[0];
            let dy = ai.coords[1] - aj.coords[1];
            let dz = ai.coords[2] - aj.coords[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 8.0 {
                let r_opt = vina_vdw_radius(ai.ad4_type, sf) + vina_vdw_radius(aj.ad4_type, sf);
                let d = r - r_opt;
                if d < -0.2 {
                    bd.intra_kcal += 0.840_245 * (d + 0.2) * (d + 0.2);
                }
            }
        }
    }

    let div = rotatable_penalty_divisor(ligand.rotatable_bonds.len());
    bd.total_affinity_kcal = (bd.raw_inter_kcal + 0.35 * bd.intra_kcal) / div;
    bd
}

pub(crate) fn are_topologically_close(mol: &Molecule3D, u: usize, v: usize) -> bool {
    if u == v {
        return true;
    }
    for &(n1, _) in &mol.adj[u] {
        if n1 == v {
            return true;
        }
        for &(n2, _) in &mol.adj[n1] {
            if n2 == v {
                return true;
            }
            for &(n3, _) in &mol.adj[n2] {
                if n3 == v {
                    return true;
                }
            }
        }
    }
    false
}

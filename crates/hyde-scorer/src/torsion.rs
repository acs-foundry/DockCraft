#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use struct_core::{BondOrder, Molecule3D};

/// SeeSAR-inspired torsion strain classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TorsionStatus {
    /// Dihedral sits within a well-populated CSD conformational minimum.
    Green,
    /// Dihedral is moderately strained (shoulder of torsion energy profile).
    Yellow,
    /// Dihedral is eclipsed or severely strained (high internal energy penalty).
    Red,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorsionAlert {
    pub bond_id: usize,
    pub atom_u: usize,
    pub atom_v: usize,
    pub dihedral_deg: f64,
    pub pattern: String,
    pub strain_kcal: f64,
    pub status: TorsionStatus,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TorsionReport {
    pub torsions: Vec<TorsionAlert>,
    pub total_strain_kcal: f64,
    pub green_count: usize,
    pub yellow_count: usize,
    pub red_count: usize,
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// Signed dihedral angle in degrees `(-180, 180]` for 4 coordinates `(p0, p1, p2, p3)`.
#[must_use]
pub fn dihedral_degrees(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let b0 = sub3(p0, p1);
    let b1 = sub3(p2, p1);
    let b2 = sub3(p3, p2);
    let b1_len = norm3(b1);
    if b1_len < 1e-8 {
        return 0.0;
    }
    let b1_u = [b1[0] / b1_len, b1[1] / b1_len, b1[2] / b1_len];
    let v_dot = dot3(b0, b1_u);
    let v = [
        b0[0] - v_dot * b1_u[0],
        b0[1] - v_dot * b1_u[1],
        b0[2] - v_dot * b1_u[2],
    ];
    let w_dot = dot3(b2, b1_u);
    let w = [
        b2[0] - w_dot * b1_u[0],
        b2[1] - w_dot * b1_u[1],
        b2[2] - w_dot * b1_u[2],
    ];
    let x = dot3(v, w);
    let y = dot3(cross3(b1_u, v), w);
    y.atan2(x).to_degrees()
}

fn is_sp2(mol: &Molecule3D, idx: usize) -> bool {
    if mol.atoms[idx].aromatic {
        return true;
    }
    mol.adj[idx].iter().any(|&(_, bid)| {
        matches!(
            mol.bonds[bid].order,
            BondOrder::Double | BondOrder::Aromatic
        )
    })
}

fn is_carbonyl_carbon(mol: &Molecule3D, idx: usize) -> bool {
    mol.atoms[idx].element == "C"
        && mol.adj[idx].iter().any(|&(nb, bid)| {
            matches!(mol.atoms[nb].element.as_str(), "O" | "S")
                && mol.bonds[bid].order == BondOrder::Double
        })
}

fn is_sulfonyl_sulfur(mol: &Molecule3D, idx: usize) -> bool {
    mol.atoms[idx].element == "S"
        && mol.adj[idx]
            .iter()
            .filter(|&&(nb, bid)| {
                mol.atoms[nb].element == "O" && mol.bonds[bid].order == BondOrder::Double
            })
            .count()
            >= 2
}

/// Minimum circular angular distance in degrees from `angle` to any preferred well in `wells`.
fn min_dist_to_wells(angle_deg: f64, wells: &[f64]) -> f64 {
    let mut best = 180.0_f64;
    for &w in wells {
        let mut d = (angle_deg - w) % 360.0;
        if d > 180.0 {
            d -= 360.0;
        } else if d < -180.0 {
            d += 360.0;
        }
        best = best.min(d.abs());
    }
    best
}

/// Evaluate every rotatable bond in `ligand` against empirical CSD-inspired torsion profiles,
/// classifying each torsion as Green / Yellow / Red with an estimated strain penalty (kcal/mol).
#[must_use]
pub fn analyze_torsions(ligand: &Molecule3D) -> TorsionReport {
    let mut report = TorsionReport::default();
    for rb in &ligand.rotatable_bonds {
        let u = rb.u;
        let v = rb.v;
        // Pick deterministic heavy reference neighbors on each end (prefer double-bonded O/N or aromatic)
        let pick_ref = |center: usize, other: usize| -> Option<usize> {
            ligand
                .adj
                .get(center)?
                .iter()
                .filter(|&&(nb, _)| nb != other && ligand.atoms[nb].is_heavy())
                .max_by_key(|&&(nb, bid)| {
                    let bo_rank = match ligand.bonds[bid].order {
                        BondOrder::Double => 3,
                        BondOrder::Aromatic => 2,
                        _ => 1,
                    };
                    let el_rank = match ligand.atoms[nb].element.as_str() {
                        "O" | "N" => 2,
                        _ => 1,
                    };
                    (bo_rank, el_rank, usize::MAX - nb)
                })
                .map(|&(nb, _)| nb)
        };
        let (Some(a), Some(d)) = (pick_ref(u, v), pick_ref(v, u)) else {
            continue;
        };
        let deg = dihedral_degrees(
            ligand.atoms[a].coords,
            ligand.atoms[u].coords,
            ligand.atoms[v].coords,
            ligand.atoms[d].coords,
        );

        let eu = ligand.atoms[u].element.as_str();
        let ev = ligand.atoms[v].element.as_str();
        let sp2_u = is_sp2(ligand, u);
        let sp2_v = is_sp2(ligand, v);
        let arom_u = ligand.atoms[u].aromatic;
        let arom_v = ligand.atoms[v].aromatic;

        // Classify torsion family and evaluate CSD-like conformational well deviation
        let (pattern, strain, status) = if (is_carbonyl_carbon(ligand, u) && ev == "N")
            || (is_carbonyl_carbon(ligand, v) && eu == "N")
        {
            // Planar amide C(=O)-N: strongly prefers 0° / 180° (trans/cis planar)
            let dev = min_dist_to_wells(deg, &[0.0, 180.0]);
            let st = if dev <= 22.0 {
                TorsionStatus::Green
            } else if dev <= 42.0 {
                TorsionStatus::Yellow
            } else {
                TorsionStatus::Red
            };
            let pen = if dev <= 18.0 {
                0.0
            } else {
                2.8 * (dev.to_radians().sin()).powi(2)
            };
            ("Amide C(=O)-N".to_string(), pen, st)
        } else if (is_carbonyl_carbon(ligand, u) && ev == "O")
            || (is_carbonyl_carbon(ligand, v) && eu == "O")
        {
            // Ester / carboxylic C(=O)-O: coplanar 0° / 180°
            let dev = min_dist_to_wells(deg, &[0.0, 180.0]);
            let st = if dev <= 25.0 {
                TorsionStatus::Green
            } else if dev <= 45.0 {
                TorsionStatus::Yellow
            } else {
                TorsionStatus::Red
            };
            let pen = if dev <= 20.0 {
                0.0
            } else {
                2.0 * (dev.to_radians().sin()).powi(2)
            };
            ("Ester C(=O)-O".to_string(), pen, st)
        } else if (is_sulfonyl_sulfur(ligand, u) && ev == "N")
            || (is_sulfonyl_sulfur(ligand, v) && eu == "N")
        {
            // Sulfonamide S(=O)2-N: prefers staggered ±60°..±120° around S=O oxygens
            let dev = min_dist_to_wells(deg, &[60.0, -60.0, 120.0, -120.0]);
            let st = if dev <= 28.0 {
                TorsionStatus::Green
            } else if dev <= 45.0 {
                TorsionStatus::Yellow
            } else {
                TorsionStatus::Red
            };
            let pen = if dev <= 25.0 {
                0.0
            } else {
                (dev - 25.0) * 0.035
            };
            ("Sulfonamide SO2-N".to_string(), pen, st)
        } else if arom_u && arom_v {
            // Biaryl / biphenyl: twisted 30–65° (or coplanar heteroaromatic down to 0° if N-adjacent)
            let dev = min_dist_to_wells(deg, &[40.0, -40.0, 140.0, -140.0]);
            let st = if dev <= 30.0 {
                TorsionStatus::Green
            } else if dev <= 42.0 {
                TorsionStatus::Yellow
            } else {
                TorsionStatus::Red
            };
            let pen = if dev <= 30.0 {
                0.0
            } else {
                (dev - 30.0) * 0.04
            };
            ("Biaryl Ar-Ar".to_string(), pen, st)
        } else if (arom_u && matches!(ev, "O" | "N")) || (arom_v && matches!(eu, "O" | "N")) {
            // Aryl ether / aniline Ar-O-C or Ar-N-C: prefers coplanar 0°/180° (or ±30° if ortho-substituted)
            let dev = min_dist_to_wells(deg, &[0.0, 180.0]);
            let st = if dev <= 35.0 {
                TorsionStatus::Green
            } else if dev <= 60.0 {
                TorsionStatus::Yellow
            } else {
                TorsionStatus::Red
            };
            let pen = if dev <= 35.0 {
                0.0
            } else {
                1.4 * ((dev - 35.0) / 55.0).powi(2)
            };
            ("Aryl-O/N Conjugated".to_string(), pen, st)
        } else if sp2_u && sp2_v {
            // Conjugated sp2-sp2: prefers coplanar 0° / 180°
            let dev = min_dist_to_wells(deg, &[0.0, 180.0]);
            let st = if dev <= 32.0 {
                TorsionStatus::Green
            } else if dev <= 55.0 {
                TorsionStatus::Yellow
            } else {
                TorsionStatus::Red
            };
            let pen = if dev <= 30.0 {
                0.0
            } else {
                1.6 * (dev.to_radians().sin()).powi(2)
            };
            ("Conjugated sp2-sp2".to_string(), pen, st)
        } else if sp2_u ^ sp2_v {
            // sp2-sp3 (e.g. Ar-CH2-R): flat 6-fold barrier; broadly favorable except extreme steric eclipse
            let dev =
                min_dist_to_wells(deg, &[0.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 180.0]);
            let st = if dev <= 22.0 {
                TorsionStatus::Green
            } else {
                TorsionStatus::Yellow
            };
            let pen = if dev <= 22.0 { 0.0 } else { 0.25 };
            ("sp2-sp3 Linker".to_string(), pen, st)
        } else {
            // Aliphatic sp3-sp3: 3-fold staggered minima at -60°, +60°, 180°; eclipsed at 0°, ±120°
            let dev = min_dist_to_wells(deg, &[60.0, -60.0, 180.0]);
            let st = if dev <= 26.0 {
                TorsionStatus::Green
            } else if dev <= 42.0 {
                TorsionStatus::Yellow
            } else {
                TorsionStatus::Red
            };
            // Classic ethane/butane barrier ~ 1.5 kcal/mol at eclipsed (dev = 60°)
            let pen = if dev <= 22.0 {
                0.0
            } else {
                1.6 * ((dev - 22.0) / 38.0).clamp(0.0, 1.0).powi(2)
            };
            ("Aliphatic sp3-sp3".to_string(), pen, st)
        };

        match status {
            TorsionStatus::Green => report.green_count += 1,
            TorsionStatus::Yellow => report.yellow_count += 1,
            TorsionStatus::Red => report.red_count += 1,
        }
        report.total_strain_kcal += strain;
        report.torsions.push(TorsionAlert {
            bond_id: rb.bond_id,
            atom_u: u,
            atom_v: v,
            dihedral_deg: deg,
            pattern,
            strain_kcal: strain,
            status,
        });
    }
    report
}

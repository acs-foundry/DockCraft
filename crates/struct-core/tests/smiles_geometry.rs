#![forbid(unsafe_code)]
//! Ground truth for the 3D structures the SMILES builder produces.
//!
//! Everything a screening run does starts here: a library arrives as SMILES, and whatever
//! geometry comes out of this builder is what gets docked. Four defects are pinned, each of which
//! made the docked pose wrong in a way nothing downstream could detect:
//!
//! * one bond length per bond order regardless of element -- every single bond 1.500 A, so C-O
//!   was 0.08 A too long, and every double bond 1.340 A, so every carbonyl was 0.11 A too long;
//! * sp3 centres placed inside a fixed 41 degree cone, leaving them nearly planar (the signed
//!   volume of alanine's alpha carbon came out 0.025 A^3 against an ideal ~7.9);
//! * saturated rings built perfectly flat, cyclohexane included;
//! * `@` and `@@` parsed and discarded, so the two enantiomers of alanine came out with
//!   *identical* coordinates.

use struct_core::{Molecule3D, StructError, parse_smiles_to_3d, perceive_all};

fn build(smiles: &str) -> Result<Molecule3D, StructError> {
    let mut m = parse_smiles_to_3d(smiles, None)?;
    perceive_all(&mut m);
    Ok(m)
}

fn distance(m: &Molecule3D, i: usize, j: usize) -> f64 {
    let (a, b) = (m.atoms[i].coords, m.atoms[j].coords);
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn angle_deg(m: &Molecule3D, i: usize, centre: usize, j: usize) -> f64 {
    let (p, q, r) = (m.atoms[i].coords, m.atoms[centre].coords, m.atoms[j].coords);
    let a = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let b = [r[0] - q[0], r[1] - q[1], r[2] - q[2]];
    let da = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let db = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    ((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (da * db))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// Signed volume of the tetrahedron formed by the four neighbours of `centre`.
fn neighbour_volume(m: &Molecule3D, centre: usize) -> f64 {
    let n: Vec<usize> = m.adj[centre].iter().map(|&(x, _)| x).collect();
    assert_eq!(n.len(), 4, "expected a four-coordinate centre");
    let p: Vec<[f64; 3]> = n.iter().map(|&i| m.atoms[i].coords).collect();
    let u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
    let v = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
    let w = [p[3][0] - p[0][0], p[3][1] - p[0][1], p[3][2] - p[0][2]];
    let cross = [
        v[1] * w[2] - v[2] * w[1],
        v[2] * w[0] - v[0] * w[2],
        v[0] * w[1] - v[1] * w[0],
    ];
    u[0] * cross[0] + u[1] * cross[1] + u[2] * cross[2]
}

#[test]
fn bond_lengths_depend_on_the_elements_not_only_on_the_bond_order() -> Result<(), StructError> {
    // (SMILES, atom i, atom j, expected length, tolerance)
    let cases = [
        ("CC", 0, 1, 1.52, 0.03),       // C-C single
        ("C=C", 0, 1, 1.33, 0.03),      // C=C
        ("C#C", 0, 1, 1.20, 0.03),      // C#C
        ("c1ccccc1", 0, 1, 1.40, 0.03), // aromatic C:C
        ("CN", 0, 1, 1.47, 0.03),       // C-N
        ("CO", 0, 1, 1.42, 0.03),       // C-O
        ("CC(=O)C", 1, 2, 1.23, 0.03),  // C=O carbonyl
        ("CC#N", 1, 2, 1.15, 0.03),     // C#N
        ("CS", 0, 1, 1.81, 0.04),       // C-S
        ("CCl", 0, 1, 1.78, 0.04),      // C-Cl
        ("CBr", 0, 1, 1.96, 0.05),      // C-Br
        ("CF", 0, 1, 1.33, 0.04),       // C-F
    ];
    for (smiles, i, j, expected, tol) in cases {
        let m = build(smiles)?;
        let d = distance(&m, i, j);
        assert!(
            (d - expected).abs() < tol,
            "{smiles}: bond {i}-{j} is {d:.3} A, expected {expected:.2} +- {tol}"
        );
    }
    Ok(())
}

#[test]
fn bond_angles_follow_the_hybridisation() -> Result<(), StructError> {
    // (SMILES, i, centre, j, expected degrees, tolerance)
    let cases = [
        ("CCC", 0, 1, 2, 109.5, 2.0),            // sp3
        ("CC(=O)C", 0, 1, 3, 120.0, 3.0),        // sp2 carbonyl
        ("CC#N", 0, 1, 2, 180.0, 2.0),           // sp
        ("c1ccccc1", 0, 1, 2, 120.0, 1.0),       // aromatic
        ("C1CCCCC1", 0, 1, 2, 109.5, 2.0),       // cyclohexane chair
        ("NC(=N)c1ccccc1", 0, 1, 2, 120.0, 3.0), // amidine
    ];
    for (smiles, i, c, j, expected, tol) in cases {
        let m = build(smiles)?;
        let a = angle_deg(&m, i, c, j);
        assert!(
            (a - expected).abs() < tol,
            "{smiles}: angle {i}-{c}-{j} is {a:.1} deg, expected {expected:.1} +- {tol}"
        );
    }
    Ok(())
}

#[test]
fn sp3_centres_are_tetrahedral_not_flat() -> Result<(), StructError> {
    // An ideal tetrahedron of neighbours gives a signed volume of several cubic angstroms; a
    // planar centre gives zero. The builder used to produce 0.025 here.
    for smiles in ["C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O", "O[C@H]1CCCCC1"] {
        let m = build(smiles)?;
        let centre = (0..m.atoms.len())
            .find(|&i| m.atoms[i].element == "C" && m.adj[i].len() == 4)
            .ok_or(StructError::EmptyInput)?;
        let v = neighbour_volume(&m, centre).abs();
        assert!(
            v > 4.0,
            "{smiles}: neighbour tetrahedron volume {v:.3} A^3 -- the centre is close to planar"
        );
    }
    Ok(())
}

#[test]
fn saturated_rings_pucker_and_aromatic_rings_stay_flat() -> Result<(), StructError> {
    // (SMILES, ring size, should pucker)
    let cases = [
        ("C1CCCCC1", 6, true),        // cyclohexane -> chair
        ("C1CCNCC1", 6, true),        // piperidine -> chair
        ("C1COCCN1", 6, true),        // morpholine -> chair
        ("c1ccccc1", 6, false),       // benzene -> flat
        ("c1ccc2ccccc2c1", 6, false), // naphthalene -> flat
        ("C1CCc2ccccc2C1", 6, true),  // tetralin saturated half -> half-chair
    ];
    for (smiles, size, should_pucker) in cases {
        let m = build(smiles)?;
        let mut best = 0.0_f64;
        for ring in &m.rings {
            if ring.len() != size {
                continue;
            }
            // Maximum displacement from the ring's own mean plane.
            let n = ring.len() as f64;
            let mut cen = [0.0_f64; 3];
            for &v in ring {
                for k in 0..3 {
                    cen[k] += m.atoms[v].coords[k] / n;
                }
            }
            let mut normal = [0.0_f64; 3];
            for i in 0..ring.len() {
                let a = m.atoms[ring[i]].coords;
                let b = m.atoms[ring[(i + 1) % ring.len()]].coords;
                normal[0] += (a[1] - cen[1]) * (b[2] - cen[2]) - (a[2] - cen[2]) * (b[1] - cen[1]);
                normal[1] += (a[2] - cen[2]) * (b[0] - cen[0]) - (a[0] - cen[0]) * (b[2] - cen[2]);
                normal[2] += (a[0] - cen[0]) * (b[1] - cen[1]) - (a[1] - cen[1]) * (b[0] - cen[0]);
            }
            let nl = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
            if nl < 1e-6 {
                continue;
            }
            let worst = ring
                .iter()
                .map(|&v| {
                    let p = m.atoms[v].coords;
                    ((p[0] - cen[0]) * normal[0]
                        + (p[1] - cen[1]) * normal[1]
                        + (p[2] - cen[2]) * normal[2])
                        .abs()
                        / nl
                })
                .fold(0.0_f64, f64::max);
            best = best.max(worst);
        }
        if should_pucker {
            assert!(
                best > 0.15,
                "{smiles}: saturated ring is flat (max out-of-plane {best:.3} A)"
            );
        } else {
            assert!(
                best < 0.05,
                "{smiles}: aromatic ring is not flat (max out-of-plane {best:.3} A)"
            );
        }
    }
    Ok(())
}

#[test]
fn enantiomers_are_built_as_mirror_images() -> Result<(), StructError> {
    for (left, right) in [
        ("C[C@@H](N)C(=O)O", "C[C@H](N)C(=O)O"),
        (
            "CC(C)NC[C@H](O)COc1cccc2ccccc12",
            "CC(C)NC[C@@H](O)COc1cccc2ccccc12",
        ),
    ] {
        let ml = build(left)?;
        let mr = build(right)?;
        let centre = (0..ml.atoms.len())
            .find(|&i| ml.atoms[i].element == "C" && ml.adj[i].len() == 4)
            .ok_or(StructError::EmptyInput)?;
        let vl = neighbour_volume(&ml, centre);
        let vr = neighbour_volume(&mr, centre);
        assert!(
            vl * vr < 0.0,
            "{left} and {right} have the same handedness (signed volumes {vl:+.3} and {vr:+.3})"
        );
        assert!(
            (vl.abs() - vr.abs()).abs() < 0.6,
            "{left} and {right} are not mirror images ({vl:+.3} vs {vr:+.3})"
        );
    }
    Ok(())
}

#[test]
fn built_structures_have_no_steric_clashes() -> Result<(), StructError> {
    for smiles in [
        "NC(=N)c1ccc(OCc2ccccc2)cc1",
        "CC(C)NCC(O)COc1cccc2ccccc12",
        "CN(C)CCCN1c2ccccc2CCc2ccccc21",
        "C1CCc2ccccc2C1",
    ] {
        let m = build(smiles)?;
        for i in 0..m.atoms.len() {
            for j in (i + 1)..m.atoms.len() {
                if !m.atoms[i].is_heavy() || !m.atoms[j].is_heavy() {
                    continue;
                }
                if m.adj[i].iter().any(|&(nb, _)| nb == j) {
                    continue;
                }
                let d = distance(&m, i, j);
                assert!(
                    d > 2.1,
                    "{smiles}: non-bonded heavy atoms {i} and {j} are {d:.2} A apart"
                );
            }
        }
    }
    Ok(())
}

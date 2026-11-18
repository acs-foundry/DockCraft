#![forbid(unsafe_code)]
//! Aromaticity perception and pH 7.4 protonation against hand-checked ground truth.
//!
//! Both of these feed everything downstream: aromaticity drives AutoDock4 atom typing and the
//! Vina hydrophobic term, and the formal charge drives molecular weight, cLogP, hydrogen-bond
//! counts, Gasteiger charges and any enrichment figure computed from them. Two specific defects
//! are pinned here because they were live and each silently corrupted a published number:
//!
//! * aromaticity leaking across a fused bond, which marked all 10 heavy atoms of tetralin
//!   aromatic instead of 6 (and, through the "aromatic nitrogens are not basic" rule, left
//!   tetrahydroisoquinoline neutral);
//! * one positive charge per nitrogen instead of one per delocalised group, which gave
//!   benzamidine +2, guanidine +3 and metformin +5.

use struct_core::{StructError, parse_smiles_to_3d, perceive_all};

fn perceived(smiles: &str) -> Result<(usize, i32, f64), StructError> {
    let mut m = parse_smiles_to_3d(smiles, None)?;
    perceive_all(&mut m);
    let aromatic = m
        .atoms
        .iter()
        .filter(|a| a.aromatic && a.is_heavy())
        .count();
    let charge: i32 = m.atoms.iter().map(|a| a.formal_charge as i32).sum();
    Ok((aromatic, charge, m.molecular_weight()))
}

#[test]
fn aromaticity_does_not_leak_across_fused_saturated_rings() -> Result<(), StructError> {
    // (name, SMILES, aromatic heavy atoms expected)
    let cases = [
        ("benzene", "c1ccccc1", 6),
        ("cyclohexane", "C1CCCCC1", 0),
        ("naphthalene", "c1ccc2ccccc2c1", 10),
        ("tetralin", "C1CCc2ccccc2C1", 6),
        ("indane", "C1Cc2ccccc2C1", 6),
        ("indoline", "C1Cc2ccccc2N1", 6),
        ("tetrahydroisoquinoline", "C1Cc2ccccc2CN1", 6),
        ("tetrahydroquinoline", "C1CCc2ccccc2N1", 6),
        ("quinoline", "c1ccc2ncccc2c1", 10),
        ("indole", "c1ccc2[nH]ccc2c1", 9),
        ("purine", "c1ncc2[nH]cnc2n1", 9),
        ("pyridine", "c1ccncc1", 6),
        ("piperidine", "C1CCNCC1", 0),
        ("pyrrole", "c1cc[nH]c1", 5),
        ("furan", "c1ccoc1", 5),
        ("thiophene", "c1ccsc1", 5),
        ("4H-pyran", "C1=CCOC=C1", 0),
        ("cyclopentadiene", "C1=CCC=C1", 0),
    ];
    for (name, smiles, expected) in cases {
        let (aromatic, _, _) = perceived(smiles)?;
        assert_eq!(
            aromatic, expected,
            "{name}: {aromatic} aromatic heavy atoms, expected {expected}"
        );
    }
    Ok(())
}

#[test]
fn ph74_protonation_charges_groups_once_not_nitrogens() -> Result<(), StructError> {
    // (name, SMILES, net formal charge, molecular weight of that species)
    let cases = [
        // Delocalised cations: one charge for the whole group.
        ("benzamidine", "NC(=N)c1ccccc1", 1, 121.16),
        ("2-naphthamidine", "NC(=N)c1ccc2ccccc2c1", 1, 171.22),
        ("guanidine", "NC(=N)N", 1, 60.08),
        ("acetamidine", "CC(=N)N", 1, 59.09),
        ("arginine", "NC(=N)NCCCC(N)C(=O)O", 1, 175.21),
        ("metformin (biguanide)", "CN(C)C(=N)NC(=N)N", 1, 130.17),
        // Ordinary bases and acids.
        ("ethylamine", "CCN", 1, 46.09),
        ("lysine", "NCCCCC(N)C(=O)O", 1, 147.20),
        ("acetic acid", "CC(=O)O", -1, 59.04),
        // Not basic at pH 7.4.
        ("urea", "NC(=O)N", 0, 60.06),
        ("acetamide", "CC(=O)N", 0, 59.07),
        ("aniline", "Nc1ccccc1", 0, 93.13),
        ("imidazole", "c1cnc[nH]1", 0, 68.08),
        ("indoline", "C1Cc2ccccc2N1", 0, 119.17),
    ];
    for (name, smiles, expected_charge, expected_mw) in cases {
        let (_, charge, mw) = perceived(smiles)?;
        assert_eq!(
            charge, expected_charge,
            "{name}: net charge {charge:+}, expected {expected_charge:+}"
        );
        assert!(
            (mw - expected_mw).abs() < 0.02,
            "{name}: MW {mw:.2}, expected {expected_mw:.2}"
        );
    }
    Ok(())
}

#[test]
fn ring_perception_covers_seven_and_eight_membered_rings() -> Result<(), StructError> {
    // (name, SMILES, expected ring sizes, sorted)
    let cases: [(&str, &str, &[usize]); 7] = [
        ("benzene", "c1ccccc1", &[6]),
        ("naphthalene", "c1ccc2ccccc2c1", &[6, 6]),
        ("cycloheptane", "C1CCCCCC1", &[7]),
        ("cyclooctane", "C1CCCCCCC1", &[8]),
        ("caprolactam", "O=C1CCCCCN1", &[7]),
        ("imipramine core", "C1Cc2ccccc2Nc2ccccc21", &[6, 6, 7]),
        ("cyclopropane", "C1CC1", &[3]),
    ];
    for (name, smiles, expected) in cases {
        let mut m = parse_smiles_to_3d(smiles, None)?;
        perceive_all(&mut m);
        let mut sizes: Vec<usize> = m.rings.iter().map(|r| r.len()).collect();
        sizes.sort_unstable();
        assert_eq!(
            sizes, expected,
            "{name}: perceived ring sizes {sizes:?}, expected {expected:?}"
        );
    }
    Ok(())
}

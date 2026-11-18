#![forbid(unsafe_code)]
//! The in-pocket substituent mutator must *replace* what you click, not bond a new group onto it.
//!
//! The previous implementation always appended: clicking a hydrogen and asking for `-CH3` left
//! the hydrogen in place and bonded a carbon to it, producing a divalent hydrogen, and clicking a
//! saturated carbon gave it a fifth bond. Its "minimisation" then translated the whole ligand
//! along the six coordinate axes, which cannot change any internal coordinate, so the new group
//! never moved relative to the scaffold.

use hyde_scorer::{SubstituentPreset, mutate_and_minimize_in_pocket};
use struct_core::{Ad4Type, Atom3D, Molecule3D, parse_smiles_to_3d, perceive_all};

fn pocket() -> Molecule3D {
    let mut rec = Molecule3D::new("rec");
    for (el, ad4, pos, don, acc, phob) in [
        ("N", Ad4Type::NA, [-6.0, 0.0, 0.0], true, false, false),
        ("C", Ad4Type::C, [6.5, 0.0, 0.0], false, false, true),
        ("O", Ad4Type::OA, [0.0, -6.2, 0.0], false, true, false),
        ("C", Ad4Type::C, [0.0, 6.2, 0.0], false, false, true),
        ("C", Ad4Type::C, [0.0, 0.0, 6.4], false, false, true),
        ("C", Ad4Type::C, [0.0, 0.0, -6.4], false, false, true),
    ] {
        let mut a = Atom3D::new(rec.atoms.len(), el, pos);
        a.ad4_type = ad4;
        a.h_donor = don;
        a.h_acceptor = acc;
        a.hydrophobic = phob;
        rec.add_atom(a);
    }
    rec
}

fn benzamidine() -> Result<Molecule3D, Box<dyn std::error::Error>> {
    let mut m = parse_smiles_to_3d("NC(=N)c1ccccc1", Some("benzamidine"))?;
    perceive_all(&mut m);
    // Centre it in the model pocket.
    let c = m.centroid_heavy();
    for a in &mut m.atoms {
        for k in 0..3 {
            a.coords[k] -= c[k];
        }
    }
    Ok(m)
}

fn max_bonds(m: &Molecule3D, element: &str) -> usize {
    (0..m.atoms.len())
        .filter(|&i| m.atoms[i].element == element)
        .map(|i| m.adj[i].len())
        .max()
        .unwrap_or(0)
}

#[test]
fn clicking_a_hydrogen_replaces_it_instead_of_bonding_to_it()
-> Result<(), Box<dyn std::error::Error>> {
    let rec = pocket();
    let lig = benzamidine()?;
    let h_on_n = (0..lig.atoms.len())
        .find(|&i| {
            lig.atoms[i].element == "H"
                && lig.adj[i]
                    .first()
                    .is_some_and(|&(nb, _)| lig.atoms[nb].element == "N")
        })
        .ok_or("benzamidine should carry polar hydrogens on its amidine nitrogens")?;
    let parent_n = lig.adj[h_on_n][0].0;
    let carbons_before = lig.atoms.iter().filter(|a| a.element == "C").count();

    let out = mutate_and_minimize_in_pocket(&rec, &lig, h_on_n, SubstituentPreset::CH3)?;
    let m = &out.mutated_ligand;

    assert!(out.replaced_anchor, "a terminal hydrogen must be replaced");
    assert_eq!(
        max_bonds(m, "H"),
        1,
        "a hydrogen ended up with more than one bond: the substituent was appended to it \
         instead of replacing it"
    );
    assert_eq!(
        m.atoms.iter().filter(|a| a.element == "C").count(),
        carbons_before + 1,
        "expected exactly one new carbon"
    );
    // The nitrogen keeps a sane valence (3 heavy/H neighbours plus at most one more).
    let n_like = (0..m.atoms.len())
        .filter(|&i| m.atoms[i].element == "N")
        .map(|i| m.adj[i].len())
        .max()
        .unwrap_or(0);
    assert!(
        n_like <= 4,
        "nitrogen {parent_n} ended up with {n_like} bonds after the mutation"
    );
    assert!(out.elapsed_ms < 2000);
    Ok(())
}

#[test]
fn mutation_never_exceeds_carbon_valence() -> Result<(), Box<dyn std::error::Error>> {
    let rec = pocket();
    let lig = benzamidine()?;
    for preset in SubstituentPreset::all() {
        for anchor in 0..lig.atoms.len() {
            let out = mutate_and_minimize_in_pocket(&rec, &lig, anchor, *preset)?;
            let m = &out.mutated_ligand;
            assert!(
                max_bonds(m, "C") <= 4,
                "{preset:?} on atom {anchor} gave a carbon {} bonds",
                max_bonds(m, "C")
            );
            assert_eq!(
                max_bonds(m, "H"),
                1,
                "{preset:?} on atom {anchor} gave a hydrogen more than one bond"
            );
        }
    }
    Ok(())
}

#[test]
fn polyatomic_substituents_get_a_real_torsional_relaxation()
-> Result<(), Box<dyn std::error::Error>> {
    let rec = pocket();
    let lig = benzamidine()?;
    let h_on_n = (0..lig.atoms.len())
        .find(|&i| {
            lig.atoms[i].element == "H"
                && lig.adj[i]
                    .first()
                    .is_some_and(|&(nb, _)| lig.atoms[nb].element == "N")
        })
        .ok_or("expected a polar hydrogen")?;

    // A single-atom substituent has no torsion, so its internal shift may legitimately be zero.
    // A polyatomic one does, and the old rigid-body-only relaxation reported exactly 0.000000 A
    // for every preset.
    let mut any_internal_motion = false;
    for preset in [
        SubstituentPreset::CF3,
        SubstituentPreset::OCH3,
        SubstituentPreset::OH,
        SubstituentPreset::NH2,
    ] {
        let out = mutate_and_minimize_in_pocket(&rec, &lig, h_on_n, preset)?;
        eprintln!(
            "{preset:?}: ddG = {:+.3} kcal/mol, internal shift = {:.4} A, replaced = {}",
            out.delta_delta_g_kcal, out.internal_rmsd_shift, out.replaced_anchor
        );
        assert!(out.internal_rmsd_shift.is_finite());
        if out.internal_rmsd_shift > 1e-6 {
            any_internal_motion = true;
        }
    }
    assert!(
        any_internal_motion,
        "no polyatomic substituent moved in the ligand's own frame: the relaxation is still a \
         rigid-body translation"
    );
    Ok(())
}

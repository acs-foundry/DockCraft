#![forbid(unsafe_code)]

use dock_engine::{
    DockingConfig, DockingConstraint, PharmacophoreKind, ScoringFunction, dock_ligand,
    metal_coord_energy_and_deriv,
};
use pocket_detect::GridBox;
use struct_core::{
    Ad4Type, Atom3D, BondOrder, Molecule3D, detect_bridging_waters, metal_chelation_tier,
    parse_pdb, parse_smiles_to_3d, perceive_all, prepare_receptor, prepare_receptor_keeping_waters,
};

#[test]
fn symmetric_rmsd_collapses_180_degree_flipped_phenyl_ring()
-> Result<(), Box<dyn std::error::Error>> {
    let a = parse_smiles_to_3d("c1ccccc1C(=O)O", Some("BenzoicA"))?;
    let mut b = a.clone();
    // Reflect phenyl across the ipso-para-COOH axis (180 deg ring flip): raw index RMSD > 1.5 A,
    // but symmetry-aware heavy RMSD must recognize the automorphism and approach 0 A.
    let ring = a.rings.iter().find(|r| r.len() == 6).ok_or("6-ring")?;
    let ipso_pos = ring
        .iter()
        .position(|&idx| {
            a.adj[idx]
                .iter()
                .filter(|&&(nb, _)| a.atoms[nb].is_heavy())
                .count()
                == 3
        })
        .ok_or("ipso carbon")?;
    let mut ord = Vec::with_capacity(6);
    for k in 0..6 {
        ord.push(ring[(ipso_pos + k) % 6]);
    }
    // Swap ortho (1 <-> 5) and meta (2 <-> 4) pairs relative to ipso (0) & para (3)
    let p1 = b.atoms[ord[1]].coords;
    let p5 = b.atoms[ord[5]].coords;
    let p2 = b.atoms[ord[2]].coords;
    let p4 = b.atoms[ord[4]].coords;
    b.atoms[ord[1]].coords = p5;
    b.atoms[ord[5]].coords = p1;
    b.atoms[ord[2]].coords = p4;
    b.atoms[ord[4]].coords = p2;

    let naive = a.rmsd_heavy(&b).ok_or("naive rmsd")?;
    let sym = a.rmsd_heavy_symmetric(&b).ok_or("sym rmsd")?;
    assert!(
        naive > 1.0,
        "expected naive RMSD > 1.0 A on swapped phenyl, got {naive:.3}"
    );
    assert!(
        sym < 0.05,
        "expected symmetry-aware RMSD ~ 0.0 A on flipped phenyl, got {sym:.3}"
    );
    Ok(())
}

#[test]
fn macrocycle_rings_and_ez_double_bond_stereo_are_perceived()
-> Result<(), Box<dyn std::error::Error>> {
    // 14-membered macrocycle ring (e.g. macrolide core)
    let mac = parse_smiles_to_3d("C1CCCCCCCCCCCCC1", Some("Cyclotetradecane"))?;
    assert!(
        mac.rings.iter().any(|r| r.len() == 14),
        "expected 14-membered macrocycle ring, got {:?}",
        mac.rings
    );

    // E vs Z 2-butene: trans (E) has larger C1-C4 distance than cis (Z)
    let trans_but = parse_smiles_to_3d("C/C=C/C", Some("E_butene"))?;
    let cis_but = parse_smiles_to_3d("C/C=C\\C", Some("Z_butene"))?;
    let dist_03 = |m: &Molecule3D| -> f64 {
        let a = m.atoms[0].coords;
        let b = m.atoms[3].coords;
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let d_trans = dist_03(&trans_but);
    let d_cis = dist_03(&cis_but);
    assert!(
        d_trans > d_cis + 0.5,
        "expected E-butene terminal distance ({d_trans:.2}) > Z-butene ({d_cis:.2})"
    );
    Ok(())
}

#[test]
fn detects_and_retains_structural_bridging_water() -> Result<(), Box<dyn std::error::Error>> {
    let pdb = "\
ATOM      1  N   ILE A  50       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  N   ILE B  50       4.200   0.000   0.000  1.00 10.00           N
HETATM    3  O   HOH A 301       2.100   1.900   0.000  1.00 12.00           O
HETATM    4  O   HOH A 999      12.000  12.000  12.000  1.00 40.00           O
END
";
    let mut raw = parse_pdb(pdb, false)?;
    let waters = detect_bridging_waters(&raw, [2.0, 1.0, 0.0], 8.0, 2);
    assert_eq!(waters.len(), 1);
    assert_eq!(waters[0].res_seq, 301);
    assert_eq!(waters[0].receptor_hbonds, 2);

    prepare_receptor_keeping_waters(&mut raw, &[301]);
    assert!(
        raw.atoms
            .iter()
            .any(|a| a.res_seq == 301 && a.h_donor && a.h_acceptor)
    );
    assert!(!raw.atoms.iter().any(|a| a.res_seq == 999));
    Ok(())
}

#[test]
fn metal_coordination_attracts_sulfonamide_warhead_to_zinc()
-> Result<(), Box<dyn std::error::Error>> {
    let azm = parse_smiles_to_3d("CC(=O)Nc1nnc(S(=O)(=O)N)s1", Some("Acetazolamide"))?;
    let sulf_n = azm
        .atoms
        .iter()
        .enumerate()
        .find(|&(i, a)| a.element == "N" && metal_chelation_tier(&azm, i) == 2)
        .map(|(i, _)| i);
    assert!(
        sulf_n.is_some(),
        "expected Tier-2 sulfonamide N on Acetazolamide"
    );
    let (e_zn, _) = metal_coord_energy_and_deriv(Ad4Type::Zn, 2, 2.05);
    assert!(
        e_zn < -2.0,
        "expected strong Zn coordination well, got {e_zn:.3}"
    );

    // Build a mini Zn active site
    let mut rec = Molecule3D::new("ZnSite");
    let mut zn = Atom3D::new(0, "Zn", [0.0, 0.0, 0.0]);
    zn.res_name = "ZN".to_string();
    zn.is_hetatm = true;
    rec.add_atom(zn);
    for (i, pos) in [
        [-2.0, 0.0, -1.5],
        [2.0, 0.0, -1.5],
        [0.0, -2.0, -1.5],
        [-3.5, 2.5, 2.5],
        [3.5, 2.5, 2.5],
    ]
    .iter()
    .enumerate()
    {
        let mut a = Atom3D::new(i + 1, "C", *pos);
        a.res_name = "HIS".to_string();
        a.res_seq = (i + 10) as i32;
        rec.add_atom(a);
    }
    prepare_receptor(&mut rec, false);

    let gbox = GridBox::from_center_size([0.0, 2.0, 2.5], [16.0, 16.0, 16.0], 0.375)?;
    let cfg = DockingConfig {
        scoring: ScoringFunction::Vina,
        exhaustiveness: 4,
        metal_coordination: true,
        ..Default::default()
    };
    let rep = dock_ligand(&rec, &azm, &gbox, &cfg)?;
    let best = rep.poses.first().ok_or("no docked poses")?;
    let sn_idx = sulf_n.ok_or("no sulfonamide N")?;
    let p = best.molecule.atoms[sn_idx].coords;
    let d_zn = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    assert!(
        d_zn < 2.6,
        "expected sulfonamide N to coordinate Zn within 2.6 A, got {d_zn:.2} A"
    );
    Ok(())
}

#[test]
fn pharmacophore_core_tether_covalent_and_flexible_sidechains_work()
-> Result<(), Box<dyn std::error::Error>> {
    // Minimal receptor with a flexible ASP sidechain and a Cys SG covalent nucleophile
    let pdb = "\
ATOM      1  N   ASP A  25       4.500   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ASP A  25       4.000   1.200   0.000  1.00 10.00           C
ATOM      3  C   ASP A  25       5.000   2.300   0.000  1.00 10.00           C
ATOM      4  O   ASP A  25       6.200   2.100   0.000  1.00 10.00           O
ATOM      5  CB  ASP A  25       2.600   1.500   0.000  1.00 10.00           C
ATOM      6  CG  ASP A  25       1.600   0.400   0.000  1.00 10.00           C
ATOM      7  OD1 ASP A  25       1.200  -0.300   1.000  1.00 10.00           O
ATOM      8  OD2 ASP A  25       1.200  -0.300  -1.000  1.00 10.00           O
ATOM      9  SG  CYS A  79      -3.800   0.000   0.000  1.00 10.00           S
ATOM     10  CB  CYS A  79      -5.300   0.500   0.000  1.00 10.00           C
END
";
    let rec = parse_pdb(pdb, true)?;
    let mut lig = parse_smiles_to_3d("c1ccccc1O", Some("Phenol"))?;
    perceive_all(&mut lig);
    let gbox = GridBox::from_center_size([0.0, 0.0, 0.0], [16.0, 16.0, 16.0], 0.375)?;

    // 1. Pharmacophore sphere constraint pulls H-donor/acceptor O into target sphere
    let target_pt = [-1.0, 2.0, 0.0];
    let o_idx = lig
        .atoms
        .iter()
        .position(|a| a.element == "O")
        .ok_or("no O")?;
    let cfg_ph = DockingConfig {
        exhaustiveness: 4,
        constraints: vec![DockingConstraint::PharmacophoreSphere {
            center: target_pt,
            radius: 1.5,
            kind: PharmacophoreKind::HAcceptor,
            weight: 6.0,
        }],
        flex_residues: vec!["A:ASP25".to_string()],
        ..Default::default()
    };
    let rep_ph = dock_ligand(&rec, &lig, &gbox, &cfg_ph)?;
    assert!(
        rep_ph.flex_receptor.is_some(),
        "expected adapted flex_receptor"
    );
    let po = rep_ph.poses.first().ok_or("no pose")?.molecule.atoms[o_idx].coords;
    let d_ph = ((po[0] - target_pt[0]).powi(2)
        + (po[1] - target_pt[1]).powi(2)
        + (po[2] - target_pt[2]).powi(2))
    .sqrt();
    assert!(
        d_ph <= 2.2,
        "pharmacophore sphere should hold O within ~2 A of target, got {d_ph:.2} A"
    );

    // 2. Covalent attachment constraint tethers ligand atom 0 to Cys79 SG at ~1.82 A
    let sg_idx = rec
        .atoms
        .iter()
        .position(|a| a.name == "SG")
        .ok_or("no SG")?;
    let cfg_cov = DockingConfig {
        exhaustiveness: 4,
        constraints: vec![DockingConstraint::CovalentAttachment {
            receptor_atom_idx: sg_idx,
            ligand_atom_idx: 0,
            target_bond_len: 1.82,
            force_k: 60.0,
        }],
        ..Default::default()
    };
    let rep_cov = dock_ligand(&rec, &lig, &gbox, &cfg_cov)?;
    let p0 = rep_cov.poses.first().ok_or("no cov pose")?.molecule.atoms[0].coords;
    let psg = rec.atoms[sg_idx].coords;
    let d_cov =
        ((p0[0] - psg[0]).powi(2) + (p0[1] - psg[1]).powi(2) + (p0[2] - psg[2]).powi(2)).sqrt();
    assert!(
        (1.50..=2.20).contains(&d_cov),
        "covalent warhead atom should tether near 1.82 A of Cys SG, got {d_cov:.2} A"
    );

    // 3. CoreTether constraint anchors benzene ring atoms near a reference template
    let mut ref_targets = Vec::new();
    for (i, a) in lig.atoms.iter().enumerate() {
        if a.aromatic {
            ref_targets.push((i, [a.coords[0] - 0.5, a.coords[1] - 1.5, a.coords[2]]));
        }
    }
    let cfg_core = DockingConfig {
        exhaustiveness: 4,
        constraints: vec![DockingConstraint::CoreTether {
            atom_targets: ref_targets.clone(),
            tolerance_a: 0.5,
            force_k: 8.0,
        }],
        ..Default::default()
    };
    let rep_core = dock_ligand(&rec, &lig, &gbox, &cfg_core)?;
    let best_core = &rep_core.poses.first().ok_or("no core pose")?.molecule;
    let mut mean_dev = 0.0;
    for &(idx, tgt) in &ref_targets {
        let q = best_core.atoms[idx].coords;
        mean_dev += ((q[0] - tgt[0]).powi(2) + (q[1] - tgt[1]).powi(2) + (q[2] - tgt[2]).powi(2))
            .sqrt()
            / (ref_targets.len() as f64);
    }
    assert!(
        mean_dev < 1.2,
        "core tether should keep scaffold near reference coordinates, got mean dev {mean_dev:.2} A"
    );
    let _ = BondOrder::Single;
    Ok(())
}

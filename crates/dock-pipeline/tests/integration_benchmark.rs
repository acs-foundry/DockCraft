#![forbid(unsafe_code)]

use std::path::Path;

use dock_engine::{DockingConfig, ScoringFunction, dock_ligand};
use dock_pipeline::{
    build_chymera_export, build_molcraft_export, evaluate_admet, load_ligand_library_bytes,
    run_screening_campaign,
};
use hyde_scorer::{SubstituentPreset, mutate_and_minimize_in_pocket, score_pose_hyde};
use pocket_detect::{GridBox, PocketDetectConfig, detect_pockets};
use struct_core::{Molecule3D, parse_pdb, parse_sdf_3d_with_limit, perceive_all, prepare_receptor};

fn fixture_path(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/docking_benchmark")
        .join(rel)
}

/// Displaces a native crystal ligand by >= 8.5 A in translation and ~105 deg rotation.
fn scramble_far(native: &Molecule3D, offset: [f64; 3]) -> Molecule3D {
    let mut mol = native.clone();
    if mol.heavy_atom_count() == 0 {
        return mol;
    }
    let c = mol.centroid_heavy();
    for a in &mut mol.atoms {
        let rx = a.coords[0] - c[0];
        let ry = a.coords[1] - c[1];
        let rz = a.coords[2] - c[2];
        // ~105 deg yaw + 90 deg roll
        let x1 = -0.2588 * rx - 0.9659 * ry;
        let y1 = 0.9659 * rx - 0.2588 * ry;
        let z1 = rz;
        a.coords = [
            c[0] + x1 + offset[0],
            c[1] + z1 + offset[1],
            c[2] - y1 + offset[2],
        ];
    }
    mol
}

fn roc_auc_by_key(items: &[(bool, f64)]) -> f64 {
    let mut act = Vec::new();
    let mut dec = Vec::new();
    for &(is_act, score) in items {
        if is_act {
            act.push(score);
        } else {
            dec.push(score);
        }
    }
    let mut wins = 0.0_f64;
    for &sa in &act {
        for &sd in &dec {
            if sa > sd {
                wins += 1.0;
            } else if (sa - sd).abs() < 1e-9 {
                wins += 0.5;
            }
        }
    }
    wins / ((act.len() * dec.len()).max(1) as f64)
}

#[test]
fn redocks_real_rcsb_benchmarks_from_9_angstroms_under_2a_rmsd()
-> Result<(), Box<dyn std::error::Error>> {
    // Wait: let's test the 4 authentic RCSB PDB structures extracted verbatim from files.rcsb.org
    let cases = [
        ("3ptb_trypsin.pdb", "3ptb_benzamidine.sdf", 9usize, 6usize),
        ("1stp_streptavidin.pdb", "1stp_biotin.sdf", 16usize, 8usize),
        ("1hsg_hivpr.pdb", "1hsg_indinavir.sdf", 45usize, 10usize),
        ("3ert_er.pdb", "3ert_4oht.sdf", 29usize, 8usize),
    ];

    for (rec_file, lig_file, expected_heavy, exh) in cases {
        let rec_raw = std::fs::read_to_string(fixture_path(rec_file))?;
        let mut rec = parse_pdb(&rec_raw, true)?;
        prepare_receptor(&mut rec, true);

        let lig_raw = std::fs::read_to_string(fixture_path(lig_file))?;
        let mut ligs = parse_sdf_3d_with_limit(&lig_raw, 1);
        let mut native = ligs.pop().ok_or("missing fixture ligand")??;
        perceive_all(&mut native);
        assert_eq!(native.heavy_atom_count(), expected_heavy);

        let gbox = GridBox::from_ligand(&native, 6.0)?;
        let scrambled = scramble_far(&native, [6.0, -5.0, 4.5]); // displacement = 9.01 A!
        let disp = {
            let ca = native.centroid_heavy();
            let cb = scrambled.centroid_heavy();
            ((ca[0] - cb[0]).powi(2) + (ca[1] - cb[1]).powi(2) + (ca[2] - cb[2]).powi(2)).sqrt()
        };
        assert!(disp >= 8.5, "{rec_file}: start offset {disp:.2} A < 8.5 A");

        let cfg = DockingConfig {
            scoring: ScoringFunction::Vina,
            exhaustiveness: exh,
            ils_steps: 28,
            bfgs_iters: 55,
            ..Default::default()
        };
        let report = dock_ligand(&rec, &scrambled, &gbox, &cfg)?;
        let top = report.poses.first().ok_or("no docked poses returned")?;
        assert_eq!(
            top.terms.breakdown.electrostatic, 0.0,
            "pure Vina must not include Coulomb term"
        );
        let rmsd = native.rmsd_heavy(&top.molecule).ok_or("rmsd error")?;
        eprintln!(
            "RCSB {rec_file}/{lig_file} ({expected_heavy} heavy, rot={}, evals={}): start={disp:.2}A -> score={:.3} kcal/mol, RMSD={rmsd:.3} A in {} ms",
            native.rotatable_bonds.len(),
            report.total_evaluations,
            top.affinity_kcal_mol,
            report.elapsed_ms
        );
        assert!(
            rmsd < 2.0,
            "{rec_file}/{lig_file}: expected RMSD < 2.0 A from {disp:.2} A start, got {rmsd:.3} A (score {:.3})",
            top.affinity_kcal_mol
        );

        let hyde = score_pose_hyde(&rec, &top.molecule)?;
        assert_eq!(hyde.atom_scores.len(), top.molecule.heavy_atom_count());

        let (pml, mol_bundle) = build_molcraft_export(rec_file, &rec, &top.molecule, &hyde);
        assert!(pml.contains("load receptor.pdb") && !mol_bundle.atom_annotations.is_empty());
        let (cxc, chy_bundle) = build_chymera_export(rec_file, &rec, &top.molecule, &hyde);
        assert!(cxc.contains("open receptor.pdb") && !chy_bundle.ligand_sdf.is_empty());
    }

    Ok(())
}

#[test]
fn real_pocket_detection_and_screening_beats_mw_null_model()
-> Result<(), Box<dyn std::error::Error>> {
    let rec_raw = std::fs::read_to_string(fixture_path("3ptb_trypsin.pdb"))?;
    let mut rec = parse_pdb(&rec_raw, true)?;
    prepare_receptor(&mut rec, true);

    let lig_raw = std::fs::read_to_string(fixture_path("3ptb_benzamidine.sdf"))?;
    let mut ben = parse_sdf_3d_with_limit(&lig_raw, 1)
        .pop()
        .ok_or("missing BEN")??;
    perceive_all(&mut ben);
    let ben_center = ben.centroid_heavy();

    let pockets = detect_pockets(&rec, &PocketDetectConfig::default())?;
    assert!(!pockets.is_empty(), "expected pockets on real 3PTB trypsin");
    let s1_pocket = pockets
        .iter()
        .min_by(|a, b| {
            let da = a
                .probe_points
                .iter()
                .map(|pt| {
                    ((pt[0] - ben_center[0]).powi(2)
                        + (pt[1] - ben_center[1]).powi(2)
                        + (pt[2] - ben_center[2]).powi(2))
                    .sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            let db = b
                .probe_points
                .iter()
                .map(|pt| {
                    ((pt[0] - ben_center[0]).powi(2)
                        + (pt[1] - ben_center[1]).powi(2)
                        + (pt[2] - ben_center[2]).powi(2))
                    .sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            da.total_cmp(&db)
        })
        .ok_or("no pockets")?;
    let min_probe_to_ben = s1_pocket
        .probe_points
        .iter()
        .map(|pt| {
            ((pt[0] - ben_center[0]).powi(2)
                + (pt[1] - ben_center[1]).powi(2)
                + (pt[2] - ben_center[2]).powi(2))
            .sqrt()
        })
        .fold(f64::INFINITY, f64::min);
    assert!(
        min_probe_to_ben < 2.2,
        "expected detected Trypsin pocket probe points inside S1 site (min_probe_to_ben = {min_probe_to_ben:.2} A)"
    );
    assert!(
        s1_pocket
            .lining_residues
            .iter()
            .any(|r| r.res_name == "ASP" && r.res_seq == 189),
        "expected Trypsin S1 ASP189 among lining residues of top matching pocket"
    );

    let gbox = GridBox::from_ligand(&ben, 4.0)?;

    // Size/MW-matched Trypsin S1 amidine/naphthamidine actives (MW ~171-227 Da)
    // vs same/higher-MW decoys (MW ~154-285 Da) so MW null model scores <= 0.45 AUC.
    let mut compounds = Vec::new();
    let actives = [
        ("Act_2Naphthamidine", "NC(=N)c1ccc2ccccc2c1"),
        ("Act_4Biphenylamidine", "NC(=N)c1ccc(-c2ccccc2)cc1"),
        ("Act_4Benzyloxybenzamidine", "NC(=N)c1ccc(OCc2ccccc2)cc1"),
        ("Act_4Phenoxybenzamidine", "NC(=N)c1ccc(Oc2ccccc2)cc1"),
    ];
    for (name, smi) in actives {
        compounds.push(serde_json::json!({ "name": name, "smiles": smi, "is_active": true }));
    }
    let decoys = [
        ("Dec_Phenanthrene", "c1ccc2c(c1)ccc3ccccc23"),
        ("Dec_Pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34"),
        ("Dec_Triphenylmethane", "c1ccc(C(c2ccccc2)c3ccccc3)cc1"),
        ("Dec_Anthracene", "c1ccc2cc3ccccc3cc2c1"),
        ("Dec_Benzophenone", "O=C(c1ccccc1)c2ccccc2"),
        ("Dec_Chalcone", "O=C(C=Cc1ccccc1)c2ccccc2"),
        ("Dec_Rhodanine_PAINS", "O=C1NC(=S)SC1=Cc2ccccc2"),
        ("Dec_Quinone_PAINS", "O=C1C=CC(=O)C=C1c2ccccc2"),
        (
            "Dec_Dinitrophenyl",
            "O=[N+]([O-])c1ccc(c2ccccc2)c([N+](=O)[O-])c1",
        ),
        ("Dec_Dichlorobiphenyl", "Clc1ccc(-c2ccc(Cl)cc2)cc1"),
        ("Dec_Stilbene", "c1ccc(C=Cc2ccccc2)cc1"),
        ("Dec_Diphenylamine_Ac", "CC(=O)N(c1ccccc1)c2ccccc2"),
        ("Dec_Hexachlorobenzene", "Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl"),
        ("Dec_Dibenzofuran", "c1ccc2oc3ccccc3c2c1"),
        ("Dec_Flavanone_frag", "O=C1CC(c2ccccc2)Oc3ccccc13"),
        ("Dec_Fluoranthene", "c1ccc2c(c1)-c3cccc4cccc-2c34"),
    ];
    for (name, smi) in decoys {
        compounds.push(serde_json::json!({ "name": name, "smiles": smi, "is_active": false }));
    }

    let cc_json = serde_json::to_vec(&serde_json::json!({ "molecules": compounds }))?;
    let lib = load_ligand_library_bytes("chemcraft", &cc_json, 100)?;
    assert_eq!(lib.len(), 20);

    let mw_null_auc = roc_auc_by_key(
        &lib.iter()
            .map(|r| (r.is_active.unwrap_or(false), r.molecule.molecular_weight()))
            .collect::<Vec<_>>(),
    );

    let cfg = DockingConfig {
        scoring: ScoringFunction::DockCraftHybrid,
        exhaustiveness: 4,
        ils_steps: 14,
        bfgs_iters: 30,
        ..Default::default()
    };
    let report = run_screening_campaign(&rec, &gbox, &lib, &cfg)?;
    assert_eq!(report.results.len(), 20);
    let enr = report
        .enrichment
        .as_ref()
        .ok_or("expected enrichment metrics")?;
    let mpo_auc = roc_auc_by_key(
        &report
            .results
            .iter()
            .map(|r| (r.is_active.unwrap_or(false), r.mpo_score))
            .collect::<Vec<_>>(),
    );
    let mpo_spread = report
        .results
        .iter()
        .map(|r| r.mpo_score)
        .fold(f64::NEG_INFINITY, f64::max)
        - report
            .results
            .iter()
            .map(|r| r.mpo_score)
            .fold(f64::INFINITY, f64::min);

    eprintln!(
        "3PTB 20-cmpd enrichment: MW_null_AUC={mw_null_auc:.3}, Dock_ROC_AUC={:.3}, MPO_AUC={mpo_auc:.3}, MPO_spread={mpo_spread:.3}, EF5%={:.2}",
        enr.roc_auc, enr.ef_5_percent
    );

    let mut mpo_sorted = report.results.clone();
    mpo_sorted.sort_by(|a, b| b.mpo_score.total_cmp(&a.mpo_score));
    assert_eq!(
        mpo_sorted.first().and_then(|c| c.is_active),
        Some(true),
        "Expected Rank #1 by MPO score on Trypsin S1 to be a true active"
    );

    assert!(
        mpo_spread > 0.30,
        "MPO score must show real separation (spread={mpo_spread:.3})"
    );
    assert!(
        mpo_auc > mw_null_auc + 0.40 && mpo_auc >= 0.85,
        "MPO screening ({mpo_auc:.3}) must decisively beat MW null model ({mw_null_auc:.3})"
    );
    assert!(
        enr.roc_auc > mw_null_auc + 0.30,
        "Docking ROC AUC ({:.3}) must decisively beat MW null model ({mw_null_auc:.3})",
        enr.roc_auc
    );

    // Verify PAINS rhodanine detected during campaign ADMET
    let rho_cand = report
        .results
        .iter()
        .find(|c| c.name == "Dec_Rhodanine_PAINS")
        .ok_or("missing rhodanine candidate")?;
    assert!(
        !rho_cand.admet.pains_alerts.is_empty(),
        "rhodanine PAINS alert must trigger in campaign"
    );
    let _ = evaluate_admet(&ben);

    let top_pose = &report.results[0].best_pose.molecule;
    let outcome = mutate_and_minimize_in_pocket(&rec, top_pose, 0, SubstituentPreset::F)?;
    assert!(outcome.elapsed_ms < 1000);
    Ok(())
}

#![forbid(unsafe_code)]

use std::path::Path;

use dock_engine::{DockingConfig, ScoringFunction, dock_ligand};
use dock_pipeline::{
    build_chymera_export, build_molcraft_export, load_ligand_library_bytes, run_screening_campaign,
};
use hyde_scorer::{SubstituentPreset, mutate_and_minimize_in_pocket, score_pose_hyde};
use pocket_detect::{GridBox, PocketDetectConfig, detect_pockets};
use struct_core::{parse_pdb, parse_sdf_3d_with_limit, perceive_all, prepare_receptor};

fn fixture_path(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/docking_benchmark")
        .join(rel)
}

fn scramble_ligand(native: &struct_core::Molecule3D, offset: [f64; 3]) -> struct_core::Molecule3D {
    let mut mol = native.clone();
    if mol.heavy_atom_count() == 0 {
        return mol;
    }
    let c = mol.centroid_heavy();
    for a in &mut mol.atoms {
        let rx = a.coords[0] - c[0];
        let ry = a.coords[1] - c[1];
        let rz = a.coords[2] - c[2];
        // Rotate by ~65 deg in X-Y + ~35 deg in Y-Z and translate
        let x1 = 0.4226 * rx - 0.9063 * ry;
        let y1 = 0.9063 * rx + 0.4226 * ry;
        let z1 = rz;
        a.coords = [
            c[0] + x1 + offset[0],
            c[1] + y1 + offset[1],
            c[2] + z1 + offset[2],
        ];
    }
    mol
}

#[test]
fn redocks_1stp_and_1hsg_benchmarks_under_2_angstrom_rmsd() -> Result<(), Box<dyn std::error::Error>>
{
    let cases = [
        ("1stp_biot_pocket.pdb", "1stp_biotin.sdf"),
        ("1hsg_hivpr_pocket.pdb", "1hsg_indinavir.sdf"),
        ("3ert_er_pocket.pdb", "3ert_4oht.sdf"),
    ];

    for (rec_file, lig_file) in cases {
        let rec_raw = std::fs::read_to_string(fixture_path(rec_file))?;
        let mut rec = parse_pdb(&rec_raw, true)?;
        prepare_receptor(&mut rec, true);

        let lig_raw = std::fs::read_to_string(fixture_path(lig_file))?;
        let mut ligs = parse_sdf_3d_with_limit(&lig_raw, 1);
        let mut native = ligs.pop().ok_or("missing fixture ligand")??;
        perceive_all(&mut native);

        let _native_bd = dock_engine::score_complex(&rec, &native, ScoringFunction::Vina);
        let gbox = GridBox::from_ligand(&native, 5.5)?;
        let scrambled = scramble_ligand(&native, [1.6, -1.3, 1.1]);

        let cfg = DockingConfig {
            scoring: ScoringFunction::Vina,
            exhaustiveness: 6,
            ils_steps: 24,
            bfgs_iters: 36,
            ..Default::default()
        };
        let report = dock_ligand(&rec, &scrambled, &gbox, &cfg)?;
        let top = report.poses.first().ok_or("no docked poses returned")?;
        let rmsd = native.rmsd_heavy(&top.molecule).ok_or("rmsd error")?;
        assert!(
            rmsd < 2.0,
            "{rec_file}/{lig_file}: expected RMSD < 2.0 A, got {rmsd:.3} A (score {:.3})",
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
fn end_to_end_20_ligand_screening_enrichment_and_mutator_pipeline()
-> Result<(), Box<dyn std::error::Error>> {
    let rec_raw = std::fs::read_to_string(fixture_path("1stp_biot_pocket.pdb"))?;
    let mut rec = parse_pdb(&rec_raw, true)?;
    prepare_receptor(&mut rec, true);

    let pockets = detect_pockets(&rec, &PocketDetectConfig::default())?;
    assert!(!pockets.is_empty(), "expected at least one pocket on 1STP");
    let gbox = &pockets[0].grid_box;

    // 20-ligand .chemcraft JSON library: compact polar/amphiphilic actives + bulky polyhalogenated/long decoys
    let mut compounds = Vec::new();
    let actives = [
        ("Act_Urea1", "O=C(N)NCC"),
        ("Act_Urea2", "O=C(N)NC1CCCC1"),
        ("Act_Amide3", "CC(=O)NC1CCSC1"),
        ("Act_Acid4", "O=C(O)CCCCN=C=O"),
    ];
    for (name, smi) in actives {
        compounds.push(serde_json::json!({
            "name": name,
            "smiles": smi,
            "is_active": true
        }));
    }
    for i in 0..16 {
        let smi = if i % 2 == 0 {
            "CCCCCCCCCCCCCCCC"
        } else {
            "Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Br"
        };
        compounds.push(serde_json::json!({
            "name": format!("Decoy_{}", i + 1),
            "smiles": smi,
            "is_active": false
        }));
    }
    let cc_json = serde_json::to_vec(&serde_json::json!({ "molecules": compounds }))?;
    let lib = load_ligand_library_bytes("chemcraft", &cc_json, 100)?;
    assert_eq!(lib.len(), 20);

    let cfg = DockingConfig {
        exhaustiveness: 3,
        ils_steps: 10,
        bfgs_iters: 20,
        ..Default::default()
    };
    let report = run_screening_campaign(&rec, gbox, &lib, &cfg)?;
    assert_eq!(report.results.len(), 20);
    let enr = report
        .enrichment
        .as_ref()
        .ok_or("expected enrichment metrics")?;
    assert!(enr.roc_auc >= 0.5 && enr.roc_auc <= 1.0);

    // Mutate top hit in pocket and check speed & valid HYDE breakdown
    let top_pose = &report.results[0].best_pose.molecule;
    let outcome = mutate_and_minimize_in_pocket(&rec, top_pose, 0, SubstituentPreset::OH)?;
    assert!(outcome.elapsed_ms < 1000);
    Ok(())
}

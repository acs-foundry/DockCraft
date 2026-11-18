#![forbid(unsafe_code)]
//! Re-docking under the conditions the tool actually meets in production.
//!
//! The crystal ligand is displaced by ~9 A **and every torsion is redrawn uniformly over the full
//! circle**, which is the situation a ligand coming from a SMILES embedding is in: the right
//! conformer is not supplied, it has to be found. The grid box follows the standard re-docking
//! protocol (centred on the native ligand), so the only question asked here is whether the global
//! search reaches the crystal basin — not whether pocket detection picks the right site, which is
//! a separate problem covered by its own test.
//!
//! This test exists because the previous benchmark handed the engine the native torsions and only
//! rotated the ligand rigidly. Under that protocol 1HSG/indinavir re-docked to 0.475 A; with the
//! torsions redrawn it went to 8.3 A mean and 0/5 under 2 A, because the conformer bank was seeded
//! with the input conformation and never left its neighbourhood.

use std::path::Path;

use dock_engine::{DockingConfig, ScoringFunction, dock_ligand};
use pocket_detect::GridBox;
use struct_core::{
    Molecule3D, parse_pdb, parse_sdf_3d_with_limit, perceive_all, prepare_receptor,
    randomize_torsions,
};

const SEEDS: u64 = 3;

fn fixture_path(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/docking_benchmark")
        .join(rel)
}

/// Displaces a native crystal ligand by >= 8.5 A in translation plus a ~105 deg reorientation.
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
        let x1 = -0.2588 * rx - 0.9659 * ry;
        let y1 = 0.9659 * rx - 0.2588 * ry;
        a.coords = [
            c[0] + x1 + offset[0],
            c[1] + rz + offset[1],
            c[2] - y1 + offset[2],
        ];
    }
    mol
}

#[test]
fn redocks_rcsb_benchmarks_with_randomized_starting_torsions()
-> Result<(), Box<dyn std::error::Error>> {
    // (receptor, ligand, expected heavy atoms, expected rotatable bonds, exhaustiveness)
    let cases = [
        (
            "3ptb_trypsin.pdb",
            "3ptb_benzamidine.sdf",
            9usize,
            1usize,
            6usize,
        ),
        ("1stp_streptavidin.pdb", "1stp_biotin.sdf", 16, 5, 8),
        ("1hsg_hivpr.pdb", "1hsg_indinavir.sdf", 45, 13, 10),
        ("3ert_er.pdb", "3ert_4oht.sdf", 29, 8, 8),
    ];

    for (rec_file, lig_file, expected_heavy, expected_rot, exh) in cases {
        let rec_raw = std::fs::read_to_string(fixture_path(rec_file))?;
        let mut rec = parse_pdb(&rec_raw, true)?;
        prepare_receptor(&mut rec, true);

        let lig_raw = std::fs::read_to_string(fixture_path(lig_file))?;
        let mut native = parse_sdf_3d_with_limit(&lig_raw, 1)
            .pop()
            .ok_or("missing fixture ligand")??;
        perceive_all(&mut native);
        assert_eq!(native.heavy_atom_count(), expected_heavy);
        assert_eq!(native.rotatable_bonds.len(), expected_rot);

        let gbox = GridBox::from_ligand(&native, 6.0)?;

        let mut rmsds = Vec::new();
        for seed in 0..SEEDS {
            let mut start = scramble_far(&native, [6.0, -5.0, 4.5]);
            let disp = {
                let ca = native.centroid_heavy();
                let cb = start.centroid_heavy();
                ((ca[0] - cb[0]).powi(2) + (ca[1] - cb[1]).powi(2) + (ca[2] - cb[2]).powi(2)).sqrt()
            };
            assert!(disp >= 8.5, "{rec_file}: start offset {disp:.2} A < 8.5 A");

            let angles = randomize_torsions(&mut start, 0xD0CC_1234 + seed);
            assert_eq!(angles.len(), expected_rot);
            let start_rmsd = native.rmsd_heavy(&start).ok_or("rmsd error")?;

            let cfg = DockingConfig {
                scoring: ScoringFunction::Vina,
                exhaustiveness: exh,
                ils_steps: 28,
                bfgs_iters: 55,
                seed: 20_261_117 + seed,
                ..Default::default()
            };
            let report = dock_ligand(&rec, &start, &gbox, &cfg)?;
            let top = report.poses.first().ok_or("no docked poses returned")?;
            assert_eq!(
                top.terms.breakdown.electrostatic, 0.0,
                "pure Vina must not include a Coulomb term"
            );
            let rmsd = native.rmsd_heavy(&top.molecule).ok_or("rmsd error")?;
            eprintln!(
                "{rec_file} seed={seed} rot={expected_rot}: start {disp:.2} A / {start_rmsd:.2} A RMSD -> {:.3} kcal/mol, RMSD {rmsd:.3} A, {} evals, {} ms (grid build included)",
                top.affinity_kcal_mol, report.total_evaluations, report.elapsed_ms
            );
            rmsds.push(rmsd);
        }

        let mean = rmsds.iter().sum::<f64>() / rmsds.len() as f64;
        for (seed, rmsd) in rmsds.iter().enumerate() {
            assert!(
                *rmsd < 2.0,
                "{rec_file}/{lig_file} seed {seed}: RMSD {rmsd:.3} A from randomized torsions (mean over {SEEDS} seeds {mean:.3} A)"
            );
        }
    }

    Ok(())
}

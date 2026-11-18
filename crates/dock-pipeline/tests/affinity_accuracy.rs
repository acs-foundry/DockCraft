#![forbid(unsafe_code)]
//! What the reported energies are worth against measured affinities.
//!
//! Both `ScoringFunction::Vina` and the HYDE decomposition return a number in kcal/mol, and
//! `ScreenedCandidate::estimated_ki_nm` turns the HYDE number into a nanomolar Ki. That
//! presentation invites the reader to treat it as a predicted affinity. On the four benchmark
//! complexes it is not one, and this test exists to keep that visible rather than buried:
//!
//! ```text
//! complex                 experiment   Vina    error    HYDE    error
//! 3PTB / benzamidine        -6.47      -5.82   +0.65   -5.64    +0.83
//! 1STP / biotin            -18.30      -7.61  +10.69  -11.32    +6.98
//! 1HSG / indinavir         -12.60     -10.52   +2.08  -19.10    -6.50
//! 3ERT / 4-hydroxytamoxifen -13.20     -9.60   +3.60  -13.12    +0.08
//! ```
//!
//! Streptavidin-biotin, at femtomolar affinity, is outside the applicability domain of every
//! empirical scoring function and dominates both mean errors; even excluding it the mean absolute
//! error is ~2.1 kcal/mol for Vina and ~2.5 for HYDE, which is three orders of magnitude in Ki.
//!
//! So the assertions here are the ones the numbers actually support: the energies are favourable,
//! they put the micromolar binder last, and the error has not grown. Four complexes cannot
//! establish a correlation and none is claimed.

use std::path::Path;

use dock_engine::{DockingConfig, ScoringFunction, dock_ligand};
use hyde_scorer::score_pose_hyde;
use pocket_detect::GridBox;
use struct_core::{parse_pdb, parse_sdf_3d_with_limit, perceive_all, prepare_receptor};

fn fixture_path(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/docking_benchmark")
        .join(rel)
}

#[test]
fn reported_energies_are_scores_not_calibrated_affinities() -> Result<(), Box<dyn std::error::Error>>
{
    // (receptor, ligand, published dG in kcal/mol, provenance)
    let cases = [
        (
            "3ptb_trypsin.pdb",
            "3ptb_benzamidine.sdf",
            -6.47,
            "Ki 18 uM",
        ),
        (
            "1stp_streptavidin.pdb",
            "1stp_biotin.sdf",
            -18.30,
            "Kd 4e-14 M",
        ),
        ("1hsg_hivpr.pdb", "1hsg_indinavir.sdf", -12.60, "Ki 0.56 nM"),
        ("3ert_er.pdb", "3ert_4oht.sdf", -13.20, "Kd 0.2 nM"),
    ];

    let mut vina_scores = Vec::new();
    let mut hyde_scores = Vec::new();
    let mut vina_abs_err = Vec::new();
    let mut hyde_abs_err = Vec::new();

    for (rec_file, lig_file, experimental, provenance) in cases {
        let rec_raw = std::fs::read_to_string(fixture_path(rec_file))?;
        let mut rec = parse_pdb(&rec_raw, true)?;
        prepare_receptor(&mut rec, true);

        let lig_raw = std::fs::read_to_string(fixture_path(lig_file))?;
        let mut native = parse_sdf_3d_with_limit(&lig_raw, 1)
            .pop()
            .ok_or("missing fixture ligand")??;
        perceive_all(&mut native);

        let gbox = GridBox::from_ligand(&native, 6.0)?;
        let cfg = DockingConfig {
            scoring: ScoringFunction::Vina,
            exhaustiveness: 8,
            ..Default::default()
        };
        let report = dock_ligand(&rec, &native, &gbox, &cfg)?;
        let top = report.poses.first().ok_or("no poses")?;
        let hyde = score_pose_hyde(&rec, &top.molecule)?;

        let v = top.affinity_kcal_mol;
        let h = hyde.total_delta_g_kcal;
        eprintln!(
            "{rec_file:24} experiment {experimental:7.2} ({provenance:11})  Vina {v:7.2} ({:+6.2})  HYDE {h:7.2} ({:+6.2})  HYDE Ki {:.3e} nM",
            v - experimental,
            h - experimental,
            hyde.estimated_ki_nm
        );

        assert!(
            v < 0.0 && h < 0.0,
            "{rec_file}: a known binder scored unfavourably (Vina {v:.2}, HYDE {h:.2})"
        );
        vina_scores.push((rec_file, v));
        hyde_scores.push((rec_file, h));
        vina_abs_err.push((v - experimental).abs());
        hyde_abs_err.push((h - experimental).abs());
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let max = |v: &[f64]| v.iter().cloned().fold(0.0_f64, f64::max);
    eprintln!(
        "mean |error|: Vina {:.2} (worst {:.2}), HYDE {:.2} (worst {:.2}) kcal/mol",
        mean(&vina_abs_err),
        max(&vina_abs_err),
        mean(&hyde_abs_err),
        max(&hyde_abs_err)
    );

    // The micromolar binder must come last under both scores. This is the only ordering claim
    // four complexes can support, and it is the one that matters for ranking a library.
    for (label, scores) in [("Vina", &vina_scores), ("HYDE", &hyde_scores)] {
        let weakest = scores
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or("empty")?;
        assert_eq!(
            weakest.0, "3ptb_trypsin.pdb",
            "{label} did not rank the micromolar benzamidine complex as the weakest of the four"
        );
    }

    // Regression guards, set above the errors measured today so they catch a real degradation
    // without pretending the current numbers are accurate.
    assert!(
        mean(&vina_abs_err) < 6.0,
        "Vina mean absolute error grew to {:.2} kcal/mol",
        mean(&vina_abs_err)
    );
    assert!(
        mean(&hyde_abs_err) < 6.0,
        "HYDE mean absolute error grew to {:.2} kcal/mol",
        mean(&hyde_abs_err)
    );
    Ok(())
}

#![forbid(unsafe_code)]

use std::path::PathBuf;

use pharm_core::{
    PharmScreenConfig, SiteAnalysisConfig, SiteBlueprint, analyze_binding_site,
    evaluate_holo_feature_recovery, screen_molecules_parallel,
};
use pocket_detect::GridBox;
use struct_core::{Molecule3D, parse_pdb, parse_sdf_3d_with_limit, perceive_all};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/docking_benchmark")
}

fn load_complex(
    pdb: &str,
    sdf: &str,
) -> Result<(Molecule3D, Molecule3D), Box<dyn std::error::Error>> {
    let dir = fixture_dir();
    let rec_raw = std::fs::read_to_string(dir.join(pdb))?;
    let mut rec = parse_pdb(&rec_raw, true)?;
    rec.name = pdb.trim_end_matches(".pdb").to_string();
    let lig_raw = std::fs::read_to_string(dir.join(sdf))?;
    let mut ligs = parse_sdf_3d_with_limit(&lig_raw, 1);
    let mut lig = ligs.pop().ok_or("empty fixture sdf")??;
    perceive_all(&mut lig);
    Ok((rec, lig))
}

#[test]
fn hybrid_and_apo_blueprints_recover_crystal_interactions_across_real_pockets()
-> Result<(), Box<dyn std::error::Error>> {
    let cases = [
        (
            "3ptb_trypsin.pdb",
            "3ptb_benzamidine.sdf",
            120.0,
            450.0,
            2usize,
        ),
        ("3ert_er.pdb", "3ert_4oht.sdf", 400.0, 1200.0, 3usize),
        (
            "1hsg_hivpr.pdb",
            "1hsg_indinavir.sdf",
            600.0,
            1600.0,
            3usize,
        ),
        (
            "1stp_streptavidin.pdb",
            "1stp_biotin.sdf",
            200.0,
            700.0,
            3usize,
        ),
    ];

    let mut total_rec = 0usize;
    let mut total_all = 0usize;

    for (pdb, sdf, min_vol, max_vol, min_match) in cases {
        let (rec, lig) = load_complex(pdb, sdf)?;
        let gbox = GridBox::from_ligand(&lig, 5.5)?;

        // 1. Hybrid Apo+Holo blueprint
        let bp = analyze_binding_site(&rec, &gbox, &[&lig], &SiteAnalysisConfig::default())?;
        assert!(
            (min_vol..=max_vol).contains(&bp.geometry.accessible_volume_a3),
            "{pdb}: unexpected pocket volume {} A^3",
            bp.geometry.accessible_volume_a3
        );
        assert!(
            bp.features.len() >= 5,
            "{pdb}: expected >= 5 pharmacophore features"
        );
        assert!(
            !bp.exclusion_spheres.is_empty(),
            "{pdb}: expected non-empty wall exclusion spheres"
        );
        for d in &bp.distance_matrix {
            assert!(
                d.min_distance_a <= d.ideal_distance_a + 1e-6
                    && d.ideal_distance_a <= d.max_distance_a + 1e-6,
                "{pdb}: invalid distance interval {}..={} for ideal {}",
                d.min_distance_a,
                d.max_distance_a,
                d.ideal_distance_a
            );
        }

        // JSON round-trip check
        let js = bp.to_json_pretty()?;
        let bp2 = SiteBlueprint::from_json_str(&js)?;
        assert_eq!(bp2.features.len(), bp.features.len());
        assert_eq!(bp2.distance_matrix.len(), bp.distance_matrix.len());

        // Crystallographic feature recovery check
        let (rec_n, all_n) = evaluate_holo_feature_recovery(&bp, &rec, &lig, 2.2);
        total_rec += rec_n;
        total_all += all_n;

        // Screening must recover the cognate crystal ligand cleanly
        let cfg = PharmScreenConfig {
            min_matched_features: min_match,
            max_rmsd_a: 1.85,
            max_exclusion_clashes: 4,
            max_conformers: 1,
            ..Default::default()
        };
        let rep = screen_molecules_parallel(std::slice::from_ref(&lig), &bp, &cfg);
        assert_eq!(
            rep.passed_count, 1,
            "{pdb}: cognate crystal ligand must pass its own blueprint"
        );
        assert!(
            rep.hits[0].pharm_rmsd_a <= 1.85,
            "{pdb}: unexpected RMSD {}",
            rep.hits[0].pharm_rmsd_a
        );

        // 2. Pure Apo blueprint (no holo ligand supplied) must also recover hotspots cleanly
        let apo_bp = analyze_binding_site(&rec, &gbox, &[], &SiteAnalysisConfig::default())?;
        assert!(
            apo_bp.features.len() >= 5,
            "{pdb} apo: expected >= 5 features"
        );
    }

    let frac = (total_rec as f64) / (total_all.max(1) as f64);
    assert!(
        frac >= 0.80,
        "expected >= 80% crystallographic feature recovery across real pockets, got {total_rec}/{total_all} ({:.1}%)",
        frac * 100.0
    );
    Ok(())
}

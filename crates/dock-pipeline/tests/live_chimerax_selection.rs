#![forbid(unsafe_code)]
//! Live verification of the ChimeraX channel: take the docking box from what a person has
//! selected on screen, and check it finds the crystal pose.
//!
//! Ignored by default because it needs a running ChimeraX. To run it:
//!
//! ```text
//! dockcraft-cli chimerax --receptor tests/fixtures/docking_benchmark/3ert_er.pdb \
//!                        --ligand   tests/fixtures/docking_benchmark/3ert_4oht.sdf \
//!                        --name 3ERT_ER --launch
//! # inside ChimeraX, once per session:
//! #   remotecontrol rest start port 60958 json true
//! # then select the binding site, e.g. for 3ERT:
//! #   select /A:346,347,350,353,387,388,391,394,404,421,424,428,521,524,525,528
//! DOCKCRAFT_LIVE_COMPLEX=3ert cargo test --release -p dock-pipeline \
//!     --test live_chimerax_selection -- --ignored --nocapture
//! ```
//!
//! # Why this exists
//!
//! Pocket detection ranks cavities by druggability, not by the site you care about, so on a
//! receptor with several plausible cavities it hands back the wrong one and every number
//! downstream is computed in the wrong place. A person looking at the structure already knows
//! where the site is. Measured on 2026-11-18:
//!
//! ```text
//! 3PTB  selection box centre 4.56 A from the native ligand, 3/3 under 2 A, mean 1.120 A
//!       best automatic pocket 7.92 A away
//! 3ERT  selection box centre 2.48 A from the native ligand, 3/3 under 2 A, mean 1.521 A
//!       best automatic pocket 11.19 A away
//! ```
//!
//! In both cases the selection box docks as well as a box centred on the answer (1.129 A and
//! 1.491 A respectively), which is the comparison that matters: the human-chosen box costs
//! nothing in accuracy and does not need to know the answer in advance.

use dock_engine::{DockingConfig, ScoringFunction, dock_ligand};
use dock_pipeline::{ChimeraXRest, default_chimerax_export_dir, grid_box_from_chimerax_selection};
use pocket_detect::{GridBox, PocketDetectConfig, detect_pockets};
use struct_core::{
    Molecule3D, parse_pdb, parse_sdf_3d_with_limit, perceive_all, prepare_receptor,
    randomize_torsions,
};

fn fixture(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/docking_benchmark")
        .join(rel)
}

/// Displaces the native ligand ~9 A and reorients it.
fn scramble(native: &Molecule3D, off: [f64; 3]) -> Molecule3D {
    let mut m = native.clone();
    let c = m.centroid_heavy();
    for a in &mut m.atoms {
        let (rx, ry, rz) = (a.coords[0] - c[0], a.coords[1] - c[1], a.coords[2] - c[2]);
        let x1 = -0.2588 * rx - 0.9659 * ry;
        let y1 = 0.9659 * rx - 0.2588 * ry;
        a.coords = [c[0] + x1 + off[0], c[1] + rz + off[1], c[2] - y1 + off[2]];
    }
    m
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[test]
#[ignore = "needs a running ChimeraX with a selection; see the module docs"]
fn box_taken_from_the_chimerax_selection_finds_the_crystal_pose()
-> Result<(), Box<dyn std::error::Error>> {
    let which = std::env::var("DOCKCRAFT_LIVE_COMPLEX").unwrap_or_else(|_| "3ptb".to_string());
    let (rec_file, lig_file) = match which.as_str() {
        "3ert" => ("3ert_er.pdb", "3ert_4oht.sdf"),
        "1hsg" => ("1hsg_hivpr.pdb", "1hsg_indinavir.sdf"),
        "1stp" => ("1stp_streptavidin.pdb", "1stp_biotin.sdf"),
        _ => ("3ptb_trypsin.pdb", "3ptb_benzamidine.sdf"),
    };

    let rec_raw = std::fs::read_to_string(fixture(rec_file))?;
    let mut rec = parse_pdb(&rec_raw, true)?;
    prepare_receptor(&mut rec, true);
    let lig_raw = std::fs::read_to_string(fixture(lig_file))?;
    let mut native = parse_sdf_3d_with_limit(&lig_raw, 1)
        .pop()
        .ok_or("missing fixture ligand")??;
    perceive_all(&mut native);
    let native_centre = native.centroid_heavy();

    let root = default_chimerax_export_dir();
    std::fs::create_dir_all(&root)?;
    let client = ChimeraXRest::with_default_port(root.clone());
    assert!(
        client.is_live(),
        "ChimeraX is not answering on 127.0.0.1:{}; see the module docs",
        client.port()
    );

    let selection = grid_box_from_chimerax_selection(&client, &root, 5.0)?;
    let selection_offset = distance(selection.grid_box.center, native_centre);
    eprintln!(
        "{which}: selection box from {} residues ({} heavy atoms), centre {selection_offset:.2} A from the native ligand, size {:?}",
        selection.residues.len(),
        selection.heavy_atoms,
        selection.grid_box.size.map(|v| (v * 10.0).round() / 10.0)
    );

    // For contrast, what the automatic detector would have chosen.
    if let Some(top) = detect_pockets(&rec, &PocketDetectConfig::default())?.first() {
        let n = top.probe_points.len().max(1) as f64;
        let sum = top
            .probe_points
            .iter()
            .fold([0.0_f64; 3], |a, p| [a[0] + p[0], a[1] + p[1], a[2] + p[2]]);
        let centre = [sum[0] / n, sum[1] / n, sum[2] / n];
        eprintln!(
            "{which}: best automatic pocket is {:.2} A from the native ligand",
            distance(centre, native_centre)
        );
    }

    // The box must at least contain the answer, whatever its centre.
    let heavy: Vec<[f64; 3]> = native
        .atoms
        .iter()
        .filter(|a| a.is_heavy())
        .map(|a| a.coords)
        .collect();
    let lo = selection.grid_box.min_corner();
    let hi = selection.grid_box.max_corner();
    let inside = heavy
        .iter()
        .filter(|p| (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]))
        .count();
    assert_eq!(
        inside,
        heavy.len(),
        "{which}: the selection box holds only {inside} of {} native heavy atoms; \
         select more of the binding site or raise the padding",
        heavy.len()
    );

    // And docking into it has to reach the crystal basin, as well as a box centred on the answer.
    for (label, gbox) in [
        ("chimerax selection", selection.grid_box.clone()),
        ("native-centred box", GridBox::from_ligand(&native, 6.0)?),
    ] {
        let mut rmsds = Vec::new();
        for seed in 0..3u64 {
            let mut start = scramble(&native, [6.0, -5.0, 4.5]);
            randomize_torsions(&mut start, 0xD0CC_1234 + seed);
            let cfg = DockingConfig {
                scoring: ScoringFunction::Vina,
                exhaustiveness: 8,
                seed: 20_261_117 + seed,
                ..Default::default()
            };
            let report = dock_ligand(&rec, &start, &gbox, &cfg)?;
            let top = report.poses.first().ok_or("no poses")?;
            rmsds.push(native.rmsd_heavy(&top.molecule).ok_or("rmsd")?);
        }
        let mean = rmsds.iter().sum::<f64>() / rmsds.len() as f64;
        let hits = rmsds.iter().filter(|r| **r < 2.0).count();
        eprintln!(
            "{which}: {label:20} mean {mean:.3} A, {hits}/3 under 2 A, rmsds {:?}",
            rmsds
                .iter()
                .map(|r| (r * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        if label == "chimerax selection" {
            assert_eq!(
                hits, 3,
                "{which}: docking into the selection box missed the crystal pose (mean {mean:.3} A)"
            );
        }
    }
    Ok(())
}

#![forbid(unsafe_code)]
//! How far the trilinear affinity grid departs from the exact pairwise score.
//!
//! The existing finite-difference test compares the analytical gradient of the interpolant with a
//! finite difference *of the same interpolant*, so inside a voxel the two agree to ~1e-12 by
//! construction and nothing is learned about the grid itself. What matters for docking is the
//! other question: how wrong is the interpolated energy compared with evaluating the scoring
//! function directly, and does the error concentrate at the voxel faces and edges where trilinear
//! interpolation is only C0.
//!
//! The answer is measured here rather than assumed, at four positions within the voxel, and
//! asserted against bounds. At 0.375 A spacing the grid is exact at its nodes to ~1e-7 kcal/mol, its mean absolute
//! error over the cell is ~0.03 kcal/mol, and the worst case inside the pose-relevant energy band
//! (|E| <= 2 kcal/mol) is ~0.09 kcal/mol. Deep in the repulsive wall the error reaches ~1 kcal/mol,
//! which does not affect ranking because those poses are rejected anyway, but it is real and is
//! the reason this test does not claim a uniform bound.

use dock_engine::grid::{AffinityGridSet, AtomChannelKey};
use dock_engine::vina::{ScoringFunction, score_complex};
use pocket_detect::GridBox;
use struct_core::{Ad4Type, Atom3D, Molecule3D};

fn receptor() -> Molecule3D {
    let mut rec = Molecule3D::new("rec");
    for (el, ad4, pos, don, acc, phob) in [
        ("N", Ad4Type::NA, [-3.2, 0.2, 0.1], true, false, false),
        ("C", Ad4Type::C, [2.8, 3.1, -0.5], false, false, true),
        ("O", Ad4Type::OA, [0.4, -2.9, 1.3], false, true, false),
        ("C", Ad4Type::A, [-1.1, 1.7, -2.2], false, false, true),
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

#[test]
fn trilinear_grid_matches_exact_scoring_at_nodes_and_is_bounded_between_them()
-> Result<(), Box<dyn std::error::Error>> {
    let rec = receptor();
    let spacing = 0.375_f64;
    let gbox = GridBox::from_center_size([0.0, 0.0, 0.0], [14.0, 14.0, 14.0], spacing)?;

    let mut probe = Molecule3D::new("probe");
    let mut pa = Atom3D::new(0, "C", [0.0, 0.0, 0.0]);
    pa.ad4_type = Ad4Type::C;
    pa.hydrophobic = true;
    probe.add_atom(pa);
    let key = AtomChannelKey::from_atom(&probe.atoms[0]);
    let grids = AffinityGridSet::build(&rec, &gbox, &[key], ScoringFunction::Vina)?;
    let ch = grids.channel_index(key).ok_or("probe channel missing")?;

    let min_c = gbox.min_corner();
    // Offsets as a fraction of one voxel: node, cell centre, face centre, edge midpoint.
    let positions: [(&str, [f64; 3], f64, f64); 4] = [
        // (label, offset, max |err| allowed where |E| <= 2, max mean |err| allowed)
        ("node", [0.0, 0.0, 0.0], 1e-6, 1e-6),
        ("cell_centre", [0.5, 0.5, 0.5], 0.15, 0.05),
        ("face_centre", [0.5, 0.5, 0.0], 0.15, 0.05),
        ("edge_mid", [0.5, 0.0, 0.0], 0.15, 0.05),
    ];

    for (label, frac, max_bound, mean_bound) in positions {
        let mut max_err_in_band = 0.0_f64;
        let mut worst_overall = 0.0_f64;
        let mut sum_err = 0.0_f64;
        let mut n = 0usize;
        for ix in 10..28 {
            for iy in 10..28 {
                for iz in 10..28 {
                    let p = [
                        min_c[0] + (ix as f64 + frac[0]) * spacing,
                        min_c[1] + (iy as f64 + frac[1]) * spacing,
                        min_c[2] + (iz as f64 + frac[2]) * spacing,
                    ];
                    let (grid_e, _) = grids.sample_atom(ch, 0.0, p);
                    let mut m = probe.clone();
                    m.atoms[0].coords = p;
                    let exact = score_complex(&rec, &m, ScoringFunction::Vina).total_affinity_kcal;
                    let err = (grid_e - exact).abs();
                    if exact.abs() <= 2.0 {
                        max_err_in_band = max_err_in_band.max(err);
                    }
                    worst_overall = worst_overall.max(err);
                    sum_err += err;
                    n += 1;
                }
            }
        }
        let mean_err = sum_err / n as f64;
        eprintln!(
            "{label:>12}: n={n}  mean|err|={mean_err:.3e}  max|err| where |E|<=2 = {max_err_in_band:.3e}  worst anywhere = {worst_overall:.3e} kcal/mol"
        );
        assert!(
            max_err_in_band <= max_bound,
            "{label}: interpolation error {max_err_in_band:.6} kcal/mol in the pose-relevant band exceeds {max_bound}"
        );
        assert!(
            mean_err <= mean_bound,
            "{label}: mean interpolation error {mean_err:.6} kcal/mol exceeds {mean_bound}"
        );
    }

    Ok(())
}

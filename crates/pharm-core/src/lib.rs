#![forbid(unsafe_code)]

//! Active-site Molecular Interaction Field (MIF) profiling, optimal inter-feature distance
//! blueprints (`.pharm.json`), ultra-fast 3D pharmacophore screening (`pharm-screen`), and
//! centralized `CRAFT_DATA_DIR` layout management.

pub mod data_dir;
pub mod kabsch;
pub mod mif;
pub mod screen;
pub mod selectivity;

use thiserror::Error;

pub use data_dir::{
    building_blocks_dir, campaigns_dir, craft_data_dir, ensure_craft_data_layout, envs_dir,
    libraries_dir, models_dir,
};
pub use kabsch::{RigidAlignment, kabsch_align, symmetric_eigen_3x3};
pub use mif::{
    ExclusionSphere, FeatureDistanceConstraint, FeatureSource, SiteAnalysisConfig, SiteBlueprint,
    SiteFeature, SiteGeometry, WaterAssessment, WaterRole, analyze_binding_site,
    build_distance_matrix, evaluate_holo_feature_recovery,
};
pub use screen::{
    LigandFeature, PharmScreenConfig, PharmScreenHit, PharmScreenReport, extract_ligand_features,
    screen_ligand_against_blueprint, screen_molecules_parallel, screen_smiles_stream,
};
pub use selectivity::{
    OffTargetComparisonSummary, OffTargetPocketInput, PocketSelectivityConfig, ResidueSubstitution,
    SelectivityAnalysisOutput, SelectivityHotspot, SelectivityHotspotKind, SelectivityReport,
    analyze_pocket_selectivity,
};

#[derive(Debug, Error)]
pub enum PharmError {
    #[error("empty receptor or ligand structure")]
    EmptyStructure,
    #[error("no receptor atoms found inside or near the specified binding-site GridBox")]
    NoPocketAtoms,
    #[error("serialization error: {0}")]
    Serialization(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use dock_engine::PharmacophoreKind;
    use pocket_detect::GridBox;
    use struct_core::{Ad4Type, Atom3D, Molecule3D, parse_smiles_to_3d};

    #[test]
    fn horn_kabsch_aligns_3d_and_coplanar_triangles_without_reflection() {
        let tgt = [[1.0, 2.0, 3.0], [5.2, 2.0, 3.0], [2.4, 5.7, 3.0]];
        // Apply a known 90-deg rotation around Z + translation (7, -4, 2)
        let mob: Vec<[f64; 3]> = tgt
            .iter()
            .map(|p| [-p[1] + 7.0, p[0] - 4.0, p[2] + 2.0])
            .collect();
        let aln = kabsch_align(&mob, &tgt);
        assert!(
            aln.rmsd < 1e-9,
            "expected exact alignment, got {}",
            aln.rmsd
        );
        // Verify det(R) == +1 (proper rotation, never mirror reflection)
        let r = aln.rotation;
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        assert!((det - 1.0).abs() < 1e-6, "expected det(R)=+1, got {det}");
    }

    #[test]
    fn analyzes_synthetic_hinge_and_hydrophobic_pocket_and_screens_matching_ligand()
    -> Result<(), Box<dyn std::error::Error>> {
        // Synthetic kinase-like pocket:
        // - Backbone H-bond acceptor O (like LEU83 O) at (-4.2, 0.0, 0.0)
        // - Backbone H-bond donor N-H (like LEU83 N) at (-3.5, 2.8, 0.0)
        // - Hydrophobic shell (VAL/ALA/PHE) enclosing (+2..+5, -3..+3, -3..+3)
        let mut rec = Molecule3D::new("synth_kinase");
        let add = |m: &mut Molecule3D,
                   el: &str,
                   rn: &str,
                   seq: i32,
                   pos: [f64; 3],
                   hd: bool,
                   ha: bool,
                   hyd: bool| {
            let mut a = Atom3D::new(m.atoms.len() + 1, el, pos);
            a.chain_id = 'A';
            a.res_name = rn.to_string();
            a.res_seq = seq;
            a.name = el.to_string();
            a.ad4_type = match el {
                "O" => Ad4Type::OA,
                "N" => Ad4Type::N,
                _ => Ad4Type::C,
            };
            a.h_donor = hd;
            a.h_acceptor = ha;
            a.hydrophobic = hyd;
            m.add_atom(a)
        };
        add(
            &mut rec,
            "O",
            "LEU",
            83,
            [-4.2, 0.0, 0.0],
            false,
            true,
            false,
        );
        let n_idx = add(
            &mut rec,
            "N",
            "LEU",
            83,
            [-3.6, 2.8, 0.0],
            true,
            false,
            false,
        );
        let mut h = Atom3D::new(rec.atoms.len() + 1, "H", [-2.6, 2.6, 0.0]);
        h.ad4_type = Ad4Type::HD;
        let h_idx = rec.add_atom(h);
        let _ = rec.add_bond(n_idx, h_idx, struct_core::BondOrder::Single);

        let wall_pts = [
            [4.5, 0.0, 0.0],
            [4.0, 3.2, 0.0],
            [4.0, -3.2, 0.0],
            [0.5, 0.0, 3.8],
            [0.5, 0.0, -3.8],
            [2.0, 3.5, 2.5],
            [2.0, -3.5, -2.5],
            [2.0, 3.5, -2.5],
            [2.0, -3.5, 2.5],
            [-2.0, -3.6, 0.0],
            [-2.0, 0.0, 3.8],
            [-2.0, 0.0, -3.8],
        ];
        for (i, &pt) in wall_pts.iter().enumerate() {
            add(&mut rec, "C", "VAL", 10 + i as i32, pt, false, false, true);
        }

        let gbox = GridBox::from_center_size([0.0, 0.5, 0.0], [12.0, 10.0, 10.0], 0.375)?;
        let bp = analyze_binding_site(&rec, &gbox, &[], &SiteAnalysisConfig::default())?;

        assert!(bp.cavity_voxel_count > 30);
        assert!(bp.geometry.accessible_volume_a3 > 20.0);
        assert!(
            bp.features
                .iter()
                .any(|f| f.kind == PharmacophoreKind::HDonor),
            "expected HBD hotspot near LEU83 O"
        );
        assert!(
            bp.features
                .iter()
                .any(|f| f.kind == PharmacophoreKind::HAcceptor),
            "expected HBA hotspot near LEU83 N-H"
        );
        assert!(
            bp.features
                .iter()
                .any(|f| f.kind == PharmacophoreKind::Hydrophobic),
            "expected HYD hotspot inside hydrophobic pocket"
        );
        assert!(!bp.distance_matrix.is_empty());
        assert!(!bp.exclusion_spheres.is_empty());

        // Screen a matching aminopyridine hinge binder vs methane (which must be rejected)
        let hinge_binder = parse_smiles_to_3d("Nc1ncccc1c2ccccc2", Some("AminopyridineHit"))?;
        let methane = parse_smiles_to_3d("C", Some("Methane"))?;
        let rep = screen_molecules_parallel(
            &[hinge_binder, methane],
            &bp,
            &PharmScreenConfig {
                min_matched_features: 3,
                max_exclusion_clashes: 4,
                ..Default::default()
            },
        );
        assert_eq!(rep.total_evaluated, 2);
        assert_eq!(
            rep.passed_count, 1,
            "expected only AminopyridineHit to match 3-point hinge blueprint"
        );
        assert_eq!(rep.hits[0].name, "AminopyridineHit");
        assert!(!bp.render_markdown().is_empty());
        Ok(())
    }
}

#![forbid(unsafe_code)]

//! Native Bidirectional **`DockCraft ↔ BiomolCraft`** Craft Link v1 Bridge (`ViewDockX` +
//! Synchronous 3D `GridBox` Selection over authenticated Unix sockets — supersedes Flatpak ChimeraX HTTP).

use std::path::{Path, PathBuf};
use std::process::Command;

use craftbio_scene::{
    DockCraftSceneBundle, DockPoseEntry, GridBox3D, HydeAtomCorona, LineOverlay, ProvenanceEntry,
    ProvenanceKind, SphereOverlay, TorsionBondLight, discover_and_request_selection_box,
    discover_craft_instances, push_dockcraft_bundle_via_craft_link,
};
use dock_engine::{DockingConstraint, PharmacophoreKind};
use hyde_scorer::{HydeReport, InteractionKind, TorsionStatus, consensus_affinity_kcal};
use pharm_core::{SelectivityHotspotKind, SelectivityReport, SiteBlueprint};
use pocket_detect::GridBox;
use struct_core::{BridgingWater, Molecule3D, write_pdb, write_sdf_3d};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BiomolLinkError {
    #[error("could not prepare BiomolCraft handoff directory '{dir}': {msg}")]
    Directory { dir: String, msg: String },
    #[error("I/O error writing BiomolCraft bundle: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Craft Link v1 IPC error: {0}")]
    Link(String),
    #[error("could not launch `biomolcraft` (bundle saved to {bundle_path}): {msg}")]
    Launch { bundle_path: String, msg: String },
}

/// Delivery outcome when pushing a `DockCraftSceneBundle` to `BiomolCraft`.
#[derive(Debug, Clone)]
pub struct BiomolDeliveryReport {
    pub bundle_path: PathBuf,
    pub pushed_live: bool,
    pub launched_new_instance: bool,
    pub pose_count: usize,
}

/// 3D `GridBox` pulled synchronously from a live `BiomolCraft` instance over Craft Link v1.
#[derive(Debug, Clone)]
pub struct BiomolSelectionBox {
    pub grid_box: GridBox,
    pub source_model: String,
    pub selected_residues: Vec<String>,
    pub heavy_atoms: usize,
}

/// Input pose record used to assemble a rich `DockCraftSceneBundle` for `BiomolCraft ViewDockX`.
pub struct DockPoseInput<'a> {
    pub pose_id: usize,
    pub name: String,
    pub canonical_smiles: Option<String>,
    pub molecule: &'a Molecule3D,
    pub vina_kcal: f64,
    pub hyde: Option<&'a HydeReport>,
    pub cnn_score: Option<f64>,
    pub mpo_score: Option<f64>,
}

/// Optional rich SBDD context overlays attached to a `DockCraftSceneBundle`.
#[derive(Default)]
pub struct BiomolBundleExtras<'a> {
    pub grid_box: Option<&'a GridBox>,
    pub constraints: &'a [DockingConstraint],
    pub site_blueprint: Option<&'a SiteBlueprint>,
    pub selectivity: Option<&'a SelectivityReport>,
    pub bridging_waters: &'a [BridgingWater],
    pub residue_metric_overrides: Vec<(char, i32, f64, String)>,
    pub trajectory_dcd_path: Option<String>,
}

/// Default handoff directory shared cleanly between `DockCraft` and `BiomolCraft`.
#[must_use]
pub fn default_biomolcraft_handoff_dir() -> PathBuf {
    if let Ok(xdg_data) = std::env::var("XDG_DATA_HOME")
        && !xdg_data.trim().is_empty()
    {
        return PathBuf::from(xdg_data)
            .join("BiomolCraft")
            .join("dockcraft_handoff");
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("BiomolCraft")
            .join("dockcraft_handoff");
    }
    std::env::temp_dir().join("biomolcraft_dockcraft_handoff")
}

fn sanitize_stem(raw: &str) -> String {
    let s: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(48)
        .collect();
    let trimmed = s.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "dockcraft_viewdock".to_string()
    } else {
        trimmed
    }
}

fn pharm_rgba(kind: PharmacophoreKind) -> [f32; 4] {
    match kind {
        PharmacophoreKind::HDonor => [0.20, 0.65, 1.0, 0.30],
        PharmacophoreKind::HAcceptor => [1.0, 0.30, 0.30, 0.30],
        PharmacophoreKind::Aromatic => [0.95, 0.55, 0.18, 0.30],
        PharmacophoreKind::Hydrophobic => [0.95, 0.85, 0.25, 0.28],
        PharmacophoreKind::PositiveIonizable => [0.28, 0.45, 1.0, 0.32],
        PharmacophoreKind::NegativeIonizable => [1.0, 0.25, 0.65, 0.32],
        PharmacophoreKind::MetalChelate => [0.88, 0.35, 0.98, 0.34],
        PharmacophoreKind::AnyHeavy => [0.65, 0.90, 0.70, 0.25],
    }
}

fn interaction_rgba(kind: InteractionKind) -> [f32; 4] {
    match kind {
        InteractionKind::HydrogenBond => [0.18, 0.88, 0.96, 0.95],
        InteractionKind::SaltBridge => [0.98, 0.28, 0.78, 0.95],
        InteractionKind::PiPiStacking | InteractionKind::CationPi => [0.98, 0.62, 0.18, 0.95],
        InteractionKind::HalogenBond => [0.24, 0.92, 0.42, 0.95],
        InteractionKind::MetalCoordination => [0.76, 0.36, 0.98, 0.98],
        InteractionKind::WaterBridge => [0.30, 0.62, 1.0, 0.95],
        InteractionKind::HydrophobicContact => [0.85, 0.85, 0.35, 0.75],
    }
}

fn torsion_rgba(st: TorsionStatus) -> [f32; 4] {
    match st {
        TorsionStatus::Green => [0.18, 0.84, 0.44, 0.92],
        TorsionStatus::Yellow => [0.96, 0.78, 0.14, 0.94],
        TorsionStatus::Red => [0.94, 0.24, 0.24, 0.96],
    }
}

/// Build a lossless `DockCraftSceneBundle` (`ViewDockX`) containing receptor PDB, all docked poses,
/// per-atom HYDE coronas, torsion traffic-lights, 3D interactions, pharmacophores, ΔMIF hotspots,
/// bridging waters, and 3D `GridBox`.
#[must_use]
pub fn build_dockcraft_scene_bundle(
    title: &str,
    receptor: &Molecule3D,
    receptor_pdb_path: Option<String>,
    poses: &[DockPoseInput<'_>],
    extras: &BiomolBundleExtras<'_>,
) -> DockCraftSceneBundle {
    let mut pose_entries = Vec::with_capacity(poses.len());
    for p in poses {
        let mut coronas = Vec::new();
        let mut torsion_lights = Vec::new();
        let mut interactions = Vec::new();
        let mut hyde_kcal = None;
        let mut consensus = None;
        let mut est_ki = None;
        let mut le = None;
        let mut pb_valid = None;
        let mut pb_sum = None;
        let mut tors_counts = None;

        if let Some(h) = p.hyde {
            hyde_kcal = Some(h.total_delta_g_kcal);
            consensus = Some(consensus_affinity_kcal(
                p.vina_kcal,
                p.vina_kcal,
                h.total_delta_g_kcal,
            ));
            est_ki = Some(h.estimated_ki_nm);
            le = Some(h.ligand_efficiency);
            if let Some(ref pb) = h.posebusters {
                pb_valid = Some(pb.valid);
                pb_sum = Some(pb.summary.clone());
            }
            for sc in &h.atom_scores {
                if let Some(atom) = p.molecule.atoms.get(sc.atom_index) {
                    coronas.push(HydeAtomCorona {
                        atom_index: sc.atom_index,
                        atom_name: sc.atom_name.clone(),
                        element: sc.element.clone(),
                        coords: atom.coords,
                        delta_g_kcal: sc.delta_g_kcal,
                        corona_radius_a: sc.corona_radius.clamp(0.45, 2.0),
                        corona_rgba: sc.corona_color,
                        diagnostic: sc.diagnostic.clone(),
                    });
                }
            }
            if let Some(ref tr) = h.torsion_report {
                tors_counts = Some([tr.green_count, tr.yellow_count, tr.red_count]);
                for t in &tr.torsions {
                    if let (Some(au), Some(av)) = (
                        p.molecule.atoms.get(t.atom_u),
                        p.molecule.atoms.get(t.atom_v),
                    ) {
                        torsion_lights.push(TorsionBondLight {
                            from: au.coords,
                            to: av.coords,
                            status: format!("{:?}", t.status),
                            pattern: t.pattern.clone(),
                            dihedral_deg: t.dihedral_deg,
                            strain_kcal: t.strain_kcal,
                            rgba: torsion_rgba(t.status),
                        });
                    }
                }
            }
            for inter in &h.interactions {
                if let (Some(&li), Some(&ri)) = (
                    inter.lig_atom_indices.first(),
                    inter.rec_atom_indices.first(),
                ) && let (Some(la), Some(ra)) =
                    (p.molecule.atoms.get(li), receptor.atoms.get(ri))
                {
                    interactions.push(LineOverlay {
                        label: format!(
                            "{:?} {} ({:.2} Å)",
                            inter.kind, inter.rec_residue, inter.distance_a
                        ),
                        from: la.coords,
                        to: ra.coords,
                        dashed: true,
                        rgba: interaction_rgba(inter.kind),
                    });
                }
            }
        }

        pose_entries.push(DockPoseEntry {
            pose_id: p.pose_id,
            name: p.name.clone(),
            canonical_smiles: p.canonical_smiles.clone(),
            sdf_block: write_sdf_3d(&[p.molecule]),
            vina_kcal: p.vina_kcal,
            hyde_kcal,
            consensus_kcal: consensus,
            cnn_score: p.cnn_score,
            estimated_ki_nm: est_ki,
            ligand_efficiency: le,
            mpo_score: p.mpo_score,
            posebusters_valid: pb_valid,
            posebusters_summary: pb_sum,
            torsion_counts: tors_counts,
            hyde_coronas: coronas,
            torsion_lights,
            interaction_lines: interactions,
        });
    }

    // Pharmacophores from explicit constraints + SiteBlueprint
    let mut pharm_spheres = Vec::new();
    let mut pharm_vectors = Vec::new();
    for (ci, c) in extras.constraints.iter().enumerate() {
        if let DockingConstraint::PharmacophoreSphere {
            center,
            radius,
            kind,
            ..
        } = c
        {
            pharm_spheres.push(SphereOverlay {
                label: format!("C{} {:?}", ci + 1, kind),
                category: "pharmacophore".to_string(),
                coords: *center,
                radius_a: *radius,
                rgba: pharm_rgba(*kind),
            });
        }
    }
    if let Some(bp) = extras.site_blueprint {
        if pharm_spheres.is_empty() {
            for f in bp.features.iter().filter(|f| f.enabled) {
                pharm_spheres.push(SphereOverlay {
                    label: format!("{} {:?} ({})", f.id, f.kind, f.key_residue),
                    category: "pharmacophore".to_string(),
                    coords: f.center,
                    radius_a: f.radius_a,
                    rgba: pharm_rgba(f.kind),
                });
            }
        }
        for f in bp.features.iter().filter(|f| f.enabled) {
            if let Some(dir) = f.direction {
                let mut col = pharm_rgba(f.kind);
                col[3] = 0.95;
                pharm_vectors.push(LineOverlay {
                    label: format!("{}_vec", f.id),
                    from: f.center,
                    to: [
                        f.center[0] + dir[0] * 1.85,
                        f.center[1] + dir[1] * 1.85,
                        f.center[2] + dir[2] * 1.85,
                    ],
                    dashed: false,
                    rgba: col,
                });
            }
        }
    }

    // ΔMIF Selectivity Hotspots (`SEL_*`)
    let mut dmif_hotspots = Vec::new();
    if let Some(sel) = extras.selectivity {
        for hs in &sel.selectivity_hotspots {
            let rgba = match hs.kind {
                SelectivityHotspotKind::StericExclusivitySubpocket => [0.16, 0.88, 0.56, 0.42],
                SelectivityHotspotKind::DifferentialPharmacophore(_) => [0.94, 0.28, 0.82, 0.42],
                SelectivityHotspotKind::ConservedAnchor(_) => [0.68, 0.72, 0.78, 0.28],
            };
            dmif_hotspots.push(SphereOverlay {
                label: format!("{} (ΔE={:+.2} kcal)", hs.id, hs.delta_energy_kcal),
                category: "dmif_selectivity".to_string(),
                coords: hs.center,
                radius_a: hs.radius_a.clamp(0.8, 3.0),
                rgba,
            });
        }
    }

    // Structural bridging waters
    let bridging_waters: Vec<SphereOverlay> = extras
        .bridging_waters
        .iter()
        .map(|bw| SphereOverlay {
            label: format!("HOH {} ({} HB)", bw.res_seq, bw.receptor_hbonds),
            category: "bridging_water".to_string(),
            coords: bw.coords,
            radius_a: 0.72,
            rgba: [0.22, 0.64, 0.98, 0.65],
        })
        .collect();

    DockCraftSceneBundle {
        schema_version: "dockcraft-bundle/v1".to_string(),
        title: title.to_string(),
        source_app: format!("dockcraft v{}", env!("CARGO_PKG_VERSION")),
        receptor_name: receptor.name.clone(),
        receptor_pdb_path,
        receptor_pdb_inline: Some(write_pdb(receptor)),
        grid_box: extras.grid_box.map(|g| GridBox3D {
            center: g.center,
            size: g.size,
            spacing: g.spacing,
        }),
        poses: pose_entries,
        pharmacophore_spheres: pharm_spheres,
        pharmacophore_vectors: pharm_vectors,
        selectivity_dmif_hotspots: dmif_hotspots,
        bridging_waters,
        residue_metric_overrides: extras.residue_metric_overrides.clone(),
        trajectory_dcd_path: extras.trajectory_dcd_path.clone(),
        provenance: vec![ProvenanceEntry {
            kind: ProvenanceKind::ComputationalPrediction,
            source_app: "dockcraft".to_string(),
            summary: format!(
                "Exported {} pose(s) + HYDE v2 + overlays for '{}'",
                poses.len(),
                receptor.name
            ),
        }],
    }
}

/// Write a `DockCraftSceneBundle` JSON file into the BiomolCraft handoff directory (or `dir_override`).
pub fn write_biomol_handoff_bundle(
    stem: &str,
    bundle: &DockCraftSceneBundle,
    dir_override: Option<&Path>,
) -> Result<PathBuf, BiomolLinkError> {
    let dir = dir_override.map_or_else(default_biomolcraft_handoff_dir, Path::to_path_buf);
    std::fs::create_dir_all(&dir).map_err(|e| BiomolLinkError::Directory {
        dir: dir.display().to_string(),
        msg: e.to_string(),
    })?;
    let out_path = dir.join(format!("{}.dockbundle.json", sanitize_stem(stem)));
    let text = serde_json::to_string_pretty(bundle)?;
    std::fs::write(&out_path, text.as_bytes())?;
    Ok(out_path)
}

/// Push a `DockCraftSceneBundle` live to a running `BiomolCraft` over Craft Link v1,
/// or cold-start `biomolcraft --dock-bundle <path>` when no live instance is found.
pub fn send_or_launch_biomolcraft(
    stem: &str,
    bundle: &DockCraftSceneBundle,
    dir_override: Option<&Path>,
) -> Result<BiomolDeliveryReport, BiomolLinkError> {
    let bundle_path = write_biomol_handoff_bundle(stem, bundle, dir_override)?;
    if let Ok(live) = discover_craft_instances(None, Some("biomolcraft"))
        && let Some(target) = live.first()
        && push_dockcraft_bundle_via_craft_link(target, bundle).is_ok()
    {
        return Ok(BiomolDeliveryReport {
            bundle_path,
            pushed_live: true,
            launched_new_instance: false,
            pose_count: bundle.poses.len(),
        });
    }

    Command::new("biomolcraft")
        .arg("--dock-bundle")
        .arg(&bundle_path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| BiomolLinkError::Launch {
            bundle_path: bundle_path.display().to_string(),
            msg: e.to_string(),
        })?;
    Ok(BiomolDeliveryReport {
        bundle_path,
        pushed_live: false,
        launched_new_instance: true,
        pose_count: bundle.poses.len(),
    })
}

/// Synchronously fetch the active 3D selection or editable `GridBox` from a live `BiomolCraft`
/// instance over Craft Link v1 (`biomol.get_selection_box`, < 1 ms, zero TCP ports!).
pub fn grid_box_from_biomolcraft_selection(
    padding_a: f64,
) -> Result<BiomolSelectionBox, BiomolLinkError> {
    let snap = discover_and_request_selection_box(None, Some(padding_a))
        .map_err(|e| BiomolLinkError::Link(e.to_string()))?;
    let gb = GridBox::from_center_size(
        snap.grid_box.center,
        snap.grid_box.size,
        snap.grid_box.spacing.max(0.25),
    )
    .map_err(|e| BiomolLinkError::Link(e.to_string()))?;
    Ok(BiomolSelectionBox {
        grid_box: gb,
        source_model: snap.source_model,
        selected_residues: snap.selected_residues,
        heavy_atoms: snap.heavy_atoms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::{
        Atom3D, BondOrder, assign_ad4_types_and_flags, perceive_all, prepare_receptor,
    };

    #[test]
    fn builds_viewdockx_bundle_and_writes_handoff_json() -> Result<(), Box<dyn std::error::Error>> {
        let mut rec = Molecule3D::new("1KE6_test");
        let mut a = Atom3D::new(1, "O", [0.0, 0.0, 0.0]);
        a.chain_id = 'A';
        a.res_name = "LEU".to_string();
        a.res_seq = 83;
        a.name = "O".to_string();
        rec.add_atom(a);
        prepare_receptor(&mut rec, false);

        let mut lig = Molecule3D::new("CDK2_Inhibitor");
        let id0 = lig.add_atom(Atom3D::new(1, "C", [2.5, 0.0, 0.0]));
        let id1 = lig.add_atom(Atom3D::new(2, "N", [3.9, 0.0, 0.0]));
        let _ = lig.add_bond(id0, id1, BondOrder::Single);
        assign_ad4_types_and_flags(&mut lig);
        perceive_all(&mut lig);

        let hyde = hyde_scorer::score_pose_hyde(&rec, &lig)?;
        let gb = GridBox::from_center_size([2.0, 0.0, 0.0], [18.0, 18.0, 18.0], 0.375)?;
        let bundle = build_dockcraft_scene_bundle(
            "CDK2 Pilot",
            &rec,
            None,
            &[DockPoseInput {
                pose_id: 1,
                name: "Mode_1".to_string(),
                canonical_smiles: Some("CN".to_string()),
                molecule: &lig,
                vina_kcal: -8.6,
                hyde: Some(&hyde),
                cnn_score: None,
                mpo_score: Some(4.9),
            }],
            &BiomolBundleExtras {
                grid_box: Some(&gb),
                ..Default::default()
            },
        );
        assert_eq!(bundle.poses.len(), 1);
        assert_eq!(bundle.poses[0].hyde_coronas.len(), 2);
        assert!(bundle.grid_box.is_some());

        let tmp = std::env::temp_dir().join(format!("dock_biomol_test_{}", std::process::id()));
        let out = write_biomol_handoff_bundle("cdk2_pilot", &bundle, Some(&tmp))?;
        let raw = std::fs::read_to_string(&out)?;
        let rt: DockCraftSceneBundle = serde_json::from_str(&raw)?;
        assert_eq!(rt.title, "CDK2 Pilot");
        assert_eq!(rt.poses.len(), 1);
        let _ = std::fs::remove_dir_all(&tmp);
        Ok(())
    }
}

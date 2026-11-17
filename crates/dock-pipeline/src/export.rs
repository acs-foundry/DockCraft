#![forbid(unsafe_code)]

use std::path::Path;

use hyde_scorer::HydeReport;
use serde::{Deserialize, Serialize};
use struct_core::{Molecule3D, write_pdb, write_sdf_3d};

use crate::sandbox::{PathSandbox, SandboxError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportAtomAnnotation {
    pub atom_index: usize,
    pub element: String,
    pub delta_g_kcal: f64,
    pub corona_color: [f32; 4],
    pub diagnostic: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteropSceneBundle {
    pub format_target: String,
    pub complex_name: String,
    pub receptor_pdb: String,
    pub ligand_sdf: String,
    pub total_delta_g_kcal: f64,
    pub estimated_ki_nm: f64,
    pub atom_annotations: Vec<ExportAtomAnnotation>,
    pub interactions: Vec<hyde_scorer::Interaction>,
}

#[must_use]
pub fn build_molcraft_export(
    complex_name: &str,
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    hyde: &HydeReport,
) -> (String, InteropSceneBundle) {
    let mut pml = String::new();
    pml.push_str(&format!(
        "# DockCraft -> MolCraft export ({complex_name})\n"
    ));
    pml.push_str("load receptor.pdb, receptor\n");
    pml.push_str("load ligand.sdf, ligand\n");
    pml.push_str("show cartoon, receptor\n");
    pml.push_str("show sticks, ligand\n");
    for sc in &hyde.atom_scores {
        let hex = format!(
            "0x{:02X}{:02X}{:02X}",
            (sc.corona_color[0].clamp(0.0, 1.0) * 255.0) as u8,
            (sc.corona_color[1].clamp(0.0, 1.0) * 255.0) as u8,
            (sc.corona_color[2].clamp(0.0, 1.0) * 255.0) as u8,
        );
        pml.push_str(&format!(
            "color {hex}, ligand and id {}\n",
            sc.atom_index + 1
        ));
    }
    for (idx, inter) in hyde.interactions.iter().enumerate() {
        if let (Some(&li), Some(&ri)) = (
            inter.lig_atom_indices.first(),
            inter.rec_atom_indices.first(),
        ) {
            pml.push_str(&format!(
                "distance inter_{}, ligand and id {}, receptor and id {}\n",
                idx + 1,
                li + 1,
                ri + 1
            ));
        }
    }
    pml.push_str("zoom ligand, 6.0\n");

    let bundle = make_bundle("MolCraft", complex_name, receptor, ligand, hyde);
    (pml, bundle)
}

#[must_use]
pub fn build_chymera_export(
    complex_name: &str,
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    hyde: &HydeReport,
) -> (String, InteropSceneBundle) {
    let mut cxc = String::new();
    cxc.push_str(&format!(
        "# DockCraft -> ChimeraX/Chymera (.cxc) script ({complex_name})\n"
    ));
    cxc.push_str("open receptor.pdb\n");
    cxc.push_str("open ligand.sdf\n");
    cxc.push_str("cartoon #1\n");
    cxc.push_str("style #2 stick\n");
    for sc in &hyde.atom_scores {
        let r = (sc.corona_color[0].clamp(0.0, 1.0) * 100.0) as u8;
        let g = (sc.corona_color[1].clamp(0.0, 1.0) * 100.0) as u8;
        let b = (sc.corona_color[2].clamp(0.0, 1.0) * 100.0) as u8;
        cxc.push_str(&format!(
            "color #2@@serial_number={} rgb({r},{g},{b})\n",
            sc.atom_index + 1
        ));
    }
    for inter in &hyde.interactions {
        if let (Some(&li), Some(&ri)) = (
            inter.lig_atom_indices.first(),
            inter.rec_atom_indices.first(),
        ) {
            cxc.push_str(&format!(
                "distance #2@@serial_number={} #1@@serial_number={}\n",
                li + 1,
                ri + 1
            ));
        }
    }
    cxc.push_str("view #2\n");

    let bundle = make_bundle("Chymera", complex_name, receptor, ligand, hyde);
    (cxc, bundle)
}

pub struct SceneExportRequest<'a> {
    pub target: &'a str,
    pub complex_name: &'a str,
    pub out_script_path: &'a Path,
    pub out_bundle_path: Option<&'a Path>,
}

/// Writes the full self-contained scene directory (`script`, `receptor.pdb`, `ligand.sdf`, optional JSON bundle)
/// through the `openat2` sandbox so ChimeraX / MolCraft never encounter missing `receptor.pdb` / `ligand.sdf`.
pub fn write_sandboxed_scene_package(
    sandbox: &PathSandbox,
    req: &SceneExportRequest<'_>,
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    hyde: &HydeReport,
) -> Result<Vec<String>, SandboxError> {
    let (script_text, bundle) = if req.target == "chymera" {
        build_chymera_export(req.complex_name, receptor, ligand, hyde)
    } else {
        build_molcraft_export(req.complex_name, receptor, ligand, hyde)
    };

    let parent_dir = req
        .out_script_path
        .parent()
        .unwrap_or_else(|| Path::new(""));
    let rec_rel = if parent_dir.as_os_str().is_empty() {
        Path::new("receptor.pdb").to_path_buf()
    } else {
        parent_dir.join("receptor.pdb")
    };
    let lig_rel = if parent_dir.as_os_str().is_empty() {
        Path::new("ligand.sdf").to_path_buf()
    } else {
        parent_dir.join("ligand.sdf")
    };

    sandbox.write_string(&rec_rel, &bundle.receptor_pdb)?;
    sandbox.write_string(&lig_rel, &bundle.ligand_sdf)?;
    sandbox.write_string(req.out_script_path, &script_text)?;

    let mut written = vec![
        rec_rel.display().to_string(),
        lig_rel.display().to_string(),
        req.out_script_path.display().to_string(),
    ];
    if let Some(bp) = req.out_bundle_path {
        let js = serde_json::to_string_pretty(&bundle)
            .map_err(|e| SandboxError::RejectedPath(format!("bundle serialize error: {e}")))?;
        sandbox.write_string(bp, &js)?;
        written.push(bp.display().to_string());
    }
    Ok(written)
}

fn make_bundle(
    format_target: &str,
    complex_name: &str,
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    hyde: &HydeReport,
) -> InteropSceneBundle {
    let annotations = hyde
        .atom_scores
        .iter()
        .map(|s| ExportAtomAnnotation {
            atom_index: s.atom_index,
            element: s.element.clone(),
            delta_g_kcal: s.delta_g_kcal,
            corona_color: s.corona_color,
            diagnostic: s.diagnostic.clone(),
        })
        .collect();
    InteropSceneBundle {
        format_target: format_target.to_string(),
        complex_name: complex_name.to_string(),
        receptor_pdb: write_pdb(receptor),
        ligand_sdf: write_sdf_3d(&[ligand]),
        total_delta_g_kcal: hyde.total_delta_g_kcal,
        estimated_ki_nm: hyde.estimated_ki_nm,
        atom_annotations: annotations,
        interactions: hyde.interactions.clone(),
    }
}

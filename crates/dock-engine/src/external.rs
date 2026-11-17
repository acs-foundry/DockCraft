#![forbid(unsafe_code)]

use std::path::Path;
use std::process::Command;

use pocket_detect::GridBox;
use serde::{Deserialize, Serialize};
use struct_core::{Molecule3D, parse_pdbqt, write_pdbqt_ligand, write_pdbqt_receptor};

use crate::DockError;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExternalEngineKind {
    Vina,
    Smina,
    Gnina,
}

impl ExternalEngineKind {
    #[must_use]
    pub fn binary_name(&self) -> &'static str {
        match self {
            Self::Vina => "vina",
            Self::Smina => "smina",
            Self::Gnina => "gnina",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalPoseResult {
    pub mode: usize,
    pub affinity_kcal_mol: f64,
    pub molecule: Molecule3D,
}

#[must_use]
pub fn detect_available_external_engines() -> Vec<ExternalEngineKind> {
    [
        ExternalEngineKind::Vina,
        ExternalEngineKind::Smina,
        ExternalEngineKind::Gnina,
    ]
    .into_iter()
    .filter(|kind| {
        Command::new(kind.binary_name())
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
    .collect()
}

pub fn run_external_docking(
    kind: ExternalEngineKind,
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    grid_box: &GridBox,
    exhaustiveness: usize,
    num_modes: usize,
    work_dir: &Path,
) -> Result<Vec<ExternalPoseResult>, DockError> {
    let rec_path = work_dir.join("receptor.pdbqt");
    let lig_path = work_dir.join("ligand.pdbqt");
    let out_path = work_dir.join("out.pdbqt");

    std::fs::write(&rec_path, write_pdbqt_receptor(receptor))
        .map_err(|e| DockError::ExternalEngine(e.to_string()))?;
    std::fs::write(&lig_path, write_pdbqt_ligand(ligand))
        .map_err(|e| DockError::ExternalEngine(e.to_string()))?;

    let output = Command::new(kind.binary_name())
        .arg("--receptor")
        .arg(&rec_path)
        .arg("--ligand")
        .arg(&lig_path)
        .arg("--center_x")
        .arg(format!("{:.3}", grid_box.center[0]))
        .arg("--center_y")
        .arg(format!("{:.3}", grid_box.center[1]))
        .arg("--center_z")
        .arg(format!("{:.3}", grid_box.center[2]))
        .arg("--size_x")
        .arg(format!("{:.3}", grid_box.size[0]))
        .arg("--size_y")
        .arg(format!("{:.3}", grid_box.size[1]))
        .arg("--size_z")
        .arg(format!("{:.3}", grid_box.size[2]))
        .arg("--exhaustiveness")
        .arg(exhaustiveness.to_string())
        .arg("--num_modes")
        .arg(num_modes.to_string())
        .arg("--out")
        .arg(&out_path)
        .output()
        .map_err(|e| {
            DockError::ExternalEngine(format!("failed to spawn {}: {e}", kind.binary_name()))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(DockError::ExternalEngine(format!(
            "{} exited {}: {}",
            kind.binary_name(),
            output.status,
            stderr.trim()
        )));
    }

    let raw_out =
        std::fs::read_to_string(&out_path).map_err(|e| DockError::ExternalEngine(e.to_string()))?;
    parse_external_pdbqt_models(&raw_out)
}

pub(crate) fn parse_external_pdbqt_models(
    pdbqt_text: &str,
) -> Result<Vec<ExternalPoseResult>, DockError> {
    let mut results = Vec::new();
    let mut current_block = String::new();
    let mut current_aff: Option<f64> = None;

    for line in pdbqt_text.lines() {
        if (line.starts_with("REMARK VINA RESULT:") || line.starts_with("REMARK  VINA RESULT:"))
            && let Some(tok) = line.split_whitespace().nth(3)
        {
            current_aff = tok.parse::<f64>().ok();
        }
        current_block.push_str(line);
        current_block.push('\n');

        if line.starts_with("ENDMDL") {
            let mut mol = parse_pdbqt(&current_block)?;
            let aff = current_aff.unwrap_or(0.0);
            mol.properties
                .insert("external_affinity_kcal".to_string(), format!("{aff:.3}"));
            results.push(ExternalPoseResult {
                mode: results.len() + 1,
                affinity_kcal_mol: aff,
                molecule: mol,
            });
            current_block.clear();
            current_aff = None;
        }
    }

    if results.is_empty() && !current_block.trim().is_empty() {
        let mol = parse_pdbqt(&current_block)?;
        results.push(ExternalPoseResult {
            mode: 1,
            affinity_kcal_mol: current_aff.unwrap_or(0.0),
            molecule: mol,
        });
    }

    Ok(results)
}

//! The return leg: docking results in a form ChemCraft can fold back into its SAR table.
//!
//! DockCraft deliberately does **not** write a `.chemcraft` project. Doing so would mean
//! fabricating `MolGraph` and `MorganFp`, which are ChemCraft's internal types, and the two
//! schemas would drift apart at the first change on either side. Each tool writes only its own
//! format; the merge belongs to whoever owns the document, which for a `.chemcraft` project is
//! ChemCraft.
//!
//! So the results travel as a small, versioned, self-describing JSON sidecar keyed by the
//! structure's canonical SMILES **as ChemCraft spelled it**, with the SAR row id as a tiebreaker.
//! Poses ride along as 3D SDF files next to it, referenced by relative path so the whole directory
//! can be moved or copied without breaking.
//!
//! The directory mirrors the convention ChemCraft already established for the other direction
//! (`$XDG_DATA_HOME/DockCraft/chemcraft_handoff`), so the pair reads symmetrically:
//!
//! ```text
//! ChemCraft -> DockCraft   $XDG_DATA_HOME/DockCraft/chemcraft_handoff
//! DockCraft -> ChemCraft   $XDG_DATA_HOME/ChemCraft/dockcraft_results
//! ```

use std::path::{Path, PathBuf};

use pocket_detect::GridBox;
use serde::{Deserialize, Serialize};
use struct_core::write_sdf_3d;
use thiserror::Error;

use crate::campaign::{CampaignReport, ScreenedCandidate};

/// Bumped when a consumer would have to change to read the file correctly.
pub const SIDECAR_FORMAT: &str = "dockcraft-results";
pub const SIDECAR_VERSION: &str = "1";

#[derive(Debug, Error)]
pub enum SidecarError {
    #[error("could not prepare results directory '{dir}': {msg}")]
    Directory { dir: String, msg: String },
    #[error("could not write '{path}': {msg}")]
    Write { path: String, msg: String },
    #[error("serialisation failed: {0}")]
    Serialize(String),
}

/// How the run was set up, so a reader can tell two runs apart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarRun {
    pub receptor_name: String,
    pub scoring: String,
    pub exhaustiveness: usize,
    pub box_center: [f64; 3],
    pub box_size: [f64; 3],
    pub dockcraft_version: String,
    pub elapsed_ms: u128,
}

/// One compound's results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarRow {
    /// Canonical SMILES exactly as the source library spelled it. The join key.
    pub canonical_smiles: String,
    /// SAR row id from the source `.chemcraft`, when there was one.
    pub source_row_id: Option<u64>,
    pub name: String,
    pub rank: usize,
    pub affinity_kcal: f64,
    pub hyde_delta_g_kcal: f64,
    pub estimated_ki_nm: f64,
    pub ligand_efficiency: f64,
    pub mpo_score: f64,
    /// Measured potency carried in from the source library, echoed back untouched so a reader can
    /// line computed against measured without a second lookup.
    pub measured_pic50: Option<f64>,
    /// Path to the docked pose, relative to the sidecar file.
    pub pose_sdf: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultsSidecar {
    pub format: String,
    pub version: String,
    pub run: SidecarRun,
    pub rows: Vec<SidecarRow>,
}

/// Where DockCraft leaves results for ChemCraft, mirroring ChemCraft's own handoff directory.
#[must_use]
pub fn default_chemcraft_results_dir() -> PathBuf {
    if let Ok(xdg_data) = std::env::var("XDG_DATA_HOME")
        && !xdg_data.trim().is_empty()
    {
        return PathBuf::from(xdg_data)
            .join("ChemCraft")
            .join("dockcraft_results");
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("ChemCraft")
            .join("dockcraft_results");
    }
    PathBuf::from("dockcraft_results")
}

fn sanitize(raw: &str) -> String {
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
        "compound".to_string()
    } else {
        trimmed
    }
}

fn row_from(candidate: &ScreenedCandidate, pose_sdf: String) -> SidecarRow {
    SidecarRow {
        canonical_smiles: candidate
            .source_properties
            .get("ChemCraft_canonical_smiles")
            .cloned()
            .unwrap_or_default(),
        source_row_id: candidate
            .source_properties
            .get("ChemCraft_row_id")
            .and_then(|v| v.trim().parse::<u64>().ok()),
        name: candidate.name.clone(),
        rank: candidate.rank,
        affinity_kcal: candidate.best_affinity_kcal,
        hyde_delta_g_kcal: candidate.hyde_delta_g_kcal,
        estimated_ki_nm: candidate.estimated_ki_nm,
        ligand_efficiency: candidate.ligand_efficiency,
        mpo_score: candidate.mpo_score,
        measured_pic50: candidate.measured_pic50,
        pose_sdf,
    }
}

/// Writes `<dir>/<stem>.dockcraft.json` plus `<dir>/<stem>_poses/*.sdf`.
///
/// Returns the path of the sidecar itself.
pub fn write_results_sidecar(
    report: &CampaignReport,
    receptor_name: &str,
    grid_box: &GridBox,
    scoring: &str,
    exhaustiveness: usize,
    dir: &Path,
    stem: &str,
) -> Result<PathBuf, SidecarError> {
    let stem = sanitize(stem);
    let poses_dir_name = format!("{stem}_poses");
    let poses_dir = dir.join(&poses_dir_name);
    std::fs::create_dir_all(&poses_dir).map_err(|e| SidecarError::Directory {
        dir: poses_dir.display().to_string(),
        msg: e.to_string(),
    })?;

    let mut rows = Vec::with_capacity(report.results.len());
    for candidate in &report.results {
        let file = format!("{}_{}.sdf", candidate.rank, sanitize(&candidate.name));
        let path = poses_dir.join(&file);
        std::fs::write(
            &path,
            write_sdf_3d(&[&candidate.best_pose.molecule]).as_bytes(),
        )
        .map_err(|e| SidecarError::Write {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;
        rows.push(row_from(candidate, format!("{poses_dir_name}/{file}")));
    }

    let sidecar = ResultsSidecar {
        format: SIDECAR_FORMAT.to_string(),
        version: SIDECAR_VERSION.to_string(),
        run: SidecarRun {
            receptor_name: receptor_name.to_string(),
            scoring: scoring.to_string(),
            exhaustiveness,
            box_center: grid_box.center,
            box_size: grid_box.size,
            dockcraft_version: env!("CARGO_PKG_VERSION").to_string(),
            elapsed_ms: report.elapsed_ms,
        },
        rows,
    };

    let path = dir.join(format!("{stem}.dockcraft.json"));
    let text = serde_json::to_string_pretty(&sidecar)
        .map_err(|e| SidecarError::Serialize(e.to_string()))?;
    std::fs::write(&path, text.as_bytes()).map_err(|e| SidecarError::Write {
        path: path.display().to_string(),
        msg: e.to_string(),
    })?;
    Ok(path)
}

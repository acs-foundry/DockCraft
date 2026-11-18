//! UCSF ChimeraX interop: write scene bundles where ChimeraX is actually allowed to read them.
//!
//! ChimeraX runs here as a Flatpak started with `--sandbox` and an explicit allow-list
//! (`common/vault/claude/chimerax/chimerax.sh`): only its own `xdg-{config,cache,data}/ChimeraX`
//! directories, `~/Downloads` read-only, `~/Downloads/ChimeraX`, and
//! `~/Projects/research/data/outputs`. A `.cxc` written next to the DockCraft project, which is
//! where `export_complex` put it, sits outside every one of those, so ChimeraX could not open it
//! however correct the script was. ChemCraft already solved this with
//! `$XDG_DATA_HOME/ChimeraX/chemcraft_export`; this is the same move for DockCraft.
//!
//! Each complex gets its own directory, and the `PathSandbox` is rooted at that directory rather
//! than widened from the project root, so the second write root is confined exactly like the
//! first: one `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` root per scene, nothing above it
//! reachable.

use std::path::{Path, PathBuf};

use hyde_scorer::HydeReport;
use struct_core::Molecule3D;
use thiserror::Error;

use crate::export::{SceneExportRequest, write_sandboxed_scene_package};
use crate::sandbox::{PathSandbox, SandboxError};

/// Read cap for the per-scene sandbox. Scene files are small; this is a backstop, not a budget.
const SCENE_READ_CAP_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum ChimeraXError {
    #[error("could not prepare ChimeraX export directory '{dir}': {msg}")]
    Directory { dir: String, msg: String },
    #[error("sandboxed write failed: {0}")]
    Sandbox(#[from] SandboxError),
    #[error("could not launch `chimerax` (scene written to {scene_dir}): {msg}")]
    Launch { scene_dir: String, msg: String },
}

/// Where a scene was written, and the command that makes ChimeraX show it.
#[derive(Debug, Clone)]
pub struct ChimeraXSceneBundle {
    pub scene_dir: PathBuf,
    pub script_path: PathBuf,
    pub receptor_path: PathBuf,
    pub ligand_path: PathBuf,
    pub bundle_path: PathBuf,
    /// A complete ChimeraX command, built here from a fixed vocabulary. It is never assembled
    /// from caller-supplied text: a ChimeraX command can run Python through `runscript`, so a
    /// pass-through command string would turn any remote caller into remote code execution.
    pub open_command: String,
}

/// Directory ChimeraX is allowed to read, mirroring ChemCraft's `chemcraft_export`.
#[must_use]
pub fn default_chimerax_export_dir() -> PathBuf {
    if let Ok(xdg_data) = std::env::var("XDG_DATA_HOME")
        && !xdg_data.trim().is_empty()
    {
        return PathBuf::from(xdg_data)
            .join("ChimeraX")
            .join("dockcraft_export");
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("ChimeraX")
            .join("dockcraft_export");
    }
    PathBuf::from("dockcraft_chimerax_export")
}

/// Is a `chimerax` launcher on `$PATH`?
#[must_use]
pub fn is_chimerax_available() -> bool {
    let Ok(path_var) = std::env::var("PATH") else {
        return false;
    };
    std::env::split_paths(&path_var).any(|dir| dir.join("chimerax").is_file())
}

/// Reduces a complex name to a safe single directory component.
#[must_use]
pub fn sanitize_stem(raw: &str) -> String {
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
        "dockcraft_scene".to_string()
    } else {
        trimmed
    }
}

/// Writes a self-contained ChimeraX scene (`scene.cxc` + `receptor.pdb` + `ligand.sdf` +
/// `scene.chimerax.json`) into a per-complex directory under an export root ChimeraX can read.
pub fn write_chimerax_session_bundle(
    complex_name: &str,
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    hyde: &HydeReport,
    dir_override: Option<&Path>,
) -> Result<ChimeraXSceneBundle, ChimeraXError> {
    let base = dir_override.map_or_else(default_chimerax_export_dir, Path::to_path_buf);
    let scene_dir = base.join(sanitize_stem(complex_name));
    std::fs::create_dir_all(&scene_dir).map_err(|e| ChimeraXError::Directory {
        dir: scene_dir.display().to_string(),
        msg: e.to_string(),
    })?;
    let scene_dir = scene_dir
        .canonicalize()
        .map_err(|e| ChimeraXError::Directory {
            dir: scene_dir.display().to_string(),
            msg: e.to_string(),
        })?;

    // Second sandbox root, as tightly confined as the project one.
    let sandbox = PathSandbox::new(
        Some(scene_dir.clone()),
        Some(scene_dir.clone()),
        SCENE_READ_CAP_BYTES,
    )?;

    write_sandboxed_scene_package(
        &sandbox,
        &SceneExportRequest {
            target: "chimerax",
            complex_name,
            out_script_path: Path::new("scene.cxc"),
            out_bundle_path: Some(Path::new("scene.chimerax.json")),
        },
        receptor,
        ligand,
        hyde,
    )?;

    let script_path = scene_dir.join("scene.cxc");
    Ok(ChimeraXSceneBundle {
        open_command: format!("open {}", script_path.display()),
        script_path,
        receptor_path: scene_dir.join("receptor.pdb"),
        ligand_path: scene_dir.join("ligand.sdf"),
        bundle_path: scene_dir.join("scene.chimerax.json"),
        scene_dir,
    })
}

/// Writes the scene and spawns `chimerax` on it. Prefer the REST channel when a session is
/// already open; this is the cold-start path.
pub fn launch_chimerax_for_scene(
    complex_name: &str,
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    hyde: &HydeReport,
    dir_override: Option<&Path>,
) -> Result<ChimeraXSceneBundle, ChimeraXError> {
    let bundle = write_chimerax_session_bundle(complex_name, receptor, ligand, hyde, dir_override)?;
    std::process::Command::new("chimerax")
        .arg(&bundle.script_path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| ChimeraXError::Launch {
            scene_dir: bundle.scene_dir.display().to_string(),
            msg: e.to_string(),
        })?;
    Ok(bundle)
}

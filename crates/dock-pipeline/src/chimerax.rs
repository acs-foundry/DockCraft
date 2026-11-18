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
    #[error("live ChimeraX channel: {0}")]
    Rest(String),
    #[error("nothing is selected in ChimeraX; pick the binding-site residues first")]
    EmptySelection,
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

/// A docking box derived from what the person has selected in ChimeraX.
#[derive(Debug, Clone)]
pub struct SelectionBox {
    pub grid_box: pocket_detect::GridBox,
    /// Residues ChimeraX reported as selected, as `A:189 ASP`.
    pub residues: Vec<String>,
    pub heavy_atoms: usize,
    /// Where the selection's PDB was written, kept for inspection.
    pub selection_pdb: PathBuf,
}

/// Builds a docking box from the current ChimeraX selection.
///
/// This is the answer to the pocket detector picking the wrong site: it ranks cavities by
/// druggability, and on 3ERT the best-scoring one sits 22 A from the estrogen site. A person
/// looking at the structure already knows where the site is, so let them point at it.
///
/// The selection is pulled across as a PDB written by ChimeraX into the export root and parsed
/// here, rather than scraped out of `info selection` text: the parser already exists and is
/// tested, and a format is a better interface than a log.
///
/// `padding` is added around the selected atoms' bounding box. Around a binding-site residue
/// shell, 4-5 A is a reasonable starting point.
pub fn grid_box_from_chimerax_selection(
    client: &crate::chimerax_rest::ChimeraXRest,
    scene_dir: &Path,
    padding: f64,
) -> Result<SelectionBox, ChimeraXError> {
    use crate::chimerax_rest::ChimeraXCommand;

    let selection_pdb = scene_dir.join("selection.pdb");
    let _ = std::fs::remove_file(&selection_pdb);

    let info = client
        .send(&ChimeraXCommand::InfoSelection)
        .map_err(|e| ChimeraXError::Rest(e.to_string()))?;
    let mut residues: Vec<String> = info
        .notes
        .iter()
        .flat_map(|note| note.lines())
        .filter_map(parse_residue_line)
        .collect();
    residues.sort_unstable();
    residues.dedup();
    if residues.is_empty() {
        return Err(ChimeraXError::EmptySelection);
    }

    client
        .send(&ChimeraXCommand::SaveSelectionPdb(selection_pdb.clone()))
        .map_err(|e| ChimeraXError::Rest(e.to_string()))?;

    let text = std::fs::read_to_string(&selection_pdb).map_err(|e| ChimeraXError::Directory {
        dir: selection_pdb.display().to_string(),
        msg: e.to_string(),
    })?;
    let selection = struct_core::parse_pdb(&text, true)
        .map_err(|e| ChimeraXError::Rest(format!("could not parse the selection PDB: {e}")))?;

    let heavy: Vec<[f64; 3]> = selection
        .atoms
        .iter()
        .filter(|a| a.is_heavy())
        .map(|a| a.coords)
        .collect();
    if heavy.is_empty() {
        return Err(ChimeraXError::EmptySelection);
    }

    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in &heavy {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let center = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let size = [
        (hi[0] - lo[0]) + 2.0 * padding,
        (hi[1] - lo[1]) + 2.0 * padding,
        (hi[2] - lo[2]) + 2.0 * padding,
    ];
    let grid_box = pocket_detect::GridBox::from_center_size(center, size, 0.375)
        .map_err(|e| ChimeraXError::Rest(e.to_string()))?;

    Ok(SelectionBox {
        grid_box,
        residues,
        heavy_atoms: heavy.len(),
        selection_pdb,
    })
}

/// Pulls `A:189 ASP` out of a ChimeraX `info selection level residue` line, which reads
/// `residue id #1/A:189 name ASP index 170`.
fn parse_residue_line(line: &str) -> Option<String> {
    let rest = line.trim().strip_prefix("residue id ")?;
    let mut parts = rest.split_whitespace();
    let spec = parts.next()?;
    let chain_res = spec.split('/').nth(1)?;
    let name = rest.split(" name ").nth(1)?.split_whitespace().next()?;
    Some(format!("{chain_res} {name}"))
}

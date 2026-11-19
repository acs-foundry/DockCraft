#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use dock_engine::{DockingConstraint, ScoringFunction};
use pocket_detect::GridBox;
use serde::{Deserialize, Serialize};
use struct_core::Molecule3D;

/// Persistent DockCraft project file (`.dockcraft`, JSON format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockCraftProject {
    pub format: String,
    pub version: String,
    pub receptor: Option<Molecule3D>,
    pub ligand: Option<Molecule3D>,
    pub grid_box: GridBox,
    pub scoring: ScoringFunction,
    pub exhaustiveness: usize,
    pub metal_coordination: bool,
    pub hyde_rerank: bool,
    pub constraints: Vec<DockingConstraint>,
    pub flex_residues: Vec<String>,
    pub kept_water_resseqs: Vec<i32>,
}

/// Launch native Linux open-file dialog (`zenity` or `kdialog`).
pub fn pick_open_file(title: &str, filter_label: &str, filter_glob: &str) -> Option<String> {
    let title_arg = format!("--title={title}");
    let filter_arg = format!("--file-filter={filter_label} ({filter_glob}) | {filter_glob}");
    if let Ok(out) = ProcessCommand::new("zenity")
        .args([
            "--file-selection",
            &title_arg,
            &filter_arg,
            "--file-filter=All files | *",
        ])
        .output()
        && out.status.success()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    let kd_filter = format!("{filter_glob}|{filter_label}");
    if let Ok(out) = ProcessCommand::new("kdialog")
        .args(["--getopenfilename", ".", &kd_filter])
        .output()
        && out.status.success()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    None
}

/// Launch native Linux save-file dialog (`zenity` or `kdialog`).
pub fn pick_save_file(
    title: &str,
    default_path: &str,
    filter_label: &str,
    filter_glob: &str,
) -> Option<String> {
    let title_arg = format!("--title={title}");
    let fname_arg = format!("--filename={default_path}");
    let filter_arg = format!("--file-filter={filter_label} ({filter_glob}) | {filter_glob}");
    if let Ok(out) = ProcessCommand::new("zenity")
        .args([
            "--file-selection",
            "--save",
            "--confirm-overwrite",
            &title_arg,
            &fname_arg,
            &filter_arg,
            "--file-filter=All files | *",
        ])
        .output()
        && out.status.success()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    let kd_filter = format!("{filter_glob}|{filter_label}");
    if let Ok(out) = ProcessCommand::new("kdialog")
        .args(["--getsavefilename", default_path, &kd_filter])
        .output()
        && out.status.success()
    {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    None
}

fn find_executable(name: &str) -> Option<PathBuf> {
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let cand = dir.join(name);
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let local = PathBuf::from(home).join(".local").join("bin").join(name);
        if local.is_file() {
            return Some(local);
        }
    }
    None
}

fn xdg_data_root() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME")
        && !xdg.trim().is_empty()
    {
        return PathBuf::from(xdg);
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.trim().is_empty()
    {
        return PathBuf::from(home).join(".local").join("share");
    }
    PathBuf::from(".")
}

/// Write a handoff `.smi` or `.sdf` file and launch `admetcraft` GUI on demand.
pub fn launch_admetcraft_handoff(
    entries: &[(String, String)], // (smiles_or_fallback, name)
) -> Result<String, String> {
    if entries.is_empty() {
        return Err("No ligand available to send to AdmetCraft.".to_string());
    }
    let dir = xdg_data_root().join("AdmetCraft").join("dockcraft_handoff");
    std::fs::create_dir_all(&dir).map_err(|e| format!("Failed creating handoff dir: {e}"))?;
    let smi_path = dir.join("dockcraft_handoff.smi");
    let mut buf = String::new();
    for (smi, name) in entries {
        if !smi.trim().is_empty() {
            buf.push_str(&format!("{} {}\n", smi.trim(), name.replace(' ', "_")));
        }
    }
    if buf.is_empty() {
        return Err(
            "Current ligand has no SMILES representation for AdmetCraft handoff.".to_string(),
        );
    }
    std::fs::write(&smi_path, buf).map_err(|e| format!("Failed writing handoff SMILES: {e}"))?;

    if let Some(bin) = find_executable("admetcraft") {
        ProcessCommand::new(bin)
            .arg(&smi_path)
            .spawn()
            .map_err(|e| {
                format!(
                    "Saved {}, failed launching admetcraft: {e}",
                    smi_path.display()
                )
            })?;
        Ok(format!("Launched AdmetCraft with {}", smi_path.display()))
    } else {
        Ok(format!(
            "Saved AdmetCraft handoff at {} (install `admetcraft` to auto-launch)",
            smi_path.display()
        ))
    }
}

/// Write a 3D SDF or results sidecar for ChemCraft and launch `chemcraft` GUI on demand.
pub fn launch_chemcraft_with_path(path: &Path) -> Result<String, String> {
    if let Some(bin) = find_executable("chemcraft") {
        ProcessCommand::new(bin)
            .arg(path)
            .spawn()
            .map_err(|e| format!("Saved {}, failed launching chemcraft: {e}", path.display()))?;
        Ok(format!("Launched ChemCraft with {}", path.display()))
    } else {
        Ok(format!(
            "Saved ChemCraft handoff at {} (install `chemcraft` to auto-launch)",
            path.display()
        ))
    }
}

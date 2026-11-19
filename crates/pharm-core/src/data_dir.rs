#![forbid(unsafe_code)]

use std::path::PathBuf;

/// Resolves the centralized CraftBio data directory (`CRAFT_DATA_DIR`).
///
/// Priority order:
/// 1. `CRAFT_DATA_DIR` environment variable if set and non-empty.
/// 2. `$HOME/Datos/CraftBio` when `$HOME/Datos` exists (protects small root `/` partitions).
/// 3. `$XDG_DATA_HOME/CraftBio` or `$HOME/.local/share/CraftBio`.
#[must_use]
pub fn craft_data_dir() -> PathBuf {
    if let Ok(explicit) = std::env::var("CRAFT_DATA_DIR") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let trimmed = home.trim();
        if !trimmed.is_empty() {
            let datos = PathBuf::from(trimmed).join("Datos");
            if datos.is_dir() {
                return datos.join("CraftBio");
            }
        }
    }
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        let trimmed = xdg.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("CraftBio");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let trimmed = home.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed)
                .join(".local")
                .join("share")
                .join("CraftBio");
        }
    }
    PathBuf::from("CraftBio")
}

#[must_use]
pub fn libraries_dir() -> PathBuf {
    craft_data_dir().join("libraries")
}

#[must_use]
pub fn models_dir() -> PathBuf {
    craft_data_dir().join("models")
}

#[must_use]
pub fn building_blocks_dir() -> PathBuf {
    craft_data_dir().join("building_blocks")
}

#[must_use]
pub fn envs_dir() -> PathBuf {
    craft_data_dir().join("envs")
}

#[must_use]
pub fn campaigns_dir() -> PathBuf {
    craft_data_dir().join("campaigns")
}

/// Ensures the standard CraftBio directory layout exists on disk.
pub fn ensure_craft_data_layout() -> std::io::Result<PathBuf> {
    let root = craft_data_dir();
    for sub in [
        "libraries",
        "models",
        "building_blocks",
        "envs",
        "campaigns",
    ] {
        std::fs::create_dir_all(root.join(sub))?;
    }
    Ok(root)
}

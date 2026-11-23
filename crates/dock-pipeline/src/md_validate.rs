//! Opt-in OpenMM GPU Molecular Dynamics validation bridge (`md-validate`) for `DockCraft`.
//!
//! Keeps all Rust binaries 100% pure, portable, and `#![forbid(unsafe_code)]` while delegating
//! explicit-solvent MD (Amber14 + OpenFF Sage + TIP3P-FB + 150 mM NaCl), multi-replicate
//! consensus, per-residue Cα RMSF, and true GPU Single-Trajectory MM/GBSA (OBC2/GBn2 + ACE SASA,
//! block-averaged SEM, and per-residue pocket decomposition) via the isolated OpenMM environment
//! in `CRAFT_DATA_DIR/envs/openmm` (tuned for 12 GB VRAM RTX 4080 GPUs).

#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MdError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "OpenMM environment not found at '{0}'. Install into CRAFT_DATA_DIR/envs/openmm or set CRAFT_OPENMM_PYTHON."
    )]
    EnvNotFound(String),
    #[error("OpenMM MD run failed (exit {code:?}): {stderr}")]
    ExecutionFailed { code: Option<i32>, stderr: String },
}

/// Target compute platform for OpenMM MD validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MdPlatform {
    #[default]
    Auto,
    Cuda,
    OpenCl,
    Cpu,
}

impl MdPlatform {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cuda => "CUDA",
            Self::OpenCl => "OpenCL",
            Self::Cpu => "CPU",
        }
    }
}

/// Stability classification of a docked pose after explicit-solvent MD production.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MdStabilityVerdict {
    /// Ligand heavy-atom RMSD mean < 2.5 Å and final < 3.0 Å across all replicates.
    StableBound,
    /// Multi-replicate run where at least one replicate stayed bound while another drifted.
    MixedReplicates,
    /// Ligand heavy-atom RMSD mean in 2.5–4.0 Å (induced-fit rearrangement).
    ModerateDrift,
    /// Ligand heavy-atom RMSD > 4.0 Å (pose unstable or dissociating).
    Unstable,
}

impl MdStabilityVerdict {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StableBound => "Stable (Bound)",
            Self::MixedReplicates => "Mixed Across Replicates (Review)",
            Self::ModerateDrift => "Moderate Drift (Induced Fit)",
            Self::Unstable => "Unstable / Dissociating",
        }
    }
}

fn default_replicates() -> usize {
    1
}
fn default_seed() -> u64 {
    2026_1121
}
fn default_mmgbsa_model() -> String {
    "obc2".to_string()
}
fn default_solute_dielectric() -> f64 {
    1.0
}
fn default_mmgbsa_max_frames() -> usize {
    40
}

/// Configuration for an explicit-solvent OpenMM GPU MD validation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdValidateConfig {
    pub receptor_path: PathBuf,
    pub ligand_path: PathBuf,
    pub out_dir: PathBuf,
    /// Production MD length in nanoseconds per replicate (default `20.0` ns; use `0.05–0.5` ns for rapid checks).
    pub production_ns: f64,
    /// NVT equilibration duration in picoseconds (default `50.0` ps).
    pub nvt_ps: f64,
    /// NPT equilibration duration in picoseconds (default `100.0` ps).
    pub npt_ps: f64,
    /// Simulation temperature in Kelvin (default `300.0` K).
    pub temperature_k: f64,
    /// Explicit solvent box padding in nanometers (default `1.0` nm — safe for 12 GB RTX 4080 VRAM).
    pub padding_nm: f64,
    /// Ionic strength in molar NaCl (default `0.15` M).
    pub ionic_strength_m: f64,
    /// Integration timestep in femtoseconds (default `2.0` fs).
    pub timestep_fs: f64,
    /// Frame recording interval in picoseconds (default `10.0` ps; auto-scaled down for short runs).
    pub report_interval_ps: f64,
    /// Partial charge model for small-molecule OpenFF parameterization (`nagl` | `mmff94` | `gasteiger` | `am1bcc`).
    pub charge_method: String,
    pub platform: MdPlatform,
    /// Number of independent MD replicates (`1..=5`, distinct Langevin & Maxwell-Boltzmann seeds).
    #[serde(default = "default_replicates")]
    pub replicates: usize,
    /// Base random seed for Langevin thermostat & initial velocity generation.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Implicit-solvent Generalized Born model for GPU MM/GBSA (`obc2` | `gbn2`).
    #[serde(default = "default_mmgbsa_model")]
    pub mmgbsa_model: String,
    /// Interior solute dielectric constant for MM/GBSA (`1.0` standard; `2.0` for highly polar pockets).
    #[serde(default = "default_solute_dielectric")]
    pub solute_dielectric: f64,
    /// Maximum equispaced snapshots evaluated per replicate in GPU MM/GBSA (default `40`).
    #[serde(default = "default_mmgbsa_max_frames")]
    pub mmgbsa_max_frames: usize,
    /// Optional explicit Python interpreter inside the OpenMM virtual environment.
    pub python_override: Option<PathBuf>,
    /// When `true`, writes the complete runnable OpenMM bundle & scripts without launching GPU simulation.
    pub dry_run: bool,
}

impl Default for MdValidateConfig {
    fn default() -> Self {
        Self {
            receptor_path: PathBuf::new(),
            ligand_path: PathBuf::new(),
            out_dir: PathBuf::from("./md_validation"),
            production_ns: 20.0,
            nvt_ps: 50.0,
            npt_ps: 100.0,
            temperature_k: 300.0,
            padding_nm: 1.0,
            ionic_strength_m: 0.15,
            timestep_fs: 2.0,
            report_interval_ps: 10.0,
            charge_method: "nagl".to_string(),
            platform: MdPlatform::Auto,
            replicates: default_replicates(),
            seed: default_seed(),
            mmgbsa_model: default_mmgbsa_model(),
            solute_dielectric: default_solute_dielectric(),
            mmgbsa_max_frames: default_mmgbsa_max_frames(),
            python_override: None,
            dry_run: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HBondPersistence {
    pub residue: String,
    pub contact_type: String,
    pub occupancy_pct: f64,
    pub mean_distance_a: f64,
}

/// Per-residue backbone Cα root-mean-square fluctuation (RMSF) over production MD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueRmsf {
    pub residue: String,
    pub rmsf_a: f64,
    pub in_pocket: bool,
}

/// Exact single-trajectory MM/GBSA energetic contribution for a single pocket residue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueEnergyDecomp {
    pub residue: String,
    pub total_kcal: f64,
    pub vdw_kcal: f64,
    pub elec_gb_kcal: f64,
}

/// Full physical Single-Trajectory GPU MM/GBSA breakdown with 5-block autocorrelation-corrected SEM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmgbsaBreakdown {
    pub model: String,
    pub solute_dielectric: f64,
    pub total_dg_kcal: f64,
    pub block_sem_kcal: f64,
    pub delta_vdw_kcal: f64,
    pub delta_elec_kcal: f64,
    pub delta_gb_kcal: f64,
    pub delta_sa_kcal: f64,
    pub frames_evaluated: usize,
    #[serde(default)]
    pub per_residue_top: Vec<ResidueEnergyDecomp>,
}

/// Per-replicate summary recorded during multi-replicate MD validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicateSummary {
    pub replicate_id: usize,
    pub seed: u64,
    pub ligand_rmsd_mean_a: f64,
    pub ligand_rmsd_final_a: f64,
    pub mmgbsa_dg_kcal: Option<f64>,
    pub stability_verdict: MdStabilityVerdict,
    pub trajectory_dcd_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdEnvStatus {
    pub python_path: String,
    pub available: bool,
    pub openmm_version: Option<String>,
    pub platforms: Vec<String>,
    pub has_cuda: bool,
    pub has_openff: bool,
    pub has_pdbfixer: bool,
    pub has_rdkit: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdValidateReport {
    pub receptor_name: String,
    pub ligand_name: String,
    pub dry_run: bool,
    pub platform_used: String,
    pub total_atoms_solvated: usize,
    pub water_molecules: usize,
    pub production_ns: f64,
    pub frames_analyzed: usize,
    pub throughput_ns_per_day: f64,
    pub wall_seconds: f64,
    pub protein_ca_rmsd_mean_a: f64,
    pub protein_ca_rmsd_max_a: f64,
    pub ligand_rmsd_mean_a: f64,
    pub ligand_rmsd_std_a: f64,
    pub ligand_rmsd_final_a: f64,
    pub ligand_rmsd_max_a: f64,
    pub stability_verdict: MdStabilityVerdict,
    pub mmgbsa_dg_kcal: Option<f64>,
    pub mmgbsa_sem_kcal: Option<f64>,
    pub interaction_vdw_kcal: Option<f64>,
    pub interaction_elec_kcal: Option<f64>,
    #[serde(default)]
    pub mmgbsa: Option<MmgbsaBreakdown>,
    #[serde(default)]
    pub residue_rmsf_top: Vec<ResidueRmsf>,
    #[serde(default = "default_replicates")]
    pub replicates_run: usize,
    #[serde(default)]
    pub replicate_summaries: Vec<ReplicateSummary>,
    pub hbond_persistence: Vec<HBondPersistence>,
    pub out_dir: String,
    pub script_path: String,
    pub chimerax_script_path: String,
    pub trajectory_dcd_path: Option<String>,
    pub initial_pdb_path: Option<String>,
}

impl MdValidateReport {
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();
        let _ = writeln!(
            md,
            "# OpenMM GPU Molecular Dynamics & MM/GBSA Validation: {} + {}\n",
            self.receptor_name, self.ligand_name
        );
        if self.dry_run {
            let _ = writeln!(
                md,
                "> **Dry-Run Bundle Prepared:** Execute with the isolated OpenMM environment to run GPU MD + MM/GBSA.\n"
            );
            let _ = writeln!(md, "- **Runner Script:** `{}`", self.script_path);
            let _ = writeln!(
                md,
                "- **ChimeraX Viewer Script:** `{}`",
                self.chimerax_script_path
            );
            let _ = writeln!(
                md,
                "- **Planned Production:** `{} replicate(s) × {:.2} ns` | **Output Dir:** `{}`",
                self.replicates_run, self.production_ns, self.out_dir
            );
            return md;
        }
        let _ = writeln!(
            md,
            "- **Stability Verdict:** **{}** ({} replicate(s))",
            self.stability_verdict.as_str(),
            self.replicates_run
        );
        let _ = writeln!(
            md,
            "- **OpenMM Platform:** `{}` | **Throughput:** `{:.1} ns/day` | **Wall Time:** `{:.1} s`",
            self.platform_used, self.throughput_ns_per_day, self.wall_seconds
        );
        let _ = writeln!(
            md,
            "- **Solvated System:** `{}` atoms (`{}` TIP3P-FB waters, 150 mM NaCl)",
            self.total_atoms_solvated, self.water_molecules
        );
        let _ = writeln!(
            md,
            "- **Production Length:** `{} × {:.3} ns` ({} total analyzed frames)\n",
            self.replicates_run, self.production_ns, self.frames_analyzed
        );

        md.push_str("## 1. Conformational Stability (Pocket-Aligned RMSD)\n\n");
        md.push_str(
            "| Metric | Mean (Å) | Std (Å) | Final (Å) | Max (Å) |\n|---|---:|---:|---:|---:|\n",
        );
        let _ = writeln!(
            md,
            "| **Ligand Heavy-Atom RMSD** | **{:.2}** | {:.2} | {:.2} | {:.2} |",
            self.ligand_rmsd_mean_a,
            self.ligand_rmsd_std_a,
            self.ligand_rmsd_final_a,
            self.ligand_rmsd_max_a
        );
        let _ = writeln!(
            md,
            "| **Protein Backbone Cα RMSD** | {:.2} | — | — | {:.2} |\n",
            self.protein_ca_rmsd_mean_a, self.protein_ca_rmsd_max_a
        );

        if self.replicate_summaries.len() > 1 {
            md.push_str("### Multi-Replicate Consensus\n\n");
            md.push_str("| Replicate | Seed | Lig RMSD Mean (Å) | Lig RMSD Final (Å) | MM/GBSA ΔG (kcal/mol) | Verdict |\n|---:|---:|---:|---:|---:|---|\n");
            for r in &self.replicate_summaries {
                let dg_s = r
                    .mmgbsa_dg_kcal
                    .map_or_else(|| "—".to_string(), |v| format!("{v:.2}"));
                let _ = writeln!(
                    md,
                    "| Rep {} | `{}` | {:.2} | {:.2} | {} | {} |",
                    r.replicate_id,
                    r.seed,
                    r.ligand_rmsd_mean_a,
                    r.ligand_rmsd_final_a,
                    dg_s,
                    r.stability_verdict.as_str()
                );
            }
            md.push('\n');
        }

        if let Some(gb) = &self.mmgbsa {
            let _ = writeln!(
                md,
                "## 2. Physical Single-Trajectory GPU MM/GBSA (`{}` · ε_in={:.1}, {} snapshots)\n",
                gb.model.to_uppercase(),
                gb.solute_dielectric,
                gb.frames_evaluated
            );
            md.push_str("| Energy Component | Mean (kcal/mol) | Physical Role |\n|---|---:|---|\n");
            let _ = writeln!(
                md,
                "| **ΔE_vdW** | {:.2} | Intermolecular Lennard-Jones shape & dispersion |",
                gb.delta_vdw_kcal
            );
            let _ = writeln!(
                md,
                "| **ΔE_elec (vac)** | {:.2} | Direct gas-phase Coulomb (÷ ε_in) |",
                gb.delta_elec_kcal
            );
            let _ = writeln!(
                md,
                "| **ΔG_GB (polar)** | {:+.2} | Generalized Born polar desolvation penalty |",
                gb.delta_gb_kcal
            );
            let _ = writeln!(
                md,
                "| **ΔG_SA (nonpolar)** | {:.2} | ACE buried solvent-accessible surface area |",
                gb.delta_sa_kcal
            );
            let _ = writeln!(
                md,
                "| **ΔG_bind (Total)** | **{:.2} ± {:.2}** | 5-block autocorrelation-corrected SEM |\n",
                gb.total_dg_kcal, gb.block_sem_kcal
            );

            if !gb.per_residue_top.is_empty() {
                md.push_str("### Per-Residue Pocket Energy Decomposition (Top Contributors)\n\n");
                md.push_str("| Residue | ΔG_total (kcal/mol) | ΔE_vdW | ΔE_elec + ΔG_GB |\n|---|---:|---:|---:|\n");
                for r in gb.per_residue_top.iter().take(15) {
                    let _ = writeln!(
                        md,
                        "| `{}` | **{:.2}** | {:.2} | {:+.2} |",
                        r.residue, r.total_kcal, r.vdw_kcal, r.elec_gb_kcal
                    );
                }
                md.push('\n');
            }
        } else if let Some(dg) = self.mmgbsa_dg_kcal {
            let sem = self.mmgbsa_sem_kcal.unwrap_or(0.0);
            md.push_str("## 2. Post-MD Binding Energy Estimate (Single-Trajectory MM/GBSA)\n\n");
            let _ = writeln!(
                md,
                "- **Estimated ΔG_bind:** **{dg:.2} ± {sem:.2} kcal/mol**\n"
            );
        }

        if !self.residue_rmsf_top.is_empty() {
            md.push_str("## 3. Backbone Flexibility (Per-Residue Cα RMSF)\n\n");
            md.push_str("| Residue | Cα RMSF (Å) | Pocket Lining | Flexibility Flag |\n|---|---:|:---:|---|\n");
            for r in self.residue_rmsf_top.iter().take(15) {
                let pock = if r.in_pocket { "Yes" } else { "No" };
                let flag = if r.rmsf_a > 2.5 {
                    "Flexible loop (>2.5 Å)"
                } else if r.rmsf_a > 1.5 {
                    "Moderate mobility"
                } else {
                    "Rigid"
                };
                let _ = writeln!(
                    md,
                    "| `{}` | **{:.2}** | {} | {} |",
                    r.residue, r.rmsf_a, pock, flag
                );
            }
            md.push('\n');
        }

        if !self.hbond_persistence.is_empty() {
            md.push_str("## 4. Protein–Ligand H-Bond & Water-Bridge Persistence\n\n");
            md.push_str("| Residue | Contact Type | Occupancy (%) | Mean Distance (Å) |\n|---|---|---:|---:|\n");
            for h in self.hbond_persistence.iter().take(20) {
                let _ = writeln!(
                    md,
                    "| `{}` | {} | **{:.1}%** | {:.2} |",
                    h.residue, h.contact_type, h.occupancy_pct, h.mean_distance_a
                );
            }
            md.push('\n');
        }

        md.push_str("## 5. Output Artifacts & ChimeraX Trajectory Playback\n\n");
        if let Some(p) = &self.initial_pdb_path {
            let _ = writeln!(md, "- **Equilibrated Complex PDB:** `{p}`");
        }
        if let Some(t) = &self.trajectory_dcd_path {
            let _ = writeln!(md, "- **Primary Production Trajectory (DCD):** `{t}`");
        }
        let _ = writeln!(
            md,
            "- **ChimeraX Playback Script:** `{}` (`chimerax {}`)",
            self.chimerax_script_path, self.chimerax_script_path
        );
        md
    }
}

/// Resolves the centralized CraftBio data directory.
///
/// `CRAFT_DATA_DIR` if set, otherwise `$XDG_DATA_HOME/CraftBio`, otherwise
/// `$HOME/.local/share/CraftBio`. Set `CRAFT_DATA_DIR` to keep multi-gigabyte libraries,
/// models and campaigns off the root partition. No path is hardcoded for any machine.
#[must_use]
pub fn craft_data_dir() -> PathBuf {
    let non_blank = |k: &str| {
        std::env::var(k)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    };

    if let Some(dir) = non_blank("CRAFT_DATA_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(xdg) = non_blank("XDG_DATA_HOME") {
        return PathBuf::from(xdg).join("CraftBio");
    }
    if let Some(home) = non_blank("HOME") {
        return PathBuf::from(home).join(".local/share/CraftBio");
    }
    PathBuf::from("CraftBio")
}

/// Resolves the isolated OpenMM Python interpreter path (`CRAFT_OPENMM_PYTHON` or
/// `CRAFT_DATA_DIR/envs/openmm/bin/python`).
#[must_use]
pub fn resolve_openmm_python(override_path: Option<&Path>) -> PathBuf {
    if let Some(p) = override_path {
        return p.to_path_buf();
    }
    if let Ok(env_py) = std::env::var("CRAFT_OPENMM_PYTHON") {
        let p = PathBuf::from(env_py.trim());
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    craft_data_dir()
        .join("envs")
        .join("openmm")
        .join("bin")
        .join("python")
}

/// Probes whether the isolated OpenMM GPU environment is installed and ready.
#[must_use]
pub fn check_openmm_env(override_path: Option<&Path>) -> MdEnvStatus {
    let py = resolve_openmm_python(override_path);
    let py_str = py.display().to_string();
    if !py.exists() {
        return MdEnvStatus {
            python_path: py_str,
            available: false,
            openmm_version: None,
            platforms: Vec::new(),
            has_cuda: false,
            has_openff: false,
            has_pdbfixer: false,
            has_rdkit: false,
            message:
                "Isolated OpenMM venv not installed yet (expected at CRAFT_DATA_DIR/envs/openmm)."
                    .to_string(),
        };
    }
    let probe = r#"
import json
out = {"openmm_version": None, "platforms": [], "has_cuda": False, "has_openff": False, "has_pdbfixer": False, "has_rdkit": False}
try:
    import openmm
    out["openmm_version"] = openmm.__version__
    plats = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
    out["platforms"] = plats
    out["has_cuda"] = "CUDA" in plats
except Exception as e:
    out["error"] = str(e)
for mod, key in [("openff.toolkit", "has_openff"), ("pdbfixer", "has_pdbfixer"), ("rdkit", "has_rdkit")]:
    try:
        __import__(mod)
        out[key] = True
    except Exception:
        pass
print(json.dumps(out))
"#;
    let Ok(output) = Command::new(&py).arg("-c").arg(probe).output() else {
        return MdEnvStatus {
            python_path: py_str,
            available: false,
            openmm_version: None,
            platforms: Vec::new(),
            has_cuda: false,
            has_openff: false,
            has_pdbfixer: false,
            has_rdkit: false,
            message: "Failed to spawn OpenMM Python interpreter.".to_string(),
        };
    };
    if !output.status.success() {
        return MdEnvStatus {
            python_path: py_str,
            available: false,
            openmm_version: None,
            platforms: Vec::new(),
            has_cuda: false,
            has_openff: false,
            has_pdbfixer: false,
            has_rdkit: false,
            message: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        };
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(stdout.trim()) {
        let ver = v
            .get("openmm_version")
            .and_then(|x| x.as_str())
            .map(ToString::to_string);
        let platforms: Vec<String> = v
            .get("platforms")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(ToString::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let has_cuda = v.get("has_cuda").and_then(|x| x.as_bool()).unwrap_or(false);
        let has_openff = v
            .get("has_openff")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let has_pdbfixer = v
            .get("has_pdbfixer")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let has_rdkit = v
            .get("has_rdkit")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let available = ver.is_some();
        return MdEnvStatus {
            python_path: py_str,
            available,
            openmm_version: ver.clone(),
            platforms,
            has_cuda,
            has_openff,
            has_pdbfixer,
            has_rdkit,
            message: if available {
                format!(
                    "OpenMM {} ready (CUDA={}, OpenFF={}, PDBFixer={}, RDKit={})",
                    ver.as_deref().unwrap_or("?"),
                    has_cuda,
                    has_openff,
                    has_pdbfixer,
                    has_rdkit
                )
            } else {
                "OpenMM Python module missing in venv.".to_string()
            },
        };
    }
    MdEnvStatus {
        python_path: py_str,
        available: false,
        openmm_version: None,
        platforms: Vec::new(),
        has_cuda: false,
        has_openff: false,
        has_pdbfixer: false,
        has_rdkit: false,
        message: "Unparseable OpenMM probe output.".to_string(),
    }
}

/// Generates a ChimeraX `.cxc` playback script for the equilibrated complex + production `.dcd` trajectory.
#[must_use]
pub fn build_chimerax_md_script(initial_pdb_name: &str, dcd_name: &str) -> String {
    format!(
        "# DockCraft OpenMM MD Trajectory Viewer (UCSF ChimeraX)\n\
         set bgColor white\n\
         open {initial_pdb_name}\n\
         open {dcd_name} structureModel #1\n\
         hide solvent\n\
         hide ions\n\
         cartoon #1 & protein\n\
         color #1 & protein #d0d7de target c\n\
         show #1 & ligand\n\
         style #1 & ligand stick\n\
         color #1 & ligand byhetero\n\
         show #1 & (protein & :<4.5 & ligand)\n\
         style #1 & (protein & :<4.5 & ligand) stick\n\
         hbonds #1 & ligand restrict both color #2563eb radius 0.08\n\
         view #1 & ligand pad 0.35\n\
         coordset slider #1\n"
    )
}

/// Runs (or dry-run stages) explicit-solvent OpenMM GPU MD + physical MM/GBSA validation.
pub fn run_md_validation(cfg: &MdValidateConfig) -> Result<MdValidateReport, MdError> {
    std::fs::create_dir_all(&cfg.out_dir)?;

    let rec_name = cfg
        .receptor_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("receptor")
        .to_string();
    let lig_name = cfg
        .ligand_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("ligand")
        .to_string();

    let replicates = cfg.replicates.clamp(1, 5);
    let model = if cfg.mmgbsa_model.eq_ignore_ascii_case("gbn2") {
        "gbn2".to_string()
    } else {
        "obc2".to_string()
    };

    let script_path = cfg.out_dir.join("run_md_validate.py");
    let config_path = cfg.out_dir.join("md_config.json");
    let cxc_path = cfg.out_dir.join("view_trajectory.cxc");
    let metrics_path = cfg.out_dir.join("md_metrics.json");
    let report_md_path = cfg.out_dir.join("md_report.md");

    std::fs::write(&script_path, OPENMM_MD_RUNNER_PY)?;
    let primary_dcd_rel = if replicates > 1 {
        "rep_1/production.dcd"
    } else {
        "production.dcd"
    };
    std::fs::write(
        &cxc_path,
        build_chimerax_md_script("complex_equil.pdb", primary_dcd_rel),
    )?;

    let json_cfg = serde_json::json!({
        "receptor_path": cfg.receptor_path.display().to_string(),
        "ligand_path": cfg.ligand_path.display().to_string(),
        "out_dir": cfg.out_dir.display().to_string(),
        "receptor_name": rec_name,
        "ligand_name": lig_name,
        "production_ns": cfg.production_ns,
        "nvt_ps": cfg.nvt_ps,
        "npt_ps": cfg.npt_ps,
        "temperature_k": cfg.temperature_k,
        "padding_nm": cfg.padding_nm,
        "ionic_strength_m": cfg.ionic_strength_m,
        "timestep_fs": cfg.timestep_fs,
        "report_interval_ps": cfg.report_interval_ps,
        "charge_method": cfg.charge_method,
        "platform": cfg.platform.as_str(),
        "replicates": replicates,
        "seed": cfg.seed,
        "mmgbsa_model": model,
        "solute_dielectric": cfg.solute_dielectric.clamp(1.0, 8.0),
        "mmgbsa_max_frames": cfg.mmgbsa_max_frames.clamp(5, 200),
    });
    std::fs::write(&config_path, serde_json::to_string_pretty(&json_cfg)?)?;

    if cfg.dry_run {
        let rep = MdValidateReport {
            receptor_name: rec_name,
            ligand_name: lig_name,
            dry_run: true,
            platform_used: format!("dry-run ({})", cfg.platform.as_str()),
            total_atoms_solvated: 0,
            water_molecules: 0,
            production_ns: cfg.production_ns,
            frames_analyzed: 0,
            throughput_ns_per_day: 0.0,
            wall_seconds: 0.0,
            protein_ca_rmsd_mean_a: 0.0,
            protein_ca_rmsd_max_a: 0.0,
            ligand_rmsd_mean_a: 0.0,
            ligand_rmsd_std_a: 0.0,
            ligand_rmsd_final_a: 0.0,
            ligand_rmsd_max_a: 0.0,
            stability_verdict: MdStabilityVerdict::StableBound,
            mmgbsa_dg_kcal: None,
            mmgbsa_sem_kcal: None,
            interaction_vdw_kcal: None,
            interaction_elec_kcal: None,
            mmgbsa: None,
            residue_rmsf_top: Vec::new(),
            replicates_run: replicates,
            replicate_summaries: Vec::new(),
            hbond_persistence: Vec::new(),
            out_dir: cfg.out_dir.display().to_string(),
            script_path: script_path.display().to_string(),
            chimerax_script_path: cxc_path.display().to_string(),
            trajectory_dcd_path: None,
            initial_pdb_path: None,
        };
        std::fs::write(&report_md_path, rep.render_markdown())?;
        return Ok(rep);
    }

    let py = resolve_openmm_python(cfg.python_override.as_deref());
    if !py.exists() {
        return Err(MdError::EnvNotFound(py.display().to_string()));
    }

    let out = Command::new(&py)
        .arg(&script_path)
        .arg(&config_path)
        .output()?;
    if !out.status.success() {
        return Err(MdError::ExecutionFailed {
            code: out.status.code(),
            stderr: format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ),
        });
    }

    let metrics_raw = std::fs::read_to_string(&metrics_path)?;
    let mut rep: MdValidateReport = serde_json::from_str(&metrics_raw)?;
    rep.script_path = script_path.display().to_string();
    rep.chimerax_script_path = cxc_path.display().to_string();
    std::fs::write(&report_md_path, rep.render_markdown())?;
    Ok(rep)
}

/// Self-contained OpenMM GPU MD + True Single-Trajectory MM/GBSA + RMSF + Multi-Replicate engine.
/// Designed for 12 GB VRAM GPUs (RTX 4080 Laptop) with graceful OpenCL/CPU fallback.
pub const OPENMM_MD_RUNNER_PY: &str = r#"#!/usr/bin/env python3
"""DockCraft v0.7.0 OpenMM GPU Molecular Dynamics, Multi-Replicate Consensus, RMSF & True GPU MM/GBSA Engine."""
import io
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

KJ_TO_KCAL = 1.0 / 4.184
K_COUL_KJ_NM = 138.935458
ACE_SURFTEN_KJ_NM2 = 2.267728  # 0.00542 kcal/(mol*A^2) standard Amber ACE nonpolar tension


def kabsch_align(P, Q):
    """Returns rotation R and centroids cP, cQ mapping P onto Q (minimizing RMSD)."""
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    P0 = P - cP
    Q0 = Q - cQ
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, -1.0 if d < 0 else 1.0])
    R = Vt.T @ S @ U.T
    return R, cP, cQ


def select_platform(requested):
    available = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    order = [requested] if requested != "auto" else ["CUDA", "OpenCL", "CPU"]
    for name in order:
        if name in available:
            plat = mm.Platform.getPlatformByName(name)
            props = {"Precision": "mixed"} if name in ("CUDA", "OpenCL") else {}
            return plat, props, name
    return mm.Platform.getPlatformByName("CPU"), {}, "CPU"


def load_clean_protein(pdb_path, ph=7.4):
    """Loads protein PDB via PDBFixer after stripping generic H lines & normalizing Amber aliases."""
    from pdbfixer import PDBFixer

    alias_map = {
        "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
        "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
        "CYM": "CYS", "CYX": "CYS",
    }
    clean_lines = []
    with open(pdb_path, "r") as fh:
        for line in fh:
            if line.startswith(("ATOM  ", "HETATM")):
                el = line[76:78].strip().upper() if len(line) >= 78 else ""
                aname = line[12:16].strip().upper()
                if el == "H" or aname.startswith("H"):
                    continue
                rname = line[17:20].strip().upper()
                if rname in ("HOH", "WAT", "DOD"):
                    continue
                if rname in alias_map:
                    line = line[:17] + f"{alias_map[rname]:<3}" + line[20:]
                clean_lines.append(line)
            elif line.startswith(("TER", "END")):
                clean_lines.append(line)
    if not clean_lines or not clean_lines[-1].startswith("END"):
        clean_lines.append("END\n")

    fixer = PDBFixer(pdbfile=io.StringIO("".join(clean_lines)))
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    return fixer.topology, fixer.positions


def load_ligand_openff(ligand_path, charge_method="nagl"):
    """Loads first molecule from SDF/MOL2/PDB via RDKit + OpenFF with NAGL AM1-BCC / MMFF94 charges."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from openff.toolkit import Molecule as OffMolecule

    p = Path(ligand_path)
    ext = p.suffix.lower()
    if ext in (".sdf", ".sd"):
        suppl = Chem.SDMolSupplier(str(p), removeHs=False, sanitize=True)
        rdmol = next((m for m in suppl if m is not None), None)
    elif ext == ".mol2":
        rdmol = Chem.MolFromMol2File(str(p), removeHs=False, sanitize=True)
    else:
        rdmol = Chem.MolFromPDBFile(str(p), removeHs=False, sanitize=True)
    if rdmol is None:
        raise RuntimeError(f"RDKit could not parse ligand '{ligand_path}'")

    rdmol = Chem.AddHs(rdmol, addCoords=True)
    if rdmol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(rdmol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(rdmol)

    off_mol = OffMolecule.from_rdkit(rdmol, allow_undefined_stereo=True)
    cm = charge_method.lower()
    assigned = False
    if cm in ("nagl", "am1bcc"):
        try:
            from openff.nagl_models import list_available_nagl_models
            models = list_available_nagl_models()
            if models:
                off_mol.assign_partial_charges(partial_charge_method=str(models[-1]))
                assigned = True
                print(f"[DockCraft MD] Assigned ligand charges with OpenFF NAGL ({models[-1].name})", flush=True)
        except Exception as exc:
            print(f"[DockCraft MD] NAGL charge fallback ({exc}); trying MMFF94/Gasteiger", flush=True)
    if not assigned:
        charges = None
        if cm in ("nagl", "mmff94"):
            props = AllChem.MMFFGetMoleculeProperties(rdmol)
            if props is not None:
                charges = [float(props.GetMMFFPartialCharge(i)) for i in range(rdmol.GetNumAtoms())]
        if charges is None:
            AllChem.ComputeGasteigerCharges(rdmol)
            charges = []
            for a in rdmol.GetAtoms():
                q = float(a.GetDoubleProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0
                charges.append(0.0 if (math.isnan(q) or math.isinf(q)) else q)
        target_q = float(sum(a.GetFormalCharge() for a in rdmol.GetAtoms()))
        drift = (sum(charges) - target_q) / max(len(charges), 1)
        charges = [q - drift for q in charges]
        off_mol.partial_charges = np.array(charges) * unit.elementary_charge

    off_top = off_mol.to_topology()
    omm_top = off_top.to_openmm()
    for res in omm_top.residues():
        res.name = "LIG"
        for idx, atom in enumerate(res.atoms()):
            atom.name = f"L{idx}"
    conf = rdmol.GetConformer()
    pos_nm = [
        mm.Vec3(conf.GetAtomPosition(i).x / 10.0, conf.GetAtomPosition(i).y / 10.0, conf.GetAtomPosition(i).z / 10.0)
        for i in range(rdmol.GetNumAtoms())
    ] * unit.nanometers
    return off_mol, omm_top, pos_nm


def ligand_interchange_to_ffxml(off_mol, lig_top):
    """Parameterizes `off_mol` with OpenFF Sage 2.2.1 via OpenFF Interchange and emits an OpenMM ForceField XML."""
    from openff.toolkit import ForceField as OffForceField
    from openff.interchange import Interchange

    off_ff = OffForceField("openff_unconstrained-2.2.1.offxml")
    ic = Interchange.from_smirnoff(off_ff, off_mol.to_topology(), charge_from_molecules=[off_mol])
    lsys = ic.to_openmm_system()

    atoms = list(lig_top.atoms())
    lines = ["<ForceField>", "  <AtomTypes>"]
    for i, a in enumerate(atoms):
        mass = lsys.getParticleMass(i).value_in_unit(unit.dalton)
        sym = a.element.symbol if a.element is not None else "C"
        lines.append(f'    <Type name="LIG_{i}" class="LIG_{i}" element="{sym}" mass="{mass:.6f}"/>')
    lines.append("  </AtomTypes>")

    lines.append("  <Residues>")
    lines.append('    <Residue name="LIG">')
    for i, a in enumerate(atoms):
        lines.append(f'      <Atom name="{a.name}" type="LIG_{i}"/>')
    for b in lig_top.bonds():
        lines.append(f'      <Bond atomName1="{b[0].name}" atomName2="{b[1].name}"/>')
    lines.append("    </Residue>")
    lines.append("  </Residues>")

    bond_set = set()
    for b in lig_top.bonds():
        u, v = b[0].index, b[1].index
        bond_set.add((u, v))
        bond_set.add((v, u))

    hb_lines, ha_lines, nb_lines = [], [], []
    pt_proper = defaultdict(list)
    pt_improper = defaultdict(list)
    for f in lsys.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for bi in range(f.getNumBonds()):
                p1, p2, length, k = f.getBondParameters(bi)
                hb_lines.append(
                    f'    <Bond type1="LIG_{p1}" type2="LIG_{p2}" '
                    f'length="{length.value_in_unit(unit.nanometers):.6f}" '
                    f'k="{k.value_in_unit(unit.kilojoules_per_mole / unit.nanometers**2):.4f}"/>'
                )
        elif isinstance(f, mm.HarmonicAngleForce):
            for ai in range(f.getNumAngles()):
                p1, p2, p3, ang, k = f.getAngleParameters(ai)
                ha_lines.append(
                    f'    <Angle type1="LIG_{p1}" type2="LIG_{p2}" type3="LIG_{p3}" '
                    f'angle="{ang.value_in_unit(unit.radians):.6f}" '
                    f'k="{k.value_in_unit(unit.kilojoules_per_mole / unit.radians**2):.4f}"/>'
                )
        elif isinstance(f, mm.PeriodicTorsionForce):
            for ti in range(f.getNumTorsions()):
                p1, p2, p3, p4, per, phase, k = f.getTorsionParameters(ti)
                term = (int(per), phase.value_in_unit(unit.radians), k.value_in_unit(unit.kilojoules_per_mole))
                if (p1, p2) in bond_set and (p2, p3) in bond_set and (p3, p4) in bond_set:
                    pt_proper[(p1, p2, p3, p4)].append(term)
                else:
                    pt_improper[(p1, p2, p3, p4)].append(term)
        elif isinstance(f, mm.NonbondedForce):
            for pi in range(f.getNumParticles()):
                q, sig, eps = f.getParticleParameters(pi)
                nb_lines.append(
                    f'    <Atom type="LIG_{pi}" '
                    f'charge="{q.value_in_unit(unit.elementary_charge):.8f}" '
                    f'sigma="{sig.value_in_unit(unit.nanometers):.6f}" '
                    f'epsilon="{eps.value_in_unit(unit.kilojoules_per_mole):.6f}"/>'
                )

    if hb_lines:
        lines.extend(["  <HarmonicBondForce>", *hb_lines, "  </HarmonicBondForce>"])
    if ha_lines:
        lines.extend(["  <HarmonicAngleForce>", *ha_lines, "  </HarmonicAngleForce>"])
    if pt_proper or pt_improper:
        lines.append("  <PeriodicTorsionForce>")
        for (p1, p2, p3, p4), terms in pt_proper.items():
            attrs = [f'type1="LIG_{p1}"', f'type2="LIG_{p2}"', f'type3="LIG_{p3}"', f'type4="LIG_{p4}"']
            for idx, (per, ph, kv) in enumerate(terms, start=1):
                attrs.append(f'periodicity{idx}="{per}" phase{idx}="{ph:.6f}" k{idx}="{kv:.6f}"')
            lines.append(f'    <Proper {" ".join(attrs)}/>')
        for (p1, p2, p3, p4), terms in pt_improper.items():
            cand = [p1, p2, p3, p4]
            center = max(cand, key=lambda c: sum(1 for o in cand if o != c and (c, o) in bond_set))
            others = [o for o in cand if o != center]
            attrs = [
                f'type1="LIG_{center}"',
                f'type2="LIG_{others[0]}"',
                f'type3="LIG_{others[1]}"',
                f'type4="LIG_{others[2]}"',
            ]
            for idx, (per, ph, kv) in enumerate(terms, start=1):
                attrs.append(f'periodicity{idx}="{per}" phase{idx}="{ph:.6f}" k{idx}="{kv:.6f}"')
            lines.append(f'    <Improper {" ".join(attrs)}/>')
        lines.append("  </PeriodicTorsionForce>")
    lines.append('  <NonbondedForce coulomb14scale="0.8333333333333334" lj14scale="0.5">')
    lines.extend(nb_lines)
    lines.append("  </NonbondedForce>")
    lines.append("</ForceField>")
    return "\n".join(lines)


def build_forcefield(off_mol, lig_top):
    lig_xml = ligand_interchange_to_ffxml(off_mol, lig_top)
    return app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml", io.StringIO(lig_xml))


def assign_mbondi_radii_and_scales(complex_top, model="obc2"):
    """Assigns Amber mbondi2 (OBC2) / mbondi3 (GBn2) intrinsic Born radii (nm) and HCT/GBn2 scale factors."""
    use_mbondi3 = (model.lower() == "gbn2")
    bonded_syms = defaultdict(list)
    for b in complex_top.bonds():
        a1, a2 = b[0], b[1]
        s1 = a1.element.symbol if a1.element is not None else "C"
        s2 = a2.element.symbol if a2.element is not None else "C"
        bonded_syms[a1.index].append(s2)
        bonded_syms[a2.index].append(s1)

    radii_nm = []
    scales = []
    # GBn2 element scaling factors (Nguyen-Roe-Simmerling 2013) vs standard HCT/OBC2
    scale_map_obc2 = {"H": 0.85, "C": 0.72, "N": 0.79, "O": 0.85, "F": 0.88, "P": 0.86, "S": 0.96, "Cl": 0.80, "Br": 0.80, "I": 0.80}
    scale_map_gbn2 = {"H": 0.8548, "C": 0.72, "N": 0.79, "O": 0.85, "F": 0.88, "P": 0.86, "S": 0.96, "Cl": 0.80, "Br": 0.80, "I": 0.80}
    smap = scale_map_gbn2 if use_mbondi3 else scale_map_obc2
    base_r = {"C": 0.170, "N": 0.155, "O": 0.150, "S": 0.180, "P": 0.185, "F": 0.150, "Cl": 0.170, "Br": 0.185, "I": 0.198, "Zn": 0.139, "Mg": 0.118}

    for atom in complex_top.atoms():
        sym = atom.element.symbol if atom.element is not None else "C"
        rname = atom.residue.name.upper()
        aname = atom.name.upper()
        if sym == "H":
            partners = bonded_syms.get(atom.index, [])
            if "N" in partners:
                r = 0.117 if (use_mbondi3 and rname == "ARG" and aname.startswith("HH")) else 0.130
            elif "O" in partners:
                r = 0.105 if use_mbondi3 else 0.080
            else:
                r = 0.120
        elif sym == "O" and use_mbondi3 and rname in ("ASP", "GLU") and aname in ("OD1", "OD2", "OE1", "OE2"):
            r = 0.140
        else:
            r = base_r.get(sym, 0.170)
        radii_nm.append(r)
        scales.append(smap.get(sym, 0.80))
    return radii_nm, scales


def make_gbsa_context(charges, radii_nm, scales, solute_eps, surften_kj_nm2, platform, props):
    """Builds a lightweight standalone OpenMM GPU GBSAOBCForce context for fast snapshot evaluation."""
    sys_gb = mm.System()
    gb = mm.GBSAOBCForce()
    gb.setNonbondedMethod(mm.GBSAOBCForce.NoCutoff)
    gb.setSoluteDielectric(float(solute_eps))
    gb.setSolventDielectric(78.5)
    gb.setSurfaceAreaEnergy(float(surften_kj_nm2))
    for q, r, s in zip(charges, radii_nm, scales):
        sys_gb.addParticle(12.0)
        gb.addParticle(float(q), float(r), float(s))
    sys_gb.addForce(gb)
    integ = mm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = mm.Context(sys_gb, integ, platform, props)
    return ctx, gb


def block_averaged_sem(values, n_blocks=5):
    """Computes autocorrelation-corrected block-averaged SEM over contiguous trajectory blocks."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n <= 1:
        return 0.0
    if n < n_blocks * 2:
        return float(arr.std(ddof=1) / math.sqrt(n))
    blocks = np.array_split(arr, n_blocks)
    b_means = np.array([b.mean() for b in blocks if len(b) > 0], dtype=np.float64)
    if len(b_means) <= 1:
        return float(arr.std(ddof=1) / math.sqrt(n))
    return float(b_means.std(ddof=1) / math.sqrt(len(b_means)))


def compute_true_mmgbsa(
    system_solv,
    complex_top,
    prot_atom_count,
    lig_atom_count,
    frames_nm,
    platform,
    props,
    model="obc2",
    solute_eps=1.0,
    max_frames=40,
):
    """
    True Single-Trajectory GPU MM/GBSA:
      dG_bind = dE_vdW + dE_elec/eps_in + dG_GB(Complex - Receptor - Ligand) + dG_SA(Complex - Receptor - Ligand)
    plus 5-block autocorrelation-corrected SEM and exact GPU per-residue pocket decomposition.
    """
    nb = next((f for f in system_solv.getForces() if isinstance(f, mm.NonbondedForce)), None)
    if nb is None or not frames_nm:
        return None

    complex_count = prot_atom_count + lig_atom_count
    q_arr = np.zeros(complex_count, dtype=np.float64)
    sig_arr = np.zeros(complex_count, dtype=np.float64)
    eps_arr = np.zeros(complex_count, dtype=np.float64)
    for i in range(complex_count):
        q, sig, eps = nb.getParticleParameters(i)
        q_arr[i] = q.value_in_unit(unit.elementary_charge)
        sig_arr[i] = sig.value_in_unit(unit.nanometers)
        eps_arr[i] = eps.value_in_unit(unit.kilojoules_per_mole)

    radii_nm, scales = assign_mbondi_radii_and_scales(complex_top, model=model)
    qP, sP, eP = q_arr[:prot_atom_count], sig_arr[:prot_atom_count], eps_arr[:prot_atom_count]
    qL, sL, eL = q_arr[prot_atom_count:], sig_arr[prot_atom_count:], eps_arr[prot_atom_count:]
    rP, scP = radii_nm[:prot_atom_count], scales[:prot_atom_count]
    rL, scL = radii_nm[prot_atom_count:], scales[prot_atom_count:]

    # Build 6 lightweight GPU GBSA contexts (GB-only vs GB+SA for Complex, Receptor, Ligand)
    ctx_c_gb, gb_c = make_gbsa_context(q_arr, radii_nm, scales, solute_eps, 0.0, platform, props)
    ctx_c_sa, _ = make_gbsa_context(q_arr, radii_nm, scales, solute_eps, ACE_SURFTEN_KJ_NM2, platform, props)
    ctx_r_gb, gb_r = make_gbsa_context(qP, rP, scP, solute_eps, 0.0, platform, props)
    ctx_r_sa, _ = make_gbsa_context(qP, rP, scP, solute_eps, ACE_SURFTEN_KJ_NM2, platform, props)
    ctx_l_gb, _ = make_gbsa_context(qL, rL, scL, solute_eps, 0.0, platform, props)
    ctx_l_sa, _ = make_gbsa_context(qL, rL, scL, solute_eps, ACE_SURFTEN_KJ_NM2, platform, props)

    stride = max(1, len(frames_nm) // max(max_frames, 5))
    sel_frames = frames_nm[::stride][:max_frames]

    # Map protein atoms to residue labels for per-residue decomposition
    atoms_complex = list(complex_top.atoms())
    res_to_atoms = defaultdict(list)
    for idx in range(prot_atom_count):
        a = atoms_complex[idx]
        lbl = f"{a.residue.chain.id}:{a.residue.name}{a.residue.id}"
        res_to_atoms[lbl].append(idx)

    vdw_vals, elec_vals, gb_vals, sa_vals, tot_vals = [], [], [], [], []
    res_vdw_acc = defaultdict(list)
    res_elec_acc = defaultdict(list)
    pocket_res_set = set()

    for pos in sel_frames:
        xP = pos[:prot_atom_count]
        xL = pos[prot_atom_count:complex_count]

        # Exact intermolecular vdW + vacuum Coulomb (all pairs, no cutoff)
        diff = xL[:, None, :] - xP[None, :, :]  # (NL, NP, 3)
        r2 = np.sum(diff * diff, axis=2)
        r = np.sqrt(np.maximum(r2, 1e-8))
        sig_ij = 0.5 * (sL[:, None] + sP[None, :])
        eps_ij = np.sqrt(eL[:, None] * eP[None, :])
        sr6 = (sig_ij / r) ** 6
        e_vdw_mat = 4.0 * eps_ij * (sr6 * sr6 - sr6)
        e_elec_mat = (K_COUL_KJ_NM * qL[:, None] * qP[None, :]) / (float(solute_eps) * r)

        vdw_kcal = float(np.sum(e_vdw_mat)) * KJ_TO_KCAL
        elec_kcal = float(np.sum(e_elec_mat)) * KJ_TO_KCAL

        # Exact GPU Generalized Born (polar) and ACE SASA (nonpolar)
        ctx_c_gb.setPositions(pos)
        ctx_c_sa.setPositions(pos)
        ctx_r_gb.setPositions(xP)
        ctx_r_sa.setPositions(xP)
        ctx_l_gb.setPositions(xL)
        ctx_l_sa.setPositions(xL)

        e_c_gb = ctx_c_gb.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        e_c_tot = ctx_c_sa.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        e_r_gb = ctx_r_gb.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        e_r_tot = ctx_r_sa.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        e_l_gb = ctx_l_gb.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        e_l_tot = ctx_l_sa.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        d_gb_kcal = (e_c_gb - e_r_gb - e_l_gb) * KJ_TO_KCAL
        d_sa_kcal = ((e_c_tot - e_c_gb) - (e_r_tot - e_r_gb) - (e_l_tot - e_l_gb)) * KJ_TO_KCAL

        vdw_vals.append(vdw_kcal)
        elec_vals.append(elec_kcal)
        gb_vals.append(d_gb_kcal)
        sa_vals.append(d_sa_kcal)
        tot_vals.append(vdw_kcal + elec_kcal + d_gb_kcal + d_sa_kcal)

        # Accumulate per-residue vdW & vacuum Coulomb + identify pocket residues (<= 0.55 nm)
        min_d_prot = np.min(r, axis=0)  # (NP,)
        vdw_per_prot = np.sum(e_vdw_mat, axis=0) * KJ_TO_KCAL
        elec_per_prot = np.sum(e_elec_mat, axis=0) * KJ_TO_KCAL
        for lbl, idxs in res_to_atoms.items():
            if float(np.min(min_d_prot[idxs])) <= 0.55:
                pocket_res_set.add(lbl)
            res_vdw_acc[lbl].append(float(np.sum(vdw_per_prot[idxs])))
            res_elec_acc[lbl].append(float(np.sum(elec_per_prot[idxs])))

    # Exact GPU per-residue polar GB decomposition over up to 10 equispaced frames via charge-zeroing identity:
    # dG_GB(r) = [G_GB(C) - G_GB(C_{q_r=0})] - [G_GB(R) - G_GB(R_{q_r=0})]
    decomp_frames = sel_frames[::max(1, len(sel_frames) // 10)][:10]
    res_gb_acc = defaultdict(list)
    pocket_res_list = sorted(pocket_res_set)
    for pos in decomp_frames:
        xP = pos[:prot_atom_count]
        ctx_c_gb.setPositions(pos)
        ctx_r_gb.setPositions(xP)
        base_c = ctx_c_gb.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        base_r = ctx_r_gb.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        for lbl in pocket_res_list:
            idxs = res_to_atoms[lbl]
            for ai in idxs:
                gb_c.setParticleParameters(ai, 0.0, float(radii_nm[ai]), float(scales[ai]))
                gb_r.setParticleParameters(ai, 0.0, float(radii_nm[ai]), float(scales[ai]))
            gb_c.updateParametersInContext(ctx_c_gb)
            gb_r.updateParametersInContext(ctx_r_gb)
            zero_c = ctx_c_gb.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            zero_r = ctx_r_gb.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            dgb_r = ((base_c - zero_c) - (base_r - zero_r)) * KJ_TO_KCAL
            res_gb_acc[lbl].append(dgb_r)
            for ai in idxs:
                gb_c.setParticleParameters(ai, float(q_arr[ai]), float(radii_nm[ai]), float(scales[ai]))
                gb_r.setParticleParameters(ai, float(q_arr[ai]), float(radii_nm[ai]), float(scales[ai]))
            gb_c.updateParametersInContext(ctx_c_gb)
            gb_r.updateParametersInContext(ctx_r_gb)

    per_res = []
    for lbl in pocket_res_list:
        v = float(np.mean(res_vdw_acc[lbl]))
        el = float(np.mean(res_elec_acc[lbl]))
        gb_r_val = float(np.mean(res_gb_acc[lbl])) if res_gb_acc[lbl] else 0.0
        pol = el + gb_r_val
        per_res.append({
            "residue": lbl,
            "total_kcal": round(v + pol, 2),
            "vdw_kcal": round(v, 2),
            "elec_gb_kcal": round(pol, 2),
        })
    per_res.sort(key=lambda x: x["total_kcal"])

    return {
        "model": model,
        "solute_dielectric": round(float(solute_eps), 2),
        "total_dg_kcal": round(float(np.mean(tot_vals)), 2),
        "block_sem_kcal": round(block_averaged_sem(tot_vals, n_blocks=5), 2),
        "delta_vdw_kcal": round(float(np.mean(vdw_vals)), 2),
        "delta_elec_kcal": round(float(np.mean(elec_vals)), 2),
        "delta_gb_kcal": round(float(np.mean(gb_vals)), 2),
        "delta_sa_kcal": round(float(np.mean(sa_vals)), 2),
        "frames_evaluated": len(sel_frames),
        "per_residue_top": per_res[:15],
        "raw_tot_vals": tot_vals,
    }


def classify_verdict(lig_mean, lig_final):
    if lig_mean < 2.5 and lig_final < 3.0:
        return "StableBound"
    if lig_mean <= 4.0:
        return "ModerateDrift"
    return "Unstable"


def main():
    t0 = time.time()
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    replicates = max(1, min(5, int(cfg.get("replicates", 1))))
    base_seed = int(cfg.get("seed", 20261121))
    mmgbsa_model = str(cfg.get("mmgbsa_model", "obc2")).lower()
    solute_eps = float(cfg.get("solute_dielectric", 1.0))
    mmgbsa_max_frames = int(cfg.get("mmgbsa_max_frames", 40))

    print(f"[DockCraft MD] Loading receptor {cfg['receptor_path']} and ligand {cfg['ligand_path']} ({replicates} replicate(s))...", flush=True)
    prot_top, prot_pos = load_clean_protein(cfg["receptor_path"])
    off_mol, lig_top, lig_pos = load_ligand_openff(cfg["ligand_path"], cfg.get("charge_method", "nagl"))

    modeller = app.Modeller(prot_top, prot_pos)
    modeller.add(lig_top, lig_pos)
    ff = build_forcefield(off_mol, lig_top)

    prot_atom_count = sum(1 for _ in prot_top.atoms())
    lig_atom_count = sum(1 for _ in lig_top.atoms())
    complex_atom_count = prot_atom_count + lig_atom_count

    # Snapshot clean unsolvated complex topology before adding solvent
    complex_mod_ref = app.Modeller(prot_top, prot_pos)
    complex_mod_ref.add(lig_top, lig_pos)
    complex_top = complex_mod_ref.topology

    modeller.addSolvent(
        ff,
        model="tip3p",
        padding=float(cfg.get("padding_nm", 1.0)) * unit.nanometers,
        ionicStrength=float(cfg.get("ionic_strength_m", 0.15)) * unit.molar,
        neutralize=True,
    )
    total_atoms = modeller.topology.getNumAtoms()
    water_mols = sum(1 for r in modeller.topology.residues() if r.name in ("HOH", "WAT"))
    print(f"[DockCraft MD] Solvated system: {total_atoms} atoms ({water_mols} waters)", flush=True)

    platform, props, plat_name = select_platform(cfg.get("platform", "auto"))
    print(f"[DockCraft MD] Selected OpenMM platform: {plat_name} {props}", flush=True)

    atoms_list = list(modeller.topology.atoms())
    ca_atoms = [(a.index, f"{a.residue.chain.id}:{a.residue.name}{a.residue.id}") for a in atoms_list[:prot_atom_count] if a.name == "CA"]
    ca_idx = [x[0] for x in ca_atoms]
    lig_heavy_idx = [
        a.index for a in atoms_list[prot_atom_count:complex_atom_count]
        if a.element is not None and a.element.symbol != "H"
    ]
    lig_polar_idx = [
        a.index for a in atoms_list[prot_atom_count:complex_atom_count]
        if a.element is not None and a.element.symbol in ("N", "O", "S")
    ]
    prot_polar = [
        (a.index, f"{a.residue.chain.id}:{a.residue.name}{a.residue.id}")
        for a in atoms_list[:prot_atom_count]
        if a.element is not None and a.element.symbol in ("N", "O")
    ]
    water_o_idx = [
        a.index for a in atoms_list[complex_atom_count:]
        if a.residue.name in ("HOH", "WAT") and a.element is not None and a.element.symbol == "O"
    ]

    temp = float(cfg.get("temperature_k", 300.0)) * unit.kelvin
    dt_fs = float(cfg.get("timestep_fs", 2.0))
    prod_ns = float(cfg.get("production_ns", 20.0))
    nvt_steps = max(500, int(round(float(cfg.get("nvt_ps", 50.0)) * 1000.0 / dt_fs)))
    npt_steps = max(500, int(round(float(cfg.get("npt_ps", 100.0)) * 1000.0 / dt_fs)))
    total_prod_steps = max(500, int(round(prod_ns * 1_000_000.0 / dt_fs)))
    req_rep_ps = float(cfg.get("report_interval_ps", 10.0))
    target_frames = max(20, min(500, int(round(prod_ns * 1000.0 / max(req_rep_ps, 0.1)))))
    report_stride = max(10, total_prod_steps // target_frames)

    all_frames_complex_nm = []
    all_aligned_ca_nm = []
    all_ca_rmsds = []
    all_lig_rmsds = []
    hb_counts = defaultdict(int)
    hb_dists = defaultdict(list)
    wb_counts = defaultdict(int)
    wb_dists = defaultdict(list)
    rep_summaries = []
    prod_elapsed_total = 0.0
    primary_dcd_path = None
    init_pdb_path = out_dir / "complex_equil.pdb"
    pocket_ca_set = set()
    ref_system_for_mmgbsa = None

    for rep_i in range(replicates):
        rep_seed = base_seed + rep_i * 1009
        rep_dir = out_dir / f"rep_{rep_i + 1}" if replicates > 1 else out_dir
        rep_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DockCraft MD] === Replicate {rep_i + 1}/{replicates} (seed={rep_seed}) ===", flush=True)

        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
        )
        restraint = mm.CustomExternalForce("0.5 * k_res * periodicdistance(x, y, z, x0, y0, z0)^2")
        restraint.addGlobalParameter("k_res", 0.0)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        init_pos_nm = modeller.positions.value_in_unit(unit.nanometers)
        restraint_indices = []
        for idx, atom in enumerate(modeller.topology.atoms()):
            if idx < prot_atom_count and atom.name == "CA":
                p = init_pos_nm[idx]
                restraint.addParticle(idx, [p[0], p[1], p[2]])
                restraint_indices.append(idx)
        system.addForce(restraint)

        integrator = mm.LangevinMiddleIntegrator(temp, 2.0 / unit.picoseconds, dt_fs * unit.femtoseconds)
        integrator.setRandomNumberSeed(rep_seed)
        sim = app.Simulation(modeller.topology, system, integrator, platform, props)
        sim.context.setPositions(modeller.positions)

        sim.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometers, maxIterations=0)
        min_pos_nm = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        for pi_idx, atom_idx in enumerate(restraint_indices):
            p = min_pos_nm[atom_idx]
            restraint.setParticleParameters(pi_idx, atom_idx, [float(p[0]), float(p[1]), float(p[2])])
        restraint.updateParametersInContext(sim.context)

        # Gentle 2-stage NVT ramp
        sim.context.setParameter("k_res", 250.0)
        integrator.setStepSize(1.0 * unit.femtoseconds)
        integrator.setTemperature(100.0 * unit.kelvin)
        sim.context.setVelocitiesToTemperature(100.0 * unit.kelvin, rep_seed)
        sim.step(max(250, nvt_steps // 3))
        integrator.setTemperature(temp)
        integrator.setStepSize(dt_fs * unit.femtoseconds)
        sim.step(nvt_steps - max(250, nvt_steps // 3))

        # NPT equilibration
        barostat = mm.MonteCarloBarostat(1.0 * unit.bar, temp, 25)
        barostat.setRandomNumberSeed(rep_seed + 17)
        system.addForce(barostat)
        sim.context.reinitialize(preserveState=True)
        sim.context.setParameter("k_res", 100.0)
        sim.step(npt_steps)

        eq_state = sim.context.getState(getPositions=True)
        eq_pos_full = eq_state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        if rep_i == 0:
            sub_mod = app.Modeller(modeller.topology, eq_state.getPositions())
            solvent_atoms = [a for a in sub_mod.topology.atoms() if a.index >= complex_atom_count]
            sub_mod.delete(solvent_atoms)
            with open(init_pdb_path, "w") as fh:
                app.PDBFile.writeFile(sub_mod.topology, sub_mod.positions, fh)
            ref_system_for_mmgbsa = system

        sim.context.setParameter("k_res", 0.0)
        dcd_path = rep_dir / "production.dcd"
        log_path = rep_dir / "md_state.csv"
        if primary_dcd_path is None:
            primary_dcd_path = dcd_path
        sim.reporters.append(app.DCDReporter(str(dcd_path), report_stride, atomSubset=list(range(complex_atom_count))))
        sim.reporters.append(
            app.StateDataReporter(str(log_path), report_stride, step=True, time=True, potentialEnergy=True, temperature=True, speed=True)
        )

        ref_ca = eq_pos_full[ca_idx].copy()
        ref_lig = eq_pos_full[lig_heavy_idx].copy()
        lig_c0 = ref_lig.mean(axis=0)
        pocket_ca_mask = np.linalg.norm(ref_ca - lig_c0, axis=1) < 1.2
        pocket_ca_idx = [ca_idx[i] for i in range(len(ca_idx)) if pocket_ca_mask[i]]
        if len(pocket_ca_idx) < 4:
            pocket_ca_idx = ca_idx
        for i, m in enumerate(pocket_ca_mask):
            if m:
                pocket_ca_set.add(ca_atoms[i][1])
        ref_pock_ca = eq_pos_full[pocket_ca_idx].copy()

        t_prod0 = time.time()
        rep_frames_nm = []
        rep_lig_rmsds = []
        steps_done = 0
        while steps_done < total_prod_steps:
            chunk = min(report_stride, total_prod_steps - steps_done)
            sim.step(chunk)
            steps_done += chunk
            pos_nm = sim.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True).value_in_unit(unit.nanometers)
            rep_frames_nm.append(pos_nm[:complex_atom_count].copy())
            all_frames_complex_nm.append(pos_nm[:complex_atom_count].copy())

            cur_pock = pos_nm[pocket_ca_idx]
            R, cP, cQ = kabsch_align(cur_pock, ref_pock_ca)
            cur_ca_al = (pos_nm[ca_idx] - cP) @ R.T + cQ
            cur_lig_al = (pos_nm[lig_heavy_idx] - cP) @ R.T + cQ
            all_aligned_ca_nm.append(cur_ca_al)

            ca_rmsd = float(np.sqrt(np.mean(np.sum((cur_ca_al - ref_ca) ** 2, axis=1)))) * 10.0
            lig_rmsd = float(np.sqrt(np.mean(np.sum((cur_lig_al - ref_lig) ** 2, axis=1)))) * 10.0
            all_ca_rmsds.append(ca_rmsd)
            all_lig_rmsds.append(lig_rmsd)
            rep_lig_rmsds.append(lig_rmsd)

            if lig_polar_idx:
                xL_pol = pos_nm[lig_polar_idx]
                seen_res = set()
                for p_idx, r_label in prot_polar:
                    d_min = float(np.min(np.linalg.norm(xL_pol - pos_nm[p_idx], axis=1)))
                    if d_min <= 0.35:
                        if r_label not in seen_res:
                            seen_res.add(r_label)
                            hb_counts[r_label] += 1
                        hb_dists[r_label].append(d_min * 10.0)
                if water_o_idx:
                    xW = pos_nm[water_o_idx]
                    lmin = xL_pol.min(axis=0) - 0.36
                    lmax = xL_pol.max(axis=0) + 0.36
                    near_w = np.where(np.all((xW >= lmin) & (xW <= lmax), axis=1))[0]
                    seen_wb = set()
                    for wi in near_w:
                        wpos = xW[wi]
                        if float(np.min(np.linalg.norm(xL_pol - wpos, axis=1))) <= 0.35:
                            for p_idx, r_label in prot_polar:
                                dp = float(np.linalg.norm(pos_nm[p_idx] - wpos))
                                if dp <= 0.35 and r_label not in seen_wb:
                                    seen_wb.add(r_label)
                                    wb_counts[r_label] += 1
                                    wb_dists[r_label].append(dp * 10.0)

        prod_elapsed_total += max(time.time() - t_prod0, 1e-3)
        rep_mean = float(np.mean(rep_lig_rmsds))
        rep_final = float(rep_lig_rmsds[-1])
        rep_verdict = classify_verdict(rep_mean, rep_final)
        rep_mmgbsa = compute_true_mmgbsa(
            system,
            complex_top,
            prot_atom_count,
            lig_atom_count,
            rep_frames_nm,
            platform,
            props,
            model=mmgbsa_model,
            solute_eps=solute_eps,
            max_frames=max(10, mmgbsa_max_frames // replicates),
        )
        rep_dg = rep_mmgbsa["total_dg_kcal"] if rep_mmgbsa else None
        rep_summaries.append({
            "replicate_id": rep_i + 1,
            "seed": rep_seed,
            "ligand_rmsd_mean_a": round(rep_mean, 3),
            "ligand_rmsd_final_a": round(rep_final, 3),
            "mmgbsa_dg_kcal": rep_dg,
            "stability_verdict": rep_verdict,
            "trajectory_dcd_path": str(dcd_path),
        })
        print(f"[DockCraft MD] Rep {rep_i + 1} finished: Lig RMSD={rep_mean:.2f} A (final {rep_final:.2f} A), MM/GBSA={rep_dg} kcal/mol ({rep_verdict})", flush=True)

    ns_per_day = ((prod_ns * replicates) / max(prod_elapsed_total, 1e-3)) * 86400.0
    n_frames = max(len(all_lig_rmsds), 1)
    lig_mean = float(np.mean(all_lig_rmsds))
    lig_std = float(np.std(all_lig_rmsds))
    lig_final = float(np.mean([r["ligand_rmsd_final_a"] for r in rep_summaries]))
    lig_max = float(np.max(all_lig_rmsds))
    ca_mean = float(np.mean(all_ca_rmsds))
    ca_max = float(np.max(all_ca_rmsds))

    rep_verdicts = {r["stability_verdict"] for r in rep_summaries}
    if replicates > 1 and len(rep_verdicts) > 1 and "StableBound" in rep_verdicts:
        overall_verdict = "MixedReplicates"
    else:
        overall_verdict = classify_verdict(lig_mean, lig_final)

    # Per-residue Cα RMSF across pooled pocket-aligned frames
    ca_stack = np.asarray(all_aligned_ca_nm, dtype=np.float64)  # (F, N_ca, 3)
    mean_ca = np.mean(ca_stack, axis=0, keepdims=True)
    rmsf_a = np.sqrt(np.mean(np.sum((ca_stack - mean_ca) ** 2, axis=2), axis=0)) * 10.0
    rmsf_list = [
        {
            "residue": ca_atoms[i][1],
            "rmsf_a": round(float(rmsf_a[i]), 3),
            "in_pocket": ca_atoms[i][1] in pocket_ca_set,
        }
        for i in range(len(ca_atoms))
    ]
    # Surface most mobile pocket residues + global high-mobility loops
    rmsf_pocket = sorted([r for r in rmsf_list if r["in_pocket"]], key=lambda x: -x["rmsf_a"])[:10]
    rmsf_global = sorted([r for r in rmsf_list if not r["in_pocket"]], key=lambda x: -x["rmsf_a"])[:5]
    rmsf_top = sorted(rmsf_pocket + rmsf_global, key=lambda x: -x["rmsf_a"])

    # Pooled true GPU MM/GBSA over all replicates
    print(f"[DockCraft MD] Running pooled GPU Single-Trajectory MM/GBSA ({mmgbsa_model.upper()}, {mmgbsa_max_frames} snapshots)...", flush=True)
    mmgbsa_full = compute_true_mmgbsa(
        ref_system_for_mmgbsa,
        complex_top,
        prot_atom_count,
        lig_atom_count,
        all_frames_complex_nm,
        platform,
        props,
        model=mmgbsa_model,
        solute_eps=solute_eps,
        max_frames=mmgbsa_max_frames,
    )
    if mmgbsa_full is not None:
        mmgbsa_full.pop("raw_tot_vals", None)

    contacts = []
    for r_label, cnt in hb_counts.items():
        occ = 100.0 * cnt / n_frames
        if occ >= 10.0:
            contacts.append({
                "residue": r_label,
                "contact_type": "Direct H-Bond",
                "occupancy_pct": round(occ, 1),
                "mean_distance_a": round(float(np.mean(hb_dists[r_label])), 2),
            })
    for r_label, cnt in wb_counts.items():
        occ = 100.0 * cnt / n_frames
        if occ >= 15.0:
            contacts.append({
                "residue": r_label,
                "contact_type": "Water Bridge",
                "occupancy_pct": round(occ, 1),
                "mean_distance_a": round(float(np.mean(wb_dists[r_label])), 2),
            })
    contacts.sort(key=lambda x: -x["occupancy_pct"])

    metrics = {
        "receptor_name": cfg["receptor_name"],
        "ligand_name": cfg["ligand_name"],
        "dry_run": False,
        "platform_used": plat_name,
        "total_atoms_solvated": total_atoms,
        "water_molecules": water_mols,
        "production_ns": prod_ns,
        "frames_analyzed": len(all_lig_rmsds),
        "throughput_ns_per_day": round(ns_per_day, 1),
        "wall_seconds": round(time.time() - t0, 2),
        "protein_ca_rmsd_mean_a": round(ca_mean, 3),
        "protein_ca_rmsd_max_a": round(ca_max, 3),
        "ligand_rmsd_mean_a": round(lig_mean, 3),
        "ligand_rmsd_std_a": round(lig_std, 3),
        "ligand_rmsd_final_a": round(lig_final, 3),
        "ligand_rmsd_max_a": round(lig_max, 3),
        "stability_verdict": overall_verdict,
        "mmgbsa_dg_kcal": mmgbsa_full["total_dg_kcal"] if mmgbsa_full else None,
        "mmgbsa_sem_kcal": mmgbsa_full["block_sem_kcal"] if mmgbsa_full else None,
        "interaction_vdw_kcal": mmgbsa_full["delta_vdw_kcal"] if mmgbsa_full else None,
        "interaction_elec_kcal": mmgbsa_full["delta_elec_kcal"] if mmgbsa_full else None,
        "mmgbsa": mmgbsa_full,
        "residue_rmsf_top": rmsf_top,
        "replicates_run": replicates,
        "replicate_summaries": rep_summaries,
        "hbond_persistence": contacts,
        "out_dir": str(out_dir),
        "script_path": str(out_dir / "run_md_validate.py"),
        "chimerax_script_path": str(out_dir / "view_trajectory.cxc"),
        "trajectory_dcd_path": str(primary_dcd_path) if primary_dcd_path else None,
        "initial_pdb_path": str(init_pdb_path),
    }
    (out_dir / "md_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"[DockCraft MD] Done in {metrics['wall_seconds']:.1f}s ({ns_per_day:.1f} ns/day) | "
        f"Lig RMSD={lig_mean:.2f}±{lig_std:.2f} A ({overall_verdict}) | "
        f"MM/GBSA={metrics['mmgbsa_dg_kcal']}±{metrics['mmgbsa_sem_kcal']} kcal/mol",
        flush=True,
    )


if __name__ == "__main__":
    main()
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn md_validate_dry_run_generates_multi_replicate_bundle_and_renders_mmgbsa_rmsf_markdown()
    -> Result<(), Box<dyn std::error::Error>> {
        let base = std::env::temp_dir().join(format!("dockcraft_md_dry_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let cfg = MdValidateConfig {
            receptor_path: PathBuf::from("1KE6_protprep.pdb"),
            ligand_path: PathBuf::from("top1_hit.sdf"),
            out_dir: base.clone(),
            production_ns: 20.0,
            replicates: 3,
            seed: 4242,
            mmgbsa_model: "obc2".to_string(),
            solute_dielectric: 2.0,
            dry_run: true,
            ..Default::default()
        };
        let rep = run_md_validation(&cfg)?;
        assert!(rep.dry_run);
        assert_eq!(rep.replicates_run, 3);
        assert!(base.join("run_md_validate.py").exists());
        assert!(base.join("md_config.json").exists());
        assert!(base.join("view_trajectory.cxc").exists());
        assert!(base.join("md_report.md").exists());

        let cfg_raw = std::fs::read_to_string(base.join("md_config.json"))?;
        let cfg_json: serde_json::Value = serde_json::from_str(&cfg_raw)?;
        assert_eq!(cfg_json["replicates"], 3);
        assert_eq!(cfg_json["seed"], 4242);
        assert_eq!(cfg_json["mmgbsa_model"], "obc2");
        assert_eq!(cfg_json["solute_dielectric"], 2.0);

        let cxc = std::fs::read_to_string(base.join("view_trajectory.cxc"))?;
        assert!(cxc.contains("open rep_1/production.dcd") && cxc.contains("coordset slider #1"));

        // Verify rich Markdown rendering of physical MM/GBSA, per-residue decomp, RMSF, and replicates
        let rich = MdValidateReport {
            dry_run: false,
            stability_verdict: MdStabilityVerdict::MixedReplicates,
            replicates_run: 2,
            mmgbsa: Some(MmgbsaBreakdown {
                model: "obc2".to_string(),
                solute_dielectric: 1.0,
                total_dg_kcal: -32.4,
                block_sem_kcal: 1.1,
                delta_vdw_kcal: -44.2,
                delta_elec_kcal: -18.5,
                delta_gb_kcal: 35.8,
                delta_sa_kcal: -5.5,
                frames_evaluated: 30,
                per_residue_top: vec![ResidueEnergyDecomp {
                    residue: "A:LEU83".to_string(),
                    total_kcal: -3.45,
                    vdw_kcal: -2.10,
                    elec_gb_kcal: -1.35,
                }],
            }),
            residue_rmsf_top: vec![ResidueRmsf {
                residue: "A:GLY13".to_string(),
                rmsf_a: 2.85,
                in_pocket: true,
            }],
            replicate_summaries: vec![
                ReplicateSummary {
                    replicate_id: 1,
                    seed: 4242,
                    ligand_rmsd_mean_a: 1.4,
                    ligand_rmsd_final_a: 1.6,
                    mmgbsa_dg_kcal: Some(-33.1),
                    stability_verdict: MdStabilityVerdict::StableBound,
                    trajectory_dcd_path: Some("rep_1/production.dcd".to_string()),
                },
                ReplicateSummary {
                    replicate_id: 2,
                    seed: 5251,
                    ligand_rmsd_mean_a: 2.8,
                    ligand_rmsd_final_a: 3.1,
                    mmgbsa_dg_kcal: Some(-31.7),
                    stability_verdict: MdStabilityVerdict::ModerateDrift,
                    trajectory_dcd_path: Some("rep_2/production.dcd".to_string()),
                },
            ],
            ..rep
        };
        let md = rich.render_markdown();
        assert!(md.contains("ΔG_GB (polar)"));
        assert!(md.contains("A:LEU83"));
        assert!(md.contains("A:GLY13") && md.contains("Flexible loop"));
        assert!(md.contains("Multi-Replicate Consensus"));
        assert!(md.contains("Mixed Across Replicates"));

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}

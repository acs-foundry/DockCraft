#![forbid(unsafe_code)]

//! Opt-in Relative Binding Free Energy (RBFE / Alchemical FEP) bridge (`fep`) for `DockCraft`.
//!
//! Combines a 100% pure-Rust perturbation network planner (`plan_rbfe_network`, computing 3D core
//! overlap matrices and star + cycle-closure thermodynamic graphs over congeneric SDF series) with
//! two opt-in execution backends:
//! - **`OpenmmAlchemical` (default):** runs directly inside `CRAFT_DATA_DIR/envs/openmm` on CUDA GPU
//!   using RDKit `rdFMCS` atom mapping, OpenFF Sage 2.2.1 + NAGL charges, exact-neutrality dual-softcore
//!   Beutler Lennard-Jones hybrid topologies, staged electrostatic-before-sterics λ-schedules,
//!   **MBAR** (`pymbar` with built-in self-consistent NumPy fallback) + **TI** cross-check, adjacent-window
//!   overlap diagnostics ($O_{i,i+1} \ge 0.03$), and optional 3rd-leg **Selectivity FEP** ($\Delta\Delta G_{\text{sel}}$).
//! - **`OpenFe` (opt-in):** exports ready-to-run `openfe` network specs and runner scripts when an external
//!   OpenFE environment is available.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use struct_core::{Molecule3D, parse_sdf_3d_with_limit};
use thiserror::Error;

use crate::md_validate::{MdPlatform, craft_data_dir, resolve_openmm_python};

#[derive(Debug, Error)]
pub enum FepError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("FEP setup error: {0}")]
    Setup(String),
    #[error(
        "OpenMM environment not found at '{0}'. Install into CRAFT_DATA_DIR/envs/openmm or set CRAFT_OPENMM_PYTHON."
    )]
    EnvNotFound(String),
    #[error("FEP execution failed (exit {code:?}): {stderr}")]
    ExecutionFailed { code: Option<i32>, stderr: String },
}

/// Execution backend for alchemical RBFE/FEP perturbations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FepBackend {
    /// Native OpenMM GPU dual-softcore alchemical engine inside `CRAFT_DATA_DIR/envs/openmm`.
    #[default]
    OpenmmAlchemical,
    /// Export manifest & runner script for external OpenFE (`CRAFT_DATA_DIR/envs/openfe`).
    OpenFe,
}

impl FepBackend {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenmmAlchemical => "openmm",
            Self::OpenFe => "openfe",
        }
    }
}

/// Environment probe for RBFE/FEP tooling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepEnvStatus {
    pub python_path: String,
    pub openmm_available: bool,
    pub has_cuda: bool,
    pub has_openff: bool,
    pub has_rdkit: bool,
    pub has_pymbar: bool,
    pub has_scipy: bool,
    pub has_openfe: bool,
    pub message: String,
}

/// Configuration for an RBFE/FEP perturbation network run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepConfig {
    pub receptor_path: PathBuf,
    /// Optional off-target receptor PDB for 3-leg Selectivity FEP (`ΔΔG_sel = ΔΔG_target - ΔΔG_offtarget`).
    pub offtarget_receptor_path: Option<PathBuf>,
    /// Congeneric 3D aligned ligand series SDF (2..12 ligands, e.g. `lead_opt_series.sdf` or `fep_pair.sdf`).
    pub ligands_sdf_path: PathBuf,
    pub out_dir: PathBuf,
    pub backend: FepBackend,
    /// Number of λ-windows per alchemical leg (`5..=21`, default `11`).
    pub n_windows: usize,
    /// Production sampling length per λ-window in nanoseconds (default `0.5` ns; use `0.05` ns for rapid checks).
    pub ns_per_window: f64,
    /// Equilibration length per λ-window in picoseconds (default `20.0` ps).
    pub equil_ps_per_window: f64,
    pub temperature_k: f64,
    pub padding_nm: f64,
    pub ionic_strength_m: f64,
    pub charge_method: String,
    pub platform: MdPlatform,
    pub max_edges: usize,
    pub python_override: Option<PathBuf>,
    pub dry_run: bool,
}

impl Default for FepConfig {
    fn default() -> Self {
        Self {
            receptor_path: PathBuf::new(),
            offtarget_receptor_path: None,
            ligands_sdf_path: PathBuf::new(),
            out_dir: PathBuf::from("./fep_run"),
            backend: FepBackend::OpenmmAlchemical,
            n_windows: 11,
            ns_per_window: 0.5,
            equil_ps_per_window: 20.0,
            temperature_k: 300.0,
            padding_nm: 0.95,
            ionic_strength_m: 0.15,
            charge_method: "nagl".to_string(),
            platform: MdPlatform::Auto,
            max_edges: 12,
            python_override: None,
            dry_run: false,
        }
    }
}

/// Pure-Rust planned alchemical edge between two aligned 3D congeneric ligands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbfePlannedEdge {
    pub edge_id: String,
    pub from_idx: usize,
    pub to_idx: usize,
    pub from_name: String,
    pub to_name: String,
    pub shared_core_atoms: usize,
    pub core_overlap_ratio: f64,
}

/// Thermodynamic result for one leg (`complex`, `solvent`, or `offtarget_complex`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepLegResult {
    pub leg_name: String,
    pub dg_mbar_kcal: f64,
    pub dg_mbar_sem_kcal: f64,
    pub dg_ti_kcal: f64,
    pub min_adjacent_overlap: f64,
    pub overlap_healthy: bool,
    #[serde(default)]
    pub adjacent_overlaps: Vec<f64>,
    pub wall_seconds: f64,
}

/// Complete RBFE result for a single perturbation edge $A \to B$.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepEdgeResult {
    pub edge_id: String,
    pub from_name: String,
    pub to_name: String,
    pub mcs_heavy_atoms: usize,
    pub unique_a_atoms: usize,
    pub unique_b_atoms: usize,
    pub complex_leg: FepLegResult,
    pub solvent_leg: FepLegResult,
    #[serde(default)]
    pub offtarget_leg: Option<FepLegResult>,
    /// $\Delta\Delta G_{\text{bind}}(A \to B) = \Delta G_{\text{complex}} - \Delta G_{\text{solvent}}$ (negative = $B$ binds tighter).
    pub ddg_bind_kcal: f64,
    pub ddg_sem_kcal: f64,
    pub ddg_ti_kcal: f64,
    /// Relative binding constant ratio $K_i(B) / K_i(A) = \exp(\Delta\Delta G / RT)$ (`< 1.0` = more potent).
    pub fold_change_ki: f64,
    /// Optional Selectivity FEP: $\Delta\Delta G_{\text{target}} - \Delta\Delta G_{\text{offtarget}}$ (negative = better selectivity).
    #[serde(default)]
    pub ddg_selectivity_kcal: Option<f64>,
    #[serde(default)]
    pub ddg_selectivity_sem_kcal: Option<f64>,
    pub convergence_status: String,
}

/// Thermodynamic cycle closure diagnostic for 3-member cycles in the perturbation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepCycleClosure {
    pub nodes: Vec<String>,
    pub closure_error_kcal: f64,
}

/// Complete RBFE/FEP campaign report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepReport {
    pub receptor_name: String,
    #[serde(default)]
    pub offtarget_name: Option<String>,
    pub backend: FepBackend,
    pub dry_run: bool,
    pub platform_used: String,
    pub n_windows: usize,
    pub ns_per_window: f64,
    pub ligand_names: Vec<String>,
    pub planned_edges: Vec<RbfePlannedEdge>,
    #[serde(default)]
    pub edge_results: Vec<FepEdgeResult>,
    #[serde(default)]
    pub cycle_closures: Vec<FepCycleClosure>,
    pub wall_seconds: f64,
    pub out_dir: String,
    pub script_path: String,
}

impl FepReport {
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();
        let _ = writeln!(
            md,
            "# Alchemical Relative Binding Free Energy (RBFE / FEP) Report: {}\n",
            self.receptor_name
        );
        if let Some(ot) = &self.offtarget_name {
            let _ = writeln!(md, "- **Selectivity Off-Target Leg:** `{ot}`");
        }
        let _ = writeln!(
            md,
            "- **Backend:** `{}` | **Platform:** `{}` | **Schedule:** `{} λ-windows × {:.3} ns`",
            self.backend.as_str(),
            self.platform_used,
            self.n_windows,
            self.ns_per_window
        );
        let _ = writeln!(
            md,
            "- **Congeneric Series:** `{}` ligands | **Planned Perturbation Edges:** `{}` | **Wall Time:** `{:.1} s`\n",
            self.ligand_names.len(),
            self.planned_edges.len(),
            self.wall_seconds
        );

        md.push_str("## 1. Perturbation Network Plan (3D Core Overlap)\n\n");
        md.push_str("| Edge | From (λ=0) | To (λ=1) | Shared 3D Core Atoms | Core Overlap |\n|---|---|---|---:|---:|\n");
        for e in &self.planned_edges {
            let _ = writeln!(
                md,
                "| `{}` | `{}` | `{}` | {} | {:.1}% |",
                e.edge_id,
                e.from_name,
                e.to_name,
                e.shared_core_atoms,
                e.core_overlap_ratio * 100.0
            );
        }
        md.push('\n');

        if self.dry_run {
            let _ = writeln!(
                md,
                "> **Dry-Run Bundle Staged:** Run `{}` with the isolated OpenMM/OpenFE environment to execute GPU alchemical sampling.\n",
                self.script_path
            );
            return md;
        }

        if !self.edge_results.is_empty() {
            md.push_str("## 2. Alchemical Free Energy Results (MBAR + TI Cross-Check)\n\n");
            md.push_str(
                "| Edge (A → B) | MCS / ΔA / ΔB | ΔG_complex | ΔG_solvent | **ΔΔG_bind (MBAR)** | ΔΔG (TI) | K_i(B)/K_i(A) | Min Overlap | Status |\n|---|---|---:|---:|---:|---:|---:|---:|---|\n",
            );
            for r in &self.edge_results {
                let min_ov = r
                    .complex_leg
                    .min_adjacent_overlap
                    .min(r.solvent_leg.min_adjacent_overlap);
                let _ = writeln!(
                    md,
                    "| `{}` → `{}` | {}/{}/{} | {:.2}±{:.2} | {:.2}±{:.2} | **{:.2} ± {:.2}** | {:.2} | {:.2}× | {:.3} | {} |",
                    r.from_name,
                    r.to_name,
                    r.mcs_heavy_atoms,
                    r.unique_a_atoms,
                    r.unique_b_atoms,
                    r.complex_leg.dg_mbar_kcal,
                    r.complex_leg.dg_mbar_sem_kcal,
                    r.solvent_leg.dg_mbar_kcal,
                    r.solvent_leg.dg_mbar_sem_kcal,
                    r.ddg_bind_kcal,
                    r.ddg_sem_kcal,
                    r.ddg_ti_kcal,
                    r.fold_change_ki,
                    min_ov,
                    r.convergence_status
                );
            }
            md.push('\n');

            if self
                .edge_results
                .iter()
                .any(|r| r.ddg_selectivity_kcal.is_some())
            {
                md.push_str("### Selectivity FEP (ΔΔG_sel = ΔΔG_target − ΔΔG_off-target)\n\n");
                md.push_str("| Edge | ΔΔG_target | ΔΔG_offtarget | **ΔΔG_selectivity** | Interpretation |\n|---|---:|---:|---:|---|\n");
                for r in &self.edge_results {
                    if let (Some(sel), Some(ot)) = (r.ddg_selectivity_kcal, &r.offtarget_leg) {
                        let ot_ddg = ot.dg_mbar_kcal - r.solvent_leg.dg_mbar_kcal;
                        let sel_sem = r.ddg_selectivity_sem_kcal.unwrap_or(0.0);
                        let interp = if sel < -0.5 {
                            "Improves target selectivity"
                        } else if sel > 0.5 {
                            "Favors off-target"
                        } else {
                            "Selectivity neutral"
                        };
                        let _ = writeln!(
                            md,
                            "| `{}` → `{}` | {:.2} | {:.2} | **{:.2} ± {:.2}** | {} |",
                            r.from_name, r.to_name, r.ddg_bind_kcal, ot_ddg, sel, sel_sem, interp
                        );
                    }
                }
                md.push('\n');
            }
        }

        if !self.cycle_closures.is_empty() {
            md.push_str("## 3. Thermodynamic Cycle Closure Audit\n\n");
            md.push_str("| Cycle | Hysteresis Error (kcal/mol) | Quality |\n|---|---:|---|\n");
            for c in &self.cycle_closures {
                let q = if c.closure_error_kcal.abs() <= 0.8 {
                    "Excellent (<0.8 kcal/mol)"
                } else if c.closure_error_kcal.abs() <= 1.5 {
                    "Acceptable"
                } else {
                    "Review (>1.5 kcal/mol)"
                };
                let _ = writeln!(
                    md,
                    "| {} | {:+.2} | {} |",
                    c.nodes.join(" → "),
                    c.closure_error_kcal,
                    q
                );
            }
            md.push('\n');
        }

        md
    }
}

/// Probes whether OpenMM CUDA + OpenFF + RDKit + pymbar (and optional OpenFE) are installed.
#[must_use]
pub fn check_fep_env(override_path: Option<&Path>) -> FepEnvStatus {
    let py = resolve_openmm_python(override_path);
    let py_str = py.display().to_string();
    let has_openfe_env = craft_data_dir()
        .join("envs")
        .join("openfe")
        .join("bin")
        .join("openfe")
        .exists();
    if !py.exists() {
        return FepEnvStatus {
            python_path: py_str,
            openmm_available: false,
            has_cuda: false,
            has_openff: false,
            has_rdkit: false,
            has_pymbar: false,
            has_scipy: false,
            has_openfe: has_openfe_env,
            message: "Isolated OpenMM environment not found at CRAFT_DATA_DIR/envs/openmm."
                .to_string(),
        };
    }
    let probe = r#"
import json
out = {"openmm": False, "cuda": False, "openff": False, "rdkit": False, "pymbar": False, "scipy": False, "openfe": False}
try:
    import openmm
    out["openmm"] = True
    plats = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
    out["cuda"] = "CUDA" in plats
except Exception:
    pass
for mod, key in [("openff.toolkit", "openff"), ("rdkit", "rdkit"), ("pymbar", "pymbar"), ("scipy", "scipy"), ("openfe", "openfe")]:
    try:
        __import__(mod)
        out[key] = True
    except Exception:
        pass
print(json.dumps(out))
"#;
    let Ok(output) = Command::new(&py).arg("-c").arg(probe).output() else {
        return FepEnvStatus {
            python_path: py_str,
            openmm_available: false,
            has_cuda: false,
            has_openff: false,
            has_rdkit: false,
            has_pymbar: false,
            has_scipy: false,
            has_openfe: has_openfe_env,
            message: "Failed to execute OpenMM Python probe.".to_string(),
        };
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(stdout.trim()) {
        let openmm_available = v.get("openmm").and_then(|x| x.as_bool()).unwrap_or(false);
        let has_cuda = v.get("cuda").and_then(|x| x.as_bool()).unwrap_or(false);
        let has_openff = v.get("openff").and_then(|x| x.as_bool()).unwrap_or(false);
        let has_rdkit = v.get("rdkit").and_then(|x| x.as_bool()).unwrap_or(false);
        let has_pymbar = v.get("pymbar").and_then(|x| x.as_bool()).unwrap_or(false);
        let has_scipy = v.get("scipy").and_then(|x| x.as_bool()).unwrap_or(false);
        let has_openfe =
            has_openfe_env || v.get("openfe").and_then(|x| x.as_bool()).unwrap_or(false);
        let ready = openmm_available && has_openff && has_rdkit;
        return FepEnvStatus {
            python_path: py_str,
            openmm_available,
            has_cuda,
            has_openff,
            has_rdkit,
            has_pymbar,
            has_scipy,
            has_openfe,
            message: if ready {
                format!(
                    "OpenMM Alchemical RBFE ready (CUDA={has_cuda}, OpenFF={has_openff}, RDKit={has_rdkit}, pymbar={has_pymbar}, scipy={has_scipy}, openfe={has_openfe})"
                )
            } else {
                "Missing OpenMM/OpenFF/RDKit prerequisites for RBFE/FEP.".to_string()
            },
        };
    }
    FepEnvStatus {
        python_path: py_str,
        openmm_available: false,
        has_cuda: false,
        has_openff: false,
        has_rdkit: false,
        has_pymbar: false,
        has_scipy: false,
        has_openfe: has_openfe_env,
        message: "Unparseable FEP environment probe output.".to_string(),
    }
}

fn core_overlap_3d(a: &Molecule3D, b: &Molecule3D) -> (usize, f64) {
    let ha: Vec<&struct_core::Atom3D> = a.atoms.iter().filter(|x| x.is_heavy()).collect();
    let hb: Vec<&struct_core::Atom3D> = b.atoms.iter().filter(|x| x.is_heavy()).collect();
    if ha.is_empty() || hb.is_empty() {
        return (0, 0.0);
    }
    let mut used_b = vec![false; hb.len()];
    let mut matched = 0usize;
    for at_a in &ha {
        let mut best_j = None;
        let mut best_d2 = 0.90_f64 * 0.90_f64;
        for (j, at_b) in hb.iter().enumerate() {
            if used_b[j] || at_a.element != at_b.element {
                continue;
            }
            let dx = at_a.coords[0] - at_b.coords[0];
            let dy = at_a.coords[1] - at_b.coords[1];
            let dz = at_a.coords[2] - at_b.coords[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 <= best_d2 {
                best_d2 = d2;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            used_b[j] = true;
            matched += 1;
        }
    }
    let denom = ha.len().max(hb.len()) as f64;
    (matched, (matched as f64 / denom).clamp(0.0, 1.0))
}

/// Plans a star + cycle-closure RBFE perturbation network from `2..12` aligned 3D congeneric ligands.
#[must_use]
pub fn plan_rbfe_network(ligands: &[Molecule3D], max_edges: usize) -> Vec<RbfePlannedEdge> {
    let n = ligands.len().min(12);
    if n < 2 || max_edges == 0 {
        return Vec::new();
    }
    let names: Vec<String> = ligands
        .iter()
        .take(n)
        .enumerate()
        .map(|(i, m)| {
            if m.name.trim().is_empty() {
                format!("lig_{i}")
            } else {
                m.name.trim().to_string()
            }
        })
        .collect();

    let mut edges = Vec::new();
    // 1. Star edges anchored at ligand 0 (Parent) -> all analogs 1..n
    for j in 1..n {
        if edges.len() >= max_edges {
            break;
        }
        let (shared, ratio) = core_overlap_3d(&ligands[0], &ligands[j]);
        edges.push(RbfePlannedEdge {
            edge_id: format!("edge_{:02}_{}_to_{}", edges.len() + 1, names[0], names[j]),
            from_idx: 0,
            to_idx: j,
            from_name: names[0].clone(),
            to_name: names[j].clone(),
            shared_core_atoms: shared,
            core_overlap_ratio: (ratio * 1000.0).round() / 1000.0,
        });
    }

    // 2. High-overlap cycle-closure cross-edges between analogs (for n >= 3)
    if n >= 3 && edges.len() < max_edges {
        let mut cross = Vec::new();
        for i in 1..n {
            for j in (i + 1)..n {
                let (shared, ratio) = core_overlap_3d(&ligands[i], &ligands[j]);
                cross.push((ratio, shared, i, j));
            }
        }
        cross.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (ratio, shared, i, j) in cross {
            if edges.len() >= max_edges {
                break;
            }
            edges.push(RbfePlannedEdge {
                edge_id: format!("edge_{:02}_{}_to_{}", edges.len() + 1, names[i], names[j]),
                from_idx: i,
                to_idx: j,
                from_name: names[i].clone(),
                to_name: names[j].clone(),
                shared_core_atoms: shared,
                core_overlap_ratio: (ratio * 1000.0).round() / 1000.0,
            });
        }
    }
    edges
}

/// Computes 3-node thermodynamic cycle-closure errors from completed RBFE edge results.
#[must_use]
pub fn compute_cycle_closures(results: &[FepEdgeResult]) -> Vec<FepCycleClosure> {
    let mut ddg_map = std::collections::HashMap::<(String, String), f64>::new();
    let mut nodes_set = std::collections::BTreeSet::<String>::new();
    for r in results {
        ddg_map.insert((r.from_name.clone(), r.to_name.clone()), r.ddg_bind_kcal);
        ddg_map.insert((r.to_name.clone(), r.from_name.clone()), -r.ddg_bind_kcal);
        nodes_set.insert(r.from_name.clone());
        nodes_set.insert(r.to_name.clone());
    }
    let nodes: Vec<String> = nodes_set.into_iter().collect();
    let mut closures = Vec::new();
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            for k in (j + 1)..nodes.len() {
                if let (Some(&ab), Some(&bc), Some(&ca)) = (
                    ddg_map.get(&(nodes[i].clone(), nodes[j].clone())),
                    ddg_map.get(&(nodes[j].clone(), nodes[k].clone())),
                    ddg_map.get(&(nodes[k].clone(), nodes[i].clone())),
                ) {
                    closures.push(FepCycleClosure {
                        nodes: vec![
                            nodes[i].clone(),
                            nodes[j].clone(),
                            nodes[k].clone(),
                            nodes[i].clone(),
                        ],
                        closure_error_kcal: ((ab + bc + ca) * 100.0).round() / 100.0,
                    });
                }
            }
        }
    }
    closures
}

/// Plans and executes (or dry-run stages) an alchemical RBFE/FEP perturbation network.
pub fn run_fep_network(cfg: &FepConfig) -> Result<FepReport, FepError> {
    std::fs::create_dir_all(&cfg.out_dir)?;

    let sdf_raw = std::fs::read_to_string(&cfg.ligands_sdf_path)?;
    let parsed_ligs: Vec<Molecule3D> = parse_sdf_3d_with_limit(&sdf_raw, 12)
        .into_iter()
        .filter_map(Result::ok)
        .collect();
    if parsed_ligs.len() < 2 {
        return Err(FepError::Setup(format!(
            "RBFE/FEP requires at least 2 aligned 3D congeneric ligands in '{}', found {}",
            cfg.ligands_sdf_path.display(),
            parsed_ligs.len()
        )));
    }

    let rec_name = cfg
        .receptor_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("receptor")
        .to_string();
    let ot_name = cfg.offtarget_receptor_path.as_ref().map(|p| {
        p.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("offtarget")
            .to_string()
    });
    let ligand_names: Vec<String> = parsed_ligs
        .iter()
        .enumerate()
        .map(|(i, m)| {
            if m.name.trim().is_empty() {
                format!("lig_{i}")
            } else {
                m.name.trim().to_string()
            }
        })
        .collect();

    let planned_edges = plan_rbfe_network(&parsed_ligs, cfg.max_edges.clamp(1, 24));
    let n_windows = cfg.n_windows.clamp(5, 21);

    let config_path = cfg.out_dir.join("fep_config.json");
    let metrics_path = cfg.out_dir.join("fep_metrics.json");
    let report_md_path = cfg.out_dir.join("fep_report.md");

    let script_path = match cfg.backend {
        FepBackend::OpenmmAlchemical => {
            let p = cfg.out_dir.join("run_fep_openmm.py");
            std::fs::write(&p, OPENMM_RBFE_RUNNER_PY)?;
            p
        }
        FepBackend::OpenFe => {
            let p = cfg.out_dir.join("run_openfe_network.py");
            std::fs::write(&p, OPENFE_RUNNER_PY)?;
            p
        }
    };

    let json_cfg = serde_json::json!({
        "receptor_path": cfg.receptor_path.display().to_string(),
        "receptor_name": rec_name,
        "offtarget_receptor_path": cfg.offtarget_receptor_path.as_ref().map(|p| p.display().to_string()),
        "offtarget_name": ot_name,
        "ligands_sdf_path": cfg.ligands_sdf_path.display().to_string(),
        "out_dir": cfg.out_dir.display().to_string(),
        "backend": cfg.backend.as_str(),
        "n_windows": n_windows,
        "ns_per_window": cfg.ns_per_window,
        "equil_ps_per_window": cfg.equil_ps_per_window,
        "temperature_k": cfg.temperature_k,
        "padding_nm": cfg.padding_nm,
        "ionic_strength_m": cfg.ionic_strength_m,
        "charge_method": cfg.charge_method,
        "platform": cfg.platform.as_str(),
        "ligand_names": ligand_names,
        "planned_edges": planned_edges,
    });
    std::fs::write(&config_path, serde_json::to_string_pretty(&json_cfg)?)?;

    if cfg.dry_run || cfg.backend == FepBackend::OpenFe {
        let rep = FepReport {
            receptor_name: rec_name,
            offtarget_name: ot_name,
            backend: cfg.backend,
            dry_run: true,
            platform_used: format!("staged ({})", cfg.platform.as_str()),
            n_windows,
            ns_per_window: cfg.ns_per_window,
            ligand_names,
            planned_edges,
            edge_results: Vec::new(),
            cycle_closures: Vec::new(),
            wall_seconds: 0.0,
            out_dir: cfg.out_dir.display().to_string(),
            script_path: script_path.display().to_string(),
        };
        std::fs::write(&report_md_path, rep.render_markdown())?;
        std::fs::write(&metrics_path, serde_json::to_string_pretty(&rep)?)?;
        return Ok(rep);
    }

    let py = resolve_openmm_python(cfg.python_override.as_deref());
    if !py.exists() {
        return Err(FepError::EnvNotFound(py.display().to_string()));
    }

    let out = Command::new(&py)
        .arg(&script_path)
        .arg(&config_path)
        .output()?;
    if !out.status.success() {
        return Err(FepError::ExecutionFailed {
            code: out.status.code(),
            stderr: format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ),
        });
    }

    let raw = std::fs::read_to_string(&metrics_path)?;
    let mut rep: FepReport = serde_json::from_str(&raw)?;
    rep.script_path = script_path.display().to_string();
    if rep.cycle_closures.is_empty() && rep.edge_results.len() >= 3 {
        rep.cycle_closures = compute_cycle_closures(&rep.edge_results);
    }
    std::fs::write(&metrics_path, serde_json::to_string_pretty(&rep)?)?;
    std::fs::write(&report_md_path, rep.render_markdown())?;
    Ok(rep)
}

/// Optional external OpenFE export runner (when `CRAFT_DATA_DIR/envs/openfe` is installed).
pub const OPENFE_RUNNER_PY: &str = r#"#!/usr/bin/env python3
"""DockCraft v0.7.0 OpenFE External RBFE Network Exporter."""
import json
import sys
from pathlib import Path


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    try:
        import openfe
        from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
    except ImportError as exc:
        raise SystemExit(
            f"OpenFE is not installed in this Python environment ({exc}). "
            "Use `--backend openmm` (native OpenMM CUDA) or install OpenFE into CRAFT_DATA_DIR/envs/openfe."
        )
    print(f"[DockCraft OpenFE] OpenFE {openfe.__version__} ready for {len(cfg['planned_edges'])} edge(s).")
    print("[DockCraft OpenFE] Launch `openfe quickrun` on generated transformation JSONs in", cfg["out_dir"])


if __name__ == "__main__":
    main()
"#;

/// Self-contained OpenMM GPU Dual-Softcore Alchemical RBFE Engine (RDKit MCS + OpenFF Sage + NAGL + MBAR/TI).
pub const OPENMM_RBFE_RUNNER_PY: &str = r#"#!/usr/bin/env python3
"""DockCraft v0.7.0 OpenMM GPU Dual-Softcore Alchemical RBFE/FEP Engine (MBAR + TI + Overlap Diagnostics)."""
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
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS

KJ_TO_KCAL = 1.0 / 4.184
R_KCAL = 0.001987204259


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


def parameterize_rdmol_openff(rdmol, charge_method="nagl"):
    from openff.toolkit import ForceField as OffForceField, Molecule as OffMolecule
    from openff.interchange import Interchange

    mol = Chem.AddHs(rdmol, addCoords=True)
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
    off_mol = OffMolecule.from_rdkit(mol, allow_undefined_stereo=True)
    assigned = False
    if charge_method.lower() in ("nagl", "am1bcc"):
        try:
            from openff.nagl_models import list_available_nagl_models
            models = list_available_nagl_models()
            if models:
                off_mol.assign_partial_charges(partial_charge_method=str(models[-1]))
                assigned = True
        except Exception:
            pass
    if not assigned:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is not None:
            charges = [float(props.GetMMFFPartialCharge(i)) for i in range(mol.GetNumAtoms())]
        else:
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for a in mol.GetAtoms():
                q = float(a.GetDoubleProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0
                charges.append(0.0 if (math.isnan(q) or math.isinf(q)) else q)
        target_q = float(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
        drift = (sum(charges) - target_q) / max(len(charges), 1)
        off_mol.partial_charges = np.array([q - drift for q in charges]) * unit.elementary_charge

    off_ff = OffForceField("openff_unconstrained-2.2.1.offxml")
    ic = Interchange.from_smirnoff(off_ff, off_mol.to_topology(), charge_from_molecules=[off_mol])
    lsys = ic.to_openmm_system()
    conf = mol.GetConformer()
    pos_nm = np.array(
        [[conf.GetAtomPosition(i).x / 10.0, conf.GetAtomPosition(i).y / 10.0, conf.GetAtomPosition(i).z / 10.0]
         for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )
    return mol, lsys, pos_nm


def map_mcs_atoms(molA, molB, posA, posB):
    """Computes heavy-atom MCS via RDKit rdFMCS (ring-preserving) + equivalent core hydrogens."""
    heavyA = [a.GetIdx() for a in molA.GetAtoms() if a.GetAtomicNum() > 1]
    heavyB = [a.GetIdx() for a in molB.GetAtoms() if a.GetAtomicNum() > 1]
    noH_A = Chem.RemoveHs(molA)
    noH_B = Chem.RemoveHs(molB)

    res = rdFMCS.FindMCS(
        [noH_A, noH_B],
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        timeout=10,
    )
    if res.numAtoms < 3:
        raise RuntimeError(f"MCS too small ({res.numAtoms} atoms) for RBFE perturbation")
    patt = Chem.MolFromSmarts(res.smartsString)
    matchesA = molA.GetSubstructMatches(patt, uniquify=False)
    matchesB = molB.GetSubstructMatches(patt, uniquify=False)
    if not matchesA or not matchesB:
        raise RuntimeError("Failed to map MCS SMARTS onto full 3D ligands")

    # Select the atom correspondence minimizing 3D heavy-atom RMSD
    best_pair = None
    best_d2 = float("inf")
    mA0 = matchesA[0]
    for mB in matchesB[:128]:
        d2 = float(np.mean(np.sum((posA[list(mA0)] - posB[list(mB)]) ** 2, axis=1)))
        if d2 < best_d2:
            best_d2 = d2
            best_pair = (mA0, mB)
    for mA in matchesA[1:32]:
        for mB in matchesB[:32]:
            d2 = float(np.mean(np.sum((posA[list(mA)] - posB[list(mB)]) ** 2, axis=1)))
            if d2 < best_d2:
                best_d2 = d2
                best_pair = (mA, mB)

    core_pairs = list(zip(best_pair[0], best_pair[1]))
    core_heavy_count = len(core_pairs)

    # Also map hydrogens attached to core heavy atoms that have identical H-count & identical heavy neighbor set in core
    mappedA = {a for a, _ in core_pairs}
    mappedB = {b for _, b in core_pairs}
    for iA, iB in list(core_pairs):
        aA = molA.GetAtomWithIdx(iA)
        aB = molB.GetAtomWithIdx(iB)
        hA = [n.GetIdx() for n in aA.GetNeighbors() if n.GetAtomicNum() == 1]
        hB = [n.GetIdx() for n in aB.GetNeighbors() if n.GetAtomicNum() == 1]
        if len(hA) == len(hB) and len(hA) > 0:
            used_hb = set()
            for ha_idx in hA:
                best_hb = min((hb_idx for hb_idx in hB if hb_idx not in used_hb),
                              key=lambda hb_idx: float(np.sum((posA[ha_idx] - posB[hb_idx]) ** 2)))
                used_hb.add(best_hb)
                core_pairs.append((ha_idx, best_hb))
                mappedA.add(ha_idx)
                mappedB.add(best_hb)

    unique_A = [i for i in range(molA.GetNumAtoms()) if i not in mappedA]
    unique_B = [i for i in range(molB.GetNumAtoms()) if i not in mappedB]
    return core_pairs, unique_A, unique_B, core_heavy_count


def extract_lig_params(lsys):
    n = lsys.getNumParticles()
    masses = [lsys.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n)]
    bonds, angles, torsions = [], [], []
    q = np.zeros(n)
    sig = np.zeros(n)
    eps = np.zeros(n)
    exceptions = {}
    for f in lsys.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for bi in range(f.getNumBonds()):
                p1, p2, r0, k = f.getBondParameters(bi)
                bonds.append((p1, p2, r0.value_in_unit(unit.nanometers), k.value_in_unit(unit.kilojoules_per_mole / unit.nanometers**2)))
        elif isinstance(f, mm.HarmonicAngleForce):
            for ai in range(f.getNumAngles()):
                p1, p2, p3, th, k = f.getAngleParameters(ai)
                angles.append((p1, p2, p3, th.value_in_unit(unit.radians), k.value_in_unit(unit.kilojoules_per_mole / unit.radians**2)))
        elif isinstance(f, mm.PeriodicTorsionForce):
            for ti in range(f.getNumTorsions()):
                p1, p2, p3, p4, per, ph, k = f.getTorsionParameters(ti)
                torsions.append((p1, p2, p3, p4, int(per), ph.value_in_unit(unit.radians), k.value_in_unit(unit.kilojoules_per_mole)))
        elif isinstance(f, mm.NonbondedForce):
            for pi in range(n):
                qi, si, ei = f.getParticleParameters(pi)
                q[pi] = qi.value_in_unit(unit.elementary_charge)
                sig[pi] = si.value_in_unit(unit.nanometers)
                eps[pi] = ei.value_in_unit(unit.kilojoules_per_mole)
            for ei in range(f.getNumExceptions()):
                p1, p2, qq, s14, e14 = f.getExceptionParameters(ei)
                key = (min(p1, p2), max(p1, p2))
                exceptions[key] = (
                    qq.value_in_unit(unit.elementary_charge**2),
                    s14.value_in_unit(unit.nanometers),
                    e14.value_in_unit(unit.kilojoules_per_mole),
                )
    return {"n": n, "masses": masses, "bonds": bonds, "angles": angles, "torsions": torsions,
            "q": q, "sig": sig, "eps": eps, "exceptions": exceptions}


def build_hybrid_system_and_positions(
    prot_top, prot_pos, molA, molB, pA, pB, posA, posB, core_pairs, unique_A, unique_B, padding_nm, ionic_strength_m
):
    """
    Constructs a solvated OpenMM System with exact-neutrality core charge offsets and
    dual Beutler-softcore CustomNonbondedForces for unique_A and unique_B.
    """
    n_core = len(core_pairs)
    n_uA = len(unique_A)
    n_uB = len(unique_B)
    n_hyb = n_core + n_uA + n_uB

    # Hybrid indexing: 0..n_core-1 = core, n_core..n_core+n_uA-1 = unique_A, rest = unique_B
    mapA_to_h = {}
    mapB_to_h = {}
    hyb_pos_nm = np.zeros((n_hyb, 3), dtype=np.float64)
    hyb_elems = []
    hyb_masses = []
    for h_idx, (iA, iB) in enumerate(core_pairs):
        mapA_to_h[iA] = h_idx
        mapB_to_h[iB] = h_idx
        hyb_pos_nm[h_idx] = 0.5 * (posA[iA] + posB[iB])
        hyb_elems.append(app.Element.getByAtomicNumber(molA.GetAtomWithIdx(iA).GetAtomicNum()))
        hyb_masses.append(max(pA["masses"][iA], pB["masses"][iB]))

    for k, iA in enumerate(unique_A):
        h_idx = n_core + k
        mapA_to_h[iA] = h_idx
        hyb_pos_nm[h_idx] = posA[iA]
        hyb_elems.append(app.Element.getByAtomicNumber(molA.GetAtomWithIdx(iA).GetAtomicNum()))
        hyb_masses.append(pA["masses"][iA])

    # Place unique_B using posB aligned onto hybrid core centroid shift
    core_shift = hyb_pos_nm[:n_core].mean(axis=0) - posB[[iB for _, iB in core_pairs]].mean(axis=0)
    for k, iB in enumerate(unique_B):
        h_idx = n_core + n_uA + k
        mapB_to_h[iB] = h_idx
        hyb_pos_nm[h_idx] = posB[iB] + core_shift
        hyb_elems.append(app.Element.getByAtomicNumber(molB.GetAtomWithIdx(iB).GetAtomicNum()))
        hyb_masses.append(pB["masses"][iB])

    # Build a dummy-parametrized topology for solvation box sizing (so waters don't clash with hybrid ligand)
    hyb_top = app.Topology()
    ch = hyb_top.addChain("L")
    res = hyb_top.addResidue("HYB", ch)
    top_atoms = [hyb_top.addAtom(f"H{i}", hyb_elems[i], res) for i in range(n_hyb)]
    bond_seen = set()
    for p1, p2, _, _ in pA["bonds"]:
        u, v = mapA_to_h[p1], mapA_to_h[p2]
        if (min(u, v), max(u, v)) not in bond_seen:
            bond_seen.add((min(u, v), max(u, v)))
            hyb_top.addBond(top_atoms[u], top_atoms[v])
    for p1, p2, _, _ in pB["bonds"]:
        u, v = mapB_to_h[p1], mapB_to_h[p2]
        if (min(u, v), max(u, v)) not in bond_seen:
            bond_seen.add((min(u, v), max(u, v)))
            hyb_top.addBond(top_atoms[u], top_atoms[v])

    # Emit lightweight neutral placeholder FFXML for HYB solely so Modeller.addSolvent places waters cleanly
    xml_lines = ["<ForceField>", "  <AtomTypes>"]
    for i in range(n_hyb):
        xml_lines.append(f'    <Type name="HYB_{i}" class="HYB_{i}" element="{hyb_elems[i].symbol}" mass="{hyb_masses[i]:.4f}"/>')
    xml_lines.extend(["  </AtomTypes>", "  <Residues>", '    <Residue name="HYB">'])
    for i in range(n_hyb):
        xml_lines.append(f'      <Atom name="H{i}" type="HYB_{i}"/>')
    for u, v in sorted(bond_seen):
        xml_lines.append(f'      <Bond atomName1="H{u}" atomName2="H{v}"/>')
    xml_lines.extend(["    </Residue>", "  </Residues>", '  <NonbondedForce coulomb14scale="0.8333333333333334" lj14scale="0.5">'])
    for i in range(n_hyb):
        xml_lines.append(f'    <Atom type="HYB_{i}" charge="0.0" sigma="0.30" epsilon="0.20"/>')
    xml_lines.extend(["  </NonbondedForce>", "</ForceField>"])

    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml", io.StringIO("\n".join(xml_lines)))
    if prot_top is not None:
        mod = app.Modeller(prot_top, prot_pos)
        lig_offset = mod.topology.getNumAtoms()
        mod.add(hyb_top, [mm.Vec3(*p) for p in hyb_pos_nm] * unit.nanometers)
    else:
        mod = app.Modeller(hyb_top, [mm.Vec3(*p) for p in hyb_pos_nm] * unit.nanometers)
        lig_offset = 0

    mod.addSolvent(
        ff,
        model="tip3p",
        padding=float(padding_nm) * unit.nanometers,
        ionicStrength=float(ionic_strength_m) * unit.molar,
        neutralize=True,
    )

    # Create base system (protein + solvent fully parameterized; HYB has zero bonds & placeholder NB)
    system = ff.createSystem(
        mod.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
    )
    total_atoms = system.getNumParticles()

    # Locate existing bonded/nonbonded forces
    hb_force = next((f for f in system.getForces() if isinstance(f, mm.HarmonicBondForce)), None)
    if hb_force is None:
        hb_force = mm.HarmonicBondForce()
        system.addForce(hb_force)
    ha_force = next((f for f in system.getForces() if isinstance(f, mm.HarmonicAngleForce)), None)
    if ha_force is None:
        ha_force = mm.HarmonicAngleForce()
        system.addForce(ha_force)
    pt_force = next((f for f in system.getForces() if isinstance(f, mm.PeriodicTorsionForce)), None)
    if pt_force is None:
        pt_force = mm.PeriodicTorsionForce()
        system.addForce(pt_force)
    nb_force = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))

    # Remove any auto-generated constraint on HYB atoms so hybrid H-bonds stay unconstrained (dt=1.0 fs)
    for ci in reversed(range(system.getNumConstraints())):
        a1, a2, _ = system.getConstraintParameters(ci)
        if lig_offset <= a1 < lig_offset + n_hyb or lig_offset <= a2 < lig_offset + n_hyb:
            system.removeConstraint(ci)

    # Add hybrid bonded terms: core terms from A (exact shared scaffold), plus all unique_A and unique_B bonded terms
    core_set = set(range(n_core))
    added_bonds = set()
    for p1, p2, r0, k in pA["bonds"]:
        u, v = lig_offset + mapA_to_h[p1], lig_offset + mapA_to_h[p2]
        hb_force.addBond(u, v, r0, k)
        added_bonds.add((min(u, v), max(u, v)))
    for p1, p2, r0, k in pB["bonds"]:
        u, v = lig_offset + mapB_to_h[p1], lig_offset + mapB_to_h[p2]
        if (min(u, v), max(u, v)) not in added_bonds:
            hb_force.addBond(u, v, r0, k)
            added_bonds.add((min(u, v), max(u, v)))

    added_angles = set()
    for p1, p2, p3, th, k in pA["angles"]:
        u, v, w = lig_offset + mapA_to_h[p1], lig_offset + mapA_to_h[p2], lig_offset + mapA_to_h[p3]
        key = (min(u, w), v, max(u, w))
        ha_force.addAngle(u, v, w, th, k)
        added_angles.add(key)
    for p1, p2, p3, th, k in pB["angles"]:
        u, v, w = lig_offset + mapB_to_h[p1], lig_offset + mapB_to_h[p2], lig_offset + mapB_to_h[p3]
        key = (min(u, w), v, max(u, w))
        if key not in added_angles:
            ha_force.addAngle(u, v, w, th, k)
            added_angles.add(key)

    core_tors_seen = set()
    for p1, p2, p3, p4, per, ph, k in pA["torsions"]:
        h1, h2, h3, h4 = mapA_to_h[p1], mapA_to_h[p2], mapA_to_h[p3], mapA_to_h[p4]
        pt_force.addTorsion(lig_offset + h1, lig_offset + h2, lig_offset + h3, lig_offset + h4, per, ph, k)
        if h1 in core_set and h2 in core_set and h3 in core_set and h4 in core_set:
            core_tors_seen.add((h1, h2, h3, h4, per))
            core_tors_seen.add((h4, h3, h2, h1, per))
    for p1, p2, p3, p4, per, ph, k in pB["torsions"]:
        h1, h2, h3, h4 = mapB_to_h[p1], mapB_to_h[p2], mapB_to_h[p3], mapB_to_h[p4]
        if h1 in core_set and h2 in core_set and h3 in core_set and h4 in core_set:
            if (h1, h2, h3, h4, per) in core_tors_seen:
                continue
        pt_force.addTorsion(lig_offset + h1, lig_offset + h2, lig_offset + h3, lig_offset + h4, per, ph, k)

    # Exact-neutrality NonbondedForce alchemical parameters
    nb_force.addGlobalParameter("lam_elec_A", 1.0)
    nb_force.addGlobalParameter("lam_elec_B", 0.0)
    nb_force.addGlobalParameter("lam_core", 0.0)

    # Ensure identical net formal charge between A and B (zeroing tiny numerical charge drift)
    q_uA_sum = float(sum(pA["q"][iA] for iA in unique_A))
    q_uB_sum = float(sum(pB["q"][iB] for iB in unique_B))
    net_A = float(np.sum(pA["q"]))
    net_B = float(np.sum(pB["q"]))
    dq_net = (net_A - net_B) / max(n_core, 1)

    for h_idx, (iA, iB) in enumerate(core_pairs):
        g_idx = lig_offset + h_idx
        qA_i = float(pA["q"][iA])
        qB_i = float(pB["q"][iB]) + dq_net
        sigA_i, epsA_i = float(pA["sig"][iA]), float(pA["eps"][iA])
        sigB_i, epsB_i = float(pB["sig"][iB]), float(pB["eps"][iB])
        # Base at (lam_elec_A=0, lam_core=0, lam_elec_B=0) = qA_i + q_uA_sum/n_core
        q_base = qA_i + q_uA_sum / max(n_core, 1)
        nb_force.setParticleParameters(g_idx, q_base, sigA_i, epsA_i)
        nb_force.addParticleParameterOffset("lam_elec_A", g_idx, -q_uA_sum / max(n_core, 1), 0.0, 0.0)
        nb_force.addParticleParameterOffset(
            "lam_core",
            g_idx,
            (qB_i + q_uB_sum / max(n_core, 1)) - q_base,
            sigB_i - sigA_i,
            epsB_i - epsA_i,
        )
        nb_force.addParticleParameterOffset("lam_elec_B", g_idx, -q_uB_sum / max(n_core, 1), 0.0, 0.0)

    for k, iA in enumerate(unique_A):
        g_idx = lig_offset + n_core + k
        nb_force.setParticleParameters(g_idx, 0.0, 0.30, 0.0)
        nb_force.addParticleParameterOffset("lam_elec_A", g_idx, float(pA["q"][iA]), 0.0, 0.0)

    for k, iB in enumerate(unique_B):
        g_idx = lig_offset + n_core + n_uA + k
        nb_force.setParticleParameters(g_idx, 0.0, 0.30, 0.0)
        nb_force.addParticleParameterOffset("lam_elec_B", g_idx, float(pB["q"][iB]), 0.0, 0.0)

    # Index existing HYB exceptions in nb_force and update/add exact exclusions & 1-4 scaling
    ex_idx_map = {}
    for ei in range(nb_force.getNumExceptions()):
        a1, a2, _, _, _ = nb_force.getExceptionParameters(ei)
        if lig_offset <= a1 < lig_offset + n_hyb and lig_offset <= a2 < lig_offset + n_hyb:
            ex_idx_map[(min(a1, a2), max(a1, a2))] = ei

    def set_nb_exc(u, v, qq, s14, e14):
        key = (min(u, v), max(u, v))
        if key in ex_idx_map:
            nb_force.setExceptionParameters(ex_idx_map[key], u, v, qq, s14, e14)
            return ex_idx_map[key]
        idx_new = nb_force.addException(u, v, qq, s14, e14, True)
        ex_idx_map[key] = idx_new
        return idx_new

    # Collect softcore 1-4 LJ & Coulomb terms for unique_A and unique_B cleanly via CustomBondForce
    c14_force = mm.CustomBondForce(
        "lam_ster * 4.0 * eps14 * ((sig14 / r)^12 - (sig14 / r)^6) + lam_elec * 138.935458 * qq14 / r"
    )
    # Use two clean CustomBondForces (one for A-involving 1-4 pairs, one for B-involving 1-4 pairs)
    cb14_A = mm.CustomBondForce(
        "lam_ster_A * 4.0 * eps14 * ((sig14 / r)^12 - (sig14 / r)^6) + lam_elec_A * 138.935458 * qq14 / r"
    )
    cb14_A.addGlobalParameter("lam_ster_A", 1.0)
    cb14_A.addGlobalParameter("lam_elec_A", 1.0)
    cb14_A.addPerBondParameter("qq14")
    cb14_A.addPerBondParameter("sig14")
    cb14_A.addPerBondParameter("eps14")

    cb14_B = mm.CustomBondForce(
        "lam_ster_B * 4.0 * eps14 * ((sig14 / r)^12 - (sig14 / r)^6) + lam_elec_B * 138.935458 * qq14 / r"
    )
    cb14_B.addGlobalParameter("lam_ster_B", 0.0)
    cb14_B.addGlobalParameter("lam_elec_B", 0.0)
    cb14_B.addPerBondParameter("qq14")
    cb14_B.addPerBondParameter("sig14")
    cb14_B.addPerBondParameter("eps14")

    sc_excl_pairs = set()
    # Exclude ALL (unique_A, unique_B) pairs in NonbondedForce
    for kA in range(n_uA):
        gA = lig_offset + n_core + kA
        for kB in range(n_uB):
            gB = lig_offset + n_core + n_uA + kB
            set_nb_exc(gA, gB, 0.0, 0.30, 0.0)
            sc_excl_pairs.add((min(gA, gB), max(gA, gB)))

    # Process exceptions of A
    core_14_handled = set()
    for (p1, p2), (qq, s14, e14) in pA["exceptions"].items():
        h1, h2 = mapA_to_h[p1], mapA_to_h[p2]
        g1, g2 = lig_offset + h1, lig_offset + h2
        pair = (min(g1, g2), max(g1, g2))
        sc_excl_pairs.add(pair)
        if h1 in core_set and h2 in core_set:
            ei = set_nb_exc(g1, g2, qq, max(s14, 0.1), e14)
            core_14_handled.add(pair)
            if (min(core_pairs[h1][1], core_pairs[h2][1]), max(core_pairs[h1][1], core_pairs[h2][1])) in pB["exceptions"]:
                qqB, s14B, e14B = pB["exceptions"][(min(core_pairs[h1][1], core_pairs[h2][1]), max(core_pairs[h1][1], core_pairs[h2][1]))]
                nb_force.addExceptionParameterOffset("lam_core", ei, qqB - qq, max(s14B, 0.1) - max(s14, 0.1), e14B - e14)
        else:
            set_nb_exc(g1, g2, 0.0, 0.30, 0.0)
            if abs(qq) > 1e-10 or e14 > 1e-10:
                cb14_A.addBond(g1, g2, [qq, max(s14, 0.1), e14])

    # Process exceptions of B
    for (p1, p2), (qq, s14, e14) in pB["exceptions"].items():
        h1, h2 = mapB_to_h[p1], mapB_to_h[p2]
        g1, g2 = lig_offset + h1, lig_offset + h2
        pair = (min(g1, g2), max(g1, g2))
        sc_excl_pairs.add(pair)
        if h1 in core_set and h2 in core_set:
            if pair not in core_14_handled:
                ei = set_nb_exc(g1, g2, 0.0, max(s14, 0.1), 0.0)
                nb_force.addExceptionParameterOffset("lam_core", ei, qq, 0.0, e14)
        else:
            set_nb_exc(g1, g2, 0.0, 0.30, 0.0)
            if abs(qq) > 1e-10 or e14 > 1e-10:
                cb14_B.addBond(g1, g2, [qq, max(s14, 0.1), e14])

    if cb14_A.getNumBonds() > 0:
        system.addForce(cb14_A)
    if cb14_B.getNumBonds() > 0:
        system.addForce(cb14_B)

    # Dual Beutler Softcore LJ CustomNonbondedForces for unique_A and unique_B
    uA_global = {lig_offset + n_core + k for k in range(n_uA)}
    uB_global = {lig_offset + n_core + n_uA + k for k in range(n_uB)}
    all_global = set(range(total_atoms))

    # Extract LJ params for all particles
    sig_all = np.zeros(total_atoms)
    eps_all = np.zeros(total_atoms)
    for idx in range(total_atoms):
        _, s_i, e_i = nb_force.getParticleParameters(idx)
        sig_all[idx] = s_i.value_in_unit(unit.nanometers)
        eps_all[idx] = e_i.value_in_unit(unit.kilojoules_per_mole)
    for k, iA in enumerate(unique_A):
        g = lig_offset + n_core + k
        sig_all[g] = float(pA["sig"][iA])
        eps_all[g] = float(pA["eps"][iA])
    for k, iB in enumerate(unique_B):
        g = lig_offset + n_core + n_uA + k
        sig_all[g] = float(pB["sig"][iB])
        eps_all[g] = float(pB["eps"][iB])

    # Collect exact exclusion list from NonbondedForce (CUDA requires identical exclusions across all nonbonded forces)
    nb_excl_pairs = []
    for ei in range(nb_force.getNumExceptions()):
        a1, a2, _, _, _ = nb_force.getExceptionParameters(ei)
        nb_excl_pairs.append((a1, a2))

    def add_softcore_force(param_name, init_val, active_set, excluded_set):
        if not active_set:
            return
        expr = (
            f"4.0 * {param_name} * eps * "
            f"(1.0 / (0.5 * (1.0 - {param_name}) + (r / sig)^6)^2 - "
            f"1.0 / (0.5 * (1.0 - {param_name}) + (r / sig)^6)); "
            "sig = 0.5 * (sig1 + sig2); eps = sqrt(eps1 * eps2)"
        )
        sc = mm.CustomNonbondedForce(expr)
        sc.addGlobalParameter(param_name, init_val)
        sc.addPerParticleParameter("sig")
        sc.addPerParticleParameter("eps")
        sc.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        sc.setCutoffDistance(1.0 * unit.nanometers)
        sc.setUseSwitchingFunction(True)
        sc.setSwitchingDistance(0.9 * unit.nanometers)
        sc.setUseLongRangeCorrection(False)
        for idx in range(total_atoms):
            sc.addParticle([float(max(sig_all[idx], 0.05)), float(max(eps_all[idx], 0.0))])
        for u, v in nb_excl_pairs:
            sc.addExclusion(u, v)
        sc.addInteractionGroup(active_set, all_global - excluded_set)
        system.addForce(sc)

    add_softcore_force("lam_ster_A", 1.0, uA_global, uB_global)
    add_softcore_force("lam_ster_B", 0.0, uB_global, uA_global)
    return system, mod.topology, mod.positions


def make_lambda_schedule(n_windows):
    """Staged electrostatic-before-sterics alchemical schedule (no naked charges at any window)."""
    s_grid = np.linspace(0.0, 1.0, n_windows)
    sched = []
    for s in s_grid:
        lam_elec_A = float(np.clip(1.0 - s / 0.35, 0.0, 1.0))
        lam_ster_A = float(np.clip(1.0 - (s - 0.20) / 0.60, 0.0, 1.0))
        lam_ster_B = float(np.clip((s - 0.20) / 0.60, 0.0, 1.0))
        lam_elec_B = float(np.clip((s - 0.65) / 0.35, 0.0, 1.0))
        lam_core = float(s)
        sched.append({
            "s": float(s),
            "lam_elec_A": lam_elec_A,
            "lam_ster_A": lam_ster_A,
            "lam_ster_B": lam_ster_B,
            "lam_elec_B": lam_elec_B,
            "lam_core": lam_core,
        })
    return sched


def set_context_lambdas(ctx, lam_dict):
    params = ctx.getParameters()
    for k in ("lam_elec_A", "lam_ster_A", "lam_ster_B", "lam_elec_B", "lam_core"):
        if k in params:
            ctx.setParameter(k, float(lam_dict[k]))


def solve_mbar_and_overlap(u_kn, n_k, temp_k):
    """Solves exact MBAR free energy + overlap matrix using pymbar (with pure-NumPy BAR fallback)."""
    kbt_kcal = R_KCAL * temp_k
    K = len(n_k)
    try:
        import pymbar
        mbar = pymbar.MBAR(u_kn, n_k)
        if hasattr(mbar, "compute_free_energy_differences"):
            res = mbar.compute_free_energy_differences()
            df = float(res["Delta_f"][0, K - 1]) * kbt_kcal
            ddf = float(res["dDelta_f"][0, K - 1]) * kbt_kcal
            ov_mat = mbar.compute_overlap()["matrix"]
        else:
            df_mat, ddf_mat = mbar.getFreeEnergyDifferences()
            df = float(df_mat[0, K - 1]) * kbt_kcal
            ddf = float(ddf_mat[0, K - 1]) * kbt_kcal
            ov_mat = mbar.computeOverlap()["matrix"]
        adj_ov = [round(float(ov_mat[i, i + 1]), 4) for i in range(K - 1)]
        return df, ddf, adj_ov
    except Exception as exc:
        print(f"[DockCraft FEP] pymbar fallback ({exc}); solving adjacent BAR in NumPy", flush=True)
        # Exact self-consistent Bennett Acceptance Ratio (BAR) over adjacent window pairs
        df_total = 0.0
        var_total = 0.0
        adj_ov = []
        offset = 0
        for i in range(K - 1):
            ni = int(n_k[i])
            nj = int(n_k[i + 1])
            w_f = u_kn[i + 1, offset:offset + ni] - u_kn[i, offset:offset + ni]
            w_r = u_kn[i, offset + ni:offset + ni + nj] - u_kn[i + 1, offset + ni:offset + ni + nj]
            offset += ni
            c = 0.5 * (float(np.mean(w_f)) - float(np.mean(w_r)))
            for _ in range(50):
                f_f = 1.0 / (1.0 + np.exp(np.clip(w_f - c, -50, 50)))
                f_r = 1.0 / (1.0 + np.exp(np.clip(w_r + c, -50, 50)))
                c_new = c + math.log(max(float(np.mean(f_r)), 1e-12) / max(float(np.mean(f_f)), 1e-12))
                if abs(c_new - c) < 1e-6:
                    c = c_new
                    break
                c = c_new
            df_total += c
            # Bhattacharyya overlap proxy between adjacent windows
            ov = float(np.mean(2.0 * np.minimum(1.0 / (1.0 + np.exp(np.clip(w_f - c, -40, 40))),
                                                 1.0 - 1.0 / (1.0 + np.exp(np.clip(w_f - c, -40, 40))))))
            adj_ov.append(round(max(0.0, min(1.0, ov)), 4))
            var_total += (float(np.var(w_f)) / max(ni, 1) + float(np.var(w_r)) / max(nj, 1)) * 0.25
        return df_total * kbt_kcal, math.sqrt(var_total) * kbt_kcal, adj_ov


def run_alchemical_leg(
    leg_name, system, topology, positions, schedule, ns_per_window, equil_ps_per_window, temp_k, platform, props, seed
):
    t_leg0 = time.time()
    temp = temp_k * unit.kelvin
    beta = 1.0 / (0.008314462618 * temp_k)  # mol/kJ
    dt_fs = 1.0  # Unconditionally stable for unconstrained hybrid ligand H-bonds
    barostat = mm.MonteCarloBarostat(1.0 * unit.bar, temp, 25)
    barostat.setRandomNumberSeed(seed + 31)
    system.addForce(barostat)

    integrator = mm.LangevinMiddleIntegrator(temp, 2.0 / unit.picoseconds, dt_fs * unit.femtoseconds)
    integrator.setRandomNumberSeed(seed)
    sim = app.Simulation(topology, system, integrator, platform, props)
    sim.context.setPositions(positions)

    K = len(schedule)
    # Initial minimization at s=0.5 (both softcore regions partially present so water/pocket avoids both!)
    set_context_lambdas(sim.context, schedule[K // 2])
    sim.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometers, maxIterations=300)
    sim.context.setVelocitiesToTemperature(temp, seed)

    equil_steps = max(1000, int(round(equil_ps_per_window * 1000.0 / dt_fs)))
    prod_steps = max(2000, int(round(ns_per_window * 1_000_000.0 / dt_fs)))
    samples_per_win = max(20, min(100, int(round(ns_per_window * 1000.0 / 2.0))))
    sample_stride = max(50, prod_steps // samples_per_win)
    actual_samples = prod_steps // sample_stride

    n_k = np.full(K, actual_samples, dtype=np.int64)
    u_kn = np.zeros((K, K * actual_samples), dtype=np.float64)
    ti_duds = np.zeros(K, dtype=np.float64)
    col = 0
    ds = 1e-3

    for k_idx, lam in enumerate(schedule):
        set_context_lambdas(sim.context, lam)
        sim.minimizeEnergy(maxIterations=50)
        sim.step(equil_steps)
        duds_win = []
        for _ in range(actual_samples):
            sim.step(sample_stride)
            # Evaluate exact reduced potential u_{m, n} across all K windows at current snapshot
            for m_idx, lam_m in enumerate(schedule):
                set_context_lambdas(sim.context, lam_m)
                e_kj = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                u_kn[m_idx, col] = beta * e_kj

            # Finite-difference dU/ds for independent TI cross-check
            s0 = lam["s"]
            s_lo = max(0.0, s0 - ds)
            s_hi = min(1.0, s0 + ds)
            lam_lo = {
                "lam_elec_A": float(np.clip(1.0 - s_lo / 0.35, 0.0, 1.0)),
                "lam_ster_A": float(np.clip(1.0 - (s_lo - 0.20) / 0.60, 0.0, 1.0)),
                "lam_ster_B": float(np.clip((s_lo - 0.20) / 0.60, 0.0, 1.0)),
                "lam_elec_B": float(np.clip((s_lo - 0.65) / 0.35, 0.0, 1.0)),
                "lam_core": s_lo,
            }
            lam_hi = {
                "lam_elec_A": float(np.clip(1.0 - s_hi / 0.35, 0.0, 1.0)),
                "lam_ster_A": float(np.clip(1.0 - (s_hi - 0.20) / 0.60, 0.0, 1.0)),
                "lam_ster_B": float(np.clip((s_hi - 0.20) / 0.60, 0.0, 1.0)),
                "lam_elec_B": float(np.clip((s_hi - 0.65) / 0.35, 0.0, 1.0)),
                "lam_core": s_hi,
            }
            set_context_lambdas(sim.context, lam_lo)
            e_lo = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            set_context_lambdas(sim.context, lam_hi)
            e_hi = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            duds_win.append(((e_hi - e_lo) / max(s_hi - s_lo, 1e-6)) * KJ_TO_KCAL)

            # Restore active window lambda before continuing MD
            set_context_lambdas(sim.context, lam)
            col += 1
        ti_duds[k_idx] = float(np.mean(duds_win))
        print(f"[DockCraft FEP] {leg_name}: window {k_idx + 1}/{K} (s={lam['s']:.2f}) <dU/ds>={ti_duds[k_idx]:+.2f} kcal/mol", flush=True)

    dg_mbar, sem_mbar, adj_ov = solve_mbar_and_overlap(u_kn, n_k, temp_k)
    s_vals = np.array([l["s"] for l in schedule], dtype=np.float64)
    dg_ti = float(np.trapezoid(ti_duds, s_vals)) if hasattr(np, "trapezoid") else float(np.trapz(ti_duds, s_vals))
    min_ov = min(adj_ov) if adj_ov else 0.0
    wall_s = time.time() - t_leg0
    print(f"[DockCraft FEP] {leg_name} COMPLETE: MBAR={dg_mbar:+.2f}±{sem_mbar:.2f} | TI={dg_ti:+.2f} kcal/mol | min_overlap={min_ov:.3f} ({wall_s:.1f}s)", flush=True)
    return {
        "leg_name": leg_name,
        "dg_mbar_kcal": round(dg_mbar, 3),
        "dg_mbar_sem_kcal": round(sem_mbar, 3),
        "dg_ti_kcal": round(dg_ti, 3),
        "min_adjacent_overlap": round(min_ov, 4),
        "overlap_healthy": bool(min_ov >= 0.03),
        "adjacent_overlaps": adj_ov,
        "wall_seconds": round(wall_s, 2),
    }


def main():
    t0 = time.time()
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    platform, props, plat_name = select_platform(cfg.get("platform", "auto"))
    print(f"[DockCraft FEP] Platform: {plat_name} {props}", flush=True)

    prot_top, prot_pos = load_clean_protein(cfg["receptor_path"])
    ot_top, ot_pos = (None, None)
    if cfg.get("offtarget_receptor_path"):
        ot_top, ot_pos = load_clean_protein(cfg["offtarget_receptor_path"])

    suppl = Chem.SDMolSupplier(cfg["ligands_sdf_path"], removeHs=False, sanitize=True)
    raw_mols = [m for m in suppl if m is not None]
    param_cache = {}
    for idx, m in enumerate(raw_mols):
        mol_h, lsys, pos_nm = parameterize_rdmol_openff(m, cfg.get("charge_method", "nagl"))
        param_cache[idx] = (mol_h, extract_lig_params(lsys), pos_nm)

    n_windows = int(cfg.get("n_windows", 11))
    ns_per_win = float(cfg.get("ns_per_window", 0.5))
    equil_ps = min(float(cfg.get("equil_ps_per_window", 20.0)), max(4.0, ns_per_win * 250.0))
    temp_k = float(cfg.get("temperature_k", 300.0))
    padding_nm = float(cfg.get("padding_nm", 0.95))
    ionic_m = float(cfg.get("ionic_strength_m", 0.15))
    schedule = make_lambda_schedule(n_windows)

    edge_results = []
    for ei, edge in enumerate(cfg["planned_edges"]):
        iA, iB = int(edge["from_idx"]), int(edge["to_idx"])
        molA, pA, posA = param_cache[iA]
        molB, pB, posB = param_cache[iB]
        core_pairs, unique_A, unique_B, core_heavy = map_mcs_atoms(molA, molB, posA, posB)
        print(
            f"[DockCraft FEP] === Edge {ei + 1}/{len(cfg['planned_edges'])}: {edge['from_name']} -> {edge['to_name']} "
            f"(MCS={core_heavy} heavy, uniqueA={len(unique_A)}, uniqueB={len(unique_B)}) ===",
            flush=True,
        )

        # Leg 1: Complex (Target)
        sys_c, top_c, pos_c = build_hybrid_system_and_positions(
            prot_top, prot_pos, molA, molB, pA, pB, posA, posB, core_pairs, unique_A, unique_B, padding_nm, ionic_m
        )
        res_c = run_alchemical_leg(
            "complex", sys_c, top_c, pos_c, schedule, ns_per_win, equil_ps, temp_k, platform, props, 202600 + ei * 97
        )

        # Leg 2: Solvent (Water)
        sys_s, top_s, pos_s = build_hybrid_system_and_positions(
            None, None, molA, molB, pA, pB, posA, posB, core_pairs, unique_A, unique_B, max(1.1, padding_nm), ionic_m
        )
        res_s = run_alchemical_leg(
            "solvent", sys_s, top_s, pos_s, schedule, ns_per_win, equil_ps, temp_k, platform, props, 302600 + ei * 97
        )

        # Optional Leg 3: Off-Target Complex (Selectivity FEP)
        res_ot = None
        ddg_sel = None
        ddg_sel_sem = None
        if ot_top is not None:
            sys_ot, top_ot, pos_ot = build_hybrid_system_and_positions(
                ot_top, ot_pos, molA, molB, pA, pB, posA, posB, core_pairs, unique_A, unique_B, padding_nm, ionic_m
            )
            res_ot = run_alchemical_leg(
                "offtarget_complex", sys_ot, top_ot, pos_ot, schedule, ns_per_win, equil_ps, temp_k, platform, props, 402600 + ei * 97
            )

        ddg = res_c["dg_mbar_kcal"] - res_s["dg_mbar_kcal"]
        ddg_sem = math.sqrt(res_c["dg_mbar_sem_kcal"] ** 2 + res_s["dg_mbar_sem_kcal"] ** 2)
        ddg_ti = res_c["dg_ti_kcal"] - res_s["dg_ti_kcal"]
        fold_ki = math.exp(max(-20.0, min(20.0, ddg / (R_KCAL * temp_k))))
        if res_ot is not None:
            ddg_ot = res_ot["dg_mbar_kcal"] - res_s["dg_mbar_kcal"]
            ddg_sel = round(ddg - ddg_ot, 3)
            ddg_sel_sem = round(math.sqrt(res_c["dg_mbar_sem_kcal"] ** 2 + res_ot["dg_mbar_sem_kcal"] ** 2), 3)

        ov_ok = res_c["overlap_healthy"] and res_s["overlap_healthy"]
        ti_diff = abs(ddg - ddg_ti)
        if ov_ok and ti_diff <= 1.5:
            status = "Healthy"
        elif not ov_ok:
            status = "Warning: Low Window Overlap (<0.03)"
        else:
            status = f"Review: MBAR/TI diff ({ti_diff:.2f} kcal/mol)"

        edge_results.append({
            "edge_id": edge["edge_id"],
            "from_name": edge["from_name"],
            "to_name": edge["to_name"],
            "mcs_heavy_atoms": core_heavy,
            "unique_a_atoms": len(unique_A),
            "unique_b_atoms": len(unique_B),
            "complex_leg": res_c,
            "solvent_leg": res_s,
            "offtarget_leg": res_ot,
            "ddg_bind_kcal": round(ddg, 3),
            "ddg_sem_kcal": round(ddg_sem, 3),
            "ddg_ti_kcal": round(ddg_ti, 3),
            "fold_change_ki": round(fold_ki, 3),
            "ddg_selectivity_kcal": ddg_sel,
            "ddg_selectivity_sem_kcal": ddg_sel_sem,
            "convergence_status": status,
        })

    out_report = {
        "receptor_name": cfg["receptor_name"],
        "offtarget_name": cfg.get("offtarget_name"),
        "backend": "OpenmmAlchemical",
        "dry_run": False,
        "platform_used": plat_name,
        "n_windows": n_windows,
        "ns_per_window": ns_per_win,
        "ligand_names": cfg["ligand_names"],
        "planned_edges": cfg["planned_edges"],
        "edge_results": edge_results,
        "cycle_closures": [],
        "wall_seconds": round(time.time() - t0, 2),
        "out_dir": str(out_dir),
        "script_path": str(out_dir / "run_fep_openmm.py"),
    }
    (out_dir / "fep_metrics.json").write_text(json.dumps(out_report, indent=2))
    print(f"[DockCraft FEP] Finished {len(edge_results)} edge(s) in {out_report['wall_seconds']:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::{Atom3D, BondOrder, write_sdf_3d};

    fn make_congeneric_lig(name: &str, extra_substituent: Option<(&str, [f64; 3])>) -> Molecule3D {
        let mut m = Molecule3D::new(name);
        let ring = [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.1, 1.2, 0.0],
            [1.4, 2.4, 0.0],
            [0.0, 2.4, 0.0],
            [-0.7, 1.2, 0.0],
        ];
        for (i, &c) in ring.iter().enumerate() {
            m.add_atom(Atom3D::new(i, "C", c));
        }
        for i in 0..6 {
            m.add_bond(i, (i + 1) % 6, BondOrder::Aromatic);
        }
        if let Some((el, coords)) = extra_substituent {
            let idx = m.add_atom(Atom3D::new(m.atoms.len(), el, coords));
            m.add_bond(1, idx, BondOrder::Single);
        }
        struct_core::perceive_all(&mut m);
        m
    }

    #[test]
    fn fep_network_planner_builds_cycle_closure_graph_and_dry_run_bundles()
    -> Result<(), Box<dyn std::error::Error>> {
        let base = std::env::temp_dir().join(format!("dockcraft_fep_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base)?;

        let l0 = make_congeneric_lig("parent", None);
        let l1 = make_congeneric_lig("analog_F", Some(("F", [2.75, 0.0, 0.0])));
        let l2 = make_congeneric_lig("analog_Cl", Some(("Cl", [3.10, 0.0, 0.0])));
        let sdf_path = base.join("series.sdf");
        std::fs::write(&sdf_path, write_sdf_3d(&[&l0, &l1, &l2]))?;

        let edges = plan_rbfe_network(&[l0, l1, l2], 12);
        assert_eq!(
            edges.len(),
            3,
            "3 ligands must form 2 star + 1 cycle-closure edge"
        );
        assert!(edges.iter().all(|e| e.shared_core_atoms == 6));
        assert!(edges.iter().all(|e| e.core_overlap_ratio >= 0.85));

        let cfg = FepConfig {
            receptor_path: PathBuf::from("1KE6_protprep.pdb"),
            offtarget_receptor_path: Some(PathBuf::from("4YC6_protprep.pdb")),
            ligands_sdf_path: sdf_path.clone(),
            out_dir: base.join("openmm_stage"),
            backend: FepBackend::OpenmmAlchemical,
            n_windows: 9,
            ns_per_window: 0.25,
            dry_run: true,
            ..Default::default()
        };
        let rep = run_fep_network(&cfg)?;
        assert!(rep.dry_run);
        assert_eq!(rep.planned_edges.len(), 3);
        assert!(base.join("openmm_stage/run_fep_openmm.py").exists());
        assert!(base.join("openmm_stage/fep_config.json").exists());
        assert!(base.join("openmm_stage/fep_report.md").exists());

        // Also verify OpenFE backend export
        let cfg_ofe = FepConfig {
            backend: FepBackend::OpenFe,
            out_dir: base.join("openfe_stage"),
            ..cfg
        };
        let rep_ofe = run_fep_network(&cfg_ofe)?;
        assert!(rep_ofe.dry_run);
        assert!(base.join("openfe_stage/run_openfe_network.py").exists());

        // Verify cycle closure arithmetic
        let dummy_leg = FepLegResult {
            leg_name: "complex".to_string(),
            dg_mbar_kcal: 0.0,
            dg_mbar_sem_kcal: 0.1,
            dg_ti_kcal: 0.0,
            min_adjacent_overlap: 0.12,
            overlap_healthy: true,
            adjacent_overlaps: vec![0.12],
            wall_seconds: 1.0,
        };
        let closures = compute_cycle_closures(&[
            FepEdgeResult {
                edge_id: "e1".to_string(),
                from_name: "A".to_string(),
                to_name: "B".to_string(),
                mcs_heavy_atoms: 6,
                unique_a_atoms: 1,
                unique_b_atoms: 1,
                complex_leg: dummy_leg.clone(),
                solvent_leg: dummy_leg.clone(),
                offtarget_leg: None,
                ddg_bind_kcal: -1.2,
                ddg_sem_kcal: 0.15,
                ddg_ti_kcal: -1.1,
                fold_change_ki: 0.13,
                ddg_selectivity_kcal: None,
                ddg_selectivity_sem_kcal: None,
                convergence_status: "Healthy".to_string(),
            },
            FepEdgeResult {
                edge_id: "e2".to_string(),
                from_name: "B".to_string(),
                to_name: "C".to_string(),
                mcs_heavy_atoms: 6,
                unique_a_atoms: 1,
                unique_b_atoms: 1,
                complex_leg: dummy_leg.clone(),
                solvent_leg: dummy_leg.clone(),
                offtarget_leg: None,
                ddg_bind_kcal: 0.5,
                ddg_sem_kcal: 0.15,
                ddg_ti_kcal: 0.4,
                fold_change_ki: 2.3,
                ddg_selectivity_kcal: None,
                ddg_selectivity_sem_kcal: None,
                convergence_status: "Healthy".to_string(),
            },
            FepEdgeResult {
                edge_id: "e3".to_string(),
                from_name: "A".to_string(),
                to_name: "C".to_string(),
                mcs_heavy_atoms: 6,
                unique_a_atoms: 1,
                unique_b_atoms: 1,
                complex_leg: dummy_leg.clone(),
                solvent_leg: dummy_leg,
                offtarget_leg: None,
                ddg_bind_kcal: -0.9,
                ddg_sem_kcal: 0.15,
                ddg_ti_kcal: -0.8,
                fold_change_ki: 0.22,
                ddg_selectivity_kcal: None,
                ddg_selectivity_sem_kcal: None,
                convergence_status: "Healthy".to_string(),
            },
        ]);
        assert_eq!(closures.len(), 1);
        assert!((closures[0].closure_error_kcal - 0.2).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}

#![forbid(unsafe_code)]

//! Opt-in Neural CNN Pose Rescoring bridge (**GNINA `--score_only`**) for `DockCraft`.
//!
//! Keeps all Rust binaries 100% pure, portable, and `#![forbid(unsafe_code)]` while delegating
//! optional 3D convolutional neural network rescoring (`CNNscore`, `CNNaffinity`) to an external
//! `gnina` binary located via `CRAFT_GNINA_BIN`, `CRAFT_DATA_DIR/envs/gnina/gnina`, or `PATH`,
//! blending Vina + HYDE v2 + GNINA CNN into a calibrated 3-way consensus ranking.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use hyde_scorer::{consensus_affinity_kcal, score_pose_hyde};
use serde::{Deserialize, Serialize};
use struct_core::{Molecule3D, parse_sdf_3d_with_limit, write_sdf_3d};
use thiserror::Error;

use crate::md_validate::craft_data_dir;

#[derive(Debug, Error)]
pub enum CnnRescoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error(
        "GNINA binary not found (checked CRAFT_GNINA_BIN, CRAFT_DATA_DIR/envs/gnina/gnina, PATH)"
    )]
    GninaNotFound,
    #[error("GNINA rescoring failed (exit {code:?}): {stderr}")]
    ExecutionFailed { code: Option<i32>, stderr: String },
    #[error("rescoring error: {0}")]
    Parse(String),
}

/// Availability and version probe for the opt-in external GNINA CNN binary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GninaEnvStatus {
    pub available: bool,
    pub binary_path: String,
    pub version_line: Option<String>,
    pub message: String,
}

/// Configuration for GNINA `--score_only` CNN rescoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GninaRescoreConfig {
    pub receptor_path: PathBuf,
    pub poses_sdf_path: PathBuf,
    pub out_sdf_path: Option<PathBuf>,
    pub out_csv_path: Option<PathBuf>,
    pub force_cpu: bool,
    pub gnina_override: Option<PathBuf>,
}

/// Single pose rescored by GNINA CNN + HYDE v2 3-way consensus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnRescoredPose {
    pub rank: usize,
    pub name: String,
    pub vina_kcal: f64,
    pub hyde_kcal: Option<f64>,
    /// GNINA CNN pose probability in `[0.0, 1.0]` (higher = more native-like pose geometry).
    pub cnn_score: f64,
    /// GNINA CNN predicted affinity in pK (`-log10 Kd`) units.
    pub cnn_affinity_pk: f64,
    /// GNINA CNN affinity converted to kcal/mol (`-1.364 * pK` at 298 K).
    pub cnn_affinity_kcal: f64,
    /// Calibrated 3-way consensus score (kcal/mol; more negative = better).
    pub consensus_3way_kcal: f64,
}

/// Complete CNN rescoring report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CnnRescoreReport {
    pub engine: String,
    pub receptor_name: String,
    pub poses_evaluated: usize,
    pub elapsed_ms: u128,
    pub poses: Vec<CnnRescoredPose>,
}

/// Resolves the `gnina` binary path (`override` -> `CRAFT_GNINA_BIN` -> `CRAFT_DATA_DIR/envs/gnina/gnina` -> `PATH`).
#[must_use]
pub fn resolve_gnina_binary(override_path: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = override_path
        && p.is_file()
    {
        return Some(p.to_path_buf());
    }
    if let Ok(env_p) = std::env::var("CRAFT_GNINA_BIN") {
        let p = PathBuf::from(env_p.trim());
        if p.is_file() {
            return Some(p);
        }
    }
    let craft_p = craft_data_dir().join("envs").join("gnina").join("gnina");
    if craft_p.is_file() {
        return Some(craft_p);
    }
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let cand = dir.join("gnina");
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

/// Probes whether the opt-in external `gnina` binary is installed and executable.
#[must_use]
pub fn check_gnina_env(override_path: Option<&Path>) -> GninaEnvStatus {
    let Some(bin) = resolve_gnina_binary(override_path) else {
        return GninaEnvStatus {
            available: false,
            binary_path: craft_data_dir()
                .join("envs")
                .join("gnina")
                .join("gnina")
                .display()
                .to_string(),
            version_line: None,
            message: "Opt-in GNINA binary not installed (expected at CRAFT_DATA_DIR/envs/gnina/gnina or on PATH).".to_string(),
        };
    };
    let bin_str = bin.display().to_string();
    match Command::new(&bin).arg("--version").output() {
        Ok(out) if out.status.success() => {
            let combined = format!(
                "{} {}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            let ver = combined
                .lines()
                .find(|l| !l.trim().is_empty())
                .map(|l| l.trim().to_string());
            GninaEnvStatus {
                available: true,
                binary_path: bin_str,
                version_line: ver.clone(),
                message: format!("GNINA ready ({})", ver.as_deref().unwrap_or("installed")),
            }
        }
        Ok(out) => GninaEnvStatus {
            available: false,
            binary_path: bin_str,
            version_line: None,
            message: format!(
                "GNINA binary failed --version: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        },
        Err(e) => GninaEnvStatus {
            available: false,
            binary_path: bin_str,
            version_line: None,
            message: format!("Failed to spawn GNINA: {e}"),
        },
    }
}

/// Computes the calibrated 3-way consensus affinity (kcal/mol) combining Vina, HYDE v2, and GNINA CNN.
///
/// Uses `CNNaffinity_kcal = -1.364 * CNNaffinity_pK` weighted with pose confidence `CNNscore`,
/// blended `0.40 * Vina/HYDE + 0.60 * CNNaffinity` with a gentle penalty when `CNNscore < 0.35`.
#[must_use]
pub fn compute_3way_consensus_kcal(
    vina_kcal: f64,
    hyde_kcal: Option<f64>,
    cnn_score: f64,
    cnn_affinity_pk: f64,
) -> (f64, f64) {
    let cnn_kcal = -1.364 * cnn_affinity_pk;
    let phys_kcal = match hyde_kcal {
        Some(h) => consensus_affinity_kcal(vina_kcal, vina_kcal, h),
        None => vina_kcal,
    };
    let pose_penalty = if cnn_score < 0.35 {
        (0.35 - cnn_score) * 4.0
    } else {
        0.0
    };
    let blended = 0.45 * phys_kcal + 0.55 * cnn_kcal + pose_penalty;
    (
        (cnn_kcal * 100.0).round() / 100.0,
        (blended * 100.0).round() / 100.0,
    )
}

/// Parses GNINA `--score_only` output SDF records into ranked `CnnRescoredPose` entries,
/// optionally augmenting with live HYDE v2 scores when `receptor` is provided.
pub fn parse_gnina_sdf_output(
    sdf_text: &str,
    receptor: Option<&Molecule3D>,
) -> Result<(Vec<CnnRescoredPose>, Vec<Molecule3D>), CnnRescoreError> {
    let parsed_list = parse_sdf_3d_with_limit(sdf_text, 2_000);
    let mut rows: Vec<(CnnRescoredPose, Molecule3D)> = Vec::new();

    for (idx, item) in parsed_list.into_iter().enumerate() {
        let Ok(mut mol) = item else {
            continue;
        };
        struct_core::perceive_all(&mut mol);
        let name = if mol.name.trim().is_empty() {
            format!("pose_{}", idx + 1)
        } else {
            mol.name.clone()
        };
        let vina =
            prop_f64(&mol, &["minimizedAffinity", "VINA_AFFINITY", "affinity"]).unwrap_or(0.0);
        let cnn_score = prop_f64(&mol, &["CNNscore", "cnn_score"]).unwrap_or(0.5);
        let cnn_pk = prop_f64(&mol, &["CNNaffinity", "cnn_affinity"]).unwrap_or(0.0);
        let hyde_opt = receptor.and_then(|rec| {
            score_pose_hyde(rec, &mol)
                .ok()
                .map(|h| (h.total_delta_g_kcal * 100.0).round() / 100.0)
        });
        let (cnn_kcal, cons3) = compute_3way_consensus_kcal(vina, hyde_opt, cnn_score, cnn_pk);

        mol.properties.insert(
            "DockCraft_3Way_Consensus_kcal".to_string(),
            format!("{cons3:.2}"),
        );

        rows.push((
            CnnRescoredPose {
                rank: 0,
                name,
                vina_kcal: (vina * 100.0).round() / 100.0,
                hyde_kcal: hyde_opt,
                cnn_score: (cnn_score * 1000.0).round() / 1000.0,
                cnn_affinity_pk: (cnn_pk * 1000.0).round() / 1000.0,
                cnn_affinity_kcal: cnn_kcal,
                consensus_3way_kcal: cons3,
            },
            mol,
        ));
    }

    if rows.is_empty() {
        return Err(CnnRescoreError::Parse(
            "no valid ligand poses found in GNINA output SDF".to_string(),
        ));
    }

    rows.sort_by(|a, b| a.0.consensus_3way_kcal.total_cmp(&b.0.consensus_3way_kcal));
    let mut poses = Vec::with_capacity(rows.len());
    let mut mols = Vec::with_capacity(rows.len());
    for (i, (mut p, m)) in rows.into_iter().enumerate() {
        p.rank = i + 1;
        poses.push(p);
        mols.push(m);
    }
    Ok((poses, mols))
}

fn prop_f64(mol: &Molecule3D, keys: &[&str]) -> Option<f64> {
    for &k in keys {
        if let Some(v) = mol.properties.get(k)
            && let Ok(f) = v.trim().parse::<f64>()
            && f.is_finite()
        {
            return Some(f);
        }
        // Case-insensitive fallback
        if let Some((_, v)) = mol
            .properties
            .iter()
            .find(|(pk, _)| pk.eq_ignore_ascii_case(k))
            && let Ok(f) = v.trim().parse::<f64>()
            && f.is_finite()
        {
            return Some(f);
        }
    }
    None
}

/// Runs opt-in GNINA `--score_only` CNN rescoring on `cfg.poses_sdf_path` against `cfg.receptor_path`.
pub fn run_gnina_rescore(
    cfg: &GninaRescoreConfig,
    loaded_receptor: Option<&Molecule3D>,
) -> Result<CnnRescoreReport, CnnRescoreError> {
    let t0 = Instant::now();
    let bin = resolve_gnina_binary(cfg.gnina_override.as_deref())
        .ok_or(CnnRescoreError::GninaNotFound)?;

    let tmp_out = cfg.out_sdf_path.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!("dockcraft_gnina_out_{}.sdf", std::process::id()))
    });
    if let Some(par) = tmp_out.parent() {
        std::fs::create_dir_all(par)?;
    }

    let mut cmd = Command::new(&bin);
    cmd.arg("--receptor")
        .arg(&cfg.receptor_path)
        .arg("--ligand")
        .arg(&cfg.poses_sdf_path)
        .arg("--score_only")
        .arg("--out")
        .arg(&tmp_out);
    if cfg.force_cpu {
        cmd.arg("--no_gpu");
    }

    let out = cmd.output()?;
    if !out.status.success() {
        return Err(CnnRescoreError::ExecutionFailed {
            code: out.status.code(),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        });
    }

    let sdf_raw = std::fs::read_to_string(&tmp_out)?;
    let (poses, ranked_mols) = parse_gnina_sdf_output(&sdf_raw, loaded_receptor)?;

    if let Some(out_sdf) = &cfg.out_sdf_path {
        let refs: Vec<&Molecule3D> = ranked_mols.iter().collect();
        std::fs::write(out_sdf, write_sdf_3d(&refs))?;
    } else {
        let _ = std::fs::remove_file(&tmp_out);
    }

    let rec_name = cfg
        .receptor_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("receptor")
        .to_string();

    let report = CnnRescoreReport {
        engine: format!("GNINA ({})", bin.display()),
        receptor_name: rec_name,
        poses_evaluated: poses.len(),
        elapsed_ms: t0.elapsed().as_millis(),
        poses,
    };

    if let Some(csv_p) = &cfg.out_csv_path {
        if let Some(par) = csv_p.parent() {
            std::fs::create_dir_all(par)?;
        }
        std::fs::write(csv_p, report.to_csv())?;
    }

    Ok(report)
}

impl CnnRescoreReport {
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut s = String::from(
            "Rank,Compound,Consensus_3Way_kcal,Vina_kcal,HYDE_kcal,CNNscore,CNNaffinity_pK,CNNaffinity_kcal\n",
        );
        for p in &self.poses {
            let hyde = p.hyde_kcal.map_or_else(String::new, |v| format!("{v:.2}"));
            let _ = writeln!(
                s,
                "{},{},{:.2},{:.2},{},{:.3},{:.3},{:.2}",
                p.rank,
                p.name,
                p.consensus_3way_kcal,
                p.vina_kcal,
                hyde,
                p.cnn_score,
                p.cnn_affinity_pk,
                p.cnn_affinity_kcal
            );
        }
        s
    }

    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();
        let _ = writeln!(
            md,
            "# GNINA CNN Neural Rescoring Report: {}\n",
            self.receptor_name
        );
        let _ = writeln!(
            md,
            "- **Engine:** `{}` | **Poses Rescored:** `{}` | **Runtime:** `{} ms`\n",
            self.engine, self.poses_evaluated, self.elapsed_ms
        );
        md.push_str("| Rank | Compound | 3-Way Consensus (kcal/mol) | Vina (kcal/mol) | HYDE (kcal/mol) | CNNscore | CNNaffinity (pK / kcal) |\n");
        md.push_str("|---:|---|---:|---:|---:|---:|---:|\n");
        for p in self.poses.iter().take(25) {
            let hyde = p
                .hyde_kcal
                .map_or_else(|| "—".to_string(), |v| format!("{v:.2}"));
            let _ = writeln!(
                md,
                "| {} | `{}` | **`{:.2}`** | `{:.2}` | `{}` | `{:.3}` | `{:.2}` (`{:.2}`) |",
                p.rank,
                p.name,
                p.consensus_3way_kcal,
                p.vina_kcal,
                hyde,
                p.cnn_score,
                p.cnn_affinity_pk,
                p.cnn_affinity_kcal
            );
        }
        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gnina_sdf_tags_and_ranks_3way_consensus_cleanly()
    -> Result<(), Box<dyn std::error::Error>> {
        let fake_sdf = "\
PoseWeak
  DockCraft

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5400    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
M  END
> <minimizedAffinity>
-7.20

> <CNNscore>
0.18

> <CNNaffinity>
4.80

$$$$
PoseStrong
  DockCraft

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5400    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
M  END
> <minimizedAffinity>
-9.80

> <CNNscore>
0.89

> <CNNaffinity>
7.85

$$$$
";
        let (poses, mols) = parse_gnina_sdf_output(fake_sdf, None)?;
        assert_eq!(poses.len(), 2);
        assert_eq!(mols.len(), 2);
        assert_eq!(poses[0].name, "PoseStrong");
        assert_eq!(poses[0].rank, 1);
        assert!(poses[0].cnn_score > 0.85);
        assert!(poses[0].consensus_3way_kcal < poses[1].consensus_3way_kcal - 3.0);
        let rep = CnnRescoreReport {
            engine: "GNINA-test".to_string(),
            receptor_name: "1KE6".to_string(),
            poses_evaluated: poses.len(),
            elapsed_ms: 5,
            poses,
        };
        assert!(rep.to_csv().contains("PoseStrong"));
        assert!(rep.render_markdown().contains("3-Way Consensus"));
        Ok(())
    }
}

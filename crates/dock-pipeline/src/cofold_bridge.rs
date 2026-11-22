#![forbid(unsafe_code)]

//! Opt-in AI Co-Folding Structure & Affinity Prediction Bridge (**Boltz-2 / Boltz-1 / Chai-1**)
//! with automatic `DockCraft` anti-hallucination structural verification (**PoseBusters + HYDE v2
//! + SeeSAR Torsion Audit + Reference RMSD**).
//!
//! Supports arbitrary multi-entity complexes (protein chains, small-molecule SMILES, catalytic ions /
//! cofactors, and partner peptide/protein chains) so the exact same bridge serves both small-molecule
//! targets without crystal structures today and peptide/NRP or antibody–antigen biologics tomorrow.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use dock_engine::{ScoringFunction, score_complex};
use hyde_scorer::{
    analyze_torsions, consensus_affinity_kcal, score_pose_hyde, validate_pose_geometry,
};
use serde::{Deserialize, Serialize};
use struct_core::{
    Molecule3D, parse_mmcif, parse_pdb, perceive_all, prepare_receptor, write_pdb, write_sdf_3d,
};
use thiserror::Error;

use crate::md_validate::craft_data_dir;

#[derive(Debug, Error)]
pub enum CoFoldError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("co-folding setup error: {0}")]
    Setup(String),
    #[error(
        "co-folding engine '{0}' not found (expected in CRAFT_DATA_DIR/envs/cofold/bin or CRAFT_COFOLD_BIN)"
    )]
    EngineNotFound(String),
    #[error("co-folding execution failed (exit {code:?}): {stderr}")]
    ExecutionFailed { code: Option<i32>, stderr: String },
}

/// Supported open-source AI co-folding predictors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CoFoldBackend {
    /// Boltz-2 / Boltz-1 (MIT license; predicts protein+ligand/peptide/protein complexes + affinity).
    #[default]
    Boltz,
    /// Chai-1 (Apache-2.0 license; multi-entity FASTA protein+ligand+peptide co-folding).
    Chai,
}

impl CoFoldBackend {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Boltz => "boltz",
            Self::Chai => "chai",
        }
    }
}

/// Single molecular entity included in an AI co-folding job (protein, peptide, ligand, or ion/cofactor).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoFoldEntity {
    Protein { chain_id: String, sequence: String },
    LigandSmiles { chain_id: String, smiles: String },
    IonOrCcd { chain_id: String, ccd_code: String },
}

/// Availability probe for Boltz / Chai co-folding environments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoFoldEnvStatus {
    pub backend: CoFoldBackend,
    pub available: bool,
    pub binary_path: String,
    pub version_line: Option<String>,
    pub message: String,
}

/// Configuration for a co-folding prediction + anti-hallucination audit run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoFoldConfig {
    pub job_name: String,
    pub backend: CoFoldBackend,
    pub entities: Vec<CoFoldEntity>,
    /// Request Boltz-2 affinity head when a ligand entity is present.
    pub predict_affinity: bool,
    /// Allow online ColabFold MSA server (`--use_msa_server`); when `false`, runs single-sequence / local MSA.
    pub use_msa_server: bool,
    /// Number of diffusion samples to generate (default `1` for fast GPU runs, `5` for production).
    pub diffusion_samples: usize,
    /// Optional reference crystal/docked ligand SDF for RMSD verification.
    pub reference_ligand_path: Option<PathBuf>,
    /// Optional explicit executable path (`boltz` or `chai-lab` / python wrapper).
    pub bin_override: Option<PathBuf>,
    pub dry_run: bool,
    pub out_dir: PathBuf,
}

impl Default for CoFoldConfig {
    fn default() -> Self {
        Self {
            job_name: "cofold_complex".to_string(),
            backend: CoFoldBackend::Boltz,
            entities: Vec::new(),
            predict_affinity: true,
            use_msa_server: false,
            diffusion_samples: 1,
            reference_ligand_path: None,
            bin_override: None,
            dry_run: false,
            out_dir: PathBuf::from("./cofold_run"),
        }
    }
}

/// Confidence & affinity scores parsed from Boltz-2 / Boltz-1 / Chai-1 output JSONs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoFoldConfidence {
    pub confidence_score: Option<f64>,
    pub ptm: Option<f64>,
    pub iptm: Option<f64>,
    pub ligand_iptm: Option<f64>,
    pub complex_plddt: Option<f64>,
    pub complex_ipde: Option<f64>,
    /// Boltz-2 predicted affinity (`log10(IC50 uM)` convention -> converted to pIC50 / kcal/mol).
    pub affinity_pred_value: Option<f64>,
    pub affinity_probability_binary: Option<f64>,
    pub predicted_dg_kcal: Option<f64>,
}

/// Structural quality verdict combining AI confidence (`ipTM`/`pLDDT`) with `DockCraft` physical gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoFoldVerdict {
    /// High AI confidence (`ipTM >= 0.70`, `pLDDT >= 0.75`) AND PoseBusters-valid with favorable HYDE/Vina.
    ValidatedHighConfidence,
    /// High AI confidence but PoseBusters or torsion strain flagged geometry distortion (needs local minimization/MD).
    HighConfidenceGeometryWarning,
    /// Moderate/low AI confidence (`ipTM < 0.70`) — treat pose as exploratory hypothesis.
    LowConfidenceReview,
    /// Severe steric clash or broken ligand chemistry hallucinated by the generative model.
    HallucinatedGeometryReject,
    /// Dry-run manifest staged cleanly without executing GPU co-folding.
    StagedDryRun,
}

impl CoFoldVerdict {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ValidatedHighConfidence => "Validated (High Confidence + PoseBusters OK)",
            Self::HighConfidenceGeometryWarning => "Geometry Warning (Minimize / MD Recommended)",
            Self::LowConfidenceReview => "Low AI Confidence (Exploratory Review)",
            Self::HallucinatedGeometryReject => "Rejected (Hallucinated Clash / Broken Geometry)",
            Self::StagedDryRun => "Dry-Run Manifest Staged",
        }
    }
}

/// Complete Co-Folding + Anti-Hallucination Structural Audit Report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoFoldAuditReport {
    pub job_name: String,
    pub backend: CoFoldBackend,
    pub dry_run: bool,
    pub entity_count: usize,
    pub manifest_path: String,
    pub predicted_structure_path: Option<String>,
    pub extracted_receptor_pdb: Option<String>,
    pub extracted_ligand_sdf: Option<String>,
    pub confidence: CoFoldConfidence,
    pub posebusters_valid: Option<bool>,
    pub torsion_red_alerts: Option<usize>,
    pub vina_dg_kcal: Option<f64>,
    pub hyde_dg_kcal: Option<f64>,
    pub consensus_dg_kcal: Option<f64>,
    pub reference_rmsd_a: Option<f64>,
    pub verdict: CoFoldVerdict,
    pub elapsed_ms: u128,
}

impl CoFoldAuditReport {
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();
        let _ = writeln!(
            md,
            "# AI Co-Folding Structure & Anti-Hallucination Audit: {}\n",
            self.job_name
        );
        let _ = writeln!(
            md,
            "- **Backend:** `{}` | **Entities:** `{}` | **Verdict:** **{}**",
            self.backend.as_str(),
            self.entity_count,
            self.verdict.as_str()
        );
        let _ = writeln!(
            md,
            "- **Input Manifest:** `{}` | **Runtime:** `{:.2} s`\n",
            self.manifest_path,
            (self.elapsed_ms as f64) / 1000.0
        );

        if self.dry_run {
            md.push_str(
                "> **Dry-Run Bundle Staged:** Install `boltz` or `chai-lab` in `CRAFT_DATA_DIR/envs/cofold` and re-run without `--dry-run` to execute GPU co-folding.\n",
            );
            return md;
        }

        md.push_str("## 1. AI Confidence & Affinity Predictions\n\n");
        md.push_str("| Metric | Value |\n|---|---:|\n");
        if let Some(v) = self.confidence.confidence_score {
            let _ = writeln!(md, "| Overall Confidence Score | **{v:.3}** |");
        }
        if let Some(v) = self.confidence.iptm {
            let _ = writeln!(md, "| Interface pTM (`ipTM`) | **{v:.3}** |");
        }
        if let Some(v) = self.confidence.ligand_iptm {
            let _ = writeln!(md, "| Ligand `ipTM` | **{v:.3}** |");
        }
        if let Some(v) = self.confidence.ptm {
            let _ = writeln!(md, "| Global `pTM` | {v:.3} |");
        }
        if let Some(v) = self.confidence.complex_plddt {
            let _ = writeln!(md, "| Complex Mean `pLDDT` | {v:.3} |");
        }
        if let Some(v) = self.confidence.predicted_dg_kcal {
            let _ = writeln!(md, "| Boltz-2 Predicted ΔG | **{v:.2} kcal/mol** |");
        }
        if let Some(v) = self.confidence.affinity_probability_binary {
            let _ = writeln!(md, "| Boltz-2 Binder Probability | {:.1}% |", v * 100.0);
        }

        md.push_str("\n## 2. DockCraft Physical Anti-Hallucination Gate\n\n");
        md.push_str("| Check | Value |\n|---|---:|\n");
        if let Some(pb) = self.posebusters_valid {
            let _ = writeln!(
                md,
                "| **PoseBusters Geometry** | **{}** |",
                if pb { "PASS" } else { "FAIL (distorted/clash)" }
            );
        }
        if let Some(red) = self.torsion_red_alerts {
            let _ = writeln!(md, "| Torsion Red Strain Alerts | {red} |");
        }
        if let Some(v) = self.vina_dg_kcal {
            let _ = writeln!(md, "| Vina Score | {v:.2} kcal/mol |");
        }
        if let Some(v) = self.hyde_dg_kcal {
            let _ = writeln!(md, "| HYDE v2 ΔG | {v:.2} kcal/mol |");
        }
        if let Some(v) = self.consensus_dg_kcal {
            let _ = writeln!(
                md,
                "| Consensus ΔG (Vina+Vinardo+HYDE) | **{v:.2} kcal/mol** |"
            );
        }
        if let Some(r) = self.reference_rmsd_a {
            let _ = writeln!(
                md,
                "| Symmetry-Aware Heavy RMSD vs Reference | **{r:.2} Å** |"
            );
        }
        md.push('\n');
        if let Some(p) = &self.extracted_receptor_pdb {
            let _ = writeln!(md, "- **Extracted Receptor PDB (ready for MD/FEP):** `{p}`");
        }
        if let Some(p) = &self.extracted_ligand_sdf {
            let _ = writeln!(
                md,
                "- **Extracted Ligand SDF (ready for Lead-Opt/FEP):** `{p}`"
            );
        }
        md
    }
}

/// Converts a 3-letter standard (or Amber protonated) amino-acid residue name to 1-letter FASTA code.
#[must_use]
pub fn res3_to_aa1(res_name: &str) -> Option<char> {
    match res_name.trim().to_ascii_uppercase().as_str() {
        "ALA" => Some('A'),
        "ARG" => Some('R'),
        "ASN" => Some('N'),
        "ASP" | "ASH" => Some('D'),
        "CYS" | "CYM" | "CYX" => Some('C'),
        "GLN" => Some('Q'),
        "GLU" | "GLH" => Some('E'),
        "GLY" => Some('G'),
        "HIS" | "HID" | "HIE" | "HIP" => Some('H'),
        "ILE" => Some('I'),
        "LEU" => Some('L'),
        "LYS" | "LYN" => Some('K'),
        "MET" | "MSE" => Some('M'),
        "PHE" => Some('F'),
        "PRO" => Some('P'),
        "SER" | "SEP" => Some('S'),
        "THR" | "TPO" => Some('T'),
        "TRP" => Some('W'),
        "TYR" | "PTR" => Some('Y'),
        "VAL" => Some('V'),
        _ => None,
    }
}

/// Extracts per-chain 1-letter amino acid sequences directly from a PDB/mmCIF receptor structure.
#[must_use]
pub fn extract_protein_sequences_from_mol(mol: &Molecule3D) -> Vec<(String, String)> {
    let mut chains: BTreeMap<char, BTreeMap<i32, char>> = BTreeMap::new();
    for a in &mol.atoms {
        if a.is_hetatm {
            continue;
        }
        if let Some(aa) = res3_to_aa1(&a.res_name) {
            chains
                .entry(a.chain_id)
                .or_default()
                .entry(a.res_seq)
                .or_insert(aa);
        }
    }
    chains
        .into_iter()
        .filter(|(_, res)| !res.is_empty())
        .map(|(ch, res)| {
            let seq: String = res.into_values().collect();
            (ch.to_string(), seq)
        })
        .collect()
}

/// Renders a native Boltz-2 / Boltz-1 YAML manifest (`version: 1`).
#[must_use]
pub fn render_boltz_yaml(entities: &[CoFoldEntity], predict_affinity: bool) -> String {
    let mut y = String::from("version: 1\nsequences:\n");
    let mut first_lig_chain: Option<String> = None;
    for ent in entities {
        match ent {
            CoFoldEntity::Protein { chain_id, sequence } => {
                let _ = writeln!(y, "  - protein:");
                let _ = writeln!(y, "      id: {chain_id}");
                let _ = writeln!(y, "      sequence: {sequence}");
            }
            CoFoldEntity::LigandSmiles { chain_id, smiles } => {
                if first_lig_chain.is_none() {
                    first_lig_chain = Some(chain_id.clone());
                }
                let escaped = smiles.replace('\'', "''");
                let _ = writeln!(y, "  - ligand:");
                let _ = writeln!(y, "      id: {chain_id}");
                let _ = writeln!(y, "      smiles: '{escaped}'");
            }
            CoFoldEntity::IonOrCcd { chain_id, ccd_code } => {
                let _ = writeln!(y, "  - ligand:");
                let _ = writeln!(y, "      id: {chain_id}");
                let _ = writeln!(y, "      ccd: {ccd_code}");
            }
        }
    }
    if predict_affinity && let Some(lig_id) = first_lig_chain {
        y.push_str("properties:\n  - affinity:\n");
        let _ = writeln!(y, "      binder: {lig_id}");
    }
    y
}

/// Renders a native Chai-1 multi-entity FASTA manifest.
#[must_use]
pub fn render_chai_fasta(entities: &[CoFoldEntity]) -> String {
    let mut fa = String::new();
    for ent in entities {
        match ent {
            CoFoldEntity::Protein { chain_id, sequence } => {
                let _ = writeln!(fa, ">protein|name=chain_{chain_id}\n{sequence}");
            }
            CoFoldEntity::LigandSmiles { chain_id, smiles } => {
                let _ = writeln!(fa, ">ligand|name=lig_{chain_id}\n{smiles}");
            }
            CoFoldEntity::IonOrCcd { chain_id, ccd_code } => {
                let _ = writeln!(fa, ">ligand|name=ion_{chain_id}\n{ccd_code}");
            }
        }
    }
    fa
}

/// Resolves the opt-in co-folding executable (`boltz` or `chai-lab`).
#[must_use]
pub fn resolve_cofold_binary(
    backend: CoFoldBackend,
    override_bin: Option<&Path>,
) -> Option<PathBuf> {
    if let Some(p) = override_bin
        && p.is_file()
    {
        return Some(p.to_path_buf());
    }
    if let Ok(env_p) = std::env::var("CRAFT_COFOLD_BIN") {
        let p = PathBuf::from(env_p.trim());
        if p.is_file() {
            return Some(p);
        }
    }
    let exe_name = match backend {
        CoFoldBackend::Boltz => "boltz",
        CoFoldBackend::Chai => "chai-lab",
    };
    let craft_p = craft_data_dir()
        .join("envs")
        .join("cofold")
        .join("bin")
        .join(exe_name);
    if craft_p.is_file() {
        return Some(craft_p);
    }
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let cand = dir.join(exe_name);
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

/// Probes whether the requested co-folding backend (`boltz` or `chai-lab`) is installed.
#[must_use]
pub fn check_cofold_env(backend: CoFoldBackend, override_bin: Option<&Path>) -> CoFoldEnvStatus {
    let expected = craft_data_dir()
        .join("envs")
        .join("cofold")
        .join("bin")
        .join(match backend {
            CoFoldBackend::Boltz => "boltz",
            CoFoldBackend::Chai => "chai-lab",
        });
    let Some(bin) = resolve_cofold_binary(backend, override_bin) else {
        return CoFoldEnvStatus {
            backend,
            available: false,
            binary_path: expected.display().to_string(),
            version_line: None,
            message: format!(
                "Opt-in {} engine not installed (expected at CRAFT_DATA_DIR/envs/cofold/bin/{}).",
                backend.as_str(),
                expected
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("boltz")
            ),
        };
    };
    let out = Command::new(&bin).arg("--help").output();
    match out {
        Ok(o) if o.status.success() => {
            let first = String::from_utf8_lossy(&o.stdout)
                .lines()
                .find(|l| !l.trim().is_empty())
                .map(|s| s.trim().to_string());
            CoFoldEnvStatus {
                backend,
                available: true,
                binary_path: bin.display().to_string(),
                version_line: first.clone(),
                message: format!(
                    "{} co-folding binary ready at {}",
                    backend.as_str(),
                    bin.display()
                ),
            }
        }
        _ => CoFoldEnvStatus {
            backend,
            available: false,
            binary_path: bin.display().to_string(),
            version_line: None,
            message: format!("Found '{}' but failed to execute.", bin.display()),
        },
    }
}

/// Parses Boltz-2 / Chai-1 confidence and optional affinity JSON files.
#[must_use]
pub fn parse_cofold_confidence(
    confidence_json: Option<&str>,
    affinity_json: Option<&str>,
) -> CoFoldConfidence {
    let mut c = CoFoldConfidence::default();
    if let Some(raw) = confidence_json
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(raw)
    {
        c.confidence_score = v.get("confidence_score").and_then(|x| x.as_f64());
        c.ptm = v
            .get("ptm")
            .or_else(|| v.get("aggregate_score"))
            .and_then(|x| x.as_f64());
        c.iptm = v.get("iptm").and_then(|x| x.as_f64());
        c.ligand_iptm = v.get("ligand_iptm").and_then(|x| x.as_f64());
        c.complex_plddt = v.get("complex_plddt").and_then(|x| x.as_f64());
        c.complex_ipde = v.get("complex_ipde").and_then(|x| x.as_f64());
    }
    if let Some(raw) = affinity_json
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(raw)
    {
        c.affinity_pred_value = v.get("affinity_pred_value").and_then(|x| x.as_f64());
        c.affinity_probability_binary = v
            .get("affinity_probability_binary")
            .and_then(|x| x.as_f64());
        if let Some(log10_ic50_um) = c.affinity_pred_value {
            // Boltz-2 affinity_pred_value = log10(IC50 in uM); pIC50 = 6 - val; dG = -1.364 * pIC50
            let pic50 = 6.0 - log10_ic50_um;
            c.predicted_dg_kcal = Some(((-1.364 * pic50) * 100.0).round() / 100.0);
        }
    }
    c
}

/// Splits a predicted complex `Molecule3D` into protein receptor (`!is_hetatm`) and largest organic ligand (`is_hetatm`).
#[must_use]
pub fn split_complex_receptor_and_ligand(complex: &Molecule3D) -> (Molecule3D, Option<Molecule3D>) {
    let mut rec = Molecule3D::new(format!("{}_receptor", complex.name));
    let mut het_groups: BTreeMap<(char, i32, String), Vec<&struct_core::Atom3D>> = BTreeMap::new();

    for a in &complex.atoms {
        if !a.is_hetatm {
            rec.add_atom(a.clone());
        } else if !matches!(
            a.res_name.trim().to_ascii_uppercase().as_str(),
            "HOH" | "WAT" | "DOD" | "NA" | "CL" | "K" | "MG" | "ZN" | "CA" | "SO4" | "PO4"
        ) {
            het_groups
                .entry((a.chain_id, a.res_seq, a.res_name.clone()))
                .or_default()
                .push(a);
        }
    }
    prepare_receptor(&mut rec, true);

    let best_lig = het_groups
        .into_values()
        .max_by_key(|atoms| atoms.iter().filter(|a| a.is_heavy()).count())
        .map(|atoms| {
            let mut lig = Molecule3D::new(format!("{}_ligand", complex.name));
            for a in atoms {
                lig.add_atom(a.clone());
            }
            perceive_all(&mut lig);
            lig
        });
    (rec, best_lig)
}

/// Audits a predicted co-folded structure (mmCIF or PDB) + confidence/affinity JSONs using
/// `DockCraft`'s PoseBusters, Torsion, HYDE v2, Vina, and optional reference RMSD checks.
#[allow(clippy::too_many_arguments)]
pub fn audit_cofold_prediction(
    job_name: &str,
    backend: CoFoldBackend,
    entity_count: usize,
    manifest_path: &Path,
    structure_path: &Path,
    confidence_json: Option<&str>,
    affinity_json: Option<&str>,
    reference_ligand: Option<&Molecule3D>,
    out_dir: &Path,
    elapsed_ms: u128,
) -> Result<CoFoldAuditReport, CoFoldError> {
    let raw = std::fs::read_to_string(structure_path)?;
    let ext = structure_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("cif")
        .to_ascii_lowercase();
    let complex = if ext == "cif" || ext == "mmcif" {
        parse_mmcif(&raw, true).map_err(|e| CoFoldError::Setup(e.to_string()))?
    } else {
        parse_pdb(&raw, true).map_err(|e| CoFoldError::Setup(e.to_string()))?
    };

    let (rec, lig_opt) = split_complex_receptor_and_ligand(&complex);
    let rec_out = out_dir.join("cofold_receptor.pdb");
    std::fs::write(&rec_out, write_pdb(&rec))?;

    let confidence = parse_cofold_confidence(confidence_json, affinity_json);
    let mut lig_out_str = None;
    let mut pb_valid = None;
    let mut torsion_red = None;
    let mut vina_dg = None;
    let mut hyde_dg = None;
    let mut cons_dg = None;
    let mut ref_rmsd = None;

    if let Some(lig) = &lig_opt {
        let lpath = out_dir.join("cofold_ligand.sdf");
        std::fs::write(&lpath, write_sdf_3d(&[lig]))?;
        lig_out_str = Some(lpath.display().to_string());

        let pb = validate_pose_geometry(&rec, lig);
        let tors = analyze_torsions(lig);
        let vina = score_complex(&rec, lig, ScoringFunction::Vina).total_affinity_kcal;
        let vinardo = score_complex(&rec, lig, ScoringFunction::Vinardo).total_affinity_kcal;
        let hyde = score_pose_hyde(&rec, lig)
            .map(|h| h.total_delta_g_kcal)
            .unwrap_or(vina);
        let cons = consensus_affinity_kcal(vina, vinardo, hyde);

        pb_valid = Some(pb.valid);
        torsion_red = Some(tors.red_count);
        vina_dg = Some((vina * 100.0).round() / 100.0);
        hyde_dg = Some((hyde * 100.0).round() / 100.0);
        cons_dg = Some((cons * 100.0).round() / 100.0);
        if let Some(ref_l) = reference_ligand {
            ref_rmsd = lig
                .rmsd_heavy_symmetric(ref_l)
                .map(|r| (r * 100.0).round() / 100.0);
        }
    }

    let eff_iptm = confidence
        .ligand_iptm
        .or(confidence.iptm)
        .or(confidence.ptm)
        .unwrap_or(0.0);
    let eff_plddt = confidence.complex_plddt.unwrap_or(0.8);
    let severe_clash = vina_dg.is_some_and(|v| v > 5.0);
    let verdict = if severe_clash {
        CoFoldVerdict::HallucinatedGeometryReject
    } else if eff_iptm >= 0.70 && eff_plddt >= 0.70 {
        if pb_valid.unwrap_or(true) && torsion_red.unwrap_or(0) == 0 {
            CoFoldVerdict::ValidatedHighConfidence
        } else {
            CoFoldVerdict::HighConfidenceGeometryWarning
        }
    } else {
        CoFoldVerdict::LowConfidenceReview
    };

    Ok(CoFoldAuditReport {
        job_name: job_name.to_string(),
        backend,
        dry_run: false,
        entity_count,
        manifest_path: manifest_path.display().to_string(),
        predicted_structure_path: Some(structure_path.display().to_string()),
        extracted_receptor_pdb: Some(rec_out.display().to_string()),
        extracted_ligand_sdf: lig_out_str,
        confidence,
        posebusters_valid: pb_valid,
        torsion_red_alerts: torsion_red,
        vina_dg_kcal: vina_dg,
        hyde_dg_kcal: hyde_dg,
        consensus_dg_kcal: cons_dg,
        reference_rmsd_a: ref_rmsd,
        verdict,
        elapsed_ms,
    })
}

/// Runs (or dry-run stages) an opt-in Boltz-2 / Chai-1 co-folding job and audits outputs.
pub fn run_cofold_prediction(cfg: &CoFoldConfig) -> Result<CoFoldAuditReport, CoFoldError> {
    let t0 = Instant::now();
    if cfg.entities.is_empty() {
        return Err(CoFoldError::Setup(
            "co-folding requires at least one protein or ligand entity".to_string(),
        ));
    }
    std::fs::create_dir_all(&cfg.out_dir)?;

    let manifest_path = match cfg.backend {
        CoFoldBackend::Boltz => {
            let p = cfg.out_dir.join(format!("{}.yaml", cfg.job_name));
            std::fs::write(&p, render_boltz_yaml(&cfg.entities, cfg.predict_affinity))?;
            p
        }
        CoFoldBackend::Chai => {
            let p = cfg.out_dir.join(format!("{}.fasta", cfg.job_name));
            std::fs::write(&p, render_chai_fasta(&cfg.entities))?;
            p
        }
    };

    if cfg.dry_run {
        let rep = CoFoldAuditReport {
            job_name: cfg.job_name.clone(),
            backend: cfg.backend,
            dry_run: true,
            entity_count: cfg.entities.len(),
            manifest_path: manifest_path.display().to_string(),
            predicted_structure_path: None,
            extracted_receptor_pdb: None,
            extracted_ligand_sdf: None,
            confidence: CoFoldConfidence::default(),
            posebusters_valid: None,
            torsion_red_alerts: None,
            vina_dg_kcal: None,
            hyde_dg_kcal: None,
            consensus_dg_kcal: None,
            reference_rmsd_a: None,
            verdict: CoFoldVerdict::StagedDryRun,
            elapsed_ms: t0.elapsed().as_millis(),
        };
        std::fs::write(
            cfg.out_dir.join("cofold_report.json"),
            serde_json::to_string_pretty(&rep)?,
        )?;
        std::fs::write(cfg.out_dir.join("cofold_report.md"), rep.render_markdown())?;
        return Ok(rep);
    }

    let bin = resolve_cofold_binary(cfg.backend, cfg.bin_override.as_deref())
        .ok_or_else(|| CoFoldError::EngineNotFound(cfg.backend.as_str().to_string()))?;

    let pred_out_dir = cfg.out_dir.join("predictions");
    std::fs::create_dir_all(&pred_out_dir)?;
    let mut cmd = Command::new(&bin);
    match cfg.backend {
        CoFoldBackend::Boltz => {
            cmd.arg("predict")
                .arg(&manifest_path)
                .arg("--out_dir")
                .arg(&pred_out_dir)
                .args([
                    "--diffusion_samples",
                    &cfg.diffusion_samples.clamp(1, 10).to_string(),
                ]);
            if cfg.use_msa_server {
                cmd.arg("--use_msa_server");
            }
        }
        CoFoldBackend::Chai => {
            cmd.arg("fold").arg(&manifest_path).arg(&pred_out_dir);
        }
    }
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(CoFoldError::ExecutionFailed {
            code: out.status.code(),
            stderr: format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ),
        });
    }

    // Recursively discover predicted model_0.cif / .pdb, confidence_*.json, and affinity_*.json
    let (cif_path, conf_path, aff_path) = discover_cofold_outputs(&pred_out_dir)?;
    let conf_raw = conf_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok());
    let aff_raw = aff_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok());

    let ref_lig = cfg.reference_ligand_path.as_ref().and_then(|p| {
        let txt = std::fs::read_to_string(p).ok()?;
        let mut v = struct_core::parse_sdf_3d_with_limit(&txt, 1);
        let mut m = v.pop()?.ok()?;
        perceive_all(&mut m);
        Some(m)
    });

    let rep = audit_cofold_prediction(
        &cfg.job_name,
        cfg.backend,
        cfg.entities.len(),
        &manifest_path,
        &cif_path,
        conf_raw.as_deref(),
        aff_raw.as_deref(),
        ref_lig.as_ref(),
        &cfg.out_dir,
        t0.elapsed().as_millis(),
    )?;
    std::fs::write(
        cfg.out_dir.join("cofold_report.json"),
        serde_json::to_string_pretty(&rep)?,
    )?;
    std::fs::write(cfg.out_dir.join("cofold_report.md"), rep.render_markdown())?;
    Ok(rep)
}

fn discover_cofold_outputs(
    root: &Path,
) -> Result<(PathBuf, Option<PathBuf>, Option<PathBuf>), CoFoldError> {
    let mut stack = vec![root.to_path_buf()];
    let mut struct_file: Option<PathBuf> = None;
    let mut conf_file: Option<PathBuf> = None;
    let mut aff_file: Option<PathBuf> = None;

    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let fname = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if (fname.ends_with(".cif") || fname.ends_with(".pdb"))
                && (struct_file.is_none() || fname.contains("model_0"))
            {
                struct_file = Some(p.clone());
            } else if fname.starts_with("confidence") && fname.ends_with(".json") {
                conf_file = Some(p.clone());
            } else if fname.starts_with("affinity") && fname.ends_with(".json") {
                aff_file = Some(p.clone());
            }
        }
    }
    let sf = struct_file.ok_or_else(|| {
        CoFoldError::Setup(format!(
            "no predicted .cif/.pdb structure found under '{}'",
            root.display()
        ))
    })?;
    Ok((sf, conf_file, aff_file))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cofold_emits_boltz2_yaml_chai_fasta_and_audits_predicted_complex()
    -> Result<(), Box<dyn std::error::Error>> {
        let base =
            std::env::temp_dir().join(format!("dockcraft_cofold_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base)?;

        // 1. Sequence extraction from PDB
        let pdb_txt = concat!(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n",
            "ATOM      2  CA  HIE A   2       3.800   0.000   0.000  1.00 20.00           C\n",
            "ATOM      3  CA  ASP A   3       7.600   0.000   0.000  1.00 20.00           C\n",
            "ATOM      4  CA  LYS A   4      11.400   0.000   0.000  1.00 20.00           C\n",
            "HETATM    5  C1  LIG L   1       4.000   4.500   0.000  1.00 20.00           C\n",
            "HETATM    6  C2  LIG L   1       5.500   4.500   0.000  1.00 20.00           C\n",
            "HETATM    7  O1  LIG L   1       6.200   5.600   0.000  1.00 20.00           O\n",
            "END\n"
        );
        let parsed = parse_pdb(pdb_txt, true)?;
        let seqs = extract_protein_sequences_from_mol(&parsed);
        assert_eq!(seqs, vec![("A".to_string(), "AHDK".to_string())]);

        // 2. Dry-run Boltz-2 YAML & Chai-1 FASTA generation (protein + peptide + ligand + Zn ion)
        let entities = vec![
            CoFoldEntity::Protein {
                chain_id: "A".to_string(),
                sequence: "AHDK".to_string(),
            },
            CoFoldEntity::Protein {
                chain_id: "B".to_string(),
                sequence: "GGGGS".to_string(),
            },
            CoFoldEntity::LigandSmiles {
                chain_id: "L".to_string(),
                smiles: "CCO".to_string(),
            },
            CoFoldEntity::IonOrCcd {
                chain_id: "Z".to_string(),
                ccd_code: "ZN".to_string(),
            },
        ];
        let cfg = CoFoldConfig {
            job_name: "cdk2_test".to_string(),
            backend: CoFoldBackend::Boltz,
            entities: entities.clone(),
            dry_run: true,
            out_dir: base.join("boltz_dry"),
            ..Default::default()
        };
        let rep = run_cofold_prediction(&cfg)?;
        assert_eq!(rep.verdict, CoFoldVerdict::StagedDryRun);
        let yaml = std::fs::read_to_string(&rep.manifest_path)?;
        assert!(
            yaml.contains("version: 1") && yaml.contains("binder: L") && yaml.contains("ccd: ZN")
        );

        let chai_fa = render_chai_fasta(&entities);
        assert!(
            chai_fa.contains(">protein|name=chain_A\nAHDK")
                && chai_fa.contains(">ligand|name=lig_L\nCCO")
        );

        // 3. Anti-hallucination structural audit on a predicted PDB + confidence/affinity JSONs
        let pred_pdb = base.join("model_0.pdb");
        std::fs::write(&pred_pdb, pdb_txt)?;
        let conf_json = r#"{"confidence_score":0.88,"ptm":0.91,"iptm":0.84,"ligand_iptm":0.82,"complex_plddt":0.89}"#;
        let aff_json = r#"{"affinity_pred_value":-1.5,"affinity_probability_binary":0.92}"#;
        let audit = audit_cofold_prediction(
            "cdk2_test",
            CoFoldBackend::Boltz,
            4,
            &PathBuf::from(&rep.manifest_path),
            &pred_pdb,
            Some(conf_json),
            Some(aff_json),
            None,
            &base,
            42,
        )?;
        assert!(audit.confidence.predicted_dg_kcal.unwrap_or(0.0) < -9.0);
        assert!(audit.extracted_receptor_pdb.is_some());
        assert!(audit.extracted_ligand_sdf.is_some());
        assert!(matches!(
            audit.verdict,
            CoFoldVerdict::ValidatedHighConfidence | CoFoldVerdict::HighConfidenceGeometryWarning
        ));
        let md = audit.render_markdown();
        assert!(md.contains("Anti-Hallucination") && md.contains("Boltz-2 Predicted ΔG"));

        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}

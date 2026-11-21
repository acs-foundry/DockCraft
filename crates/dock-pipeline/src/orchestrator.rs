//! Reproducible End-to-End SBDD → ADMET → MD → Retrosynthesis Campaign Orchestrator.
//!
//! Chains `dockcraft-cli` (`prot-prep`, `analyze-site`, `pharm-screen`, `screen` / `ensemble-dock`,
//! optional `md-validate`), `admetcraft-cli` (`batch` QSAR + Conformal + SoM + TPP MPO), and
//! `chemcraft-cli` (`retro` Buy-Before-Make + Retro*) with pure-Rust SHA-256 provenance hashing,
//! tool version capture, seed tracking, and explicit GO/NO-GO gates.

#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::md_validate::craft_data_dir;

#[derive(Debug, Error)]
pub enum OrchestratorError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("stage '{stage}' failed: {message}")]
    StageFailed { stage: String, message: String },
}

/// Declarative specification for a fully reproducible CraftBio drug discovery campaign.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignSpec {
    pub campaign_name: String,
    pub target_name: String,
    pub out_dir: PathBuf,
    /// Primary raw PDB/mmCIF receptor path.
    pub receptor_pdb: PathBuf,
    /// Optional chain filter for primary receptor (empty = all chains).
    #[serde(default)]
    pub receptor_chains: Vec<char>,
    /// Optional additional receptor conformations for Boltzmann ensemble re-docking.
    #[serde(default)]
    pub ensemble_receptors: Vec<EnsembleSpecEntry>,
    /// Optional cocrystallized reference ligand for hybrid Apo+Holo MIF consensus.
    pub holo_ligand: Option<PathBuf>,
    /// Screening library (`.smi`, `.smi.gz`, `.sdf`, `.sdf.gz`, `.chemcraft`).
    pub library_path: PathBuf,
    /// Active-site box center `[x, y, z]` (auto-detected if omitted).
    pub box_center: Option<[f64; 3]>,
    /// Active-site box size `[sx, sy, sz]` in Å.
    #[serde(default = "default_box_size")]
    pub box_size: [f64; 3],
    /// Physiological pH for `prot-prep`.
    #[serde(default = "default_ph")]
    pub ph: f64,
    /// Minimum blueprint pharmacophore features required to pass Stage 3 funnel.
    #[serde(default = "default_pharm_min_match")]
    pub pharm_min_match: usize,
    /// Maximum hits forwarded from Stage 3 (`pharm-screen`) to Stage 4 (`screen` / docking).
    #[serde(default = "default_pharm_top")]
    pub pharm_top: usize,
    /// Docking exhaustiveness for Stage 4.
    #[serde(default = "default_exhaustiveness")]
    pub dock_exhaustiveness: usize,
    /// Top docking hits forwarded to Stage 5 (`AdmetCraft` QSAR + TPP MPO).
    #[serde(default = "default_admet_top")]
    pub admet_top: usize,
    /// AdmetCraft Target Product Profile (`oral` | `cns` | `peripheral` | `oncology`).
    #[serde(default = "default_tpp")]
    pub tpp_profile: String,
    /// Minimum TPP MPO score required to pass the GO/NO-GO ADMET gate.
    #[serde(default = "default_min_mpo")]
    pub min_tpp_mpo: f64,
    /// Top ADMET-passing candidates forwarded to Stage 6 (MD) and Stage 7 (ChemCraft Retro*).
    #[serde(default = "default_final_top")]
    pub final_top: usize,
    /// Whether to execute live OpenMM GPU MD in Stage 6 (`false` = stage dry-run MD bundles only).
    #[serde(default)]
    pub run_md: bool,
    /// Production MD length (ns) for Stage 6.
    #[serde(default = "default_md_ns")]
    pub md_production_ns: f64,
    /// Target synthesis scale in mg for Stage 7 (`ChemCraft retro`).
    #[serde(default = "default_synth_scale_mg")]
    pub synth_scale_mg: f64,
    /// Optional off-target / paralog receptors for Stage 2b structural selectivity (ΔMIF) analysis.
    #[serde(default)]
    pub offtarget_selectivity_receptors: Vec<EnsembleSpecEntry>,
    /// Whether to run Stage 5b Inverse Docking Safety Panel against Tier-1 human anti-targets.
    #[serde(default)]
    pub run_safety_panel: bool,
    /// Optional explicit safety-panel manifest path (defaults to `CRAFT_DATA_DIR/safety_panel/panel.json`).
    #[serde(default)]
    pub safety_panel_manifest: Option<PathBuf>,
    /// Docking exhaustiveness for Stage 5b safety-panel inverse docking (default `3`).
    #[serde(default = "default_safety_exhaustiveness")]
    pub safety_panel_exhaustiveness: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleSpecEntry {
    pub receptor_pdb: PathBuf,
    #[serde(default)]
    pub chains: Vec<char>,
    pub box_center: [f64; 3],
}

fn default_box_size() -> [f64; 3] {
    [22.0, 22.0, 22.0]
}
fn default_ph() -> f64 {
    7.4
}
fn default_pharm_min_match() -> usize {
    3
}
fn default_pharm_top() -> usize {
    100
}
fn default_exhaustiveness() -> usize {
    6
}
fn default_admet_top() -> usize {
    20
}
fn default_tpp() -> String {
    "oncology".to_string()
}
fn default_min_mpo() -> f64 {
    0.65
}
fn default_final_top() -> usize {
    3
}
fn default_md_ns() -> f64 {
    20.0
}
fn default_synth_scale_mg() -> f64 {
    100.0
}
fn default_safety_exhaustiveness() -> usize {
    3
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateVerdict {
    Go,
    Review,
    NoGo,
    Skipped,
}

impl GateVerdict {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Go => "GO",
            Self::Review => "REVIEW",
            Self::NoGo => "NO-GO",
            Self::Skipped => "SKIPPED",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileProvenance {
    pub role: String,
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageRecord {
    pub stage_number: usize,
    pub stage_id: String,
    pub tool: String,
    pub verdict: GateVerdict,
    pub elapsed_ms: u128,
    pub summary: String,
    pub outputs: Vec<FileProvenance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignLeadSummary {
    pub rank: usize,
    pub name: String,
    pub smiles: String,
    pub dock_affinity_kcal: Option<f64>,
    pub tpp_mpo: Option<f64>,
    pub tpp_decision: Option<String>,
    #[serde(default)]
    pub tox21_alerts: Option<String>,
    #[serde(default)]
    pub sea_high_risk: Option<String>,
    #[serde(default)]
    pub metabolite_bioactivation: Option<usize>,
    #[serde(default)]
    pub safety_panel_verdict: Option<String>,
    #[serde(default)]
    pub safety_panel_min_margin_kcal: Option<f64>,
    pub md_verdict: Option<String>,
    pub synth_md_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignManifest {
    pub campaign_name: String,
    pub target_name: String,
    pub unix_timestamp: u64,
    pub tool_versions: Vec<(String, String)>,
    pub input_provenance: Vec<FileProvenance>,
    pub stages: Vec<StageRecord>,
    pub final_leads: Vec<CampaignLeadSummary>,
    pub total_elapsed_ms: u128,
}

impl CampaignManifest {
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();
        let _ = writeln!(
            md,
            "# CraftBio End-to-End Campaign Dossier: {}\n",
            self.campaign_name
        );
        let _ = writeln!(md, "- **Target:** {}", self.target_name);
        let _ = writeln!(
            md,
            "- **Total Runtime:** `{:.2} s` | **Timestamp (UTC epoch):** `{}`",
            (self.total_elapsed_ms as f64) / 1000.0,
            self.unix_timestamp
        );
        let vers: Vec<String> = self
            .tool_versions
            .iter()
            .map(|(k, v)| format!("`{k}` ({v})"))
            .collect();
        let _ = writeln!(md, "- **Suite Versions:** {}\n", vers.join(" · "));

        md.push_str("## 1. Input Provenance (SHA-256)\n\n");
        md.push_str("| Role | File | Bytes | SHA-256 |\n|---|---|---:|---|\n");
        for fp in &self.input_provenance {
            let _ = writeln!(
                md,
                "| {} | `{}` | {} | `{}` |",
                fp.role, fp.path, fp.bytes, fp.sha256
            );
        }
        md.push('\n');

        md.push_str("## 2. Pipeline Stages & GO/NO-GO Gates\n\n");
        md.push_str("| # | Stage | Tool | Gate | Time | Summary |\n|---:|---|---|---|---:|---|\n");
        for st in &self.stages {
            let _ = writeln!(
                md,
                "| {} | `{}` | `{}` | **{}** | {} ms | {} |",
                st.stage_number,
                st.stage_id,
                st.tool,
                st.verdict.as_str(),
                st.elapsed_ms,
                st.summary
            );
        }
        md.push('\n');

        if !self.final_leads.is_empty() {
            md.push_str("## 3. Final Prioritized Leads (SBDD → Selectivity → ADMET/Guard → Safety Panel → MD → Synthesis)\n\n");
            md.push_str("| Rank | Compound | Dock ΔG | TPP MPO | Decision | Tox21 / SEA / Bioact | Safety Panel (ΔΔG) | MD | Synthesis Protocol |\n|---:|---|---:|---:|---|---|---|---|---|\n");
            for l in &self.final_leads {
                let dg = l
                    .dock_affinity_kcal
                    .map_or_else(|| "—".to_string(), |v| format!("{v:.2}"));
                let mpo = l
                    .tpp_mpo
                    .map_or_else(|| "—".to_string(), |v| format!("{v:.3}"));
                let dec = l.tpp_decision.as_deref().unwrap_or("—");
                let guard = format!(
                    "{} / {} / {}",
                    l.tox21_alerts.as_deref().unwrap_or("—"),
                    l.sea_high_risk.as_deref().unwrap_or("—"),
                    l.metabolite_bioactivation
                        .map_or_else(|| "—".to_string(), |b| b.to_string())
                );
                let sp = match (&l.safety_panel_verdict, l.safety_panel_min_margin_kcal) {
                    (Some(v), Some(m)) => format!("**{v}** ({m:+.2})"),
                    (Some(v), None) => format!("**{v}**"),
                    _ => "—".to_string(),
                };
                let mdv = l.md_verdict.as_deref().unwrap_or("staged");
                let syn = l
                    .synth_md_path
                    .as_deref()
                    .map_or_else(|| "—".to_string(), |p| format!("`{p}`"));
                let _ = writeln!(
                    md,
                    "| {} | `{}` | {} | {} | {} | {} | {} | {} | {} |",
                    l.rank, l.name, dg, mpo, dec, guard, sp, mdv, syn
                );
            }
            md.push('\n');
        }
        md
    }
}

/// Builds a ready-to-run CDK2 template `CampaignSpec` pointing at `/home/acs/Datos/CraftBio`.
#[must_use]
pub fn default_cdk2_campaign_template() -> CampaignSpec {
    let base = craft_data_dir().join("campaigns").join("cdk2_pilot");
    CampaignSpec {
        campaign_name: "cdk2_e2e_reproducible".to_string(),
        target_name: "Human CDK2 (1KE6 + 1H1S)".to_string(),
        out_dir: base.join("orchestrated_run"),
        receptor_pdb: base.join("raw/1KE6.pdb"),
        receptor_chains: vec!['A'],
        ensemble_receptors: vec![EnsembleSpecEntry {
            receptor_pdb: base.join("raw/1H1S.pdb"),
            chains: vec!['A', 'B'],
            box_center: [5.37, 43.75, 50.94],
        }],
        holo_ligand: Some(base.join("prepared/1KE6_lig_LS2.pdb")),
        library_path: base.join("top20_cdk2_hits.smi"),
        box_center: Some([26.96, 11.18, 11.17]),
        box_size: [22.0, 22.0, 22.0],
        ph: 7.4,
        pharm_min_match: 2,
        pharm_top: 20,
        dock_exhaustiveness: 4,
        admet_top: 10,
        tpp_profile: "oncology".to_string(),
        min_tpp_mpo: 0.65,
        final_top: 3,
        run_md: false,
        md_production_ns: 20.0,
        synth_scale_mg: 100.0,
        offtarget_selectivity_receptors: Vec::new(),
        run_safety_panel: false,
        safety_panel_manifest: None,
        safety_panel_exhaustiveness: 3,
    }
}

/// Executes the full 7-stage reproducible CraftBio campaign described by `spec`.
pub fn run_campaign(spec: &CampaignSpec) -> Result<CampaignManifest, OrchestratorError> {
    let t_total = Instant::now();
    std::fs::create_dir_all(&spec.out_dir)?;

    let dock_cli = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("dockcraft-cli"));
    let admet_cli = find_installed_tool("admetcraft-cli");
    let chem_cli = find_installed_tool("chemcraft-cli");

    let mut tool_versions = vec![("dockcraft-cli".to_string(), probe_version(&dock_cli))];
    if let Some(p) = &admet_cli {
        tool_versions.push(("admetcraft-cli".to_string(), probe_version(p)));
    }
    if let Some(p) = &chem_cli {
        tool_versions.push(("chemcraft-cli".to_string(), probe_version(p)));
    }

    let mut input_provenance = Vec::new();
    for (role, p) in [
        ("Primary Receptor", &spec.receptor_pdb),
        ("Screening Library", &spec.library_path),
    ] {
        if let Ok(fp) = provenance_of(role, p) {
            input_provenance.push(fp);
        }
    }
    if let Some(hl) = &spec.holo_ligand
        && let Ok(fp) = provenance_of("Holo Reference Ligand", hl)
    {
        input_provenance.push(fp);
    }

    let mut stages: Vec<StageRecord> = Vec::new();

    // --- STAGE 1: prot-prep ---
    let t1 = Instant::now();
    let prep_dir = spec.out_dir.join("01_prot_prep");
    std::fs::create_dir_all(&prep_dir)?;
    let prep_pdb = prep_dir.join("receptor_prepared.pdb");
    let prep_md = prep_dir.join("receptor_audit.md");
    let prep_json = prep_dir.join("receptor_audit.json");
    let mut cmd1 = Command::new(&dock_cli);
    cmd1.args(["prot-prep", "--input"])
        .arg(&spec.receptor_pdb)
        .arg("--out-pdb")
        .arg(&prep_pdb)
        .arg("--out-md")
        .arg(&prep_md)
        .arg("--out-report")
        .arg(&prep_json)
        .args(["--ph", &format!("{:.1}", spec.ph)]);
    if !spec.receptor_chains.is_empty() {
        let ch: String = spec
            .receptor_chains
            .iter()
            .map(char::to_string)
            .collect::<Vec<_>>()
            .join(",");
        cmd1.args(["--chains", &ch]);
    }
    run_checked_cmd("01_prot_prep", &mut cmd1)?;
    let grade = std::fs::read_to_string(&prep_json)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| {
            v.get("audit")
                .and_then(|a| a.get("quality_grade"))
                .and_then(|g| g.as_str())
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| "Unknown".to_string());
    let v1 = if grade.starts_with('D') {
        GateVerdict::Review
    } else {
        GateVerdict::Go
    };
    stages.push(StageRecord {
        stage_number: 1,
        stage_id: "prot-prep".to_string(),
        tool: "dockcraft-cli prot-prep".to_string(),
        verdict: v1,
        elapsed_ms: t1.elapsed().as_millis(),
        summary: format!("Prepared receptor (Quality Grade: {grade})"),
        outputs: collect_provenance(&[
            ("Prepared Receptor PDB", &prep_pdb),
            ("Audit Markdown", &prep_md),
        ]),
    });

    // --- STAGE 2: analyze-site (7-probe MIF blueprint) ---
    let t2 = Instant::now();
    let bp_dir = spec.out_dir.join("02_site_blueprint");
    std::fs::create_dir_all(&bp_dir)?;
    let bp_json = bp_dir.join("site.pharm.json");
    let bp_md = bp_dir.join("site_blueprint.md");
    let mut cmd2 = Command::new(&dock_cli);
    cmd2.args(["analyze-site", "--receptor"])
        .arg(&prep_pdb)
        .arg("--out-json")
        .arg(&bp_json)
        .arg("--out-md")
        .arg(&bp_md);
    if let Some(hl) = &spec.holo_ligand {
        cmd2.arg("--ligand").arg(hl);
    }
    if let Some(c) = spec.box_center {
        cmd2.args([
            "--center",
            &format!("{:.3},{:.3},{:.3}", c[0], c[1], c[2]),
            "--size",
            &format!(
                "{:.2},{:.2},{:.2}",
                spec.box_size[0], spec.box_size[1], spec.box_size[2]
            ),
        ]);
    } else {
        cmd2.arg("--auto-pocket");
    }
    run_checked_cmd("02_site_blueprint", &mut cmd2)?;
    let bp_center = spec.box_center.unwrap_or_else(|| {
        std::fs::read_to_string(&bp_json)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| {
                let c = v.get("grid_box")?.get("center")?.as_array()?;
                Some([c[0].as_f64()?, c[1].as_f64()?, c[2].as_f64()?])
            })
            .unwrap_or([0.0, 0.0, 0.0])
    });
    stages.push(StageRecord {
        stage_number: 2,
        stage_id: "analyze-site".to_string(),
        tool: "dockcraft-cli analyze-site".to_string(),
        verdict: GateVerdict::Go,
        elapsed_ms: t2.elapsed().as_millis(),
        summary: format!(
            "Generated 7-probe MIF blueprint centered at ({:.1}, {:.1}, {:.1})",
            bp_center[0], bp_center[1], bp_center[2]
        ),
        outputs: collect_provenance(&[
            ("Site Blueprint JSON", &bp_json),
            ("Site Blueprint Markdown", &bp_md),
        ]),
    });

    // --- OPTIONAL STAGE 2b: Structural Pocket Selectivity (ΔMIF) ---
    let mut active_bp_json = bp_json.clone();
    if !spec.offtarget_selectivity_receptors.is_empty() {
        let t2b = Instant::now();
        let sel_dir = spec.out_dir.join("02b_selectivity");
        std::fs::create_dir_all(&sel_dir)?;
        let mut ot_paths = Vec::new();
        let mut ot_centers = Vec::new();
        for (oi, ot) in spec.offtarget_selectivity_receptors.iter().enumerate() {
            let ot_pdb = sel_dir.join(format!("offtarget_{}.pdb", oi + 1));
            let mut c_ot = Command::new(&dock_cli);
            c_ot.args(["prot-prep", "--input"])
                .arg(&ot.receptor_pdb)
                .arg("--out-pdb")
                .arg(&ot_pdb)
                .args(["--ph", &format!("{:.1}", spec.ph)]);
            if !ot.chains.is_empty() {
                let ch: String = ot
                    .chains
                    .iter()
                    .map(char::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                c_ot.args(["--chains", &ch]);
            }
            run_checked_cmd("02b_offtarget_prot_prep", &mut c_ot)?;
            ot_paths.push(ot_pdb.display().to_string());
            ot_centers.push(format!(
                "{:.3},{:.3},{:.3}",
                ot.box_center[0], ot.box_center[1], ot.box_center[2]
            ));
        }
        let sel_json = sel_dir.join("selectivity_report.json");
        let sel_md = sel_dir.join("selectivity_report.md");
        let sel_cxc = sel_dir.join("selectivity_overlay.cxc");
        let sel_bp = sel_dir.join("site_selective.pharm.json");
        let mut cmd2b = Command::new(&dock_cli);
        cmd2b
            .args(["compare-pockets", "--target"])
            .arg(&prep_pdb)
            .args([
                "--off-targets",
                &ot_paths.join(","),
                "--target-center",
                &format!(
                    "{:.3},{:.3},{:.3}",
                    bp_center[0], bp_center[1], bp_center[2]
                ),
                "--off-target-centers",
                &ot_centers.join(";"),
                "--size",
                &format!(
                    "{:.2},{:.2},{:.2}",
                    spec.box_size[0], spec.box_size[1], spec.box_size[2]
                ),
            ])
            .arg("--in-blueprint")
            .arg(&bp_json)
            .arg("--out-blueprint")
            .arg(&sel_bp)
            .arg("--out-json")
            .arg(&sel_json)
            .arg("--out-md")
            .arg(&sel_md)
            .arg("--out-cxc")
            .arg(&sel_cxc);
        run_checked_cmd("02b_selectivity", &mut cmd2b)?;
        if sel_bp.exists() {
            active_bp_json = sel_bp.clone();
        }
        let (feas, n_hs) = std::fs::read_to_string(&sel_json)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .map_or((0.0, 0usize), |v| {
                (
                    v.get("selectivity_feasibility_score")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0),
                    v.get("selectivity_hotspots")
                        .and_then(serde_json::Value::as_array)
                        .map_or(0, Vec::len),
                )
            });
        stages.push(StageRecord {
            stage_number: 2,
            stage_id: "pocket-selectivity".to_string(),
            tool: "dockcraft-cli compare-pockets".to_string(),
            verdict: if n_hs > 0 {
                GateVerdict::Go
            } else {
                GateVerdict::Review
            },
            elapsed_ms: t2b.elapsed().as_millis(),
            summary: format!(
                "ΔMIF vs {} paralog(s): {n_hs} selectivity hotspot(s), feasibility={feas:.3}",
                spec.offtarget_selectivity_receptors.len()
            ),
            outputs: collect_provenance(&[
                ("Selectivity Report JSON", &sel_json),
                ("Selectivity Report MD", &sel_md),
                ("Selectivity ChimeraX Overlay", &sel_cxc),
                ("Selective Blueprint JSON", &sel_bp),
            ]),
        });
    }

    // --- STAGE 3: pharm-screen ---
    let t3 = Instant::now();
    let ps_dir = spec.out_dir.join("03_pharm_funnel");
    std::fs::create_dir_all(&ps_dir)?;
    let ps_csv = ps_dir.join("pharm_hits.csv");
    let ps_sdf = ps_dir.join("pharm_hits.sdf");
    let mut cmd3 = Command::new(&dock_cli);
    cmd3.args(["pharm-screen", "--blueprint"])
        .arg(&active_bp_json)
        .arg("--library")
        .arg(&spec.library_path)
        .args([
            "--min-match",
            &spec.pharm_min_match.to_string(),
            "--top",
            &spec.pharm_top.to_string(),
        ])
        .arg("--out-csv")
        .arg(&ps_csv)
        .arg("--out-sdf")
        .arg(&ps_sdf);
    run_checked_cmd("03_pharm_funnel", &mut cmd3)?;
    let pharm_rows = count_csv_data_rows(&ps_csv);
    let stage4_lib = if pharm_rows > 0 && ps_sdf.exists() {
        ps_sdf.clone()
    } else {
        spec.library_path.clone()
    };
    stages.push(StageRecord {
        stage_number: 3,
        stage_id: "pharm-screen".to_string(),
        tool: "dockcraft-cli pharm-screen".to_string(),
        verdict: if pharm_rows > 0 {
            GateVerdict::Go
        } else {
            GateVerdict::Review
        },
        elapsed_ms: t3.elapsed().as_millis(),
        summary: format!("{pharm_rows} compound(s) passed 3D pharmacophore funnel"),
        outputs: collect_provenance(&[
            ("Pharmacophore Hits CSV", &ps_csv),
            ("Pharmacophore Hits 3D SDF", &ps_sdf),
        ]),
    });

    // --- STAGE 4: Docking / Ensemble Docking ---
    let t4 = Instant::now();
    let dock_dir = spec.out_dir.join("04_docking");
    std::fs::create_dir_all(&dock_dir)?;
    let dock_csv = dock_dir.join("docking_ranked.csv");
    let dock_sdf = dock_dir.join("docking_poses.sdf");
    let dock_summary = if !spec.ensemble_receptors.is_empty() {
        // Prepare auxiliary ensemble conformations first
        let mut rec_paths = vec![prep_pdb.display().to_string()];
        let mut centers = vec![format!(
            "{:.3},{:.3},{:.3}",
            bp_center[0], bp_center[1], bp_center[2]
        )];
        for (ei, entry) in spec.ensemble_receptors.iter().enumerate() {
            let aux_pdb = prep_dir.join(format!("ensemble_rec_{}.pdb", ei + 2));
            let mut c_aux = Command::new(&dock_cli);
            c_aux
                .args(["prot-prep", "--input"])
                .arg(&entry.receptor_pdb)
                .arg("--out-pdb")
                .arg(&aux_pdb)
                .args(["--ph", &format!("{:.1}", spec.ph)]);
            if !entry.chains.is_empty() {
                let ch: String = entry
                    .chains
                    .iter()
                    .map(char::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                c_aux.args(["--chains", &ch]);
            }
            run_checked_cmd("04_ensemble_prot_prep", &mut c_aux)?;
            rec_paths.push(aux_pdb.display().to_string());
            centers.push(format!(
                "{:.3},{:.3},{:.3}",
                entry.box_center[0], entry.box_center[1], entry.box_center[2]
            ));
        }
        let ens_md = dock_dir.join("ensemble_report.md");
        let mut cmd4 = Command::new(&dock_cli);
        cmd4.args([
            "ensemble-dock",
            "--receptors",
            &rec_paths.join(","),
            "--center",
            &centers.join(";"),
            "--size",
            &format!(
                "{:.2},{:.2},{:.2}",
                spec.box_size[0], spec.box_size[1], spec.box_size[2]
            ),
            "--aggregation",
            "boltzmann",
            "--exhaustiveness",
            &spec.dock_exhaustiveness.to_string(),
        ])
        .arg("--library")
        .arg(&stage4_lib)
        .arg("--out-csv")
        .arg(&dock_csv)
        .arg("--out-sdf")
        .arg(&dock_sdf)
        .arg("--out-md")
        .arg(&ens_md);
        run_checked_cmd("04_ensemble_dock", &mut cmd4)?;
        let n_docked = count_csv_data_rows(&dock_csv);
        format!(
            "Boltzmann ensemble docking across {} conformations ({n_docked} ligands ranked)",
            rec_paths.len()
        )
    } else {
        let mut cmd4 = Command::new(&dock_cli);
        cmd4.args([
            "screen",
            "--center",
            &format!(
                "{:.3},{:.3},{:.3}",
                bp_center[0], bp_center[1], bp_center[2]
            ),
            "--size",
            &format!(
                "{:.2},{:.2},{:.2}",
                spec.box_size[0], spec.box_size[1], spec.box_size[2]
            ),
            "--exhaustiveness",
            &spec.dock_exhaustiveness.to_string(),
        ])
        .arg("--receptor")
        .arg(&prep_pdb)
        .arg("--library")
        .arg(&stage4_lib)
        .arg("--pharmacophore-file")
        .arg(&active_bp_json)
        .arg("--out")
        .arg(&dock_csv);
        run_checked_cmd("04_docking", &mut cmd4)?;
        let n_docked = count_csv_data_rows(&dock_csv);
        format!("Pharmacophore-guided docking ({n_docked} ligands ranked)")
    };
    stages.push(StageRecord {
        stage_number: 4,
        stage_id: "docking".to_string(),
        tool: if spec.ensemble_receptors.is_empty() {
            "dockcraft-cli screen".to_string()
        } else {
            "dockcraft-cli ensemble-dock".to_string()
        },
        verdict: GateVerdict::Go,
        elapsed_ms: t4.elapsed().as_millis(),
        summary: dock_summary,
        outputs: collect_provenance(&[
            ("Ranked Docking CSV", &dock_csv),
            ("Docked 3D Poses SDF", &dock_sdf),
        ]),
    });

    // Extract ranked top candidates with SMILES (lookup from input library if SDF/SMI)
    let ranked_hits = extract_top_docked_with_smiles(&dock_csv, &spec.library_path, spec.admet_top);

    // --- STAGE 5: AdmetCraft v1.0 QSAR + Conformal + SoM + TPP MPO ---
    let t5 = Instant::now();
    let admet_dir = spec.out_dir.join("05_admet");
    std::fs::create_dir_all(&admet_dir)?;
    let top_smi_path = admet_dir.join("top_candidates.smi");
    let admet_csv = admet_dir.join("admet_qsar_tpp.csv");
    {
        let mut smi_buf = String::new();
        for (name, smi, _) in &ranked_hits {
            let _ = writeln!(smi_buf, "{smi} {name}");
        }
        std::fs::write(&top_smi_path, smi_buf)?;
    }

    let mut final_leads: Vec<CampaignLeadSummary> = Vec::new();
    if let Some(ac) = &admet_cli
        && !ranked_hits.is_empty()
    {
        let mut cmd5 = Command::new(ac);
        cmd5.arg("batch")
            .arg(&top_smi_path)
            .arg("--out")
            .arg(&admet_csv)
            .args(["--format", "csv", "--tpp", &spec.tpp_profile]);
        run_checked_cmd("05_admet", &mut cmd5)?;
        let admet_map = parse_admet_csv(&admet_csv);
        let mut passing = 0usize;
        for (name, smi, dg) in &ranked_hits {
            let row = admet_map.get(name).cloned().unwrap_or_default();
            let passes = row.mpo.is_some_and(|v| v >= spec.min_tpp_mpo)
                && !row.decision.as_deref().is_some_and(|d| d.contains("Hold"));
            if passes {
                passing += 1;
            }
            final_leads.push(CampaignLeadSummary {
                rank: 0,
                name: name.clone(),
                smiles: smi.clone(),
                dock_affinity_kcal: *dg,
                tpp_mpo: row.mpo,
                tpp_decision: row.decision,
                tox21_alerts: row.tox21_alerts,
                sea_high_risk: row.sea_high_risk,
                metabolite_bioactivation: row.metabolite_bioactivation,
                safety_panel_verdict: None,
                safety_panel_min_margin_kcal: None,
                md_verdict: None,
                synth_md_path: None,
            });
        }
        // Sort passing leads first, then by docking affinity
        final_leads.sort_by(|a, b| {
            let pa = a.tpp_mpo.is_some_and(|v| v >= spec.min_tpp_mpo)
                && !a
                    .tpp_decision
                    .as_deref()
                    .is_some_and(|d| d.contains("Hold"));
            let pb = b.tpp_mpo.is_some_and(|v| v >= spec.min_tpp_mpo)
                && !b
                    .tpp_decision
                    .as_deref()
                    .is_some_and(|d| d.contains("Hold"));
            pb.cmp(&pa).then_with(|| {
                a.dock_affinity_kcal
                    .unwrap_or(0.0)
                    .total_cmp(&b.dock_affinity_kcal.unwrap_or(0.0))
            })
        });
        final_leads.truncate(spec.final_top);
        for (i, l) in final_leads.iter_mut().enumerate() {
            l.rank = i + 1;
        }
        stages.push(StageRecord {
            stage_number: 5,
            stage_id: "admet-qsar-tpp".to_string(),
            tool: "admetcraft-cli batch".to_string(),
            verdict: if passing > 0 {
                GateVerdict::Go
            } else {
                GateVerdict::Review
            },
            elapsed_ms: t5.elapsed().as_millis(),
            summary: format!(
                "{passing}/{} candidates passed TPP '{}' (MPO >= {:.2}, Tox21 + SEA + Metabolite Guard)",
                ranked_hits.len(),
                spec.tpp_profile,
                spec.min_tpp_mpo
            ),
            outputs: collect_provenance(&[("ADMET QSAR & TPP CSV", &admet_csv)]),
        });
    } else {
        for (i, (name, smi, dg)) in ranked_hits.iter().take(spec.final_top).enumerate() {
            final_leads.push(CampaignLeadSummary {
                rank: i + 1,
                name: name.clone(),
                smiles: smi.clone(),
                dock_affinity_kcal: *dg,
                tpp_mpo: None,
                tpp_decision: None,
                tox21_alerts: None,
                sea_high_risk: None,
                metabolite_bioactivation: None,
                safety_panel_verdict: None,
                safety_panel_min_margin_kcal: None,
                md_verdict: None,
                synth_md_path: None,
            });
        }
        stages.push(StageRecord {
            stage_number: 5,
            stage_id: "admet-qsar-tpp".to_string(),
            tool: "admetcraft-cli".to_string(),
            verdict: GateVerdict::Skipped,
            elapsed_ms: t5.elapsed().as_millis(),
            summary: "admetcraft-cli not found on PATH".to_string(),
            outputs: Vec::new(),
        });
    }

    // --- OPTIONAL STAGE 5b: Inverse Docking Safety Panel ---
    if spec.run_safety_panel && !final_leads.is_empty() {
        let t5b = Instant::now();
        let sp_dir = spec.out_dir.join("05b_safety_panel");
        std::fs::create_dir_all(&sp_dir)?;
        let sp_smi = sp_dir.join("leads.smi");
        let sp_json = sp_dir.join("safety_panel_report.json");
        let sp_md = sp_dir.join("safety_panel_report.md");
        let sp_poses = sp_dir.join("flagged_poses");
        {
            let mut buf = String::new();
            for l in &final_leads {
                if !l.smiles.trim().is_empty() {
                    let _ = writeln!(buf, "{} {}", l.smiles, l.name);
                }
            }
            std::fs::write(&sp_smi, buf)?;
        }
        let ref_dg = final_leads
            .first()
            .and_then(|l| l.dock_affinity_kcal)
            .unwrap_or(-10.0);
        let mut cmd5b = Command::new(&dock_cli);
        cmd5b
            .args(["safety-panel", "--ligands"])
            .arg(&sp_smi)
            .args([
                "--target-dg",
                &format!("{ref_dg:.2}"),
                "--exhaustiveness",
                &spec.safety_panel_exhaustiveness.clamp(1, 16).to_string(),
            ])
            .arg("--out-json")
            .arg(&sp_json)
            .arg("--out-md")
            .arg(&sp_md)
            .arg("--out-poses-dir")
            .arg(&sp_poses);
        if let Some(mpath) = &spec.safety_panel_manifest {
            cmd5b.arg("--manifest").arg(mpath);
        }
        run_checked_cmd("05b_safety_panel", &mut cmd5b)?;

        let mut pass_cnt = 0usize;
        let mut n_targets = 0usize;
        if let Ok(raw_sp) = std::fs::read_to_string(&sp_json)
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw_sp)
        {
            n_targets = v
                .get("targets_evaluated")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;
            if let Some(sums) = v.get("summaries").and_then(serde_json::Value::as_array) {
                for s in sums {
                    let lname = s
                        .get("ligand_name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("");
                    let verd = s
                        .get("verdict")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("Caution")
                        .to_string();
                    let min_m = s
                        .get("min_selectivity_margin_kcal")
                        .and_then(serde_json::Value::as_f64);
                    if verd.eq_ignore_ascii_case("Pass") {
                        pass_cnt += 1;
                    }
                    if let Some(lead) = final_leads.iter_mut().find(|l| l.name == lname) {
                        lead.safety_panel_verdict = Some(verd);
                        lead.safety_panel_min_margin_kcal = min_m;
                    }
                }
            }
        }
        stages.push(StageRecord {
            stage_number: 5,
            stage_id: "safety-panel".to_string(),
            tool: "dockcraft-cli safety-panel".to_string(),
            verdict: if pass_cnt > 0 {
                GateVerdict::Go
            } else {
                GateVerdict::Review
            },
            elapsed_ms: t5b.elapsed().as_millis(),
            summary: format!(
                "Inverse docked {} lead(s) across {n_targets} Bowes anti-targets ({pass_cnt} PASS)",
                final_leads.len()
            ),
            outputs: collect_provenance(&[
                ("Safety Panel JSON", &sp_json),
                ("Safety Panel Report MD", &sp_md),
            ]),
        });
    }

    // --- STAGE 6: OpenMM GPU MD Validation (live if run_md=true, dry-run bundle otherwise) ---
    let t6 = Instant::now();
    let md_root = spec.out_dir.join("06_md_validation");
    std::fs::create_dir_all(&md_root)?;
    let mut md_outs: Vec<FileProvenance> = Vec::new();
    let lig_pose_src = if dock_sdf.exists() {
        dock_sdf.clone()
    } else {
        stage4_lib.clone()
    };
    if let Some(top_lead) = final_leads.first_mut() {
        let lead_md_dir = md_root.join(&top_lead.name);
        let mut cmd6 = Command::new(&dock_cli);
        cmd6.args(["md-validate", "--receptor"])
            .arg(&prep_pdb)
            .arg("--ligand")
            .arg(&lig_pose_src)
            .arg("--out-dir")
            .arg(&lead_md_dir)
            .args(["--ns", &format!("{:.3}", spec.md_production_ns)]);
        if !spec.run_md {
            cmd6.arg("--dry-run");
        }
        if run_checked_cmd("06_md_validation", &mut cmd6).is_ok() {
            let md_rep = lead_md_dir.join("md_report.md");
            if let Ok(fp) = provenance_of("MD Validation Report", &md_rep) {
                md_outs.push(fp);
            }
            top_lead.md_verdict = Some(if spec.run_md {
                "Executed (see md_report.md)".to_string()
            } else {
                "Dry-Run Staged".to_string()
            });
        }
    }
    stages.push(StageRecord {
        stage_number: 6,
        stage_id: "md-validate".to_string(),
        tool: "dockcraft-cli md-validate".to_string(),
        verdict: if spec.run_md {
            GateVerdict::Go
        } else {
            GateVerdict::Skipped
        },
        elapsed_ms: t6.elapsed().as_millis(),
        summary: if spec.run_md {
            format!(
                "Executed OpenMM MD ({:.2} ns) on top lead",
                spec.md_production_ns
            )
        } else {
            "Staged runnable OpenMM GPU MD bundle (pass run_md=true to launch GPU)".to_string()
        },
        outputs: md_outs,
    });

    // --- STAGE 7: ChemCraft v0.3.0 Retrosynthesis (Buy-Before-Make + Retro*) ---
    let t7 = Instant::now();
    let syn_dir = spec.out_dir.join("07_synthesis");
    std::fs::create_dir_all(&syn_dir)?;
    let mut syn_outs: Vec<FileProvenance> = Vec::new();
    let mut planned_cnt = 0usize;
    if let Some(cc) = &chem_cli {
        for lead in &mut final_leads {
            if lead.smiles.trim().is_empty() {
                continue;
            }
            let out_md = syn_dir.join(format!("synth_{}.md", lead.name));
            let rxn_dir = syn_dir.join(format!("rxn_{}", lead.name));
            let mut cmd7 = Command::new(cc);
            cmd7.args([
                "retro",
                &lead.smiles,
                "--name",
                &lead.name,
                "--scale-mg",
                &format!("{:.1}", spec.synth_scale_mg),
            ])
            .arg("--out-md")
            .arg(&out_md)
            .arg("--out-rxn-dir")
            .arg(&rxn_dir);
            if run_checked_cmd("07_synthesis", &mut cmd7).is_ok() {
                planned_cnt += 1;
                lead.synth_md_path = Some(out_md.display().to_string());
                if let Ok(fp) = provenance_of(&format!("Synthesis {}", lead.name), &out_md) {
                    syn_outs.push(fp);
                }
            }
        }
        stages.push(StageRecord {
            stage_number: 7,
            stage_id: "retrosynthesis".to_string(),
            tool: "chemcraft-cli retro".to_string(),
            verdict: if planned_cnt > 0 {
                GateVerdict::Go
            } else {
                GateVerdict::Review
            },
            elapsed_ms: t7.elapsed().as_millis(),
            summary: format!(
                "Planned forward synthesis + MDL .rxn for {planned_cnt} final lead(s)"
            ),
            outputs: syn_outs,
        });
    } else {
        stages.push(StageRecord {
            stage_number: 7,
            stage_id: "retrosynthesis".to_string(),
            tool: "chemcraft-cli retro".to_string(),
            verdict: GateVerdict::Skipped,
            elapsed_ms: t7.elapsed().as_millis(),
            summary: "chemcraft-cli not found on PATH".to_string(),
            outputs: Vec::new(),
        });
    }

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let manifest = CampaignManifest {
        campaign_name: spec.campaign_name.clone(),
        target_name: spec.target_name.clone(),
        unix_timestamp: ts,
        tool_versions,
        input_provenance,
        stages,
        final_leads,
        total_elapsed_ms: t_total.elapsed().as_millis(),
    };

    std::fs::write(
        spec.out_dir.join("campaign_manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    std::fs::write(
        spec.out_dir.join("CAMPAIGN_REPORT.md"),
        manifest.render_markdown(),
    )?;
    Ok(manifest)
}

fn find_installed_tool(bin: &str) -> Option<PathBuf> {
    if let Ok(home) = std::env::var("HOME") {
        let local = PathBuf::from(home).join(".local/bin").join(bin);
        if local.exists() {
            return Some(local);
        }
    }
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let cand = dir.join(bin);
            if cand.exists() {
                return Some(cand);
            }
        }
    }
    None
}

fn probe_version(exe: &Path) -> String {
    Command::new(exe)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn run_checked_cmd(stage: &str, cmd: &mut Command) -> Result<(), OrchestratorError> {
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(OrchestratorError::StageFailed {
            stage: stage.to_string(),
            message: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        });
    }
    Ok(())
}

fn collect_provenance(items: &[(&str, &PathBuf)]) -> Vec<FileProvenance> {
    items
        .iter()
        .filter_map(|(role, p)| provenance_of(role, p).ok())
        .collect()
}

fn provenance_of(role: &str, path: &Path) -> std::io::Result<FileProvenance> {
    let bytes = std::fs::read(path)?;
    Ok(FileProvenance {
        role: role.to_string(),
        path: path.display().to_string(),
        sha256: sha256_hex(&bytes),
        bytes: bytes.len() as u64,
    })
}

fn count_csv_data_rows(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .map(|s| s.lines().skip(1).filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0)
}

fn extract_top_docked_with_smiles(
    dock_csv: &Path,
    library_path: &Path,
    top_n: usize,
) -> Vec<(String, String, Option<f64>)> {
    // Index SMILES from original library (.smi) by compound name
    let mut name_to_smi = std::collections::HashMap::new();
    if let Ok(lib_txt) = std::fs::read_to_string(library_path) {
        for line in lib_txt.lines() {
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            let mut parts = t.split_whitespace();
            if let (Some(smi), Some(nm)) = (parts.next(), parts.next()) {
                name_to_smi.insert(nm.to_string(), smi.to_string());
            }
        }
    }

    let mut out = Vec::new();
    let Ok(mut rdr) = csv::Reader::from_path(dock_csv) else {
        return out;
    };
    let Ok(headers) = rdr.headers().cloned() else {
        return out;
    };
    let name_col = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("Compound") || h.eq_ignore_ascii_case("Name"))
        .unwrap_or(1);
    let score_col = headers
        .iter()
        .position(|h| {
            h.contains("Ensemble_dG")
                || h.contains("Affinity")
                || h.contains("Vina")
                || h.contains("dG")
        })
        .unwrap_or(3);
    for rec in rdr.records().flatten().take(top_n) {
        let name = rec.get(name_col).unwrap_or("").trim().to_string();
        if name.is_empty() {
            continue;
        }
        let dg = rec
            .get(score_col)
            .and_then(|s| s.trim().parse::<f64>().ok());
        let smi = name_to_smi.get(&name).cloned().unwrap_or_default();
        out.push((name, smi, dg));
    }
    out
}

#[derive(Debug, Clone, Default)]
struct AdmetCsvRow {
    mpo: Option<f64>,
    decision: Option<String>,
    tox21_alerts: Option<String>,
    sea_high_risk: Option<String>,
    metabolite_bioactivation: Option<usize>,
}

fn parse_admet_csv(path: &Path) -> std::collections::HashMap<String, AdmetCsvRow> {
    let mut map = std::collections::HashMap::new();
    let Ok(mut rdr) = csv::Reader::from_path(path) else {
        return map;
    };
    let Ok(headers) = rdr.headers().cloned() else {
        return map;
    };
    let mol_col = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("Molecule"))
        .unwrap_or(0);
    let mpo_col = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("TPP MPO Score"));
    let dec_col = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("TPP Decision"));
    let tox21_col = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("Tox21 Alerts"));
    let sea_col = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("SEA Off-Target High"));
    let bio_col = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("Metabolite Bioactivation"));
    for rec in rdr.records().flatten() {
        let name = rec.get(mol_col).unwrap_or("").trim().to_string();
        if name.is_empty() {
            continue;
        }
        let mpo = mpo_col
            .and_then(|c| rec.get(c))
            .and_then(|s| s.trim().parse::<f64>().ok());
        let decision = dec_col
            .and_then(|c| rec.get(c))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let tox21_alerts = tox21_col
            .and_then(|c| rec.get(c))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let sea_high_risk = sea_col
            .and_then(|c| rec.get(c))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let metabolite_bioactivation = bio_col
            .and_then(|c| rec.get(c))
            .and_then(|s| s.trim().parse::<usize>().ok());
        map.insert(
            name,
            AdmetCsvRow {
                mpo,
                decision,
                tox21_alerts,
                sea_high_risk,
                metabolite_bioactivation,
            },
        );
    }
    map
}

/// Pure safe Rust FIPS 180-4 SHA-256 digest (zero external crates).
#[must_use]
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = Vec::with_capacity(data.len() + 72);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let b = &chunk[4 * i..4 * i + 4];
            w[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for word in h {
        let _ = write!(out, "{word:08x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_nist_test_vector_and_template_serializes() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let tpl = default_cdk2_campaign_template();
        let js = serde_json::to_string_pretty(&tpl);
        assert!(js.is_ok());
    }
}

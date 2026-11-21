#![forbid(unsafe_code)]

pub mod admet;
pub mod campaign;
pub mod chimerax;
pub mod chimerax_rest;
pub mod cnn_rescore;
pub mod export;
pub mod fep_bridge;
pub mod lead_opt;
pub mod md_validate;
pub mod orchestrator;
pub mod results_sidecar;
pub mod safety_panel;
pub mod sandbox;

pub use admet::{AdmetProfile, detect_sibling_clis, evaluate_admet};
pub use campaign::{
    CampaignError, CampaignReport, EnrichmentMetrics, LigandInputRecord, ScreenedCandidate,
    campaign_report_to_csv, compute_enrichment_metrics, compute_mpo_score,
    load_ligand_library_bytes, measured_pic50_of, run_screening_campaign,
};
pub use chimerax::{
    ChimeraXError, ChimeraXSceneBundle, SelectionBox, default_chimerax_export_dir,
    grid_box_from_chimerax_selection, is_chimerax_available, launch_chimerax_for_scene,
    write_chimerax_session_bundle,
};
pub use chimerax_rest::{
    ChimeraXCommand, ChimeraXReply, ChimeraXRest, DEFAULT_REST_PORT, RestError,
};
pub use cnn_rescore::{
    CnnRescoreError, CnnRescoreReport, CnnRescoredPose, GninaEnvStatus, GninaRescoreConfig,
    check_gnina_env, compute_3way_consensus_kcal, parse_gnina_sdf_output, resolve_gnina_binary,
    run_gnina_rescore,
};
pub use export::{
    InteropSceneBundle, SceneExportRequest, build_chimerax_export, build_pymol_export,
    canonical_export_target, write_sandboxed_scene_package,
};
pub use fep_bridge::{
    FepBackend, FepConfig, FepCycleClosure, FepEdgeResult, FepEnvStatus, FepError, FepLegResult,
    FepReport, RbfePlannedEdge, check_fep_env, compute_cycle_closures, plan_rbfe_network,
    run_fep_network,
};
pub use lead_opt::{
    LeadOptAnalog, LeadOptConfig, LeadOptError, LeadOptMode, LeadOptReport, run_lead_optimization,
};
pub use md_validate::{
    HBondPersistence, MdEnvStatus, MdError, MdPlatform, MdStabilityVerdict, MdValidateConfig,
    MdValidateReport, MmgbsaBreakdown, ReplicateSummary, ResidueEnergyDecomp, ResidueRmsf,
    build_chimerax_md_script, check_openmm_env, craft_data_dir, resolve_openmm_python,
    run_md_validation,
};
pub use orchestrator::{
    CampaignLeadSummary, CampaignManifest, CampaignSpec, EnsembleSpecEntry, FileProvenance,
    GateVerdict, OrchestratorError, StageRecord, default_cdk2_campaign_template, run_campaign,
    sha256_hex,
};
pub use results_sidecar::{
    ResultsSidecar, SIDECAR_FORMAT, SIDECAR_VERSION, SidecarError, SidecarRow, SidecarRun,
    default_chemcraft_results_dir, write_results_sidecar,
};
pub use safety_panel::{
    AntiTargetEntry, LigandSafetySummary, LoadedAntiTarget, OffTargetDockHit, PairRisk,
    SafetyPanelConfig, SafetyPanelError, SafetyPanelManifest, SafetyPanelReport, SafetyVerdict,
    default_bowes_tier1_catalog, default_safety_panel_dir, default_safety_panel_manifest_path,
    load_safety_panel_receptors, run_safety_panel, run_safety_panel_loaded,
};
pub use sandbox::{
    DEFAULT_MAX_FILE_BYTES, MAX_ZIP_DECOMPRESSED_BYTES, PathSandbox, SandboxError,
    read_bounded_chemcraft_entries,
};

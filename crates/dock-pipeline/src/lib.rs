#![forbid(unsafe_code)]

pub mod admet;
pub mod campaign;
pub mod chimerax;
pub mod chimerax_rest;
pub mod export;
pub mod results_sidecar;
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
pub use export::{
    InteropSceneBundle, SceneExportRequest, build_chimerax_export, build_pymol_export,
    canonical_export_target, write_sandboxed_scene_package,
};
pub use results_sidecar::{
    ResultsSidecar, SIDECAR_FORMAT, SIDECAR_VERSION, SidecarError, SidecarRow, SidecarRun,
    default_chemcraft_results_dir, write_results_sidecar,
};
pub use sandbox::{
    DEFAULT_MAX_FILE_BYTES, MAX_ZIP_DECOMPRESSED_BYTES, PathSandbox, SandboxError,
    read_bounded_chemcraft_entries,
};

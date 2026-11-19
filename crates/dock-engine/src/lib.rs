#![forbid(unsafe_code)]

pub mod constraints;
pub mod external;
pub mod grid;
pub mod search;
pub mod vina;

pub use constraints::{DockingConstraint, PharmacophoreKind};
pub use external::{ExternalEngineKind, detect_available_external_engines, run_external_docking};
pub use grid::{AffinityGridSet, AtomChannelKey};
pub use search::{
    DockingConfig, DockingPose, DockingReport, PoseTerms, dock_ligand, dock_ligand_with_grids,
};
pub use vina::{
    ScoreBreakdown, ScoringFunction, metal_coord_energy_and_deriv, score_complex,
    score_complex_with_options,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DockError {
    #[error("empty receptor or ligand coordinates")]
    EmptyStructure,
    #[error("ligand atom count {count} exceeds limit {limit}")]
    TooManyLigandAtoms { count: usize, limit: usize },
    #[error("grid error: {0}")]
    Grid(#[from] pocket_detect::PocketError),
    #[error("format error: {0}")]
    Format(#[from] struct_core::StructError),
    #[error("external engine error: {0}")]
    ExternalEngine(String),
}

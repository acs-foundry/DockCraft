use thiserror::Error;

#[derive(Debug, Error)]
pub enum StructError {
    #[error("empty input")]
    EmptyInput,
    #[error("invalid PDB at line {line}: {msg}")]
    InvalidPdb { line: usize, msg: String },
    #[error("invalid mmCIF at line {line}: {msg}")]
    InvalidCif { line: usize, msg: String },
    #[error("invalid MOL2 at line {line}: {msg}")]
    InvalidMol2 { line: usize, msg: String },
    #[error("invalid SDF at line {line}: {msg}")]
    InvalidSdf { line: usize, msg: String },
    #[error("invalid PDBQT at line {line}: {msg}")]
    InvalidPdbqt { line: usize, msg: String },
    #[error("invalid SMILES at pos {pos}: {msg}")]
    InvalidSmiles { pos: usize, msg: String },
    #[error("atom limit exceeded ({count} > {limit})")]
    TooManyAtoms { count: usize, limit: usize },
    #[error("topology error: {0}")]
    Topology(String),
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("vertex not found in PLY")]
    PlyVertexNotFound,
    #[error("vertex property {0} not found in PLY")]
    PlyVertexPropertyNotFound(String),
    #[error("variable name {0} not found in packages")]
    VarNotFound(String),
}

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
    #[error("{0}")]
    Wesl(#[from] wesl::Error),
    #[error(
        "\
        resource count and bind group layout count mismatch: \
        {resource_count} != {bind_group_layout_count}\
        "
    )]
    ResourceCountMismatch {
        resource_count: usize,
        bind_group_layout_count: usize,
    },
    #[error("missing bind group layout for compute bundle")]
    MissingBindGroupLayout,
    #[error("missing resolver for compute bundle")]
    MissingResolver,
    #[error("missing entry point for compute bundle")]
    MissingEntryPoint,
    #[error("missing main shader for compute bundle")]
    MissingMainShader,
    #[error("{0}")]
    BufferDownloadOneShotReceive(#[from] oneshot::RecvError),
    #[error("{0}")]
    BufferDownloadAsync(#[from] wgpu::BufferAsyncError),
    #[error("{0}")]
    DeviceFailedToPoll(#[from] wgpu::PollError),
}

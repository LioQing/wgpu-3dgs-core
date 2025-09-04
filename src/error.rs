use thiserror::Error;

/// The error type for reading PLY.
#[derive(Debug, Error)]
pub enum ReadPlyError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("vertex not found in PLY")]
    VertexNotFound,
    #[error("vertex property {0} not found in PLY")]
    VertexPropertyNotFound(String),
}

/// The error type for downloading buffer.
#[derive(Debug, Error)]
pub enum DownloadBufferError {
    #[error("{0}")]
    OneShotRecv(#[from] oneshot::RecvError),
    #[error("{0}")]
    Async(#[from] wgpu::BufferAsyncError),
    #[error("{0}")]
    Poll(#[from] wgpu::PollError),
}

/// The error type for [`GaussiansBuffer`](crate::GaussiansBuffer) update functions.
#[derive(Debug, Error)]
pub enum GaussiansBufferUpdateError {
    #[error("Gaussians count mismatch: {count} != {expected_count}")]
    CountMismatch { count: usize, expected_count: usize },
}

/// The error type for [`GaussiansBuffer`](crate::GaussiansBuffer) update range functions.
#[derive(Debug, Error)]
pub enum GaussiansBufferUpdateRangeError {
    #[error("Gaussians count mismatch: {count} + {start} > {expected_count}")]
    CountMismatch {
        count: usize,
        start: usize,
        expected_count: usize,
    },
}

/// The error type for [`GaussiansBuffer`](crate::GaussiansBuffer)'s [`TryFrom`] implementation for
/// [`wgpu::Buffer`].
#[derive(Debug, Error)]
pub enum GaussiansBufferTryFromBufferError {
    #[error(
        "buffer size and expected multiple size mismatch: {buffer_size} % {expected_multiple_size} != 0"
    )]
    BufferSizeNotMultiple {
        buffer_size: wgpu::BufferAddress,
        expected_multiple_size: wgpu::BufferAddress,
    },
}

/// The error type for [`FixedSizeBufferWrapper`](crate::FixedSizeBufferWrapper).
#[derive(Debug, Error)]
pub enum FixedSizeBufferWrapperError {
    #[error("buffer size and expected size mismatch: {buffer_size} != {expected_size}")]
    BufferSizeMismatched {
        buffer_size: wgpu::BufferAddress,
        expected_size: wgpu::BufferAddress,
    },
}

/// The error type for [`ComputeBundle`](crate::ComputeBundle) creation.
#[derive(Debug, Error)]
pub enum ComputeBundleCreateError {
    #[error(
        "resource count and bind group layout count mismatch: \
        {resource_count} != {bind_group_layout_count}\
        "
    )]
    ResourceCountMismatch {
        resource_count: usize,
        bind_group_layout_count: usize,
    },
}

/// The error type for [`ComputeBundleBuilder::build`](crate::ComputeBundleBuilder::build) function.
#[derive(Debug, Error)]
pub enum ComputeBundleBuildError {
    #[error("{0}")]
    Wesl(#[from] wesl::Error),
    #[error("{0}")]
    Create(#[from] ComputeBundleCreateError),
    #[error("missing bind group layout for compute bundle")]
    MissingBindGroupLayout,
    #[error("missing resolver for compute bundle")]
    MissingResolver,
    #[error("missing entry point for compute bundle")]
    MissingEntryPoint,
    #[error("missing main shader for compute bundle")]
    MissingMainShader,
}

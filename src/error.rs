use thiserror::Error;

use crate::{SpzGaussianPosition, SpzGaussianRotation, SpzGaussianSh};

/// The error type for [`SpzGaussians::from_gaussian_slice_with_options`](crate::SpzGaussians::from_gaussian_slice_with_options).
#[derive(Debug, Error)]
pub enum SpzGaussiansFromGaussianSliceError {
    #[error("{0}")]
    Fromiter(#[from] SpzGaussiansFromIterError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

/// The error type for [`SpzGaussians::from_iter`](crate::SpzGaussians::from_iter).
#[derive(Debug, Error)]
pub enum SpzGaussiansFromIterError {
    #[error("invalid mixed position variant: {0:?}")]
    InvalidMixedPositionVariant(SpzGaussiansCollectError<SpzGaussianPosition>),
    #[error("invalid mixed rotation variant: {0:?}")]
    InvalidMixedRotationVariant(SpzGaussiansCollectError<SpzGaussianRotation>),
    #[error("invalid mixed SH variant: {0:?}")]
    InvalidMixedShVariant(SpzGaussiansCollectError<SpzGaussianSh>),
    #[error("Gaussians count mismatch: {actual_count} != {header_count}")]
    CountMismatch {
        actual_count: usize,
        header_count: usize,
    },
    #[error("Position float16 format mismatch: {is_float16} != {header_uses_float16}")]
    PositionFloat16Mismatch {
        is_float16: bool,
        header_uses_float16: bool,
    },
    #[error(
        "Rotation smallest three format mismatch: \
        {is_quat_smallest_three} != {header_uses_quat_smallest_three}\
        "
    )]
    RotationQuatSmallestThreeMismatch {
        is_quat_smallest_three: bool,
        header_uses_quat_smallest_three: bool,
    },
    #[error("SH degree mismatch: {sh_degree} != {header_sh_degree}")]
    ShDegreeMismatch { sh_degree: u8, header_sh_degree: u8 },
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

/// The error type for collecting SPZ Gaussians.
#[derive(Debug, Error)]
pub enum SpzGaussiansCollectError<T> {
    #[error("invalid mixed variant: {first_variant} != {current_variant}")]
    InvalidMixedVariant {
        first_variant: T,
        current_variant: T,
    },
    #[error("empty iterator")]
    EmptyIterator,
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
    #[error(
        "workgroup size exceeds device limit: \
        {workgroup_size} > {device_limit}"
    )]
    WorkgroupSizeExceedsDeviceLimit {
        workgroup_size: u32,
        device_limit: u32,
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

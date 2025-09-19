use glam::*;
use wgpu::util::DeviceExt;

use crate::{BufferWrapper, FixedSizeBufferWrapper, FixedSizeBufferWrapperError};

/// The Gaussian display modes.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GaussianDisplayMode {
    Splat = 0,
    Ellipse = 1,
    Point = 2,
}

/// The Gaussian spherical harmonics degrees.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianShDegree(u8);

impl GaussianShDegree {
    /// Create a new Gaussian SH degree.
    ///
    /// Returns [`None`] if the degree is not in the range of \[0, 3\].
    pub const fn new(sh_deg: u8) -> Option<Self> {
        match sh_deg {
            0..=3 => Some(Self(sh_deg)),
            _ => None,
        }
    }

    /// Create a new Gaussian SH degree without checking.
    pub const fn new_unchecked(sh_deg: u8) -> Self {
        Self(sh_deg)
    }

    /// Get the degree.
    pub const fn degree(&self) -> u8 {
        self.0
    }
}

/// The Gaussian transform buffer.
///
/// This buffer holds the Gaussian transformation data, including size, display mode, SH degree,
/// and whether to show SH0.
#[derive(Debug, Clone)]
pub struct GaussianTransformBuffer(wgpu::Buffer);

impl GaussianTransformBuffer {
    /// Create a new Gaussian transform buffer.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian transform Buffer"),
            contents: bytemuck::bytes_of(&GaussianTransformPod::default()),
            usage: Self::DEFAULT_USAGES,
        });

        Self(buffer)
    }

    /// Update the Gaussian transformation buffer.
    pub fn update(
        &self,
        queue: &wgpu::Queue,
        size: f32,
        display_mode: GaussianDisplayMode,
        sh_deg: GaussianShDegree,
        no_sh0: bool,
        std_dev: f32,
    ) {
        self.update_with_pod(
            queue,
            &GaussianTransformPod::new(size, display_mode, sh_deg, no_sh0, std_dev),
        );
    }

    /// Update the Gaussian transformation buffer with [`GaussianTransformPod`].
    pub fn update_with_pod(&self, queue: &wgpu::Queue, transform: &GaussianTransformPod) {
        queue.write_buffer(&self.0, 0, bytemuck::bytes_of(transform));
    }
}

impl BufferWrapper for GaussianTransformBuffer {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<GaussianTransformBuffer> for wgpu::Buffer {
    fn from(wrapper: GaussianTransformBuffer) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for GaussianTransformBuffer {
    type Error = FixedSizeBufferWrapperError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for GaussianTransformBuffer {
    type Pod = GaussianTransformPod;
}

/// The POD representation of a Gaussian transformation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianTransformPod {
    pub size: f32,

    /// \[display_mode, sh_deg, no_sh0, std_dev\]
    pub flags: U8Vec4,
}

impl GaussianTransformPod {
    /// Create a new Gaussian transformation.
    pub const fn new(
        size: f32,
        display_mode: GaussianDisplayMode,
        sh_deg: GaussianShDegree,
        no_sh0: bool,
        std_dev: f32,
    ) -> Self {
        let display_mode = display_mode as u8;
        let sh_deg = sh_deg.0;
        let no_sh0 = no_sh0 as u8;
        let std_dev = (std_dev / 3.0 * 255.0) as u8;

        Self {
            size,
            flags: u8vec4(display_mode, sh_deg, no_sh0, std_dev),
        }
    }
}

impl Default for GaussianTransformPod {
    fn default() -> Self {
        Self::new(
            1.0,
            GaussianDisplayMode::Splat,
            GaussianShDegree::new_unchecked(3),
            false,
            3.0,
        )
    }
}

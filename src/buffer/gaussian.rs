use glam::*;

use wgpu::util::DeviceExt;

use crate::{
    BufferWrapper, Gaussian, GaussianCov3dConfig, GaussianCov3dHalfConfig,
    GaussianCov3dSingleConfig, GaussianShConfig, GaussianShHalfConfig, GaussianShNoneConfig,
    GaussianShNorm8Config, GaussianShSingleConfig,
};

/// The Gaussians storage buffer.
#[derive(Debug)]
pub struct GaussiansBuffer<G: GaussianPod>(wgpu::Buffer, std::marker::PhantomData<G>);

impl<G: GaussianPod> GaussiansBuffer<G> {
    /// Create a new Gaussians buffer.
    pub fn new(device: &wgpu::Device, gaussians: &[Gaussian]) -> Self {
        Self::new_with_pods(
            device,
            gaussians
                .iter()
                .map(G::from_gaussian)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    /// Create a new Gaussians buffer with [`GaussianPod`].
    pub fn new_with_pods(device: &wgpu::Device, gaussians: &[G]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussians Buffer"),
            contents: bytemuck::cast_slice(gaussians),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self(buffer, std::marker::PhantomData)
    }

    /// Create a new Gaussians buffer with the specified size.
    pub fn new_empty(device: &wgpu::Device, len: usize) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussians Buffer"),
            size: (len * std::mem::size_of::<G>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self(buffer, std::marker::PhantomData)
    }

    /// Get the number of Gaussians.
    pub fn len(&self) -> usize {
        self.0.size() as usize / std::mem::size_of::<G>()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Update the buffer.
    pub fn update(&self, queue: &wgpu::Queue, gaussians: &[Gaussian]) {
        if gaussians.len() != self.len() {
            log::error!(
                "Gaussians count mismatch, buffer has {}, but {} were provided",
                self.len(),
                gaussians.len()
            );
            return;
        }

        self.update_with_pod(
            queue,
            gaussians
                .iter()
                .map(G::from_gaussian)
                .collect::<Vec<_>>()
                .as_slice(),
        );
    }

    /// Update the buffer with [`GaussianPod`].
    pub fn update_with_pod(&self, queue: &wgpu::Queue, pods: &[G]) {
        if pods.len() != self.len() {
            log::error!(
                "Gaussians count mismatch, buffer has {}, but {} were provided",
                self.len(),
                pods.len()
            );
            return;
        }

        queue.write_buffer(&self.0, 0, bytemuck::cast_slice(pods));
    }

    /// Update a range of the buffer.
    pub fn update_range(&self, queue: &wgpu::Queue, start: usize, gaussians: &[Gaussian]) {
        if start + gaussians.len() > self.len() {
            log::error!(
                "Gaussians count mismatch, buffer has {}, but {} were provided starting at {}",
                self.len(),
                gaussians.len(),
                start
            );
            return;
        }

        self.update_range_with_pod(
            queue,
            start,
            gaussians
                .iter()
                .map(G::from_gaussian)
                .collect::<Vec<_>>()
                .as_slice(),
        );
    }

    /// Update a range of the buffer with [`GaussianPod`].
    pub fn update_range_with_pod(&self, queue: &wgpu::Queue, start: usize, pods: &[G]) {
        if start + pods.len() > self.len() {
            log::error!(
                "Gaussians count mismatch, buffer has {}, but {} were provided starting at {}",
                self.len(),
                pods.len(),
                start
            );
            return;
        }

        queue.write_buffer(
            &self.0,
            (start * std::mem::size_of::<G>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(pods),
        );
    }
}

impl<G: GaussianPod> BufferWrapper for GaussiansBuffer<G> {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The Gaussian POD trait.
pub trait GaussianPod: for<'a> From<&'a Gaussian> + bytemuck::NoUninit {
    /// The SH configuration.
    type ShConfig: GaussianShConfig;

    /// The covariance 3D configuration.
    type Cov3dConfig: GaussianCov3dConfig;

    /// Create a new Gaussian POD from the Gaussian.
    fn from_gaussian(gaussian: &Gaussian) -> Self {
        Self::from(gaussian)
    }

    /// Create the features for [`Wesl`](wesl::Wesl) compilation.
    fn features() -> [(&'static str, bool); 6] {
        [
            GaussianShSingleConfig::FEATURE,
            GaussianShHalfConfig::FEATURE,
            GaussianShNorm8Config::FEATURE,
            GaussianShNoneConfig::FEATURE,
            GaussianCov3dSingleConfig::FEATURE,
            GaussianCov3dHalfConfig::FEATURE,
        ]
        .map(|name| {
            (
                name,
                name == Self::ShConfig::FEATURE || name == Self::Cov3dConfig::FEATURE,
            )
        })
    }
}

/// Macro to create the POD representation of Gaussian given the configurations.
macro_rules! gaussian_pod {
    (sh = $sh:ident, cov3d = $cov3d:ident, padding_size = $padding:expr) => {
        paste::paste! {
            /// The POD representation of Gaussian.
            #[repr(C)]
            #[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
            pub struct [< GaussianPodWith Sh $sh Cov3d $cov3d Configs >] {
                pub pos: Vec3,
                pub color: U8Vec4,
                pub sh: <[< GaussianSh $sh Config >] as GaussianShConfig>::Field,
                pub cov3d: <[< GaussianCov3d $cov3d Config >] as GaussianCov3dConfig>::Field,
                _padding: [f32; $padding],
            }

            impl From<&Gaussian> for [< GaussianPodWith Sh $sh Cov3d $cov3d Configs >] {
                fn from(gaussian: &Gaussian) -> Self {
                    // Covariance
                    let r = Mat3::from_quat(gaussian.rotation);
                    let s = Mat3::from_diagonal(gaussian.scale);
                    let m = r * s;
                    let sigma = m * m.transpose();
                    let cov3d = [< GaussianCov3d $cov3d Config >]::from_cov3d([
                        sigma.x_axis.x,
                        sigma.x_axis.y,
                        sigma.x_axis.z,
                        sigma.y_axis.y,
                        sigma.y_axis.z,
                        sigma.z_axis.z,
                    ]);

                    // Color
                    let color = gaussian.color;

                    // Spherical harmonics
                    let sh = [< GaussianSh $sh Config >]::from_sh(&gaussian.sh);

                    // Position
                    let pos = gaussian.pos;

                    Self {
                        pos,
                        color,
                        sh,
                        cov3d,
                        _padding: [0.0; $padding],
                    }
                }
            }

            impl GaussianPod for [< GaussianPodWith Sh $sh Cov3d $cov3d Configs >] {
                type ShConfig = [< GaussianSh $sh Config >];
                type Cov3dConfig = [< GaussianCov3d $cov3d Config >];
            }
        }
    };
}

gaussian_pod!(sh = Single, cov3d = Single, padding_size = 1);
gaussian_pod!(sh = Single, cov3d = Half, padding_size = 0);
gaussian_pod!(sh = Half, cov3d = Single, padding_size = 3);
gaussian_pod!(sh = Half, cov3d = Half, padding_size = 2);
gaussian_pod!(sh = Norm8, cov3d = Single, padding_size = 1);
gaussian_pod!(sh = Norm8, cov3d = Half, padding_size = 0);
gaussian_pod!(sh = None, cov3d = Single, padding_size = 2);
gaussian_pod!(sh = None, cov3d = Half, padding_size = 1);

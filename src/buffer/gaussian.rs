use glam::*;

use wgpu::util::DeviceExt;

use crate::{
    BufferWrapper, DownloadableBufferWrapper, Gaussian, GaussianCov3dConfig,
    GaussianCov3dHalfConfig, GaussianCov3dRotScaleConfig, GaussianCov3dSingleConfig,
    GaussianShConfig, GaussianShHalfConfig, GaussianShNoneConfig, GaussianShNorm8Config,
    GaussianShSingleConfig,
};

/// The Gaussians storage buffer.
#[derive(Debug, Clone)]
pub struct GaussiansBuffer<G: GaussianPod>(wgpu::Buffer, std::marker::PhantomData<G>);

impl<G: GaussianPod> GaussiansBuffer<G> {
    /// The default [`wgpu::BufferUsages`] for the Gaussians buffer.
    pub const DEFAULT_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_truncate(
        wgpu::BufferUsages::STORAGE.bits() | wgpu::BufferUsages::COPY_DST.bits(),
    );

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

    /// Create a new Gaussians buffer with the specified size with [`wgpu::BufferUsages`].
    pub fn new_with_usage(
        device: &wgpu::Device,
        gaussians: &[Gaussian],
        usage: wgpu::BufferUsages,
    ) -> Self {
        Self::new_with_pods_and_usage(
            device,
            gaussians
                .iter()
                .map(G::from_gaussian)
                .collect::<Vec<_>>()
                .as_slice(),
            usage,
        )
    }

    /// Create a new Gaussians buffer with [`GaussianPod`].
    pub fn new_with_pods(device: &wgpu::Device, gaussians: &[G]) -> Self {
        Self::new_with_pods_and_usage(device, gaussians, Self::DEFAULT_USAGE)
    }

    /// Create a new Gaussians buffer with [`GaussianPod`] and the specified size with
    /// [`wgpu::BufferUsages`].
    pub fn new_with_pods_and_usage(
        device: &wgpu::Device,
        gaussians: &[G],
        usage: wgpu::BufferUsages,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussians Buffer"),
            contents: bytemuck::cast_slice(gaussians),
            usage,
        });

        Self(buffer, std::marker::PhantomData)
    }

    /// Create a new Gaussians buffer with the specified size.
    pub fn new_empty(device: &wgpu::Device, len: usize) -> Self {
        Self::new_empty_with_usage(device, len, Self::DEFAULT_USAGE)
    }

    /// Create a new Gaussians buffer with the specified size with [`wgpu::BufferUsages`].
    pub fn new_empty_with_usage(
        device: &wgpu::Device,
        len: usize,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussians Buffer"),
            size: (len * std::mem::size_of::<G>()) as u64,
            usage,
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

    /// Download the buffer data into a vector of [`Gaussian`].
    pub async fn download_gaussians(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Gaussian>, crate::Error> {
        self.download::<G>(device, queue)
            .await
            .map(|pods| pods.into_iter().map(Into::into).collect::<Vec<_>>())
    }
}

impl<G: GaussianPod> BufferWrapper for GaussiansBuffer<G> {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

/// The Gaussian POD trait.
pub trait GaussianPod:
    for<'a> From<&'a Gaussian>
    + Into<Gaussian>
    + Send
    + Sync
    + bytemuck::NoUninit
    + bytemuck::AnyBitPattern
{
    /// The SH configuration.
    type ShConfig: GaussianShConfig;

    /// The covariance 3D configuration.
    type Cov3dConfig: GaussianCov3dConfig;

    /// Convert from POD to Gaussian.
    fn into_gaussian(self) -> Gaussian {
        self.into()
    }

    /// Create a new Gaussian POD from the Gaussian.
    fn from_gaussian(gaussian: &Gaussian) -> Self {
        Self::from(gaussian)
    }

    /// Create the features for [`Wesl`](wesl::Wesl) compilation.
    fn features() -> [(&'static str, bool); 7] {
        [
            GaussianShSingleConfig::FEATURE,
            GaussianShHalfConfig::FEATURE,
            GaussianShNorm8Config::FEATURE,
            GaussianShNoneConfig::FEATURE,
            GaussianCov3dRotScaleConfig::FEATURE,
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

    /// Create the features for [`Wesl`](wesl::Wesl) compilation as a [`HashMap`].
    fn features_map() -> std::collections::HashMap<String, bool> {
        Self::features()
            .iter()
            .map(|(name, enabled)| (name.to_string(), *enabled))
            .collect()
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
                    let cov3d = <[< GaussianCov3d $cov3d Config >]>::from_rot_scale(
                        gaussian.rot,
                        gaussian.scale,
                    );

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

            impl From<[< GaussianPodWith Sh $sh Cov3d $cov3d Configs >]> for Gaussian {
                fn from(pod: [< GaussianPodWith Sh $sh Cov3d $cov3d Configs >]) -> Self {
                    // Position
                    let pos = pod.pos;

                    // Spherical harmonics
                    let sh = [< GaussianSh $sh Config >]::to_sh(&pod.sh);

                    // Color
                    let color = pod.color;

                    // Rotation
                    let (rot, scale) = <[< GaussianCov3d $cov3d Config >]>::to_rot_scale(&pod.cov3d);

                    Self {
                        rot,
                        pos,
                        color,
                        sh,
                        scale,
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

gaussian_pod!(sh = Single, cov3d = RotScale, padding_size = 0);
gaussian_pod!(sh = Single, cov3d = Single, padding_size = 1);
gaussian_pod!(sh = Single, cov3d = Half, padding_size = 0);
gaussian_pod!(sh = Half, cov3d = RotScale, padding_size = 2);
gaussian_pod!(sh = Half, cov3d = Single, padding_size = 3);
gaussian_pod!(sh = Half, cov3d = Half, padding_size = 2);
gaussian_pod!(sh = Norm8, cov3d = RotScale, padding_size = 0);
gaussian_pod!(sh = Norm8, cov3d = Single, padding_size = 1);
gaussian_pod!(sh = Norm8, cov3d = Half, padding_size = 0);
gaussian_pod!(sh = None, cov3d = RotScale, padding_size = 1);
gaussian_pod!(sh = None, cov3d = Single, padding_size = 2);
gaussian_pod!(sh = None, cov3d = Half, padding_size = 1);

use glam::*;

use wgpu::util::DeviceExt;

use crate::{
    BufferWrapper, DownloadBufferError, DownloadableBufferWrapper, Gaussian, GaussianCov3dConfig,
    GaussianCov3dHalfConfig, GaussianCov3dRotScaleConfig, GaussianCov3dSingleConfig,
    GaussianShConfig, GaussianShHalfConfig, GaussianShNoneConfig, GaussianShNorm8Config,
    GaussianShSingleConfig, GaussiansBufferTryFromBufferError, GaussiansBufferUpdateError,
    GaussiansBufferUpdateRangeError,
};

/// The Gaussians storage buffer.
///
/// This buffer holds an array of Gaussians represented by the specified [`GaussianPod`].
#[derive(Debug, Clone)]
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
        Self::new_with_pods_and_usage(device, gaussians, Self::DEFAULT_USAGES)
    }

    /// Create a new Gaussians buffer with [`GaussianPod`] and the specified size and
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
        Self::new_empty_with_usage(device, len, Self::DEFAULT_USAGES)
    }

    /// Create a new Gaussians buffer with the specified size and [`wgpu::BufferUsages`].
    pub fn new_empty_with_usage(
        device: &wgpu::Device,
        len: usize,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussians Buffer"),
            size: (len * std::mem::size_of::<G>()) as wgpu::BufferAddress,
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
    ///
    /// `gaussians` should have the same number of Gaussians as the buffer.
    pub fn update(
        &self,
        queue: &wgpu::Queue,
        gaussians: &[Gaussian],
    ) -> Result<(), GaussiansBufferUpdateError> {
        if gaussians.len() != self.len() {
            return Err(GaussiansBufferUpdateError::CountMismatch {
                count: gaussians.len(),
                expected_count: self.len(),
            });
        }

        self.update_with_pod(
            queue,
            gaussians
                .iter()
                .map(G::from_gaussian)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    /// Update the buffer with [`GaussianPod`].
    ///
    /// `pods` should have the same number of Gaussians as the buffer.
    pub fn update_with_pod(
        &self,
        queue: &wgpu::Queue,
        pods: &[G],
    ) -> Result<(), GaussiansBufferUpdateError> {
        if pods.len() != self.len() {
            return Err(GaussiansBufferUpdateError::CountMismatch {
                count: pods.len(),
                expected_count: self.len(),
            });
        }

        queue.write_buffer(&self.0, 0, bytemuck::cast_slice(pods));

        Ok(())
    }

    /// Update a range of the buffer.
    ///
    /// `gaussians` should fit in the buffer starting from `start`.
    pub fn update_range(
        &self,
        queue: &wgpu::Queue,
        start: usize,
        gaussians: &[Gaussian],
    ) -> Result<(), GaussiansBufferUpdateRangeError> {
        if start + gaussians.len() > self.len() {
            return Err(GaussiansBufferUpdateRangeError::CountMismatch {
                count: gaussians.len(),
                start,
                expected_count: self.len(),
            });
        }

        self.update_range_with_pod(
            queue,
            start,
            gaussians
                .iter()
                .map(G::from_gaussian)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    /// Update a range of the buffer with [`GaussianPod`].
    ///
    /// `pods` should fit in the buffer starting from `start`.
    pub fn update_range_with_pod(
        &self,
        queue: &wgpu::Queue,
        start: usize,
        pods: &[G],
    ) -> Result<(), GaussiansBufferUpdateRangeError> {
        if start + pods.len() > self.len() {
            return Err(GaussiansBufferUpdateRangeError::CountMismatch {
                count: pods.len(),
                start,
                expected_count: self.len(),
            });
        }

        queue.write_buffer(
            &self.0,
            (start * std::mem::size_of::<G>()) as wgpu::BufferAddress,
            bytemuck::cast_slice(pods),
        );

        Ok(())
    }

    /// Download the buffer data into a [`Vec`] of [`Gaussian`].
    pub async fn download_gaussians(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Gaussian>, DownloadBufferError> {
        self.download::<G>(device, queue)
            .await
            .map(|pods| pods.into_iter().map(Into::into).collect::<Vec<_>>())
    }
}

impl<G: GaussianPod> BufferWrapper for GaussiansBuffer<G> {
    const DEFAULT_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_retain(
        wgpu::BufferUsages::STORAGE.bits() | wgpu::BufferUsages::COPY_DST.bits(),
    );

    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl<G: GaussianPod> From<GaussiansBuffer<G>> for wgpu::Buffer {
    fn from(wrapper: GaussiansBuffer<G>) -> Self {
        wrapper.0
    }
}

impl<G: GaussianPod> TryFrom<wgpu::Buffer> for GaussiansBuffer<G> {
    type Error = GaussiansBufferTryFromBufferError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        if buffer.size() % std::mem::size_of::<G>() as wgpu::BufferAddress != 0 {
            return Err(GaussiansBufferTryFromBufferError::BufferSizeNotMultiple {
                buffer_size: buffer.size(),
                expected_multiple_size: std::mem::size_of::<G>() as wgpu::BufferAddress,
            });
        }

        Ok(Self(buffer, std::marker::PhantomData))
    }
}

/// The Gaussian POD trait.
///
/// The number of configurations for this is the combination of all the [`GaussianShConfig`]
/// and [`GaussianCov3dConfig`].
///
/// You can use the corresponding config by using the name in the following format:
/// `GaussianPodWithSh{ShConfig}Cov3d{Cov3dConfig}Configs`, e.g.
/// [`GaussianPodWithShSingleCov3dRotScaleConfigs`].
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
    ///
    /// You may want to use [`GaussianPod::wesl_features`] most of the time instead.
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

    /// Create the features for [`Wesl`](wesl::Wesl) compilation as a [`wesl::Features`].
    fn wesl_features() -> wesl::Features {
        wesl::Features {
            flags: Self::features()
                .iter()
                .map(|(name, enabled)| (name.to_string(), (*enabled).into()))
                .collect(),
            ..Default::default()
        }
    }
}

/// Macro to create the POD representation of Gaussian given the configurations.
macro_rules! gaussian_pod {
    (sh = $sh:ident, cov3d = $cov3d:ident, padding_size = $padding:expr) => {
        paste::paste! {
            #[repr(C)]
            #[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
            pub struct [< GaussianPodWith Sh $sh Cov3d $cov3d Configs >] {
                pub pos: Vec3,
                pub color: U8Vec4,
                pub sh: <[< GaussianSh $sh Config >] as GaussianShConfig>::Field,
                pub cov3d: <[< GaussianCov3d $cov3d Config >] as GaussianCov3dConfig>::Field,
                pub padding: [f32; $padding],
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
                        padding: [0.0; $padding],
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

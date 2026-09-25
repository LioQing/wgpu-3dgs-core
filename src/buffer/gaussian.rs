use glam::*;

use wgpu::util::DeviceExt;

use crate::{
    BufferWrapper, DownloadBufferError, Gaussian, GaussianCov3dConfig, GaussianCov3dHalfConfig,
    GaussianCov3dRotScaleConfig, GaussianCov3dSingleConfig, GaussianShConfig, GaussianShHalfConfig,
    GaussianShNoneConfig, GaussianShNorm8Config, GaussianShSingleConfig,
    GaussiansBufferTryFromBufferError, GaussiansBufferUpdateError, GaussiansBufferUpdateRangeError,
    IterGaussian,
};

/// The Gaussians storage buffer.
///
/// This buffer holds an array of Gaussians represented by the specified [`GaussianPod`].
#[derive(Debug, Clone)]
pub struct GaussiansBuffer<G: GaussianPod>(wgpu::Buffer, std::marker::PhantomData<G>);

impl<G: GaussianPod> GaussiansBuffer<G> {
    /// Create a new Gaussians buffer.
    pub fn new(device: &wgpu::Device, gaussians: &impl IterGaussian) -> Self {
        Self::new_with_pods(
            device,
            gaussians
                .iter_gaussian()
                .map(|g| G::from_gaussian(&g))
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    /// Create a new Gaussians buffer with the specified size with [`wgpu::BufferUsages`].
    pub fn new_with_usage(
        device: &wgpu::Device,
        gaussians: &impl IterGaussian,
        usage: wgpu::BufferUsages,
    ) -> Self {
        Self::new_with_pods_and_usage(
            device,
            gaussians
                .iter_gaussian()
                .map(|g| G::from_gaussian(&g))
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
        gaussians: &impl IterGaussian,
    ) -> Result<(), GaussiansBufferUpdateError> {
        self.update_with_pod(
            queue,
            gaussians
                .iter_gaussian()
                .map(|g| G::from_gaussian(&g))
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
        if !buffer
            .size()
            .is_multiple_of(std::mem::size_of::<G>() as wgpu::BufferAddress)
        {
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
/// Use [`PackedGaussian`] with an SH and covariance config to select a layout, e.g.
/// `PackedGaussian<ShHalf, CovHalf>`.
pub trait GaussianPod:
    for<'a> From<&'a Gaussian>
    + Into<Gaussian>
    + Send
    + Sync
    + std::fmt::Debug
    + Clone
    + Copy
    + PartialEq
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

mod sealed {
    /// This trait exists to avoid other crates from implementing
    /// [`GaussianPodLayout`](super::GaussianPodLayout).
    pub trait Sealed {}
}

/// Supported combinations of SH and covariance encoding, with explicit GPU stride padding.
///
/// Only the combinations implemented by this crate can be used as [`PackedGaussian`] layouts.
pub trait GaussianPodLayout: sealed::Sealed {
    type Padding: bytemuck::Pod + bytemuck::Zeroable + std::fmt::Debug + Copy + PartialEq;
}

/// A GPU-ready Gaussian selected by its SH and covariance encodings.
///
/// For example, `PackedGaussian<ShHalf, CovHalf>`. The padding is part of the
/// storage-buffer stride and must match the WESL `Gaussian` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackedGaussian<Sh: GaussianShConfig, Cov: GaussianCov3dConfig>
where
    (Sh, Cov): GaussianPodLayout,
{
    pub pos: Vec3,
    pub color: U8Vec4,
    pub sh: Sh::Field,
    pub cov3d: Cov::Field,
    pub padding: <(Sh, Cov) as GaussianPodLayout>::Padding,
}

// SAFETY: Every field is Zeroable. Only the sealed combinations below have a layout.
unsafe impl<Sh, Cov> bytemuck::Zeroable for PackedGaussian<Sh, Cov>
where
    Sh: GaussianShConfig + Copy,
    Cov: GaussianCov3dConfig + Copy,
    (Sh, Cov): GaussianPodLayout,
{
}

// SAFETY: The fields are Pod, and the size/offset assertions for every sealed
// combination below guarantee that repr(C) introduces no implicit padding.
unsafe impl<Sh, Cov> bytemuck::Pod for PackedGaussian<Sh, Cov>
where
    Sh: GaussianShConfig + Copy + 'static,
    Cov: GaussianCov3dConfig + Copy + 'static,
    (Sh, Cov): GaussianPodLayout,
{
}

impl<Sh, Cov> From<&Gaussian> for PackedGaussian<Sh, Cov>
where
    Sh: GaussianShConfig,
    Cov: GaussianCov3dConfig,
    (Sh, Cov): GaussianPodLayout,
{
    fn from(gaussian: &Gaussian) -> Self {
        Self {
            pos: gaussian.pos,
            color: (gaussian.color * 255.0)
                .round()
                .clamp(Vec4::ZERO, Vec4::splat(255.0))
                .as_u8vec4(),
            sh: Sh::from_sh(&gaussian.sh),
            cov3d: Cov::from_rot_scale(gaussian.rot, gaussian.scale),
            padding: bytemuck::Zeroable::zeroed(),
        }
    }
}

impl<Sh, Cov> From<PackedGaussian<Sh, Cov>> for Gaussian
where
    Sh: GaussianShConfig,
    Cov: GaussianCov3dConfig,
    (Sh, Cov): GaussianPodLayout,
{
    fn from(pod: PackedGaussian<Sh, Cov>) -> Self {
        let (rot, scale) = Cov::to_rot_scale(&pod.cov3d);
        Self {
            rot,
            pos: pod.pos,
            color: pod.color.as_vec4() / 255.0,
            sh: Sh::to_sh(&pod.sh),
            scale,
        }
    }
}

impl<Sh, Cov> GaussianPod for PackedGaussian<Sh, Cov>
where
    Sh: GaussianShConfig + std::fmt::Debug + Copy + PartialEq + Send + Sync + 'static,
    Cov: GaussianCov3dConfig + std::fmt::Debug + Copy + PartialEq + Send + Sync + 'static,
    Sh::Field: std::fmt::Debug + PartialEq + Send + Sync,
    Cov::Field: std::fmt::Debug + PartialEq + Send + Sync,
    (Sh, Cov): GaussianPodLayout,
    <(Sh, Cov) as GaussianPodLayout>::Padding: Send + Sync,
{
    type ShConfig = Sh;
    type Cov3dConfig = Cov;
}

// Preserve the old public names as aliases, only one generic POD definition is needed.
// TODO: Remove this old name in version 0.10.
macro_rules! gaussian_pod_layout {
    (sh = $sh:ident, cov3d = $cov3d:ident, padding_size = $padding:expr) => {
        paste::paste! {
            impl sealed::Sealed for ([< GaussianSh $sh Config >], [< GaussianCov3d $cov3d Config >]) {}

            impl GaussianPodLayout for ([< GaussianSh $sh Config >], [< GaussianCov3d $cov3d Config >]) {
                type Padding = [f32; $padding];
            }

            #[doc = "Compatibility alias for a `PackedGaussian` layout."]
            #[deprecated(note = "Use `PackedGaussian<Sh..., Cov...>` with the corresponding configs instead. This will be removed in 0.10.")]
            pub type [< GaussianPodWithSh $sh Cov3d $cov3d Configs >] =
                PackedGaussian<[< GaussianSh $sh Config >], [< GaussianCov3d $cov3d Config >]>;

            const _: () = {
                type G = PackedGaussian<[< GaussianSh $sh Config >], [< GaussianCov3d $cov3d Config >]>;
                assert!(std::mem::offset_of!(G, pos) == 0);
                assert!(std::mem::offset_of!(G, color) == 12);
                assert!(std::mem::offset_of!(G, sh) == 16);
                assert!(std::mem::offset_of!(G, cov3d) == 16 + std::mem::size_of::<<[< GaussianSh $sh Config >] as GaussianShConfig>::Field>());
                assert!(std::mem::offset_of!(G, padding) == std::mem::offset_of!(G, cov3d) + std::mem::size_of::<<[< GaussianCov3d $cov3d Config >] as GaussianCov3dConfig>::Field>());
                assert!(std::mem::size_of::<G>() == std::mem::offset_of!(G, padding) + $padding * 4);
                assert!(std::mem::size_of::<G>() % 16 == 0);
            };
        }
    };
}

gaussian_pod_layout!(sh = Single, cov3d = RotScale, padding_size = 0);
gaussian_pod_layout!(sh = Single, cov3d = Single, padding_size = 1);
gaussian_pod_layout!(sh = Single, cov3d = Half, padding_size = 0);
gaussian_pod_layout!(sh = Half, cov3d = RotScale, padding_size = 2);
gaussian_pod_layout!(sh = Half, cov3d = Single, padding_size = 3);
gaussian_pod_layout!(sh = Half, cov3d = Half, padding_size = 2);
gaussian_pod_layout!(sh = Norm8, cov3d = RotScale, padding_size = 1);
gaussian_pod_layout!(sh = Norm8, cov3d = Single, padding_size = 2);
gaussian_pod_layout!(sh = Norm8, cov3d = Half, padding_size = 1);
gaussian_pod_layout!(sh = None, cov3d = RotScale, padding_size = 1);
gaussian_pod_layout!(sh = None, cov3d = Single, padding_size = 2);
gaussian_pod_layout!(sh = None, cov3d = Half, padding_size = 1);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CovHalf, CovRotScale, CovSingle, ShHalf, ShNone, ShNorm8, ShSingle};

    macro_rules! test_pod_from_gaussian {
        ($name:ident, $pod_type:ty, true) => {
            paste::paste! {
                #[test]
                #[should_panic]
                fn [<test_ $name _into_gaussian_should_panic>]() {
                    let pod = <$pod_type>::from_gaussian(&Gaussian {
                        rot: Quat::from_xyzw(0.0, 0.0, 0.0, 1.0),
                        pos: Vec3::new(1.0, 2.0, 3.0),
                        color: Vec4::new(255.0, 128.0, 64.0, 32.0) / 255.0,
                        sh: [Vec3::new(0.1, 0.2, 0.3); 15],
                        scale: Vec3::new(1.0, 2.0, 3.0),
                    });

                    pod.into_gaussian();
                }
            }
        };
        ($name:ident, $pod_type:ty, false) => {
            paste::paste! {
                #[test]
                fn [<test_ $name _into_gaussian_should_equal_original_pod>]() {
                    let pod = <$pod_type>::from_gaussian(&Gaussian {
                        rot: Quat::from_xyzw(0.0, 0.0, 0.0, 1.0),
                        pos: Vec3::new(1.0, 2.0, 3.0),
                        color: Vec4::new(255.0, 128.0, 64.0, 32.0) / 255.0,
                        sh: [Vec3::new(0.1, 0.2, 0.3); 15],
                        scale: Vec3::new(1.0, 2.0, 3.0),
                    });

                    let gaussian = pod.into_gaussian();

                    assert_eq!(pod.pos, gaussian.pos);
                    assert_eq!(pod.color.as_vec4() / 255.0, gaussian.color);
                    assert_eq!(
                        pod.sh,
                        <$pod_type as GaussianPod>::ShConfig::from_sh(&gaussian.sh),
                    );
                    assert_eq!(
                        pod.cov3d,
                        <$pod_type as GaussianPod>::Cov3dConfig::from_rot_scale(
                            gaussian.rot,
                            gaussian.scale
                        ),
                    );
                }
            }
        };
    }

    macro_rules! test_pod {
        ($name:ident, $pod_type:ty, $when_into_gaussian_should_panic:expr) => {
            paste::paste! {
                #[test]
                fn [<test_ $name _from_gaussian_should_equal_original_gaussian>]() {
                    let gaussian = Gaussian {
                        rot: Quat::from_xyzw(0.0, 0.0, 0.0, 1.0),
                        pos: Vec3::new(1.0, 2.0, 3.0),
                        color: Vec4::new(255.0, 128.0, 64.0, 32.0) / 255.0,
                        sh: [Vec3::new(0.1, 0.2, 0.3); 15],
                        scale: Vec3::new(1.0, 2.0, 3.0),
                    };

                    let pod = <$pod_type>::from_gaussian(&gaussian);

                    assert_eq!(gaussian.pos, pod.pos);
                    assert_eq!(gaussian.color, pod.color.as_vec4() / 255.0);
                    assert_eq!(
                        <$pod_type as GaussianPod>::ShConfig::from_sh(&gaussian.sh),
                        pod.sh,
                    );
                    assert_eq!(
                        <$pod_type as GaussianPod>::Cov3dConfig::from_rot_scale(
                            gaussian.rot,
                            gaussian.scale
                        ),
                        pod.cov3d,
                    );
                }

                test_pod_from_gaussian!($name, $pod_type, $when_into_gaussian_should_panic);

                #[test]
                fn [<test_ $name _color_should_be_clamped_and_quantized>]() {
                    let gaussian = Gaussian {
                        rot: Quat::IDENTITY,
                        pos: Vec3::ZERO,
                        color: Vec4::new(-0.1, 1.1, 0.5, 0.1234567),
                        sh: [Vec3::ZERO; 15],
                        scale: Vec3::ONE,
                    };

                    let pod = <$pod_type>::from_gaussian(&gaussian);

                    assert_eq!(pod.color, U8Vec4::new(0, 255, 128, 31));
                }

                #[test]
                fn [<test_ $name _features_should_be_correct>]() {
                    let features = <$pod_type as GaussianPod>::features();

                    for (name, enabled) in features {
                        if name == <$pod_type as GaussianPod>::ShConfig::FEATURE
                            || name == <$pod_type as GaussianPod>::Cov3dConfig::FEATURE
                        {
                            assert!(enabled, "Feature {name} should be enabled");
                        } else {
                            assert!(!enabled, "Feature {name} should be disabled");
                        }
                    }
                }

                #[test]
                fn [<test_ $name _wesl_features_should_be_correct>]() {
                    let wesl_features = <$pod_type as GaussianPod>::wesl_features();
                    let features = <$pod_type as GaussianPod>::features();

                    for (name, enabled) in features {
                        let wesl_enabled = wesl_features
                            .flags
                            .get(name)
                            .map(|v| *v == wesl::Feature::Enable)
                            .unwrap_or(false);

                        assert_eq!(
                            enabled, wesl_enabled,
                            "Feature {name} should be {}",
                            if enabled { "enabled" } else { "disabled" }
                        );
                    }
                }
            }
        };
    }

    #[rustfmt::skip]
    mod pod {
        use super::*;

        test_pod!(single_rotscale, PackedGaussian<ShSingle, CovRotScale>, false);
        test_pod!(single_single, PackedGaussian<ShSingle, CovSingle>, true);
        test_pod!(single_half, PackedGaussian<ShSingle, CovHalf>, true);
        test_pod!(half_rotscale, PackedGaussian<ShHalf, CovRotScale>, false);
        test_pod!(test_half_single, PackedGaussian<ShHalf, CovSingle>, true);
        test_pod!(test_half_half, PackedGaussian<ShHalf, CovHalf>, true);
        test_pod!(norm8_rotscale, PackedGaussian<ShNorm8, CovRotScale>, false);
        test_pod!(norm8_single, PackedGaussian<ShNorm8, CovSingle>, true);
        test_pod!(norm8_half, PackedGaussian<ShNorm8, CovHalf>, true);
        test_pod!(none_rotscale, PackedGaussian<ShNone, CovRotScale>, true);
        test_pod!(none_single, PackedGaussian<ShNone, CovSingle>, true);
        test_pod!(none_half, PackedGaussian<ShNone, CovHalf>, true);
    }

    #[test]
    #[allow(deprecated)] // Verify the old public name still resolves to the generic layout.
    fn test_generic_pod_strides_match_existing_shader_layouts() {
        assert_eq!(
            std::mem::size_of::<PackedGaussian<ShSingle, CovRotScale>>(),
            224
        );
        assert_eq!(
            std::mem::size_of::<PackedGaussian<ShSingle, CovSingle>>(),
            224
        );
        assert_eq!(
            std::mem::size_of::<PackedGaussian<ShSingle, CovHalf>>(),
            208
        );
        assert_eq!(
            std::mem::size_of::<PackedGaussian<ShHalf, CovRotScale>>(),
            144
        );
        assert_eq!(
            std::mem::size_of::<PackedGaussian<ShHalf, CovSingle>>(),
            144
        );
        assert_eq!(std::mem::size_of::<PackedGaussian<ShHalf, CovHalf>>(), 128);
        assert_eq!(
            std::mem::size_of::<PackedGaussian<ShNorm8, CovRotScale>>(),
            96
        );
        assert_eq!(
            std::mem::size_of::<PackedGaussian<ShNorm8, CovSingle>>(),
            96
        );
        assert_eq!(std::mem::size_of::<PackedGaussian<ShNorm8, CovHalf>>(), 80);
        assert_eq!(
            std::mem::size_of::<PackedGaussian<ShNone, CovRotScale>>(),
            48
        );
        assert_eq!(std::mem::size_of::<PackedGaussian<ShNone, CovSingle>>(), 48);
        assert_eq!(std::mem::size_of::<PackedGaussian<ShNone, CovHalf>>(), 32);
        let pod = PackedGaussian::<ShHalf, CovHalf>::from_gaussian(&Gaussian {
            rot: Quat::IDENTITY,
            pos: Vec3::ZERO,
            color: Vec4::ONE,
            sh: [Vec3::ZERO; 15],
            scale: Vec3::ONE,
        });
        let legacy: GaussianPodWithShHalfCov3dHalfConfigs = pod;
        assert_eq!(legacy, pod);
    }
}

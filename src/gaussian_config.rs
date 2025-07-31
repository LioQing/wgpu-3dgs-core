use glam::*;
use half::f16;

/// The spherical harmonics configuration of Gaussian.
pub trait GaussianShConfig {
    /// The feature name of the configuration.
    ///
    /// Must match the feature name in the shader.
    const FEATURE: &'static str;

    /// The [`GaussianPod`](crate::GaussianPod) field type.
    type Field: bytemuck::Pod + bytemuck::Zeroable;

    /// Create from [`Gaussian::sh`](crate::Gaussian::sh).
    fn from_sh(sh: &[Vec3; 15]) -> Self::Field;
}

/// The single precision SH configuration of Gaussian.
pub struct GaussianShSingleConfig;

impl GaussianShConfig for GaussianShSingleConfig {
    const FEATURE: &'static str = "sh_single";

    type Field = [Vec3; 15];

    fn from_sh(sh: &[Vec3; 15]) -> Self::Field {
        *sh
    }
}

/// The half precision SH configuration of Gaussian.
pub struct GaussianShHalfConfig;

impl GaussianShConfig for GaussianShHalfConfig {
    const FEATURE: &'static str = "sh_half";

    type Field = [f16; 3 * 15 + 1];

    fn from_sh(sh: &[Vec3; 15]) -> Self::Field {
        sh.iter()
            .flat_map(|sh| sh.to_array())
            .map(f16::from_f32)
            .chain(std::iter::once(f16::from_f32(0.0)))
            .collect::<Vec<_>>()
            .try_into()
            .expect("SH half")
    }
}

/// The min max 8 bit normalized SH configuration of Gaussian.
pub struct GaussianShNorm8Config;

impl GaussianShConfig for GaussianShNorm8Config {
    const FEATURE: &'static str = "sh_norm8";

    type Field = [u8; 4 + (3 * 15 + 3)]; // ([f16; 2], [U8Vec4; (3 * 15 + 3) / 4])

    fn from_sh(sh: &[Vec3; 15]) -> Self::Field {
        let mut sh_pod = [0; 4 + (3 * 15 + 3)];

        let sh = sh.iter().flat_map(|sh| sh.to_array()).collect::<Vec<_>>();
        let (min, max) = sh.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| {
            (min.min(x), max.max(x))
        });

        sh_pod[0..2].copy_from_slice(&f16::from_f32(min).to_ne_bytes());
        sh_pod[2..4].copy_from_slice(&f16::from_f32(max).to_ne_bytes());
        sh_pod[4..].copy_from_slice(
            &sh.iter()
                .map(|&x| ((x - min) / (max - min) * 255.0).round() as u8)
                .chain(std::iter::repeat_n(0, 3))
                .collect::<Vec<_>>(),
        );

        sh_pod
    }
}

/// The none SH configuration of Gaussian.
pub struct GaussianShNoneConfig;

impl GaussianShConfig for GaussianShNoneConfig {
    const FEATURE: &'static str = "sh_none";

    type Field = ();

    fn from_sh(_sh: &[Vec3; 15]) -> Self::Field {}
}

/// The covariance 3D configuration of Gaussian.
pub trait GaussianCov3dConfig {
    /// The name of the configuration.
    ///
    /// Must match the name in the shader.
    const FEATURE: &'static str;

    /// The [`GaussianPod`](crate::GaussianPod) field type.
    type Field: bytemuck::Pod + bytemuck::Zeroable;

    /// Create from [`Gaussian::rot`](crate::Gaussian::rot) and [`Gaussian::scale`](crate::Gaussian::scale).
    fn from_rot_scale(rot: Quat, scale: Vec3) -> Self::Field;
}

/// The unconverted rotation and scale covariance 3D configuration of Gaussian.
pub struct GaussianCov3dRotScaleConfig;

impl GaussianCov3dConfig for GaussianCov3dRotScaleConfig {
    const FEATURE: &'static str = "cov3d_rot_scale";

    type Field = [f32; 7]; // (rot: [f32; 4], scale: [f32; 3])

    fn from_rot_scale(rot: Quat, scale: Vec3) -> Self::Field {
        [rot.x, rot.y, rot.z, rot.w, scale.x, scale.y, scale.z]
    }
}

/// The single precision covariance 3D configuration of Gaussian.
pub struct GaussianCov3dSingleConfig;

impl GaussianCov3dConfig for GaussianCov3dSingleConfig {
    const FEATURE: &'static str = "cov3d_single";

    type Field = [f32; 6];

    fn from_rot_scale(rot: Quat, scale: Vec3) -> Self::Field {
        let r = Mat3::from_quat(rot);
        let s = Mat3::from_diagonal(scale);
        let m = r * s;
        let sigma = m * m.transpose();

        [
            sigma.x_axis.x,
            sigma.x_axis.y,
            sigma.x_axis.z,
            sigma.y_axis.y,
            sigma.y_axis.z,
            sigma.z_axis.z,
        ]
    }
}

/// The half precision covariance 3D configuration of Gaussian.
pub struct GaussianCov3dHalfConfig;

impl GaussianCov3dConfig for GaussianCov3dHalfConfig {
    const FEATURE: &'static str = "cov3d_half";

    type Field = [f16; 6];

    fn from_rot_scale(rot: Quat, scale: Vec3) -> Self::Field {
        GaussianCov3dSingleConfig::from_rot_scale(rot, scale).map(f16::from_f32)
    }
}

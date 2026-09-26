use glam::*;
use half::f16;

/// The spherical harmonics configuration of Gaussian.
///
/// Currently, there are four configurations:
/// - Single precision [`ShSingle`]
///     - Format: 15 * [`Vec3`]
/// - Half precision [`ShHalf`]
///     - Format: (15 * 3 + 1) * [`struct@f16`]
/// - 8 bit normalized [`ShNorm8`]
///     - Format: (15 * 3 + 3) * [`prim@i8`]
/// - None [`ShNone`]
///    - Cannot be converted back to SH
pub trait GaussianShConfig {
    /// The feature name of the configuration.
    ///
    /// Must match the [`wesl::Feature`] name in the shader.
    const FEATURE: &'static str;

    /// The [`GaussianPod`](crate::GaussianPod) field type.
    type Field: bytemuck::Pod + bytemuck::Zeroable;

    /// Create from [`Gaussian::sh`](crate::Gaussian::sh).
    fn from_sh(sh: &[Vec3; 15]) -> Self::Field;

    /// Convert the field to [`Gaussian::sh`](crate::Gaussian::sh).
    fn to_sh(field: &Self::Field) -> [Vec3; 15];
}

/// Single-precision SH coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShSingle;

impl GaussianShConfig for ShSingle {
    const FEATURE: &'static str = "sh_single";

    type Field = [Vec3; 15];

    fn from_sh(sh: &[Vec3; 15]) -> Self::Field {
        *sh
    }

    fn to_sh(field: &Self::Field) -> [Vec3; 15] {
        *field
    }
}

/// Half-precision SH coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShHalf;

impl GaussianShConfig for ShHalf {
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

    fn to_sh(field: &Self::Field) -> [Vec3; 15] {
        field
            .as_chunks::<3>()
            .0
            .iter()
            .map(|chunk| {
                Vec3::new(
                    f16::to_f32(chunk[0]),
                    f16::to_f32(chunk[1]),
                    f16::to_f32(chunk[2]),
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("SH half")
    }
}

/// The 8 bit signed normalized SH configuration of Gaussian.
///
/// This is by the fact that SH coefficients are within \[-1, 1\].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShNorm8;

impl GaussianShConfig for ShNorm8 {
    const FEATURE: &'static str = "sh_norm8";

    type Field = [i8; 3 * 15 + 3];

    fn from_sh(sh: &[Vec3; 15]) -> Self::Field {
        sh.iter()
            .flat_map(|sh| sh.to_array())
            .map(|v| (v * 127.0).clamp(-127.0, 127.0) as i8)
            .chain(std::iter::repeat_n(0, 3))
            .collect::<Vec<_>>()
            .try_into()
            .expect("SH norm8")
    }

    fn to_sh(field: &Self::Field) -> [Vec3; 15] {
        field
            .as_chunks::<3>()
            .0
            .iter()
            .take(15)
            .map(|chunk| {
                Vec3::new(
                    ((chunk[0] as f32) / 127.0).max(-1.0),
                    ((chunk[1] as f32) / 127.0).max(-1.0),
                    ((chunk[2] as f32) / 127.0).max(-1.0),
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("SH norm8")
    }
}

/// The none SH configuration of Gaussian.
///
/// Calling [`GaussianShConfig::to_sh`] will panic on this config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShNone;

impl GaussianShConfig for ShNone {
    const FEATURE: &'static str = "sh_none";

    type Field = ();

    fn from_sh(_sh: &[Vec3; 15]) -> Self::Field {}

    fn to_sh(_field: &Self::Field) -> [Vec3; 15] {
        panic!("Cannot convert from SH None configuration")
    }
}

/// The covariance 3D configuration of Gaussian.
///
/// Currently, there are three configurations:
/// - Rotation and scale [`CovRotScale`]
///     - Format: [`Quat`] + [`Vec3`]
/// - Single precision [`CovSingle`]
///     - Format: 6 * [`prim@f32`]
///     - Cannot be converted back to rotation and scale
/// - Half precision [`CovHalf`]
///     - Format: 6 * [`struct@f16`]
///     - Cannot be converted back to rotation and scale
pub trait GaussianCov3dConfig {
    /// The name of the configuration.
    ///
    /// Must match the [`wesl::Feature`] name in the shader.
    const FEATURE: &'static str;

    /// The [`GaussianPod`](crate::GaussianPod) field type.
    type Field: bytemuck::Pod + bytemuck::Zeroable;

    /// Create from [`Gaussian::rot`](crate::Gaussian::rot) and [`Gaussian::scale`](crate::Gaussian::scale).
    fn from_rot_scale(rot: Quat, scale: Vec3) -> Self::Field;

    /// Convert the field to [`Gaussian::rot`](crate::Gaussian::rot) and [`Gaussian::scale`](crate::Gaussian::scale).
    fn to_rot_scale(field: &Self::Field) -> (Quat, Vec3);
}

/// The unconverted rotation and scale covariance 3D configuration of Gaussian.
///
/// Instead of storing the covariance matrix, this config stores the rotation and scale directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CovRotScale;

impl GaussianCov3dConfig for CovRotScale {
    const FEATURE: &'static str = "cov3d_rot_scale";

    type Field = [f32; 7]; // (rot: [f32; 4], scale: [f32; 3])

    fn from_rot_scale(rot: Quat, scale: Vec3) -> Self::Field {
        [rot.x, rot.y, rot.z, rot.w, scale.x, scale.y, scale.z]
    }

    fn to_rot_scale(field: &Self::Field) -> (Quat, Vec3) {
        (
            Quat::from_xyzw(field[0], field[1], field[2], field[3]),
            Vec3::new(field[4], field[5], field[6]),
        )
    }
}

/// The single precision covariance 3D configuration of Gaussian.
///
/// Calling [`GaussianCov3dConfig::to_rot_scale`] will panic on this config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CovSingle;

impl GaussianCov3dConfig for CovSingle {
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

    fn to_rot_scale(_field: &Self::Field) -> (Quat, Vec3) {
        panic!("Cannot convert from Cov3d Single configuration")
    }
}

/// The half precision covariance 3D configuration of Gaussian.
///
/// Calling [`GaussianCov3dConfig::to_rot_scale`] will panic on this config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CovHalf;

impl GaussianCov3dConfig for CovHalf {
    const FEATURE: &'static str = "cov3d_half";

    type Field = [f16; 6];

    fn from_rot_scale(rot: Quat, scale: Vec3) -> Self::Field {
        CovSingle::from_rot_scale(rot, scale).map(f16::from_f32)
    }

    fn to_rot_scale(_field: &Self::Field) -> (Quat, Vec3) {
        panic!("Cannot convert from Cov3d Half configuration")
    }
}

/// Compatibility alias for [`CovHalf`].
#[deprecated(note = "Use `CovHalf` instead. This will be removed in 0.10.")]
pub use CovHalf as GaussianCov3dHalfConfig;
/// Compatibility alias for [`CovRotScale`].
#[deprecated(note = "Use `CovRotScale` instead. This will be removed in 0.10.")]
pub use CovRotScale as GaussianCov3dRotScaleConfig;
/// Compatibility alias for [`CovSingle`].
#[deprecated(note = "Use `CovSingle` instead. This will be removed in 0.10.")]
pub use CovSingle as GaussianCov3dSingleConfig;
/// Compatibility alias for [`ShHalf`].
#[deprecated(note = "Use `ShHalf` instead. This will be removed in 0.10.")]
pub use ShHalf as GaussianShHalfConfig;
/// Compatibility alias for [`ShNone`].
#[deprecated(note = "Use `ShNone` instead. This will be removed in 0.10.")]
pub use ShNone as GaussianShNoneConfig;
/// Compatibility alias for [`ShNorm8`].
#[deprecated(note = "Use `ShNorm8` instead. This will be removed in 0.10.")]
pub use ShNorm8 as GaussianShNorm8Config;
/// Compatibility alias for [`ShSingle`].
#[deprecated(note = "Use `ShSingle` instead. This will be removed in 0.10.")]
pub use ShSingle as GaussianShSingleConfig;

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn legacy_names_remain_usable_as_types_and_values() {
        fn same_type<T>(_: T, _: T) {}

        same_type(ShSingle, GaussianShSingleConfig);
        same_type(ShHalf, GaussianShHalfConfig);
        same_type(ShNorm8, GaussianShNorm8Config);
        same_type(ShNone, GaussianShNoneConfig);
        same_type(CovRotScale, GaussianCov3dRotScaleConfig);
        same_type(CovSingle, GaussianCov3dSingleConfig);
        same_type(CovHalf, GaussianCov3dHalfConfig);
    }
}

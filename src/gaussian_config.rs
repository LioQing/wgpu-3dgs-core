use bytemuck::Zeroable;
use glam::*;
use half::f16;

use crate::{ShDegree, ShDegree0, ShDegree3};

/// The spherical harmonics configuration of Gaussian.
///
/// Select an encoding and a storage degree, e.g. `ShHalf<ShDegree4>`.
/// Packed half and normalized encodings are padded to complete 32-bit words.
pub trait GaussianShConfig {
    type Degree: ShDegree;

    /// The feature name of the configuration.
    ///
    /// Must match the [`wesl::Feature`] name in the shader.
    const FEATURE: &'static str;

    /// The [`GaussianPod`](crate::GaussianPod) field type.
    type Field: bytemuck::Pod + bytemuck::Zeroable;

    /// Create from [`Gaussian::sh`](crate::Gaussian::sh).
    fn from_sh(sh: &<Self::Degree as ShDegree>::Coefficients) -> Self::Field;

    /// Convert the field to [`Gaussian::sh`](crate::Gaussian::sh).
    fn to_sh(field: &Self::Field) -> <Self::Degree as ShDegree>::Coefficients;
}

/// Single-precision SH coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShSingle<D: ShDegree = ShDegree3>(std::marker::PhantomData<D>);

impl<D: ShDegree> GaussianShConfig for ShSingle<D> {
    type Degree = D;

    const FEATURE: &'static str = "sh_single";

    type Field = D::Coefficients;

    fn from_sh(sh: &D::Coefficients) -> Self::Field {
        *sh
    }

    fn to_sh(field: &Self::Field) -> D::Coefficients {
        *field
    }
}

/// Half-precision SH coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShHalf<D: ShDegree = ShDegree3>(std::marker::PhantomData<D>);

impl<D: ShDegree> GaussianShConfig for ShHalf<D> {
    type Degree = D;

    const FEATURE: &'static str = "sh_half";

    type Field = D::Half;

    fn from_sh(sh: &D::Coefficients) -> Self::Field {
        let mut field = Self::Field::zeroed();

        for (src, dst) in sh
            .as_ref()
            .iter()
            .flat_map(|v| v.to_array())
            .zip(field.as_mut())
        {
            *dst = f16::from_f32(src);
        }

        field
    }

    fn to_sh(field: &Self::Field) -> D::Coefficients {
        let mut sh = D::Coefficients::zeroed();

        for (src, dst) in field.as_ref().as_chunks::<3>().0.iter().zip(sh.as_mut()) {
            *dst = Vec3::new(src[0].to_f32(), src[1].to_f32(), src[2].to_f32());
        }

        sh
    }
}

/// The 8 bit signed normalized SH configuration of Gaussian.
///
/// Values outside \[-1, 1\] are clamped by this lossy encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShNorm8<D: ShDegree = ShDegree3>(std::marker::PhantomData<D>);

impl<D: ShDegree> GaussianShConfig for ShNorm8<D> {
    type Degree = D;

    const FEATURE: &'static str = "sh_norm8";

    type Field = D::Norm8;

    fn from_sh(sh: &D::Coefficients) -> Self::Field {
        let mut field = Self::Field::zeroed();

        for (src, dst) in sh
            .as_ref()
            .iter()
            .flat_map(|v| v.to_array())
            .zip(field.as_mut())
        {
            *dst = (src * 127.0).clamp(-127.0, 127.0) as i8;
        }

        field
    }

    fn to_sh(field: &Self::Field) -> D::Coefficients {
        let mut sh = D::Coefficients::zeroed();

        for (src, dst) in field.as_ref().as_chunks::<3>().0.iter().zip(sh.as_mut()) {
            *dst = Vec3::from_array([src[0], src[1], src[2]].map(|v| (v as f32 / 127.0).max(-1.0)));
        }

        sh
    }
}

/// A degree-zero layout, with no non-DC SH payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShNone;

impl GaussianShConfig for ShNone {
    type Degree = ShDegree0;

    const FEATURE: &'static str = "sh_none";

    type Field = ();

    fn from_sh(_sh: &[Vec3; 0]) -> Self::Field {}

    fn to_sh(_field: &Self::Field) -> [Vec3; 0] {
        []
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
    fn test_legacy_names_should_remain_usable_as_types_and_values() {
        fn same_type<T>(_: T, _: T) {}

        same_type(
            ShSingle::<ShDegree3>(std::marker::PhantomData),
            GaussianShSingleConfig(std::marker::PhantomData),
        );
        same_type(
            ShHalf::<ShDegree3>(std::marker::PhantomData),
            GaussianShHalfConfig(std::marker::PhantomData),
        );
        same_type(
            ShNorm8::<ShDegree3>(std::marker::PhantomData),
            GaussianShNorm8Config(std::marker::PhantomData),
        );
        same_type(ShNone, GaussianShNoneConfig);
        same_type(CovRotScale, GaussianCov3dRotScaleConfig);
        same_type(CovSingle, GaussianCov3dSingleConfig);
        same_type(CovHalf, GaussianCov3dHalfConfig);
    }
}

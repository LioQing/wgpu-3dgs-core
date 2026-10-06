use bytemuck::Zeroable;
use glam::*;

use crate::{
    GaussianShDegree, GaussianToSpzError, IterGaussian, PlyGaussian, PlyGaussiansFromIterError,
    ShDegree, ShDegree0, ShDegree1, ShDegree2, ShDegree3, ShDegree4, ShDegreeMismatchError,
    SpzGaussian, SpzGaussianPosition, SpzGaussianPositionRef, SpzGaussianRef, SpzGaussianRotation,
    SpzGaussianRotationRef, SpzGaussianSh, SpzGaussiansHeader, ToAnyGaussian,
};

/// The Gaussian.
///
/// This is an intermediate representation used by the CPU to convert to
/// [`GaussianPod`](crate::GaussianPod).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian<D: ShDegree = ShDegree3> {
    pub rot: Quat,
    pub pos: Vec3,
    pub color: Vec4,
    pub sh: D::Coefficients,
    pub scale: Vec3,
}

impl<D: ShDegree> Gaussian<D> {
    /// The storage degree, independent of the coefficient values.
    pub fn sh_degree(&self) -> GaussianShDegree {
        GaussianShDegree::new(D::DEGREE).unwrap()
    }

    /// The constant to convert from SH coefficient at degree 0 to linear color.
    pub const SH0_TO_LINEAR_FACTOR: f32 = 0.2820948;

    /// The constant to convert from SH coefficient at degree 0 to linear color in SPZ.
    pub const SPZ_SH0_TO_LINEAR_FACTOR: f32 = 0.15;

    /// Convert from PLY, rejecting a storage degree mismatch.
    pub fn from_ply(ply: &PlyGaussian) -> Result<Self, PlyGaussiansFromIterError> {
        if ply.sh.len() != D::COEFFICIENT_COUNT * 3 {
            return Err(PlyGaussiansFromIterError::ShCountMismatch {
                actual_count: ply.sh.len(),
                expected_count: D::COEFFICIENT_COUNT * 3,
            });
        }

        let pos = ply.pos;

        let rot = ply.rot.normalize();

        let scale = ply.scale.exp();

        let color = (ply.color * Self::SH0_TO_LINEAR_FACTOR + Vec3::splat(0.5))
            .extend(1.0 / (1.0 + (-ply.alpha).exp()));

        let sh_count = D::COEFFICIENT_COUNT;
        let mut sh = D::Coefficients::zeroed();

        for (i, dst) in sh.as_mut().iter_mut().enumerate() {
            *dst = Vec3::new(ply.sh[i], ply.sh[i + sh_count], ply.sh[i + 2 * sh_count]);
        }

        Ok(Self {
            rot,
            pos,
            color,
            sh,
            scale,
        })
    }

    /// Convert to PLY, preserving the storage degree.
    pub fn to_ply(&self) -> PlyGaussian {
        let pos = self.pos;

        let rot = self.rot;

        let scale = self.scale.map(|x| x.ln());

        let rgba = self.color;
        let color = (rgba.xyz() - Vec3::splat(0.5)) / Self::SH0_TO_LINEAR_FACTOR;

        let alpha = -(1.0 / rgba.w - 1.0).ln();

        let sh_count = D::COEFFICIENT_COUNT;
        let mut sh = vec![0.0; 3 * sh_count];

        for (i, src) in self.sh.as_ref().iter().enumerate() {
            sh[i] = src.x;
            sh[i + sh_count] = src.y;
            sh[i + 2 * sh_count] = src.z;
        }

        let normal = Vec3::Z;

        PlyGaussian {
            pos,
            normal,
            color,
            sh,
            alpha,
            scale,
            rot,
        }
    }

    const SPZ_COLOR_TO_LINEAR_FRAC_A_B: f32 =
        Self::SH0_TO_LINEAR_FACTOR / Self::SPZ_SH0_TO_LINEAR_FACTOR;
    const SPZ_COLOR_TO_LINEAR_C: f32 = (1.0 - Self::SPZ_COLOR_TO_LINEAR_FRAC_A_B) * 0.5;

    /// Convert from [`SpzGaussianRef`].
    pub fn from_spz(
        spz: SpzGaussianRef,
        header: &SpzGaussiansHeader,
    ) -> Result<Self, ShDegreeMismatchError> {
        for actual_degree in [header.sh_degree().get(), spz.sh.degree().get()] {
            if actual_degree != D::DEGREE {
                return Err(ShDegreeMismatchError {
                    actual_degree,
                    expected_degree: D::DEGREE,
                });
            }
        }

        let pos = match spz.position {
            SpzGaussianPositionRef::Float16(pos) => {
                // The Niantic SPZ format matches the `half` crate's f16 const conversion.
                let unpacked = pos.map(|c| half::f16::from_bits(c).to_f32_const());
                Vec3::from_array(unpacked)
            }
            SpzGaussianPositionRef::FixedPoint24(pos) => {
                let scale = 1.0 / (1 << header.fractional_bits()) as f32;
                let unpacked = pos.map(|c| {
                    let mut fixed32: i32 = c[0] as i32;
                    fixed32 |= (c[1] as i32) << 8;
                    fixed32 |= (c[2] as i32) << 16;
                    fixed32 |= if fixed32 & 0x800000 != 0 {
                        0xff000000u32 as i32
                    } else {
                        0
                    };
                    fixed32 as f32 * scale
                });
                Vec3::from_array(unpacked)
            }
        };

        let scale = Vec3::from_array(spz.scale.map(|c| c as f32 / 16.0 - 10.0)).exp();

        let rot = match spz.rotation {
            SpzGaussianRotationRef::QuatFirstThree(quat) => {
                let xyz = Vec3::from(quat.map(|c| c as f32 / 127.5 - 1.0));
                let w = (1.0 - xyz.length_squared()).max(0.0).sqrt();
                Quat::from_xyzw(xyz.x, xyz.y, xyz.z, w)
            }
            SpzGaussianRotationRef::QuatSmallestThree(quat) => {
                let mut comp: u32 = quat[0] as u32
                    | ((quat[1] as u32) << 8)
                    | ((quat[2] as u32) << 16)
                    | ((quat[3] as u32) << 24);

                const C_MASK: u32 = (1 << 9) - 1;

                let largest_index = (comp >> 30) as usize;
                let mut sum_squares = 0.0f32;
                let mut comps = std::array::from_fn(|i| {
                    if i == largest_index {
                        return 0.0;
                    }

                    let mag = comp & C_MASK;
                    let neg_bit = (comp >> 9) & 1;
                    comp >>= 10;

                    let value = std::f32::consts::FRAC_1_SQRT_2
                        * (mag as f32 / C_MASK as f32)
                        * if neg_bit != 0 { -1.0 } else { 1.0 };
                    sum_squares += value * value;

                    value
                });

                comps[largest_index] = (1.0 - sum_squares).max(0.0).sqrt();

                Quat::from_array(comps)
            }
        };

        let color = Vec3::from_array(spz.color.map(|c| {
            c as f32 / 255.0 * Self::SPZ_COLOR_TO_LINEAR_FRAC_A_B + Self::SPZ_COLOR_TO_LINEAR_C
        }))
        .extend(*spz.alpha as f32 / 255.0);

        let mut sh = D::Coefficients::zeroed();

        for (src, dst) in spz.sh.iter().zip(sh.as_mut().iter_mut()) {
            *dst = Vec3::from_array(src.map(|c| (c as f32 - 128.0) / 128.0));
        }

        Ok(Self {
            rot,
            pos,
            color,
            sh,
            scale,
        })
    }

    /// Convert to [`SpzGaussian`].
    ///
    /// User usually don't need to call this directly due to the overhead of constructing a
    /// valid [`SpzGaussiansHeader`]. Instead, use one of the following methods to convert a
    /// collection of [`Gaussian`] to [`SpzGaussians`](crate::SpzGaussians) properly:
    ///
    /// - [`SpzGaussians::from_gaussians`](crate::SpzGaussians::from_gaussians)
    /// - [`SpzGaussians::from_gaussians_with_options`](crate::SpzGaussians::from_gaussians_with_options)
    pub fn to_spz(
        &self,
        header: &SpzGaussiansHeader,
        options: &GaussianToSpzOptions,
    ) -> Result<SpzGaussian, GaussianToSpzError> {
        if header.sh_degree().get() != D::DEGREE {
            return Err(ShDegreeMismatchError {
                actual_degree: D::DEGREE,
                expected_degree: header.sh_degree().get(),
            }
            .into());
        }

        for (index, &bits) in options.sh_quantize_bits.iter().enumerate() {
            if bits > 8 {
                return Err(GaussianToSpzError::InvalidShQuantizeBits {
                    degree: index as u8 + 1,
                    bits,
                });
            }
        }

        let position = if header.uses_float16() {
            let packed = self
                .pos
                .to_array()
                .map(|c| half::f16::from_f32_const(c).to_bits());
            SpzGaussianPosition::Float16(packed)
        } else {
            let scale = (1 << header.fractional_bits()) as f32;
            let packed = self.pos.to_array().map(|c| {
                let fixed32 = (c * scale).round() as i32;
                [
                    (fixed32 & 0xff) as u8,
                    ((fixed32 >> 8) & 0xff) as u8,
                    ((fixed32 >> 16) & 0xff) as u8,
                ]
            });
            SpzGaussianPosition::FixedPoint24(packed)
        };

        let scale = self
            .scale
            .to_array()
            .map(|c| ((c.ln() + 10.0) * 16.0).round().clamp(0.0, 255.0) as u8);

        let rotation = if header.uses_quat_smallest_three() {
            let rot = self.rot.normalize().to_array();
            let largest_index = rot
                .into_iter()
                .map(f32::abs)
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("quaternion has at least one component")
                .0;

            const C_MASK: u32 = (1 << 9) - 1;

            let negate = (rot[largest_index] < 0.0) as u32;

            let mut comp = largest_index as u32;
            for (i, &value) in rot.iter().enumerate() {
                if i == largest_index {
                    continue;
                }

                let neg_bit = (value < 0.0) as u32 ^ negate;
                let mag = (C_MASK as f32 * (value.abs() * std::f32::consts::SQRT_2) + 0.5)
                    .clamp(0.0, C_MASK as f32 - 1.0) as u32;
                comp = (comp << 10) | (neg_bit << 9) | mag;
            }

            SpzGaussianRotation::QuatSmallestThree([
                (comp & 0xff) as u8,
                ((comp >> 8) & 0xff) as u8,
                ((comp >> 16) & 0xff) as u8,
                ((comp >> 24) & 0xff) as u8,
            ])
        } else {
            let rot = self.rot.normalize();
            let rot = if rot.w < 0.0 { -rot } else { rot };
            let packed = rot
                .xyz()
                .to_array()
                .map(|c| ((c + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8);
            SpzGaussianRotation::QuatFirstThree(packed)
        };

        let alpha = (self.color.w * 255.0).round().clamp(0.0, 255.0) as u8;

        let color = self.color.xyz().to_array().map(|c| {
            ((c - Self::SPZ_COLOR_TO_LINEAR_C) / Self::SPZ_COLOR_TO_LINEAR_FRAC_A_B * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8
        });

        let sh = match header.sh_degree().get() {
            0 => SpzGaussianSh::Zero,
            deg @ 1..=3 => {
                let mut sh = match deg {
                    1 => SpzGaussianSh::One([[0; 3]; 3]),
                    2 => SpzGaussianSh::Two([[0; 3]; 8]),
                    3 => SpzGaussianSh::Three([[0; 3]; 15]),
                    _ => unreachable!(),
                };

                fn quantize_sh(x: f32, bucket_size: u32) -> u8 {
                    let q = (x * 128.0 + 128.0).round() as u32;
                    let q = if bucket_size >= 8 {
                        q
                    } else {
                        (q + bucket_size / 2) / bucket_size * bucket_size
                    };
                    q.clamp(0, 255) as u8
                }

                for (src, dst) in self.sh.as_ref().iter().zip(sh.iter_mut()) {
                    let bucket_size = options
                        .sh_bucket_size(deg)
                        .expect("header SH degree is valid");
                    *dst = src.to_array().map(|x| quantize_sh(x, bucket_size));
                }

                sh
            }
            _ => {
                // SAFETY: SpzGaussianShDegree is guaranteed to be in [0, 3].
                unreachable!()
            }
        };

        Ok(SpzGaussian {
            position,
            scale,
            rotation,
            color,
            alpha,
            sh,
        })
    }

    /// Explicitly truncate higher SH bands or zero-extend to a different degree.
    pub fn convert_sh_degree<T: ShDegree>(&self) -> Gaussian<T> {
        let mut sh = T::Coefficients::zeroed();

        for (src, dst) in self.sh.as_ref().iter().zip(sh.as_mut().iter_mut()) {
            *dst = *src;
        }

        Gaussian {
            rot: self.rot,
            pos: self.pos,
            color: self.color,
            sh,
            scale: self.scale,
        }
    }
}

// It can be useful to implement `AsRef` for `Gaussian` and `&Gaussian` due to the frequent use of
// `from_iter` for other source formats.

impl<D: ShDegree> AsRef<Gaussian<D>> for Gaussian<D> {
    fn as_ref(&self) -> &Gaussian<D> {
        self
    }
}

/// A native Gaussian whose storage degree is discovered at runtime.
///
/// Use typed collection variants [`Gaussian`] for persistent storage, this enum is an iteration
/// and streaming boundary, not a compact replacement for degree-specific arrays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyGaussian {
    Zero(Gaussian<ShDegree0>),
    One(Gaussian<ShDegree1>),
    Two(Gaussian<ShDegree2>),
    Three(Gaussian<ShDegree3>),
    Four(Gaussian<ShDegree4>),
}

/// A borrowed native Gaussian whose storage degree is discovered at runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyGaussianRef<'a> {
    Zero(&'a Gaussian<ShDegree0>),
    One(&'a Gaussian<ShDegree1>),
    Two(&'a Gaussian<ShDegree2>),
    Three(&'a Gaussian<ShDegree3>),
    Four(&'a Gaussian<ShDegree4>),
}

/// A mutable borrow of a native Gaussian, preserving its storage degree.
#[derive(Debug)]
pub enum AnyGaussianMut<'a> {
    Zero(&'a mut Gaussian<ShDegree0>),
    One(&'a mut Gaussian<ShDegree1>),
    Two(&'a mut Gaussian<ShDegree2>),
    Three(&'a mut Gaussian<ShDegree3>),
    Four(&'a mut Gaussian<ShDegree4>),
}

macro_rules! with_any_gaussian {
    ($value:expr, $g:ident => $body:expr) => {
        match $value {
            AnyGaussian::Zero($g) => $body,
            AnyGaussian::One($g) => $body,
            AnyGaussian::Two($g) => $body,
            AnyGaussian::Three($g) => $body,
            AnyGaussian::Four($g) => $body,
        }
    };
}

impl AnyGaussian {
    pub fn sh_degree(&self) -> GaussianShDegree {
        let degree = match self {
            Self::Zero(_) => 0,
            Self::One(_) => 1,
            Self::Two(_) => 2,
            Self::Three(_) => 3,
            Self::Four(_) => 4,
        };

        GaussianShDegree::new(degree).unwrap()
    }

    /// Read a PLY record without padding or discarding any SH bands.
    pub fn from_ply(ply: &PlyGaussian) -> Result<Self, PlyGaussiansFromIterError> {
        match ply.sh.len() {
            0 => Gaussian::from_ply(ply).map(Self::Zero),
            9 => Gaussian::from_ply(ply).map(Self::One),
            24 => Gaussian::from_ply(ply).map(Self::Two),
            45 => Gaussian::from_ply(ply).map(Self::Three),
            72 => Gaussian::from_ply(ply).map(Self::Four),
            count => Err(PlyGaussiansFromIterError::UnsupportedShCount { count }),
        }
    }

    /// Read a SPZ record along with the header without padding or discarding any SH bands.
    pub fn from_spz(
        spz: SpzGaussianRef,
        header: &SpzGaussiansHeader,
    ) -> Result<Self, ShDegreeMismatchError> {
        match header.sh_degree().get() {
            0 => Gaussian::from_spz(spz, header).map(Self::Zero),
            1 => Gaussian::from_spz(spz, header).map(Self::One),
            2 => Gaussian::from_spz(spz, header).map(Self::Two),
            3 => Gaussian::from_spz(spz, header).map(Self::Three),
            _ => unreachable!("validated SPZ degree"),
        }
    }

    pub fn to_ply(&self) -> PlyGaussian {
        with_any_gaussian!(self, g => g.to_ply())
    }

    pub fn to_spz(
        &self,
        header: &SpzGaussiansHeader,
        options: &GaussianToSpzOptions,
    ) -> Result<SpzGaussian, GaussianToSpzError> {
        with_any_gaussian!(self, g => g.to_spz(header, options))
    }

    /// Recover a typed value, rejecting mismatched degrees.
    pub fn try_typed<D: ShDegree>(&self) -> Result<Gaussian<D>, ShDegreeMismatchError> {
        if self.sh_degree().get() != D::DEGREE {
            return Err(ShDegreeMismatchError {
                actual_degree: self.sh_degree().get(),
                expected_degree: D::DEGREE,
            });
        }

        Ok(self.convert_sh_degree())
    }

    /// Explicitly truncate or zero-extend SH coefficients.
    pub fn convert_sh_degree<D: ShDegree>(&self) -> Gaussian<D> {
        with_any_gaussian!(self, g => g.convert_sh_degree())
    }
}

/// Compact native collections, with one storage degree per model.
#[derive(Debug, Clone, PartialEq)]
pub enum InternalGaussians {
    Zero(Vec<Gaussian<ShDegree0>>),
    One(Vec<Gaussian<ShDegree1>>),
    Two(Vec<Gaussian<ShDegree2>>),
    Three(Vec<Gaussian<ShDegree3>>),
    Four(Vec<Gaussian<ShDegree4>>),
}

macro_rules! internal_sh_degree {
    ($degree:ty, $variant:ident) => {
        impl From<Vec<Gaussian<$degree>>> for InternalGaussians {
            fn from(value: Vec<Gaussian<$degree>>) -> Self {
                Self::$variant(value)
            }
        }
    };
}

internal_sh_degree!(ShDegree0, Zero);
internal_sh_degree!(ShDegree1, One);
internal_sh_degree!(ShDegree2, Two);
internal_sh_degree!(ShDegree3, Three);
internal_sh_degree!(ShDegree4, Four);

impl InternalGaussians {
    /// Collect native Gaussians, inferring their shared storage degree.
    ///
    /// Empty typed input retains its static degree. Empty runtime input defaults to
    /// degree 3. Use [`Self::from_iter`] to preserve an explicit degree instead.
    /// Mixed storage degrees are rejected.
    pub fn from_gaussians<T: ToAnyGaussian>(
        gaussians: impl IntoIterator<Item = T>,
    ) -> Result<Self, ShDegreeMismatchError> {
        let mut gaussians = gaussians
            .into_iter()
            .map(|g| g.to_any_gaussian())
            .peekable();
        let degree = gaussians
            .peek()
            .map(|g| g.sh_degree())
            .unwrap_or_else(|| GaussianShDegree::new(T::SH_DEGREE.unwrap_or(3)).unwrap());

        Self::from_iter(gaussians, degree)
    }

    /// Collect runtime values with an explicit shared degree, including empty input.
    pub fn from_iter(
        iter: impl IntoIterator<Item = AnyGaussian>,
        degree: GaussianShDegree,
    ) -> Result<Self, ShDegreeMismatchError> {
        macro_rules! collect {
            ($variant:ident) => {
                iter.into_iter()
                    .map(|g| g.try_typed())
                    .collect::<Result<Vec<_>, ShDegreeMismatchError>>()
                    .map(Self::$variant)
            };
        }

        match degree.get() {
            0 => collect!(Zero),
            1 => collect!(One),
            2 => collect!(Two),
            3 => collect!(Three),
            4 => collect!(Four),
            _ => unreachable!(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Zero(g) => g.len(),
            Self::One(g) => g.len(),
            Self::Two(g) => g.len(),
            Self::Three(g) => g.len(),
            Self::Four(g) => g.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over borrowed Gaussians without copying their coefficients.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = AnyGaussianRef<'_>> + '_ {
        let iter: Box<dyn ExactSizeIterator<Item = AnyGaussianRef<'_>> + '_> = match self {
            Self::Zero(g) => Box::new(g.iter().map(AnyGaussianRef::Zero)),
            Self::One(g) => Box::new(g.iter().map(AnyGaussianRef::One)),
            Self::Two(g) => Box::new(g.iter().map(AnyGaussianRef::Two)),
            Self::Three(g) => Box::new(g.iter().map(AnyGaussianRef::Three)),
            Self::Four(g) => Box::new(g.iter().map(AnyGaussianRef::Four)),
        };

        iter
    }

    /// Iterate over mutable Gaussian borrows without changing their storage degree.
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = AnyGaussianMut<'_>> + '_ {
        let iter: Box<dyn ExactSizeIterator<Item = AnyGaussianMut<'_>> + '_> = match self {
            Self::Zero(g) => Box::new(g.iter_mut().map(AnyGaussianMut::Zero)),
            Self::One(g) => Box::new(g.iter_mut().map(AnyGaussianMut::One)),
            Self::Two(g) => Box::new(g.iter_mut().map(AnyGaussianMut::Two)),
            Self::Three(g) => Box::new(g.iter_mut().map(AnyGaussianMut::Three)),
            Self::Four(g) => Box::new(g.iter_mut().map(AnyGaussianMut::Four)),
        };

        iter
    }
}

impl IterGaussian for InternalGaussians {
    type Gaussian = AnyGaussian;

    fn sh_degree(&self) -> GaussianShDegree {
        let degree = match self {
            Self::Zero(_) => 0,
            Self::One(_) => 1,
            Self::Two(_) => 2,
            Self::Three(_) => 3,
            Self::Four(_) => 4,
        };

        GaussianShDegree::new(degree).unwrap()
    }

    fn iter_gaussian(&self) -> impl ExactSizeIterator<Item = AnyGaussian> + '_ {
        // Dispatch once per model, no per-Gaussian allocation or degree branch.
        let iter: Box<dyn ExactSizeIterator<Item = AnyGaussian> + '_> = match self {
            Self::Zero(g) => Box::new(g.iter().copied().map(AnyGaussian::Zero)),
            Self::One(g) => Box::new(g.iter().copied().map(AnyGaussian::One)),
            Self::Two(g) => Box::new(g.iter().copied().map(AnyGaussian::Two)),
            Self::Three(g) => Box::new(g.iter().copied().map(AnyGaussian::Three)),
            Self::Four(g) => Box::new(g.iter().copied().map(AnyGaussian::Four)),
        };

        iter
    }
}

/// Extra options for [`Gaussian::to_spz`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianToSpzOptions {
    /// The quantization bits for each SH degree.
    pub sh_quantize_bits: [u32; 3],
}

impl GaussianToSpzOptions {
    /// Get the bits for the given SH degree.
    pub fn sh_bits(&self, degree: u8) -> Option<u32> {
        match degree {
            1..=3 => Some(self.sh_quantize_bits[degree as usize - 1]),
            _ => None,
        }
    }

    /// Get the quantization bucket size for the given SH degree.
    pub fn sh_bucket_size(&self, degree: u8) -> Option<u32> {
        self.sh_bits(degree).map(|bits| 1 << (8 - bits))
    }
}

impl Default for GaussianToSpzOptions {
    fn default() -> Self {
        Self {
            sh_quantize_bits: [5, 4, 4],
        }
    }
}

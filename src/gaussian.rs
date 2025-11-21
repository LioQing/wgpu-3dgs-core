use glam::*;

use crate::{
    PlyGaussianPod, SpzGaussian, SpzGaussianPosition, SpzGaussianPositionRef, SpzGaussianRef,
    SpzGaussianRotation, SpzGaussianRotationRef, SpzGaussianSh, SpzGaussiansHeader,
};

/// A trait of representing an iterable collection of [`Gaussian`].
pub trait IterGaussian {
    /// Iterate over [`Gaussian`].
    fn iter_gaussian(&self) -> impl Iterator<Item = Gaussian> + '_;
}

impl IterGaussian for Vec<Gaussian> {
    fn iter_gaussian(&self) -> impl Iterator<Item = Gaussian> + '_ {
        self.iter().copied()
    }
}

/// The Gaussian.
///
/// This is an intermediate representation used by the CPU to convert to
/// [`GaussianPod`](crate::GaussianPod).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    pub rot: Quat,
    pub pos: Vec3,
    pub color: U8Vec4,
    pub sh: [Vec3; 15],
    pub scale: Vec3,
}

impl Gaussian {
    /// Convert from [`PlyGaussianPod`].
    pub fn from_ply(ply: &PlyGaussianPod) -> Self {
        let pos = Vec3::from_array(ply.pos);

        let rot = Quat::from_xyzw(ply.rot[1], ply.rot[2], ply.rot[3], ply.rot[0]).normalize();

        let scale = Vec3::from_array(ply.scale).exp();

        const SH_C0: f32 = 0.2820948;
        let color = ((Vec3::splat(0.5) + Vec3::from_array(ply.color) * SH_C0) * 255.0)
            .extend((1.0 / (1.0 + (-ply.alpha).exp())) * 255.0)
            .clamp(Vec4::splat(0.0), Vec4::splat(255.0))
            .as_u8vec4();

        let sh = std::array::from_fn(|i| Vec3::new(ply.sh[i], ply.sh[i + 15], ply.sh[i + 30]));

        Self {
            rot,
            pos,
            color,
            sh,
            scale,
        }
    }

    /// Convert to [`PlyGaussianPod`].
    pub fn to_ply(&self) -> PlyGaussianPod {
        let pos = self.pos.to_array();

        let rot = [self.rot.w, self.rot.x, self.rot.y, self.rot.z];

        let scale = self.scale.map(|x| x.ln()).to_array();

        const SH_C0: f32 = 0.2820948;
        let rgba = self.color.as_vec4() / 255.0;
        let color = ((rgba.xyz() / SH_C0) - Vec3::splat(0.5 / SH_C0)).to_array();

        let alpha = -(1.0 / rgba.w - 1.0).ln();

        let mut sh = [0.0; 3 * 15];
        for i in 0..15 {
            sh[i] = self.sh[i].x;
            sh[i + 15] = self.sh[i].y;
            sh[i + 30] = self.sh[i].z;
        }

        let normal = [0.0, 0.0, 1.0];

        PlyGaussianPod {
            pos,
            normal,
            color,
            sh,
            alpha,
            scale,
            rot,
        }
    }

    /// Convert from [`SpzGaussianRef`].
    pub fn from_spz(spz: &SpzGaussianRef, header: &SpzGaussiansHeader) -> Self {
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

        let color = U8Vec3::from_array(*spz.color).extend(*spz.alpha);

        let mut sh = [Vec3::ZERO; 15];
        for (from_sh, to_sh) in spz.sh.iter().zip(sh.iter_mut()) {
            *to_sh = Vec3::from_array(from_sh.map(|c| (c as f32 - 128.0) / 128.0));
        }

        Self {
            rot,
            pos,
            color,
            sh,
            scale,
        }
    }

    /// Convert to [`SpzGaussian`].
    pub fn to_spz(
        &self,
        header: &SpzGaussiansHeader,
        options: &GaussianToSpzOptions,
    ) -> SpzGaussian {
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

        let alpha = self.color.w;

        let color = self.color.xyz().to_array();

        let sh = match header.sh_degree() {
            0 => SpzGaussianSh::Zero,
            deg @ 1..=3 => {
                let mut sh = match deg {
                    1 => SpzGaussianSh::One([[0; 3]; 3]),
                    2 => SpzGaussianSh::Two([[0; 3]; 8]),
                    3 => SpzGaussianSh::Three([[0; 3]; 15]),
                    _ => unreachable!(),
                };

                fn quantize_sh(x: f32, bucket_size: u32) -> i8 {
                    let q = (x * 128.0 + 128.0).round() as u32;
                    let q = if bucket_size >= 8 {
                        q
                    } else {
                        (q + bucket_size / 2) / bucket_size * bucket_size
                    };
                    q.clamp(0, 255) as u8 as i8
                }

                for (src, dst) in self.sh.iter().zip(sh.iter_mut()) {
                    let bucket_size = options
                        .sh_bucket_size(deg)
                        .expect("header SH degree is valid");
                    *dst = src.to_array().map(|x| quantize_sh(x, bucket_size));
                }

                sh
            }
            _ => unreachable!(),
        };

        SpzGaussian {
            position,
            scale,
            rotation,
            color,
            alpha,
            sh,
        }
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

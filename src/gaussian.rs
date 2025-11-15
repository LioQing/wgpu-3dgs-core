use glam::*;

use crate::PlyGaussianPod;

/// A vector of Gaussians.
///
/// This is a simple wrapper around a [`Vec`] of [`Gaussian`].
#[derive(Debug, Clone)]
pub struct Gaussians<Source>
where
    for<'a> &'a Source: Into<Gaussian>,
{
    /// The Gaussians.
    pub gaussians: Vec<Source>,
}

impl<Source> Gaussians<Source>
where
    for<'a> &'a Source: Into<Gaussian>,
{
    /// Create a new Gaussians.
    pub fn new(gaussians: Vec<Source>) -> Self {
        Self { gaussians }
    }

    /// Iterate over [`Gaussian`].
    pub fn iter(&self) -> impl Iterator<Item = Gaussian> + '_ {
        self.gaussians.iter().map(Into::into)
    }

    /// Get the number of Gaussians.
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    /// Check if there are no Gaussians.
    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }

    /// Convert to Gaussians of another source type.
    pub fn convert<Dest>(&self) -> Gaussians<Dest>
    where
        for<'a> &'a Source: Into<Dest>,
        for<'a> &'a Dest: Into<Gaussian>,
    {
        Gaussians {
            gaussians: self
                .gaussians
                .iter()
                .map(|g| Into::<Dest>::into(g))
                .collect::<Vec<_>>(),
        }
    }
}

impl<Source> FromIterator<Source> for Gaussians<Source>
where
    for<'a> &'a Source: Into<Gaussian>,
{
    fn from_iter<T: IntoIterator<Item = Source>>(iter: T) -> Self {
        Gaussians {
            gaussians: iter.into_iter().collect(),
        }
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
        // Position
        let pos = Vec3::from_array(ply.pos);

        // Rotation
        let rot = Quat::from_xyzw(ply.rot[1], ply.rot[2], ply.rot[3], ply.rot[0]).normalize();

        // Scale
        let scale = Vec3::from_array(ply.scale).exp();

        // Color
        const SH_C0: f32 = 0.2820948;
        let color = ((Vec3::splat(0.5) + Vec3::from_array(ply.color) * SH_C0) * 255.0)
            .extend((1.0 / (1.0 + (-ply.alpha).exp())) * 255.0)
            .clamp(Vec4::splat(0.0), Vec4::splat(255.0))
            .as_u8vec4();

        // Spherical harmonics
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
        // Position
        let pos = self.pos.to_array();

        // Rotation
        let rot = [self.rot.w, self.rot.x, self.rot.y, self.rot.z];

        // Scale
        let scale = self.scale.map(|x| x.ln()).to_array();

        // Color
        const SH_C0: f32 = 0.2820948;
        let rgba = self.color.as_vec4() / 255.0;
        let color = ((rgba.xyz() / SH_C0) - Vec3::splat(0.5 / SH_C0)).to_array();

        // Alpha
        let alpha = -(1.0 / rgba.w - 1.0).ln();

        // Spherical harmonics
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

    /// Convert from [`spz_rs::UnpackedGaussian`].
    pub fn from_spz(spz: &spz_rs::UnpackedGaussian) -> Self {
        let pos = Vec3::from_array(spz.position);
        let rot = Quat::from_array(spz.rotation);
        let scale = Vec3::from_array(spz.scale);
        let color = U8Vec4::from_array([
            (spz.color[0] * 255.0) as u8,
            (spz.color[1] * 255.0) as u8,
            (spz.color[2] * 255.0) as u8,
            (spz.alpha * 255.0) as u8,
        ]);
        let sh = std::array::from_fn(|i| Vec3::new(spz.sh_r[i], spz.sh_g[i], spz.sh_b[i]));

        Self {
            rot,
            pos,
            color,
            sh,
            scale,
        }
    }

    /// Convert to [`spz_rs::UnpackedGaussian`].
    pub fn to_spz(&self) -> spz_rs::UnpackedGaussian {
        let position = self.pos.to_array();
        let rotation = self.rot.to_array();
        let scale = self.scale.to_array();
        let color = [
            self.color.x as f32 / 255.0,
            self.color.y as f32 / 255.0,
            self.color.z as f32 / 255.0,
        ];
        let alpha = self.color.w as f32 / 255.0;
        let mut sh_r = [0.0; 15];
        let mut sh_g = [0.0; 15];
        let mut sh_b = [0.0; 15];
        for i in 0..15 {
            sh_r[i] = self.sh[i].x;
            sh_g[i] = self.sh[i].y;
            sh_b[i] = self.sh[i].z;
        }

        spz_rs::UnpackedGaussian {
            position,
            rotation,
            scale,
            color,
            alpha,
            sh_r,
            sh_g,
            sh_b,
        }
    }
}

impl From<&Gaussian> for Gaussian {
    fn from(gaussian: &Gaussian) -> Self {
        *gaussian
    }
}

impl From<PlyGaussianPod> for Gaussian {
    fn from(ply: PlyGaussianPod) -> Self {
        Self::from_ply(&ply)
    }
}

impl From<&PlyGaussianPod> for Gaussian {
    fn from(ply: &PlyGaussianPod) -> Self {
        Self::from_ply(ply)
    }
}

impl From<spz_rs::UnpackedGaussian> for Gaussian {
    fn from(spz: spz_rs::UnpackedGaussian) -> Self {
        Self::from_spz(&spz)
    }
}

impl From<&spz_rs::UnpackedGaussian> for Gaussian {
    fn from(spz: &spz_rs::UnpackedGaussian) -> Self {
        Self::from_spz(spz)
    }
}

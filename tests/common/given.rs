use std::io::{self, Read, Write};

use wgpu_3dgs_core::{AnyGaussian, Gaussian, PlyGaussians, ShDegree, SpzGaussians, glam::*};

/// A non-seekable reader whose short reads split headers and records.
pub struct ShortRead<'a>(pub &'a [u8]);

impl Read for ShortRead<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let count = out.len().min(3);

        self.0.read(&mut out[..count])
    }
}

/// A non-seekable writer that exercises `write_all` with short writes.
#[derive(Default)]
pub struct ShortWrite(pub Vec<u8>);

impl Write for ShortWrite {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let count = bytes.len().min(3);
        self.0.extend_from_slice(&bytes[..count]);

        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A writer that fails after accepting the specified number of bytes.
pub struct FailingWrite(pub usize);

impl Write for FailingWrite {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.0 == 0 {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "test write failure",
            ));
        }

        let count = bytes.len().min(self.0);
        self.0 -= count;

        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Wrapper for a temporary file that deletes the file on drop.
pub struct TempFile(std::path::PathBuf);

impl AsRef<std::path::PathBuf> for TempFile {
    fn as_ref(&self) -> &std::path::PathBuf {
        &self.0
    }
}

impl AsRef<std::path::Path> for TempFile {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl From<std::path::PathBuf> for TempFile {
    fn from(path: std::path::PathBuf) -> Self {
        std::fs::File::create(&path).expect("temporary file created");
        TempFile(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Gets a temporary file path with the given suffix.
///
/// Returns a [`TempFile`], which deletes the file on drop.
pub fn temp_file_path(suffix: &str) -> TempFile {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    std::env::temp_dir()
        .join(format!(
            "wgpu-3dgs-core-test-{}-{nanos}{suffix}",
            std::process::id()
        ))
        .into()
}

pub fn gaussian_with_seed(seed: u32) -> Gaussian {
    let base = seed as f32;

    let rot_x = base + 0.1;
    let rot_y = base + 0.2;
    let rot_z = base + 0.3;
    let rot_w = base + 0.4;
    let rot = Quat::from_xyzw(rot_x, rot_y, rot_z, rot_w).normalize();

    let pos = Vec3::new(base + 1.1, base + 2.2, base + 3.3);

    let color = Vec4::new(
        (base + 11.0) % 256.0,
        (base + 22.0) % 256.0,
        (base + 33.0) % 256.0,
        (base + 44.0) % 256.0,
    ) / 255.0;

    let mut sh = [Vec3::ZERO; 15];
    for (i, sh) in sh.iter_mut().enumerate() {
        let sh_base = base + (i as f32 * 0.3);
        *sh = Vec3::new(sh_base + 0.1, sh_base + 0.2, sh_base + 0.3) % Vec3::splat(2.0) - Vec3::ONE;
    }

    let scale = Vec3::new(base + 0.12, base + 0.34, base + 0.56);

    Gaussian {
        rot,
        pos,
        color,
        sh,
        scale,
    }
}

pub fn gaussians() -> Vec<Gaussian> {
    vec![gaussian_with_seed(42), gaussian_with_seed(123)]
}

pub fn ply_gaussians() -> PlyGaussians {
    PlyGaussians::from_gaussians(gaussians()).unwrap()
}

pub fn spz_gaussians() -> SpzGaussians {
    SpzGaussians::from_gaussians(gaussians()).unwrap()
}

pub fn gaussian_for_sh_degree<D: ShDegree>(seed: u32) -> Gaussian<D> {
    let mut gaussian = gaussian_with_seed(seed).convert_sh_degree::<D>();

    for (i, sh) in gaussian.sh.as_mut().iter_mut().enumerate() {
        let base = seed as f32 + i as f32 * 0.3;
        *sh = Vec3::new(base + 0.1, base + 0.2, base + 0.3) % Vec3::splat(2.0) - Vec3::ONE;
    }

    gaussian
}

/// Gets fixture data with populated coefficients for the requested runtime degree.
pub fn runtime_gaussians(degree: u8) -> Vec<AnyGaussian> {
    [42, 123]
        .into_iter()
        .map(|seed| match degree {
            0 => AnyGaussian::Zero(gaussian_for_sh_degree(seed)),
            1 => AnyGaussian::One(gaussian_for_sh_degree(seed)),
            2 => AnyGaussian::Two(gaussian_for_sh_degree(seed)),
            3 => AnyGaussian::Three(gaussian_for_sh_degree(seed)),
            4 => AnyGaussian::Four(gaussian_for_sh_degree(seed)),
            _ => panic!("unsupported fixture degree"),
        })
        .collect()
}

pub fn gaussian() -> Gaussian {
    gaussian_with_seed(42)
}

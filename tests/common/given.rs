use wgpu_3dgs_core::{Gaussian, PlyGaussians, SpzGaussians, glam::*};

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

    let pos = Vec3::new(base + 1.0, base + 2.0, base + 3.0);

    let color = U8Vec4::new(
        ((base + 10.0) % 256.0) as u8,
        ((base + 20.0) % 256.0) as u8,
        ((base + 30.0) % 256.0) as u8,
        ((base + 40.0) % 256.0) as u8,
    );

    let mut sh = [Vec3::ZERO; 15];
    for (i, sh) in sh.iter_mut().enumerate() {
        let sh_base = base + (i as f32);
        *sh = Vec3::new(sh_base + 0.1, sh_base + 0.2, sh_base + 0.3);
    }

    let scale = Vec3::new(base + 0.1, base + 0.2, base + 0.3);

    Gaussian {
        rot,
        pos,
        color,
        sh,
        scale,
    }
}

pub fn ply_gaussians() -> PlyGaussians {
    PlyGaussians(vec![
        gaussian_with_seed(42).to_ply(),
        gaussian_with_seed(123).to_ply(),
    ])
}

pub fn spz_gaussians() -> SpzGaussians {
    SpzGaussians::from(&[gaussian_with_seed(42), gaussian_with_seed(123)])
}

pub fn gaussian() -> Gaussian {
    gaussian_with_seed(42)
}

use std::io::BufRead;

use crate::{Gaussian, Gaussians};

impl Gaussians<spz_rs::UnpackedGaussian> {
    /// Read a SPZ file.
    pub fn read_spz_file(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::read_spz(&mut reader)
    }

    /// Read a SPZ from buffer.
    ///
    /// Implementation uses the [`spz_rs`] crate.
    pub fn read_spz(reader: &mut impl BufRead) -> Result<Self, std::io::Error> {
        let spz = spz_rs::load_packed_gaussians_from_spz_buffer(reader)?;
        let gaussians = (0..spz.num_points).map(|i| spz.unpack(i)).collect();

        Ok(Self { gaussians })
    }
}

impl From<Gaussian> for spz_rs::UnpackedGaussian {
    fn from(gaussian: Gaussian) -> Self {
        gaussian.to_spz()
    }
}

impl From<&Gaussian> for spz_rs::UnpackedGaussian {
    fn from(gaussian: &Gaussian) -> Self {
        gaussian.to_spz()
    }
}

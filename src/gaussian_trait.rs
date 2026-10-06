use std::io::BufRead;

use crate::{
    AnyGaussian, Gaussian, GaussianShDegree, Gaussians, GaussiansFromIterError, GaussiansSource,
    ShDegree,
};

/// A trait of representing an iterable collection of [`Gaussian`].
pub trait IterGaussian {
    type Gaussian: ToAnyGaussian;

    /// Shared storage degree, including for empty collections.
    fn sh_degree(&self) -> GaussianShDegree;

    /// Iterate over [`Gaussian`].
    fn iter_gaussian(&self) -> impl ExactSizeIterator<Item = Self::Gaussian> + '_;
}

impl<D: ShDegree> IterGaussian for Vec<Gaussian<D>> {
    type Gaussian = Gaussian<D>;

    fn sh_degree(&self) -> GaussianShDegree {
        GaussianShDegree::new(D::DEGREE).unwrap()
    }

    fn iter_gaussian(&self) -> impl ExactSizeIterator<Item = Gaussian<D>> + '_ {
        self.iter().copied()
    }
}

/// A trait of representing a [`IterGaussian`] that can be read from a buffer.
pub trait ReadIterGaussian: IterGaussian + Sized {
    /// Read from a buffer.
    fn read_from(reader: &mut impl BufRead) -> std::io::Result<Self>;

    /// Read from a file.
    fn read_from_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::read_from(&mut reader)
    }
}

/// A trait of representing a [`IterGaussian`] that can be written to a buffer.
pub trait WriteIterGaussian: IterGaussian {
    /// Write to a buffer.
    fn write_to(&self, writer: &mut impl std::io::Write) -> std::io::Result<()>;

    /// Write to a file.
    fn write_to_file(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        self.write_to(&mut writer)
    }
}

/// Conversion of typed and runtime native Gaussian values to [`AnyGaussian`].
pub trait ToAnyGaussian {
    /// Static SH degree if known, including when an iterator is empty.
    const SH_DEGREE: Option<u8>;

    /// Convert this value to an [`AnyGaussian`].
    fn to_any_gaussian(&self) -> AnyGaussian;
}

impl<D: ShDegree> ToAnyGaussian for Gaussian<D> {
    const SH_DEGREE: Option<u8> = Some(D::DEGREE);

    fn to_any_gaussian(&self) -> AnyGaussian {
        D::into_any(*self)
    }
}

impl ToAnyGaussian for AnyGaussian {
    const SH_DEGREE: Option<u8> = None;

    fn to_any_gaussian(&self) -> AnyGaussian {
        *self
    }
}

impl<T: ToAnyGaussian + ?Sized> ToAnyGaussian for &T {
    const SH_DEGREE: Option<u8> = T::SH_DEGREE;

    fn to_any_gaussian(&self) -> AnyGaussian {
        (*self).to_any_gaussian()
    }
}

/// Trait to extend [`Iterator`] of [`Gaussian`] to collect into [`Gaussians`].
pub trait IteratorGaussianExt: Iterator + Sized
where
    Self::Item: ToAnyGaussian,
{
    /// Collect the iterator into [`Gaussians`] with the given source.
    fn collect_gaussians(
        self,
        source: GaussiansSource,
    ) -> Result<Gaussians, GaussiansFromIterError> {
        Gaussians::from_gaussians_iter(self, source)
    }
}

impl<T: Iterator> IteratorGaussianExt for T where T::Item: ToAnyGaussian {}

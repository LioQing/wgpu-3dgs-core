use std::io::BufRead;

use crate::{
    AnyGaussian, Gaussian, GaussianShDegree, GaussiansFromIterError, InternalGaussians,
    IterGaussian, PlyGaussians, PlyGaussiansFromIterError, ReadIterGaussian, ShDegree,
    ShDegreeMismatchError, SpzGaussianShDegree, SpzGaussians, SpzGaussiansFromGaussianSliceOptions,
    SpzGaussiansFromGaussiansError, ToAnyGaussian, WriteIterGaussian,
};

/// A discriminant representation of [`Gaussians`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussiansSource {
    Internal,
    Ply,
    Spz,
}

impl From<&Gaussians> for GaussiansSource {
    fn from(value: &Gaussians) -> Self {
        match value {
            Gaussians::Internal(_) => GaussiansSource::Internal,
            Gaussians::Ply(_) => GaussiansSource::Ply,
            Gaussians::Spz(_) => GaussiansSource::Spz,
        }
    }
}

/// A unified Gaussian representation.
///
/// [`Gaussians::Internal`] variant contains Gaussians in the [`Gaussian`] format, which is the one
/// converted to [`GaussianPod`](crate::GaussianPod) directly.
///
/// Other variants contain Gaussians in their respective source file formats.
#[derive(Debug, Clone, PartialEq)]
pub enum Gaussians {
    Internal(InternalGaussians),
    Ply(PlyGaussians),
    Spz(SpzGaussians),
}

impl Gaussians {
    /// Convert a model to compact native storage without changing its degree.
    pub fn into_internal(self) -> Result<InternalGaussians, ShDegreeMismatchError> {
        if let Self::Internal(native) = self {
            return Ok(native);
        }

        InternalGaussians::from_iter(self.iter_gaussian(), self.sh_degree())
    }

    /// Convert a model to PLY storage without changing its degree.
    ///
    /// Existing PLY storage is returned unchanged. Other representations are converted
    /// through native values, so source-only data cannot be recovered.
    pub fn into_ply(self) -> Result<PlyGaussians, PlyGaussiansFromIterError> {
        if let Self::Ply(ply) = self {
            return Ok(ply);
        }

        PlyGaussians::new(
            self.iter_gaussian().map(|g| g.to_ply()).collect(),
            self.sh_degree().get(),
        )
    }

    /// Convert a model to SPZ storage without changing its degree.
    ///
    /// Existing SPZ storage is returned unchanged. Other representations use default
    /// encoding options with the model's degree, encoding is lossy.
    /// SH degree 4 is currently unsupported.
    pub fn into_spz(self) -> Result<SpzGaussians, SpzGaussiansFromGaussiansError> {
        if let Self::Spz(spz) = self {
            return Ok(spz);
        }

        let degree = self.sh_degree().get();
        let sh_degree = SpzGaussianShDegree::new(degree)
            .ok_or(SpzGaussiansFromGaussiansError::UnsupportedShDegree { degree })?;

        self.into_spz_with_options(&SpzGaussiansFromGaussianSliceOptions {
            sh_degree,
            ..Default::default()
        })
    }

    /// Encode a model as SPZ using explicit options, including existing SPZ models.
    ///
    /// Encoding is lossy. The options' SH degree must match the model's storage degree,
    /// including for empty models, use explicit degree conversion to change it.
    /// SH degree 4 is currently unsupported.
    pub fn into_spz_with_options(
        self,
        options: &SpzGaussiansFromGaussianSliceOptions,
    ) -> Result<SpzGaussians, SpzGaussiansFromGaussiansError> {
        let degree = self.sh_degree().get();

        if SpzGaussianShDegree::new(degree).is_none() {
            return Err(SpzGaussiansFromGaussiansError::UnsupportedShDegree { degree });
        }

        if degree != options.sh_degree.get() {
            return Err(ShDegreeMismatchError {
                actual_degree: degree,
                expected_degree: options.sh_degree.get(),
            }
            .into());
        }

        SpzGaussians::from_gaussians_with_options(self.iter_gaussian(), options)
    }

    /// Create a collection of Gaussians from an iterator of [`Gaussian`] with the given source.
    pub fn from_gaussians_iter<T: ToAnyGaussian>(
        iter: impl Iterator<Item = T>,
        source: GaussiansSource,
    ) -> Result<Self, GaussiansFromIterError> {
        let mut iter = iter.map(|g| g.to_any_gaussian()).peekable();
        let degree = iter
            .peek()
            .map(|g| g.sh_degree())
            .unwrap_or_else(|| GaussianShDegree::new(T::SH_DEGREE.unwrap_or(3)).unwrap());

        Self::from_gaussians_iter_with_sh_degree(iter, source, degree)
    }

    /// Convert runtime values with an explicit degree, preserving empty-model metadata.
    pub fn from_gaussians_iter_with_sh_degree(
        iter: impl Iterator<Item = AnyGaussian>,
        source: GaussiansSource,
        degree: GaussianShDegree,
    ) -> Result<Self, GaussiansFromIterError> {
        match source {
            GaussiansSource::Internal => InternalGaussians::from_iter(iter, degree)
                .map(Gaussians::Internal)
                .map_err(Into::into),
            GaussiansSource::Ply => PlyGaussians::new(
                iter.map(|g| {
                    if g.sh_degree() != degree {
                        return Err(ShDegreeMismatchError {
                            actual_degree: g.sh_degree().get(),
                            expected_degree: degree.get(),
                        });
                    }

                    Ok(g.to_ply())
                })
                .collect::<Result<Vec<_>, ShDegreeMismatchError>>()?,
                degree.get(),
            )
            .map(Gaussians::Ply)
            .map_err(Into::into),
            GaussiansSource::Spz => {
                let sh_degree = crate::SpzGaussianShDegree::new(degree.get()).ok_or(
                    SpzGaussiansFromGaussiansError::UnsupportedShDegree {
                        degree: degree.get(),
                    },
                )?;

                SpzGaussians::from_gaussians_with_options(
                    iter,
                    &crate::SpzGaussiansFromGaussianSliceOptions {
                        sh_degree,
                        ..Default::default()
                    },
                )
                .map(Gaussians::Spz)
                .map_err(Into::into)
            }
        }
    }

    /// Get the source representation of the Gaussians.
    pub fn source(&self) -> GaussiansSource {
        GaussiansSource::from(self)
    }

    /// Get the number of Gaussians.
    pub fn len(&self) -> usize {
        match self {
            Gaussians::Internal(gaussians) => gaussians.len(),
            Gaussians::Ply(ply_gaussians) => ply_gaussians.len(),
            Gaussians::Spz(spz_gaussians) => spz_gaussians.len(),
        }
    }

    /// Check if there is no Gaussian.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read from a file with the given source.
    pub fn read_from_file(
        path: impl AsRef<std::path::Path>,
        source: GaussiansSource,
    ) -> std::io::Result<Self> {
        match source {
            GaussiansSource::Internal => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot read Internal Gaussians from file",
            )),
            GaussiansSource::Ply => {
                let ply_gaussians = PlyGaussians::read_from_file(path)?;
                Ok(Gaussians::Ply(ply_gaussians))
            }
            GaussiansSource::Spz => {
                let spz_gaussians = SpzGaussians::read_from_file(path)?;
                Ok(Gaussians::Spz(spz_gaussians))
            }
        }
    }

    /// Read from a buffer with the given source.
    pub fn read_from(reader: &mut impl BufRead, source: GaussiansSource) -> std::io::Result<Self> {
        match source {
            GaussiansSource::Internal => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot read Internal Gaussians from buffer",
            )),
            GaussiansSource::Ply => {
                let ply_gaussians = PlyGaussians::read_from(reader)?;
                Ok(Gaussians::Ply(ply_gaussians))
            }
            GaussiansSource::Spz => {
                let spz_gaussians = SpzGaussians::read_from(reader)?;
                Ok(Gaussians::Spz(spz_gaussians))
            }
        }
    }

    /// Write to a file with the given source.
    pub fn write_to_file(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        match self {
            Gaussians::Internal(_) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot write Internal Gaussians to file",
            )),
            Gaussians::Ply(ply_gaussians) => ply_gaussians.write_to_file(path),
            Gaussians::Spz(spz_gaussians) => spz_gaussians.write_to_file(path),
        }
    }

    /// Write to a buffer with the given source.
    pub fn write_to(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        match self {
            Gaussians::Internal(_) => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot write Internal Gaussians to buffer",
            )),
            Gaussians::Ply(ply_gaussians) => ply_gaussians.write_to(writer),
            Gaussians::Spz(spz_gaussians) => spz_gaussians.write_to(writer),
        }
    }
}

impl<D: ShDegree> From<Vec<Gaussian<D>>> for Gaussians
where
    InternalGaussians: From<Vec<Gaussian<D>>>,
{
    fn from(value: Vec<Gaussian<D>>) -> Self {
        Gaussians::Internal(value.into())
    }
}

impl From<PlyGaussians> for Gaussians {
    fn from(value: PlyGaussians) -> Self {
        Gaussians::Ply(value)
    }
}

impl From<SpzGaussians> for Gaussians {
    fn from(value: SpzGaussians) -> Self {
        Gaussians::Spz(value)
    }
}

impl IterGaussian for Gaussians {
    type Gaussian = AnyGaussian;

    fn sh_degree(&self) -> GaussianShDegree {
        match self {
            Self::Internal(g) => g.sh_degree(),
            Self::Ply(g) => IterGaussian::sh_degree(g),
            Self::Spz(g) => IterGaussian::sh_degree(g),
        }
    }

    fn iter_gaussian(&self) -> impl ExactSizeIterator<Item = AnyGaussian> + '_ {
        match self {
            Gaussians::Internal(gaussians) => GaussiansIter::Internal(gaussians.iter_gaussian()),
            Gaussians::Ply(ply_gaussians) => GaussiansIter::Ply(ply_gaussians.iter_gaussian()),
            Gaussians::Spz(spz_gaussians) => GaussiansIter::Spz(spz_gaussians.iter_gaussian()),
        }
    }
}

impl<D: ShDegree> FromIterator<Gaussian<D>> for Gaussians
where
    InternalGaussians: From<Vec<Gaussian<D>>>,
{
    fn from_iter<T: IntoIterator<Item = Gaussian<D>>>(iter: T) -> Self {
        Gaussians::Internal(iter.into_iter().collect::<Vec<_>>().into())
    }
}

/// Iterator for [`Gaussians`].
#[derive(Debug, Clone)]
pub enum GaussiansIter<
    InternalIter: Iterator<Item = AnyGaussian>,
    PlyIter: Iterator<Item = AnyGaussian>,
    SpzIter: Iterator<Item = AnyGaussian>,
> {
    Internal(InternalIter),
    Ply(PlyIter),
    Spz(SpzIter),
}

impl<
    InternalIter: Iterator<Item = AnyGaussian>,
    PlyIter: Iterator<Item = AnyGaussian>,
    SpzIter: Iterator<Item = AnyGaussian>,
> Iterator for GaussiansIter<InternalIter, PlyIter, SpzIter>
{
    type Item = AnyGaussian;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            GaussiansIter::Internal(iter) => iter.next(),
            GaussiansIter::Ply(iter) => iter.next(),
            GaussiansIter::Spz(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            GaussiansIter::Internal(iter) => iter.size_hint(),
            GaussiansIter::Ply(iter) => iter.size_hint(),
            GaussiansIter::Spz(iter) => iter.size_hint(),
        }
    }
}

impl<
    InternalIter: ExactSizeIterator<Item = AnyGaussian>,
    PlyIter: ExactSizeIterator<Item = AnyGaussian>,
    SpzIter: ExactSizeIterator<Item = AnyGaussian>,
> ExactSizeIterator for GaussiansIter<InternalIter, PlyIter, SpzIter>
{
}

use std::io::{self, BufRead};

use crate::{
    AnyGaussian, BatchProgress, Gaussian, GaussianShDegree, GaussianStream, GaussiansSource,
    PlyGaussianStream, ShDegree, ShDegreeMismatchError,
};

/// Stream that delivers unified [`Gaussian`] values from a PLY source.
///
/// SPZ stores separate field arrays and cannot deliver individual Gaussians early.
pub struct GaussiansStream<R: BufRead> {
    stream: PlyGaussianStream<R>,
}

impl<R: BufRead> GaussiansStream<R> {
    /// Source storage degree, available before reading any records.
    pub fn sh_degree(&self) -> GaussianShDegree {
        GaussianShDegree::new(self.stream.header().sh_degree).unwrap()
    }

    /// Dispatch once using the header, then stream exact-size typed native values.
    pub fn into_typed<D: ShDegree>(
        self,
    ) -> Result<TypedGaussiansStream<D, R>, ShDegreeMismatchError> {
        if self.sh_degree().get() != D::DEGREE {
            return Err(ShDegreeMismatchError {
                actual_degree: self.sh_degree().get(),
                expected_degree: D::DEGREE,
            });
        }

        Ok(TypedGaussiansStream {
            stream: self.stream,
            degree: std::marker::PhantomData,
        })
    }

    /// Open a PLY source for streaming. Other sources return `InvalidInput`.
    pub fn new(reader: R, source: GaussiansSource) -> io::Result<Self> {
        match source {
            GaussiansSource::Ply => Ok(Self {
                stream: PlyGaussianStream::new(reader)?,
            }),
            _ => Err(std::io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("cannot stream {source:?} Gaussians"),
            )),
        }
    }
}

impl<R: BufRead> Iterator for GaussiansStream<R> {
    type Item = io::Result<AnyGaussian>;

    fn next(&mut self) -> Option<Self::Item> {
        self.stream.next().map(|result| {
            result.and_then(|pod| {
                AnyGaussian::from_ply(&pod)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
            })
        })
    }
}

impl<R: BufRead> GaussianStream for GaussiansStream<R> {
    type Gaussian = AnyGaussian;

    fn total_gaussians(&self) -> usize {
        self.stream.total_gaussians()
    }

    fn progress(&self) -> BatchProgress {
        self.stream.progress()
    }
}

/// A header-validated PLY stream with compact, typed native Gaussian output.
pub struct TypedGaussiansStream<D: ShDegree, R: BufRead> {
    stream: PlyGaussianStream<R>,
    degree: std::marker::PhantomData<D>,
}

impl<D: ShDegree, R: BufRead> Iterator for TypedGaussiansStream<D, R> {
    type Item = io::Result<Gaussian<D>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.stream.next().map(|g| {
            g.and_then(|g| {
                Gaussian::from_ply(&g)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
            })
        })
    }
}

impl<D: ShDegree, R: BufRead> GaussianStream for TypedGaussiansStream<D, R> {
    type Gaussian = Gaussian<D>;

    fn total_gaussians(&self) -> usize {
        self.stream.total_gaussians()
    }

    fn progress(&self) -> BatchProgress {
        self.stream.progress()
    }
}

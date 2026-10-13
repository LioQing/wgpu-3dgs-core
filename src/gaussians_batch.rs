use std::{
    io::{self, BufRead, Write},
    num::NonZeroUsize,
};

use crate::{
    BatchProgress, BatchRead, BatchWrite, Gaussians, GaussiansSource, PlyBatchReader,
    PlyBatchWriter, PlyGaussiansBatchIter, SpzBatchReader, SpzBatchWriter,
};

/// Whole-model batch reader for the unified Gaussian representation.
pub enum GaussiansBatchReader<R: BufRead> {
    Ply(Box<PlyBatchReader<R>>),
    Spz(Box<SpzBatchReader<R>>),
}

impl<R: BufRead> GaussiansBatchReader<R> {
    /// Read the source header and prepare to read batches.
    pub fn new(reader: R, source: GaussiansSource) -> io::Result<Self> {
        match source {
            GaussiansSource::Internal => Err(std::io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot batch read Internal Gaussians".to_string(),
            )),
            GaussiansSource::Ply => Ok(Self::Ply(Box::new(PlyBatchReader::new(reader)?))),
            GaussiansSource::Spz => Ok(Self::Spz(Box::new(SpzBatchReader::new(reader)?))),
        }
    }
}

impl<R: BufRead> BatchRead for GaussiansBatchReader<R> {
    type Model = Gaussians;

    fn progress(&self) -> BatchProgress {
        match self {
            Self::Ply(reader) => reader.progress(),
            Self::Spz(reader) => reader.progress(),
        }
    }

    fn step(&mut self, max_items: NonZeroUsize) -> io::Result<BatchProgress> {
        match self {
            Self::Ply(reader) => reader.step(max_items),
            Self::Spz(reader) => reader.step(max_items),
        }
    }

    fn finish(self) -> io::Result<Self::Model> {
        match self {
            Self::Ply(reader) => reader.finish().map(Gaussians::Ply),
            Self::Spz(reader) => reader.finish().map(Gaussians::Spz),
        }
    }
}

/// Whole-model batch writer for the unified Gaussian representation.
pub enum GaussiansBatchWriter<'a, W: Write> {
    Ply(PlyBatchWriter<W, PlyGaussiansBatchIter<'a>>),
    Spz(SpzBatchWriter<'a, W>),
}

impl<'a, W: Write> GaussiansBatchWriter<'a, W> {
    /// Write the source header and prepare to write batches.
    pub fn new(writer: W, gaussians: &'a Gaussians) -> io::Result<Self> {
        match gaussians {
            Gaussians::Internal(_) => Err(std::io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot batch write Internal Gaussians".to_string(),
            )),
            Gaussians::Ply(gaussians) => Ok(Self::Ply(PlyBatchWriter::new(writer, gaussians)?)),
            Gaussians::Spz(gaussians) => Ok(Self::Spz(SpzBatchWriter::new(writer, gaussians)?)),
        }
    }
}

impl<W: Write> BatchWrite for GaussiansBatchWriter<'_, W> {
    type Writer = W;

    fn progress(&self) -> BatchProgress {
        match self {
            Self::Ply(writer) => writer.progress(),
            Self::Spz(writer) => writer.progress(),
        }
    }

    fn step(&mut self, max_items: NonZeroUsize) -> io::Result<BatchProgress> {
        match self {
            Self::Ply(writer) => writer.step(max_items),
            Self::Spz(writer) => writer.step(max_items),
        }
    }

    fn finish(self) -> io::Result<Self::Writer> {
        match self {
            Self::Ply(writer) => writer.finish(),
            Self::Spz(writer) => writer.finish(),
        }
    }
}

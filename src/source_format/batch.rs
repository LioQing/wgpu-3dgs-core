use std::{io, num::NonZeroUsize};

/// Progress through a source format. Work units are format-specific, not compressed bytes.
///
/// PLY counts vertex records. SPZ counts entries in each successive field array. The units
/// are useful for progress display, but not an estimate of time or on-disk bytes remaining.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchProgress {
    /// Current field or section, or `"done"` for a completed SPZ operation.
    pub phase: &'static str,

    /// Entries processed in the current phase.
    pub completed_in_phase: usize,

    /// Entries in the current phase.
    pub total_in_phase: usize,

    /// Entries processed in all phases so far.
    pub completed_units: usize,

    /// Entries in all phases.
    pub total_units: usize,

    /// Whether the operation has processed all records and can be finished.
    pub done: bool,
}

/// Incrementally read a complete source-format model.
pub trait BatchRead {
    /// The complete source-format representation.
    type Model;

    /// Inspect current progress without doing more work.
    fn progress(&self) -> BatchProgress;

    /// Process at most `max_items` records in the current phase.
    ///
    /// Discard the reader after an I/O error. Header parsing and the underlying input may
    /// still block; the caller must yield between steps for UI responsiveness.
    fn step(&mut self, max_items: NonZeroUsize) -> io::Result<BatchProgress>;

    /// Finish reading. Returns an error if the model is incomplete.
    fn finish(self) -> io::Result<Self::Model>;
}

/// Incrementally write a complete source-format model.
pub trait BatchWrite {
    /// The output writer, returned after successful completion.
    type Writer;

    /// Inspect current progress without doing more work.
    fn progress(&self) -> BatchProgress;

    /// Process at most `max_items` records in the current phase.
    ///
    /// Discard the writer after an I/O error. The caller must yield between steps for UI
    /// responsiveness.
    fn step(&mut self, max_items: NonZeroUsize) -> io::Result<BatchProgress>;

    /// Finish writing. Returns an error if the model is incomplete.
    fn finish(self) -> io::Result<Self::Writer>;
}

/// Read individual Gaussians before the entire source file has been read.
///
/// Unlike [`BatchRead`], the caller owns the decoded data and may discard each batch after
/// reading it. Only formats supporting early delivery implement this trait. [`Iterator::next`]
/// yields one `io::Result` per Gaussian, or `None` at completion. An I/O error is yielded once;
/// subsequent calls return `None` and `progress().done` remains false. Discard the stream then.
pub trait GaussianStream: Iterator<Item = io::Result<Self::Gaussian>> {
    /// A single Gaussian in the original source format.
    type Gaussian;

    /// Number of Gaussians declared in the file header.
    fn total_gaussians(&self) -> usize;

    /// Inspect progress without reading more data.
    fn progress(&self) -> BatchProgress;

    /// Append up to `max_gaussians` items to `out`, returning the number appended.
    /// Successfully read items remain in `out` if a later read fails. A return of zero
    /// indicates completion only when `progress().done` is true.
    fn next_batch(
        &mut self,
        max_gaussians: NonZeroUsize,
        out: &mut Vec<Self::Gaussian>,
    ) -> io::Result<usize> {
        let mut count = 0;
        for _ in 0..max_gaussians.get() {
            match self.next() {
                Some(Ok(gaussian)) => {
                    out.push(gaussian);
                    count += 1;
                }
                Some(Err(error)) => return Err(error),
                None => break,
            }
        }
        Ok(count)
    }
}

pub(crate) fn incomplete() -> io::Error {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "source format is not complete",
    )
}

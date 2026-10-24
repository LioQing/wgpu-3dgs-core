use std::{
    io::{self, BufRead, Write},
    num::NonZeroUsize,
};

use glam::{Quat, Vec3};

use crate::{
    AnyGaussian, BatchProgress, BatchRead, BatchWrite, Gaussian, GaussianShDegree, GaussianStream,
    IterGaussian, PlyGaussiansFromIterError, ReadIterGaussian, ShDegree, ToAnyGaussian,
    WriteIterGaussian, source_format,
};

fn sh_count(degree: u8) -> usize {
    GaussianShDegree::new(degree)
        .expect("validated PLY degree")
        .num_coefficients()
}

fn sh_degree_from_len(len: usize) -> Option<u8> {
    if !len.is_multiple_of(3) {
        return None;
    }

    GaussianShDegree::from_coefficient_count(len / 3).map(|d| d.get())
}

fn ply_scalar_size(data_type: &ply_rs::ply::PropertyType) -> io::Result<usize> {
    use ply_rs::ply::{PropertyType, ScalarType};

    match data_type {
        PropertyType::Scalar(scalar) => Ok(match scalar {
            ScalarType::Char | ScalarType::UChar => 1,
            ScalarType::Short | ScalarType::UShort => 2,
            ScalarType::Int | ScalarType::UInt | ScalarType::Float => 4,
            ScalarType::Double => 8,
        }),
        PropertyType::List(..) => Err(invalid_data("PLY vertex list properties are unsupported")),
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// A Gaussian as stored in a PLY file. `sh` is channel-major: all red coefficients,
/// then green, then blue. The DC coefficients are stored separately in `color`.
#[derive(Debug, Clone, PartialEq)]
pub struct PlyGaussian {
    pub pos: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
    /// Changing this to an invalid count will cause [`IterGaussian::iter_gaussian`] to panic.
    pub sh: Vec<f32>,
    pub alpha: f32,
    pub scale: Vec3,
    /// Rotation in xyzw order. PLY stores it in wxyz order.
    pub rot: Quat,
}

impl PlyGaussian {
    fn values(&self) -> impl Iterator<Item = f32> + '_ {
        self.pos
            .to_array()
            .into_iter()
            .chain(self.normal.to_array())
            .chain(self.color.to_array())
            .chain(self.sh.iter().copied())
            .chain([self.alpha])
            .chain(self.scale.to_array())
            .chain([self.rot.w, self.rot.x, self.rot.y, self.rot.z])
    }
}

struct PlyReadProperty {
    offset: usize,
    /// Index in the canonical value buffer, or [`None`] for an unknown property.
    destination: Option<usize>,
}

/// A validated PLY header's vertex layout, shared by all records in a stream.
struct PlyReadLayout {
    properties: Vec<PlyReadProperty>,
    stride: usize,
    /// Number of non-DC SH coefficients across all three color channels.
    rest: usize,
    /// Whether every property is a float in canonical wire order, with normals.
    canonical: bool,
    encoding: ply_rs::ply::Encoding,
}

impl PlyReadLayout {
    /// Maximum number of scratch values for SH degree 4.
    ///
    /// Canonical wire order is position, normals, DC, SH, opacity, scale, rotation
    /// (wxyz). This is scratch storage, not the memory representation of [`PlyGaussian`].
    const MAX_FLOATS: usize = 17 + 3 * 24;

    /// Compile property offsets and destinations without reading any records.
    ///
    /// `header` must come from [`PlyGaussians::read_header`] without modification.
    fn new(header: &PlyHeader) -> Self {
        let rest = sh_count(header.sh_degree) * 3;
        let mut stride = 0usize;

        let properties: Vec<_> = header.header.elements["vertex"]
            .properties
            .iter()
            .map(|(name, property)| {
                let destination = match name.as_str() {
                    "x" => Some(0),
                    "y" => Some(1),
                    "z" => Some(2),
                    "nx" => Some(3),
                    "ny" => Some(4),
                    "nz" => Some(5),
                    "f_dc_0" => Some(6),
                    "f_dc_1" => Some(7),
                    "f_dc_2" => Some(8),
                    "opacity" => Some(9 + rest),
                    "scale_0" => Some(10 + rest),
                    "scale_1" => Some(11 + rest),
                    "scale_2" => Some(12 + rest),
                    "rot_0" => Some(13 + rest),
                    "rot_1" => Some(14 + rest),
                    "rot_2" => Some(15 + rest),
                    "rot_3" => Some(16 + rest),
                    _ => name
                        .strip_prefix("f_rest_")
                        .and_then(|s| s.parse::<usize>().ok())
                        .map(|index| 9 + index),
                };
                let offset = stride;

                // read_header validates scalar types and checks the complete stride.
                stride += ply_scalar_size(&property.data_type).expect("validated PLY scalar");

                PlyReadProperty {
                    offset,
                    destination,
                }
            })
            .collect();

        let canonical = properties.len() == 17 + rest
            && properties.iter().enumerate().all(|(index, property)| {
                property.destination == Some(index) && property.offset == index * 4
            })
            && stride == (17 + rest) * 4;

        Self {
            properties,
            stride,
            rest,
            canonical,
            encoding: header.header.encoding,
        }
    }

    /// Decode known properties from one binary record using the declared byte order.
    ///
    /// `bytes` must contain at least `self.stride` bytes, and the encoding must be
    /// binary. Unknown properties and absent normals leave `values` unchanged.
    fn decode_binary(&self, bytes: &[u8], values: &mut [f32; Self::MAX_FLOATS]) {
        for property in &self.properties {
            if let Some(destination) = property.destination {
                let bytes = bytes[property.offset..property.offset + 4]
                    .try_into()
                    .unwrap();

                values[destination] = match self.encoding {
                    ply_rs::ply::Encoding::BinaryLittleEndian => f32::from_le_bytes(bytes),
                    ply_rs::ply::Encoding::BinaryBigEndian => f32::from_be_bytes(bytes),
                    ply_rs::ply::Encoding::Ascii => unreachable!(),
                };
            }
        }
    }

    /// Read one PLY Gaussian record, defaulting absent normals to zero.
    ///
    /// Reuse `line` for ASCII records and `scratch` for non-canonical binary records
    /// that do not fit in the reader's buffer. Unknown scalar properties are skipped.
    fn read_from(
        &self,
        reader: &mut impl BufRead,
        line: &mut String,
        scratch: &mut Vec<u8>,
    ) -> io::Result<PlyGaussian> {
        use ply_rs::ply::Encoding;

        let mut values = [0.0; Self::MAX_FLOATS];
        match self.encoding {
            Encoding::Ascii => {
                line.clear();

                if reader.read_line(line)? == 0 {
                    return Err(io::ErrorKind::UnexpectedEof.into());
                }

                let mut tokens = line.split_whitespace();
                for property in &self.properties {
                    let token = tokens.next().ok_or_else(|| {
                        invalid_data("Gaussian element property invalid or missing in PLY")
                    })?;

                    if let Some(destination) = property.destination {
                        values[destination] = token.parse::<f32>().map_err(|_| {
                            invalid_data("Gaussian element property invalid or missing in PLY")
                        })?;
                    }
                }

                if tokens.next().is_some() {
                    return Err(invalid_data("Gaussian element has extra values in PLY"));
                }
            }
            Encoding::BinaryLittleEndian | Encoding::BinaryBigEndian if self.canonical => {
                let record = &mut values[..17 + self.rest];
                reader.read_exact(bytemuck::cast_slice_mut(record))?;

                let native_endian = match self.encoding {
                    Encoding::BinaryLittleEndian => cfg!(target_endian = "little"),
                    Encoding::BinaryBigEndian => cfg!(target_endian = "big"),
                    Encoding::Ascii => unreachable!(),
                };

                if !native_endian {
                    for value in record {
                        *value = f32::from_bits(value.to_bits().swap_bytes());
                    }
                }
            }
            Encoding::BinaryLittleEndian | Encoding::BinaryBigEndian => {
                let buffer = reader.fill_buf()?;
                if buffer.len() >= self.stride {
                    self.decode_binary(&buffer[..self.stride], &mut values);
                    reader.consume(self.stride);
                } else {
                    if scratch.len() != self.stride {
                        scratch
                            .try_reserve_exact(self.stride - scratch.len())
                            .map_err(io::Error::other)?;
                        scratch.resize(self.stride, 0);
                    }

                    reader.read_exact(scratch)?;
                    self.decode_binary(scratch, &mut values);
                }
            }
        }

        let tail = &values[9 + self.rest..];

        Ok(PlyGaussian {
            pos: Vec3::from_slice(&values[0..3]),
            normal: Vec3::from_slice(&values[3..6]),
            color: Vec3::from_slice(&values[6..9]),
            sh: values[9..9 + self.rest].to_vec(),
            alpha: tail[0],
            scale: Vec3::from_slice(&tail[1..4]),
            rot: Quat::from_xyzw(tail[5], tail[6], tail[7], tail[4]),
        })
    }
}

impl<D: ShDegree> From<Gaussian<D>> for PlyGaussian {
    fn from(gaussian: Gaussian<D>) -> Self {
        gaussian.to_ply()
    }
}

impl<D: ShDegree> From<&Gaussian<D>> for PlyGaussian {
    fn from(gaussian: &Gaussian<D>) -> Self {
        gaussian.to_ply()
    }
}

/// Header of PLY file.
///
/// This represents the header parsed by [`PlyGaussians::read_header`].
#[derive(Debug, Clone)]
pub struct PlyHeader {
    pub header: ply_rs::ply::Header,
    pub sh_degree: u8,
}

impl PlyHeader {
    /// Number of vertex records in the file.
    pub fn count(&self) -> usize {
        self.header.elements["vertex"].count
    }
}

/// A collection of PLY Gaussians with one shared SH degree.
#[derive(Debug, Clone, PartialEq)]
pub struct PlyGaussians {
    pub gaussians: Vec<PlyGaussian>,
    pub sh_degree: u8,
}

impl PlyGaussians {
    /// Construct a collection, checking that all records have the declared SH degree.
    pub fn new(
        gaussians: Vec<PlyGaussian>,
        sh_degree: u8,
    ) -> Result<Self, PlyGaussiansFromIterError> {
        let degree = GaussianShDegree::new(sh_degree)
            .ok_or(PlyGaussiansFromIterError::UnsupportedShDegree { degree: sh_degree })?;
        let expected_count = degree.num_coefficients() * 3;

        for gaussian in &gaussians {
            if gaussian.sh.len() != expected_count {
                return Err(PlyGaussiansFromIterError::ShCountMismatch {
                    actual_count: gaussian.sh.len(),
                    expected_count,
                });
            }
        }

        Ok(Self {
            gaussians,
            sh_degree,
        })
    }

    /// Collect PLY records with an explicit shared SH degree, including empty input.
    ///
    /// Each record's coefficient count must match the declared degree.
    /// Use [`Self::from_gaussians`] to convert native values with degree inference.
    pub fn from_iter(
        iter: impl IntoIterator<Item = PlyGaussian>,
        sh_degree: u8,
    ) -> Result<Self, PlyGaussiansFromIterError> {
        Self::new(iter.into_iter().collect(), sh_degree)
    }

    /// Convert native Gaussians, inferring their shared storage degree.
    ///
    /// Empty typed input retains its static degree. Empty runtime input defaults to
    /// degree 3. Use [`Self::from_iter`] to preserve an explicit degree instead.
    /// Mixed storage degrees are rejected.
    pub fn from_gaussians<T: ToAnyGaussian>(
        gaussians: impl IntoIterator<Item = T>,
    ) -> Result<Self, PlyGaussiansFromIterError> {
        let mut gaussians = gaussians
            .into_iter()
            .map(|g| g.to_any_gaussian())
            .peekable();

        let degree = gaussians
            .peek()
            .map(|g| g.sh_degree().get())
            .unwrap_or(T::SH_DEGREE.unwrap_or(3));

        Self::from_iter(gaussians.map(|g| g.to_ply()), degree)
    }

    /// The number of Gaussians.
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    /// Check if there are no Gaussians.
    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }

    /// Iterate over the Gaussians.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &PlyGaussian> {
        self.gaussians.iter()
    }

    /// Iterate over the Gaussians mutably.
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut PlyGaussian> {
        self.gaussians.iter_mut()
    }

    /// Read and validate the vertex properties and infer the SH degree.
    ///
    /// SH degree 0 to 4 are supported.
    pub fn read_header(reader: &mut impl BufRead) -> io::Result<PlyHeader> {
        use ply_rs::ply::{PropertyType, ScalarType};

        let header =
            ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new().read_header(reader)?;

        // The stream starts at the first element's data. Other leading elements
        // would have to be consumed before reading any vertices.
        if header.elements.keys().next().map(String::as_str) != Some("vertex") {
            return Err(invalid_data(
                "Gaussian vertex element not found in PLY header",
            ));
        }

        let vertex = header
            .elements
            .get("vertex")
            .ok_or_else(|| invalid_data("Gaussian vertex element not found in PLY header"))?;

        // Lists have variable record sizes and token counts. Validate fixed scalar
        // widths here so the stream can compile offsets without reading body data.
        vertex
            .properties
            .values()
            .try_fold(0usize, |stride, property| {
                stride
                    .checked_add(ply_scalar_size(&property.data_type)?)
                    .ok_or_else(|| invalid_data("PLY vertex record size overflow"))
            })?;

        let mut rest = 0;
        for name in vertex.properties.keys() {
            if name.starts_with("f_rest_") {
                rest += 1;
            }
        }

        let sh_degree = sh_degree_from_len(rest)
            .ok_or_else(|| invalid_data(format!("unsupported PLY SH coefficient count: {rest}")))?;

        for name in [
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
            "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ]
        .into_iter()
        .map(str::to_string)
        .chain((0..rest).map(|i| format!("f_rest_{i}")))
        {
            match vertex.properties.get(&name) {
                Some(prop) if prop.data_type == PropertyType::Scalar(ScalarType::Float) => (),
                _ => {
                    return Err(invalid_data(format!(
                        "missing or non-float PLY property: {name}"
                    )));
                }
            }
        }

        for name in ["nx", "ny", "nz"] {
            if vertex
                .properties
                .get(name)
                .is_some_and(|p| p.data_type != PropertyType::Scalar(ScalarType::Float))
            {
                return Err(invalid_data(format!("non-float PLY property: {name}")));
            }
        }

        Ok(PlyHeader { header, sh_degree })
    }

    /// Read a complete PLY Gaussian collection, preserving the header's SH degree.
    ///
    /// `reader` must be positioned after the header parsed by [`Self::read_header`].
    /// For incremental delivery, use [`PlyGaussianStream::from_header`] instead.
    pub fn read_gaussians(reader: &mut impl BufRead, header: PlyHeader) -> io::Result<Self> {
        let count = header.count();
        let sh_degree = header.sh_degree;

        log::info!(
            "Reading PLY format with {count} Gaussians (SH degree {})",
            header.sh_degree
        );

        let mut gaussians = Vec::new();
        gaussians
            .try_reserve_exact(count)
            .map_err(io::Error::other)?;
        for gaussian in PlyGaussianStream::from_header(reader, header) {
            gaussians.push(gaussian?);
        }

        Self::new(gaussians, sh_degree)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
}

impl IterGaussian for PlyGaussians {
    type Gaussian = AnyGaussian;

    fn sh_degree(&self) -> GaussianShDegree {
        GaussianShDegree::new(self.sh_degree).expect("validated PLY degree")
    }

    fn iter_gaussian(&self) -> impl ExactSizeIterator<Item = AnyGaussian> + '_ {
        self.iter()
            .map(|g| AnyGaussian::from_ply(g).expect("validated PLY coefficients"))
    }
}

impl ReadIterGaussian for PlyGaussians {
    fn read_from(reader: &mut impl BufRead) -> io::Result<Self> {
        let header = Self::read_header(reader)?;
        Self::read_gaussians(reader, header)
    }
}

impl WriteIterGaussian for PlyGaussians {
    fn write_to(&self, writer: &mut impl Write) -> io::Result<()> {
        let mut writer = PlyBatchWriter::new(writer, self)?;

        while !BatchWrite::progress(&writer).done {
            writer.step(NonZeroUsize::new(4096).unwrap())?;
        }

        writer.finish()?;

        Ok(())
    }
}

impl TryFrom<Vec<PlyGaussian>> for PlyGaussians {
    type Error = PlyGaussiansFromIterError;

    fn try_from(gaussians: Vec<PlyGaussian>) -> Result<Self, Self::Error> {
        let sh_degree = match gaussians.first() {
            Some(gaussian) => {
                sh_degree_from_len(gaussian.sh.len()).ok_or(Self::Error::UnsupportedShCount {
                    count: gaussian.sh.len(),
                })?
            }
            None => 3,
        };

        let expected_count = sh_count(sh_degree) * 3;
        if let Some(gaussian) = gaussians.iter().find(|g| g.sh.len() != expected_count) {
            return Err(Self::Error::ShCountMismatch {
                actual_count: gaussian.sh.len(),
                expected_count,
            });
        }

        Ok(Self {
            gaussians,
            sh_degree,
        })
    }
}

/// PLY stream that delivers records before the whole file has been read.
pub struct PlyGaussianStream<R: BufRead> {
    reader: R,
    header: PlyHeader,
    layout: PlyReadLayout,
    line: String,
    scratch: Vec<u8>,
    read: usize,
    total: usize,
    failed: bool,
}

impl<R: BufRead> PlyGaussianStream<R> {
    /// Parse the header and prepare to stream PLY records.
    pub fn new(mut reader: R) -> io::Result<Self> {
        let header = PlyGaussians::read_header(&mut reader)?;
        Ok(Self::from_header(reader, header))
    }

    /// Prepare to stream records after a previously parsed header.
    ///
    /// `header` must come from [`PlyGaussians::read_header`] without modification,
    /// and `reader` must be positioned at the first vertex record. This constructor
    /// does not read any records.
    pub fn from_header(reader: R, header: PlyHeader) -> Self {
        let total = header.count();
        let layout = PlyReadLayout::new(&header);

        Self {
            reader,
            header,
            layout,
            line: String::new(),
            scratch: Vec::new(),
            read: 0,
            total,
            failed: false,
        }
    }

    /// Get the header parsed from PLY.
    pub fn header(&self) -> &PlyHeader {
        &self.header
    }
}

impl<R: BufRead> Iterator for PlyGaussianStream<R> {
    type Item = io::Result<PlyGaussian>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.read == self.total {
            return None;
        }

        match self
            .layout
            .read_from(&mut self.reader, &mut self.line, &mut self.scratch)
        {
            Ok(gaussian) => {
                self.read += 1;
                Some(Ok(gaussian))
            }
            Err(error) => {
                self.failed = true;
                Some(Err(error))
            }
        }
    }
}

impl<R: BufRead> GaussianStream for PlyGaussianStream<R> {
    type Gaussian = PlyGaussian;

    fn total_gaussians(&self) -> usize {
        self.total
    }

    fn progress(&self) -> BatchProgress {
        BatchProgress {
            phase: "vertices",
            completed_in_phase: self.read,
            total_in_phase: self.total,
            completed_units: self.read,
            total_units: self.total,
            done: self.read == self.total,
        }
    }
}

/// Whole-model PLY reader built on the [`PlyGaussianStream`].
pub struct PlyBatchReader<R: BufRead> {
    stream: PlyGaussianStream<R>,
    gaussians: Vec<PlyGaussian>,
}

impl<R: BufRead> PlyBatchReader<R> {
    pub fn new(reader: R) -> io::Result<Self> {
        Ok(Self {
            stream: PlyGaussianStream::new(reader)?,
            gaussians: Vec::new(),
        })
    }
}

impl<R: BufRead> BatchRead for PlyBatchReader<R> {
    type Model = PlyGaussians;

    fn progress(&self) -> BatchProgress {
        self.stream.progress()
    }

    fn step(&mut self, max_items: NonZeroUsize) -> io::Result<BatchProgress> {
        let remaining = self.stream.total - self.stream.read;
        self.gaussians
            .try_reserve(max_items.get().min(remaining))
            .map_err(io::Error::other)?;
        self.stream.next_batch(max_items, &mut self.gaussians)?;

        Ok(self.stream.progress())
    }

    fn finish(self) -> io::Result<Self::Model> {
        if !self.stream.progress().done {
            return Err(source_format::batch::incomplete());
        }

        PlyGaussians::new(self.gaussians, self.stream.header().sh_degree)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
}

/// PLY writer that consumes an iterator incrementally. The degree and count must
/// be supplied before writing the header. Each record is checked before writing.
pub struct PlyBatchWriter<W: Write, I: Iterator<Item = io::Result<PlyGaussian>>> {
    writer: W,
    gaussians: I,
    sh_degree: GaussianShDegree,
    total: usize,
    written: usize,
}

pub type PlyGaussiansBatchIter<'a> = std::iter::Map<
    std::iter::Cloned<std::slice::Iter<'a, PlyGaussian>>,
    fn(PlyGaussian) -> io::Result<PlyGaussian>,
>;

impl<'a, W: Write> PlyBatchWriter<W, PlyGaussiansBatchIter<'a>> {
    pub fn new(writer: W, gaussians: &'a PlyGaussians) -> io::Result<Self> {
        Self::from_iter(
            writer,
            gaussians.len(),
            gaussians.sh_degree,
            gaussians.gaussians.iter().cloned().map(Ok),
        )
    }
}

impl<W: Write, I: Iterator<Item = io::Result<PlyGaussian>>> PlyBatchWriter<W, I> {
    /// Write a header and prepare to consume `count` records with the given SH degree.
    pub fn from_iter(mut writer: W, count: usize, sh_degree: u8, gaussians: I) -> io::Result<Self> {
        let sh_degree = GaussianShDegree::new(sh_degree).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported PLY SH degree: {sh_degree}"),
            )
        })?;

        Self::write_header(&mut writer, count, sh_degree)?;

        Ok(Self {
            writer,
            gaussians,
            sh_degree,
            total: count,
            written: 0,
        })
    }

    fn write_header(
        writer: &mut impl Write,
        count: usize,
        sh_degree: GaussianShDegree,
    ) -> io::Result<()> {
        const SYSTEM_ENDIANNESS: ply_rs::ply::Encoding = match cfg!(target_endian = "little") {
            true => ply_rs::ply::Encoding::BinaryLittleEndian,
            false => ply_rs::ply::Encoding::BinaryBigEndian,
        };

        writeln!(writer, "ply")?;
        writeln!(writer, "format {SYSTEM_ENDIANNESS} 1.0")?;
        writeln!(writer, "element vertex {count}")?;

        for property in [
            "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        ] {
            writeln!(writer, "property float {property}")?;
        }

        for i in 0..sh_degree.num_coefficients() * 3 {
            writeln!(writer, "property float f_rest_{i}")?;
        }

        for property in [
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ] {
            writeln!(writer, "property float {property}")?;
        }

        writeln!(writer, "end_header")
    }
}

impl<W: Write, I: Iterator<Item = io::Result<PlyGaussian>>> BatchWrite for PlyBatchWriter<W, I> {
    type Writer = W;

    fn progress(&self) -> BatchProgress {
        BatchProgress {
            phase: "vertices",
            completed_in_phase: self.written,
            total_in_phase: self.total,
            completed_units: self.written,
            total_units: self.total,
            done: self.written == self.total,
        }
    }

    fn step(&mut self, max_items: NonZeroUsize) -> io::Result<BatchProgress> {
        let count = max_items.get().min(self.total - self.written);

        for _ in 0..count {
            let gaussian = self
                .gaussians
                .next()
                .ok_or_else(source_format::batch::incomplete)??;

            if gaussian.sh.len() != self.sh_degree.num_coefficients() * 3 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "PLY Gaussian SH count does not match the degree",
                ));
            }

            for value in gaussian.values() {
                self.writer.write_all(&value.to_ne_bytes())?;
            }

            self.written += 1;
        }

        Ok(self.progress())
    }

    fn finish(self) -> io::Result<W> {
        if !self.progress().done {
            return Err(source_format::batch::incomplete());
        }

        Ok(self.writer)
    }
}

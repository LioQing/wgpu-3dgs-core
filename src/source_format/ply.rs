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

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn validate_sh_degree(degree: u8) -> io::Result<()> {
    if degree > 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported PLY SH degree: {degree}"),
        ));
    }

    Ok(())
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
    fn empty(sh_degree: u8) -> Self {
        Self {
            pos: Vec3::ZERO,
            normal: Vec3::ZERO,
            color: Vec3::ZERO,
            sh: vec![0.0; sh_count(sh_degree) * 3],
            alpha: 0.0,
            scale: Vec3::ZERO,
            rot: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        }
    }

    fn set_value(&mut self, name: &str, value: f32) {
        match name {
            "x" => self.pos.x = value,
            "y" => self.pos.y = value,
            "z" => self.pos.z = value,
            "nx" => self.normal.x = value,
            "ny" => self.normal.y = value,
            "nz" => self.normal.z = value,
            "f_dc_0" => self.color.x = value,
            "f_dc_1" => self.color.y = value,
            "f_dc_2" => self.color.z = value,
            "opacity" => self.alpha = value,
            "scale_0" => self.scale.x = value,
            "scale_1" => self.scale.y = value,
            "scale_2" => self.scale.z = value,
            "rot_0" => self.rot.w = value,
            "rot_1" => self.rot.x = value,
            "rot_2" => self.rot.y = value,
            "rot_3" => self.rot.z = value,
            _ => {
                if let Some(index) = name
                    .strip_prefix("f_rest_")
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    self.sh[index] = value; // The header validates all SH indices first.
                }
            }
        }
    }

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

    /// Read one PLY Gaussian record into [`PlyGaussian`].
    ///
    /// `header` may be parsed by calling [`PlyGaussians::read_header`].
    fn read_from(reader: &mut impl BufRead, header: &PlyHeader) -> io::Result<PlyGaussian> {
        use ply_rs::ply::{Encoding, Property};

        let vertex = &header.header.elements["vertex"];
        let mut gaussian = PlyGaussian::empty(header.sh_degree);

        match header.header.encoding {
            Encoding::Ascii => {
                let mut line = String::new();

                if reader.read_line(&mut line)? == 0 {
                    return Err(io::ErrorKind::UnexpectedEof.into());
                }

                let mut values = line.split_whitespace();

                for (name, property) in &vertex.properties {
                    let value = values.next().ok_or_else(|| {
                        invalid_data("Gaussian element property invalid or missing in PLY")
                    })?;

                    if property.data_type
                        == ply_rs::ply::PropertyType::Scalar(ply_rs::ply::ScalarType::Float)
                    {
                        let value = value.parse::<f32>().map_err(|_| {
                            invalid_data("Gaussian element property invalid or missing in PLY")
                        })?;

                        gaussian.set_value(name, value);
                    }
                }

                if values.next().is_some() {
                    return Err(invalid_data("Gaussian element has extra values in PLY"));
                }
            }
            Encoding::BinaryLittleEndian | Encoding::BinaryBigEndian => {
                let parser = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();

                let element = match header.header.encoding {
                    Encoding::BinaryLittleEndian => {
                        parser.read_little_endian_element(reader, vertex)?
                    }
                    Encoding::BinaryBigEndian => parser.read_big_endian_element(reader, vertex)?,
                    Encoding::Ascii => unreachable!(),
                };

                for (name, property) in element {
                    if let Property::Float(value) = property {
                        gaussian.set_value(&name, value);
                    }
                }
            }
        }

        Ok(gaussian)
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

        // List properties contain a variable number of ASCII tokens. Refuse
        // them rather than mistaking a later value for an SH coefficient.
        if vertex
            .properties
            .values()
            .any(|p| !matches!(p.data_type, PropertyType::Scalar(_)))
        {
            return Err(invalid_data("PLY vertex list properties are unsupported"));
        }

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

        let gaussians =
            PlyGaussianStream::from_header(reader, header).collect::<io::Result<Vec<_>>>()?;

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

        Self {
            reader,
            header,
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

        match PlyGaussian::read_from(&mut self.reader, &self.header) {
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
    sh_degree: u8,
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
        validate_sh_degree(sh_degree)?;

        Self::write_header(&mut writer, count, sh_degree)?;

        Ok(Self {
            writer,
            gaussians,
            sh_degree,
            total: count,
            written: 0,
        })
    }

    fn write_header(writer: &mut impl Write, count: usize, sh_degree: u8) -> io::Result<()> {
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

        for i in 0..sh_count(sh_degree) * 3 {
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

            if gaussian.sh.len() != sh_count(self.sh_degree) * 3 {
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

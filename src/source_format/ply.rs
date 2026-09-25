use std::{
    io::{self, BufRead, Write},
    num::NonZeroUsize,
};

use bytemuck::Zeroable;

use crate::{
    BatchProgress, BatchRead, BatchWrite, Gaussian, IterGaussian, ProgressiveGaussianRead,
    ReadIterGaussian, WriteIterGaussian,
};

/// The POD representation of Gaussian in PLY format.
///
/// Fields are stored as arrays because using glam types would add padding
/// according to C alignment rules.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlyGaussianPod {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub sh: [f32; 3 * 15],
    pub alpha: f32,
    pub scale: [f32; 3],
    pub rot: [f32; 4],
}

impl PlyGaussianPod {
    /// Set the value of a property by name.
    pub fn set_value(&mut self, name: &str, value: f32) {
        macro_rules! set_prop {
            ($name:expr, $field:expr) => {
                $field = value
            };
        }

        match name {
            "x" => set_prop!("x", self.pos[0]),
            "y" => set_prop!("y", self.pos[1]),
            "z" => set_prop!("z", self.pos[2]),
            "nx" => set_prop!("nx", self.normal[0]),
            "ny" => set_prop!("ny", self.normal[1]),
            "nz" => set_prop!("nz", self.normal[2]),
            "f_dc_0" => set_prop!("f_dc_0", self.color[0]),
            "f_dc_1" => set_prop!("f_dc_1", self.color[1]),
            "f_dc_2" => set_prop!("f_dc_2", self.color[2]),
            "f_rest_0" => set_prop!("f_rest_0", self.sh[0]),
            "f_rest_1" => set_prop!("f_rest_1", self.sh[1]),
            "f_rest_2" => set_prop!("f_rest_2", self.sh[2]),
            "f_rest_3" => set_prop!("f_rest_3", self.sh[3]),
            "f_rest_4" => set_prop!("f_rest_4", self.sh[4]),
            "f_rest_5" => set_prop!("f_rest_5", self.sh[5]),
            "f_rest_6" => set_prop!("f_rest_6", self.sh[6]),
            "f_rest_7" => set_prop!("f_rest_7", self.sh[7]),
            "f_rest_8" => set_prop!("f_rest_8", self.sh[8]),
            "f_rest_9" => set_prop!("f_rest_9", self.sh[9]),
            "f_rest_10" => set_prop!("f_rest_10", self.sh[10]),
            "f_rest_11" => set_prop!("f_rest_11", self.sh[11]),
            "f_rest_12" => set_prop!("f_rest_12", self.sh[12]),
            "f_rest_13" => set_prop!("f_rest_13", self.sh[13]),
            "f_rest_14" => set_prop!("f_rest_14", self.sh[14]),
            "f_rest_15" => set_prop!("f_rest_15", self.sh[15]),
            "f_rest_16" => set_prop!("f_rest_16", self.sh[16]),
            "f_rest_17" => set_prop!("f_rest_17", self.sh[17]),
            "f_rest_18" => set_prop!("f_rest_18", self.sh[18]),
            "f_rest_19" => set_prop!("f_rest_19", self.sh[19]),
            "f_rest_20" => set_prop!("f_rest_20", self.sh[20]),
            "f_rest_21" => set_prop!("f_rest_21", self.sh[21]),
            "f_rest_22" => set_prop!("f_rest_22", self.sh[22]),
            "f_rest_23" => set_prop!("f_rest_23", self.sh[23]),
            "f_rest_24" => set_prop!("f_rest_24", self.sh[24]),
            "f_rest_25" => set_prop!("f_rest_25", self.sh[25]),
            "f_rest_26" => set_prop!("f_rest_26", self.sh[26]),
            "f_rest_27" => set_prop!("f_rest_27", self.sh[27]),
            "f_rest_28" => set_prop!("f_rest_28", self.sh[28]),
            "f_rest_29" => set_prop!("f_rest_29", self.sh[29]),
            "f_rest_30" => set_prop!("f_rest_30", self.sh[30]),
            "f_rest_31" => set_prop!("f_rest_31", self.sh[31]),
            "f_rest_32" => set_prop!("f_rest_32", self.sh[32]),
            "f_rest_33" => set_prop!("f_rest_33", self.sh[33]),
            "f_rest_34" => set_prop!("f_rest_34", self.sh[34]),
            "f_rest_35" => set_prop!("f_rest_35", self.sh[35]),
            "f_rest_36" => set_prop!("f_rest_36", self.sh[36]),
            "f_rest_37" => set_prop!("f_rest_37", self.sh[37]),
            "f_rest_38" => set_prop!("f_rest_38", self.sh[38]),
            "f_rest_39" => set_prop!("f_rest_39", self.sh[39]),
            "f_rest_40" => set_prop!("f_rest_40", self.sh[40]),
            "f_rest_41" => set_prop!("f_rest_41", self.sh[41]),
            "f_rest_42" => set_prop!("f_rest_42", self.sh[42]),
            "f_rest_43" => set_prop!("f_rest_43", self.sh[43]),
            "f_rest_44" => set_prop!("f_rest_44", self.sh[44]),
            "opacity" => set_prop!("opacity", self.alpha),
            "scale_0" => set_prop!("scale_0", self.scale[0]),
            "scale_1" => set_prop!("scale_1", self.scale[1]),
            "scale_2" => set_prop!("scale_2", self.scale[2]),
            "rot_0" => set_prop!("rot_0", self.rot[0]),
            "rot_1" => set_prop!("rot_1", self.rot[1]),
            "rot_2" => set_prop!("rot_2", self.rot[2]),
            "rot_3" => set_prop!("rot_3", self.rot[3]),
            _ => {
                log::warn!("Unknown property: {name}");
            }
        }
    }
}

impl ply_rs::ply::PropertyAccess for PlyGaussianPod {
    fn new() -> Self {
        PlyGaussianPod::zeroed()
    }

    fn set_property(&mut self, property_name: String, property: ply_rs::ply::Property) {
        let ply_rs::ply::Property::Float(value) = property else {
            log::error!("Property {property_name} is not a float");
            return;
        };

        self.set_value(&property_name, value);
    }
}

impl From<Gaussian> for PlyGaussianPod {
    fn from(gaussian: Gaussian) -> Self {
        gaussian.to_ply()
    }
}

impl From<&Gaussian> for PlyGaussianPod {
    fn from(gaussian: &Gaussian) -> Self {
        gaussian.to_ply()
    }
}

/// Header of PLY file.
///
/// This represents the header parsed by [`PlyGaussians::read_ply_header`].
#[derive(Debug, Clone)]
pub enum PlyHeader {
    /// The Inria PLY format.
    ///
    /// The number represents the number of Gaussians.
    ///
    /// This can be directly loaded into [`PlyGaussianPod`] by [`BufReader::read_exact`](std::io::Read::read_exact).
    Inria(usize),

    /// Custom PLY format.
    Custom(ply_rs::ply::Header),
}

impl PlyHeader {
    /// Get the number of Gaussians.
    ///
    /// Returns [`None`] if the vertex element is not found in [`PlyHeader::Custom`].
    pub fn count(&self) -> Option<usize> {
        match self {
            Self::Inria(count) => Some(*count),
            Self::Custom(header) => header.elements.get("vertex").map(|vertex| vertex.count),
        }
    }
}

/// PLY Gaussian [`Result`] iterator.
pub enum PlyGaussianIter<
    I: Iterator<Item = Result<PlyGaussianPod, std::io::Error>>,
    C: Iterator<Item = Result<PlyGaussianPod, std::io::Error>>,
> {
    /// The Inria PLY format.
    Inria(I),

    /// Custom PLY format.
    ///
    /// This still is the same properties as Inria format, but may have different order.
    Custom(C),
}

impl<
    I: Iterator<Item = Result<PlyGaussianPod, std::io::Error>>,
    C: Iterator<Item = Result<PlyGaussianPod, std::io::Error>>,
> Iterator for PlyGaussianIter<I, C>
{
    type Item = Result<PlyGaussianPod, std::io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inria(iter) => iter.next(),
            Self::Custom(iter) => iter.next(),
        }
    }
}

fn vertex_element_not_found_error() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "Gaussian vertex element not found in PLY header",
    )
}

fn read_inria(reader: &mut impl BufRead) -> io::Result<PlyGaussianPod> {
    let mut gaussian = PlyGaussianPod::zeroed();
    reader.read_exact(bytemuck::bytes_of_mut(&mut gaussian))?;
    Ok(gaussian)
}

fn read_custom(
    reader: &mut impl BufRead,
    header: &ply_rs::ply::Header,
) -> io::Result<PlyGaussianPod> {
    let vertex = header
        .elements
        .get("vertex")
        .ok_or_else(vertex_element_not_found_error)?;

    Ok(match header.encoding {
        ply_rs::ply::Encoding::Ascii => {
            let mut line = String::new();
            reader.read_line(&mut line)?;

            let mut gaussian = PlyGaussianPod::zeroed();
            vertex
                .properties
                .keys()
                .zip(
                    line.split_whitespace()
                        .map(|s| Some(s.parse::<f32>()))
                        .chain(std::iter::repeat(None)),
                )
                .try_for_each(|(name, value)| match value {
                    Some(Ok(value)) => {
                        gaussian.set_value(name, value);
                        Ok(())
                    }
                    Some(Err(_)) | None => Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Gaussian element property invalid or missing in PLY",
                    )),
                })?;
            gaussian
        }
        ply_rs::ply::Encoding::BinaryLittleEndian => {
            ply_rs::parser::Parser::<PlyGaussianPod>::new()
                .read_little_endian_element(reader, vertex)?
        }
        ply_rs::ply::Encoding::BinaryBigEndian => ply_rs::parser::Parser::<PlyGaussianPod>::new()
            .read_big_endian_element(reader, vertex)?,
    })
}

fn write_header(writer: &mut impl Write, count: usize) -> io::Result<()> {
    const SYSTEM_ENDIANNESS: ply_rs::ply::Encoding = match cfg!(target_endian = "little") {
        true => ply_rs::ply::Encoding::BinaryLittleEndian,
        false => ply_rs::ply::Encoding::BinaryBigEndian,
    };

    writeln!(writer, "ply")?;
    writeln!(writer, "format {SYSTEM_ENDIANNESS} 1.0")?;
    writeln!(writer, "element vertex {count}")?;
    for property in PlyGaussians::PLY_PROPERTIES {
        writeln!(writer, "property float {property}")?;
    }
    writeln!(writer, "end_header")
}

/// A collection of Gaussians in PLY format.
///
/// The PLY file is expected to be the same format as the one used in the original Inria
/// implementation, or a custom PLY file with the same properties.
///
/// See [`PlyGaussians::PLY_PROPERTIES`] for a list of expected properties.
#[derive(Debug, Clone, PartialEq)]
pub struct PlyGaussians(pub Vec<PlyGaussianPod>);

impl PlyGaussians {
    /// The list of properties in the PLY file.
    pub const PLY_PROPERTIES: &[&str] = &[
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "f_rest_0",
        "f_rest_1",
        "f_rest_2",
        "f_rest_3",
        "f_rest_4",
        "f_rest_5",
        "f_rest_6",
        "f_rest_7",
        "f_rest_8",
        "f_rest_9",
        "f_rest_10",
        "f_rest_11",
        "f_rest_12",
        "f_rest_13",
        "f_rest_14",
        "f_rest_15",
        "f_rest_16",
        "f_rest_17",
        "f_rest_18",
        "f_rest_19",
        "f_rest_20",
        "f_rest_21",
        "f_rest_22",
        "f_rest_23",
        "f_rest_24",
        "f_rest_25",
        "f_rest_26",
        "f_rest_27",
        "f_rest_28",
        "f_rest_29",
        "f_rest_30",
        "f_rest_31",
        "f_rest_32",
        "f_rest_33",
        "f_rest_34",
        "f_rest_35",
        "f_rest_36",
        "f_rest_37",
        "f_rest_38",
        "f_rest_39",
        "f_rest_40",
        "f_rest_41",
        "f_rest_42",
        "f_rest_43",
        "f_rest_44",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ];

    /// Get the number of Gaussians.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if there are no Gaussians.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate over the Gaussians.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &PlyGaussianPod> {
        self.0.iter()
    }

    /// Iterate over the Gaussians mutably.
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut PlyGaussianPod> {
        self.0.iter_mut()
    }

    /// Read a PLY header.
    ///
    /// See [`PlyGaussians::PLY_PROPERTIES`] for a list of expected properties.
    pub fn read_header(reader: &mut impl BufRead) -> Result<PlyHeader, std::io::Error> {
        let parser = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
        let header = parser.read_header(reader)?;
        let vertex = header
            .elements
            .get("vertex")
            .ok_or_else(vertex_element_not_found_error)?;

        const SYSTEM_ENDIANNESS: ply_rs::ply::Encoding = match cfg!(target_endian = "little") {
            true => ply_rs::ply::Encoding::BinaryLittleEndian,
            false => ply_rs::ply::Encoding::BinaryBigEndian,
        };

        let ply_header = match vertex
            .properties
            .iter()
            .zip(Self::PLY_PROPERTIES.iter())
            .all(|((a, property), b)| {
                a == *b
                    && property.data_type
                        == ply_rs::ply::PropertyType::Scalar(ply_rs::ply::ScalarType::Float)
            })
            && vertex.properties.len() == Self::PLY_PROPERTIES.len()
            && header.encoding == SYSTEM_ENDIANNESS
        {
            true => PlyHeader::Inria(vertex.count),
            false => PlyHeader::Custom(header),
        };

        Ok(ply_header)
    }

    /// Read the PLY Gaussians into [`PlyGaussianPod`].
    ///
    /// `header` may be parsed by calling [`PlyGaussians::read_header`].
    pub fn read_gaussians(
        reader: &mut impl BufRead,
        header: PlyHeader,
    ) -> Result<impl Iterator<Item = Result<PlyGaussianPod, std::io::Error>>, std::io::Error> {
        let count = header.count().ok_or_else(vertex_element_not_found_error)?;
        log::info!("Reading PLY format with {count} Gaussians");

        Ok(match header {
            PlyHeader::Inria(..) => PlyGaussianIter::Inria((0..count).map(|_| read_inria(reader))),
            PlyHeader::Custom(header) => {
                PlyGaussianIter::Custom((0..count).map(move |_| read_custom(reader, &header)))
            }
        })
    }
}

impl IterGaussian for PlyGaussians {
    fn iter_gaussian(&self) -> impl ExactSizeIterator<Item = Gaussian> + '_ {
        self.iter().map(Gaussian::from_ply)
    }
}

impl ReadIterGaussian for PlyGaussians {
    fn read_from(reader: &mut impl BufRead) -> std::io::Result<Self> {
        let mut reader = PlyBatchReader::new(reader)?;
        while !BatchRead::progress(&reader).done {
            reader.step(NonZeroUsize::new(4096).unwrap())?;
        }
        reader.finish()
    }
}

impl WriteIterGaussian for PlyGaussians {
    fn write_to(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        let mut writer = PlyBatchWriter::new(writer, self)?;
        while !BatchWrite::progress(&writer).done {
            writer.step(NonZeroUsize::new(4096).unwrap())?;
        }
        writer.finish()?;
        Ok(())
    }
}

/// PLY reader that delivers records before the whole file has been read.
pub struct PlyGaussianStream<R: BufRead> {
    reader: R,
    header: PlyHeader,
    read: usize,
    total: usize,
}

impl<R: BufRead> PlyGaussianStream<R> {
    pub fn new(mut reader: R) -> io::Result<Self> {
        let header = PlyGaussians::read_header(&mut reader)?;
        let total = header.count().ok_or_else(vertex_element_not_found_error)?;

        Ok(Self {
            reader,
            header,
            read: 0,
            total,
        })
    }
}

impl<R: BufRead> ProgressiveGaussianRead for PlyGaussianStream<R> {
    type Item = PlyGaussianPod;

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

    fn read_gaussians(
        &mut self,
        max_gaussians: NonZeroUsize,
        out: &mut Vec<Self::Item>,
    ) -> io::Result<usize> {
        let count = max_gaussians.get().min(self.total - self.read);
        for _ in 0..count {
            let gaussian = match &self.header {
                PlyHeader::Inria(..) => read_inria(&mut self.reader)?,
                PlyHeader::Custom(header) => read_custom(&mut self.reader, header)?,
            };
            out.push(gaussian);
            self.read += 1;
        }
        Ok(count)
    }
}

/// Whole-model PLY reader built on the [`PlyGaussianStream`].
pub struct PlyBatchReader<R: BufRead> {
    stream: PlyGaussianStream<R>,
    gaussians: Vec<PlyGaussianPod>,
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
        self.stream.read_gaussians(max_items, &mut self.gaussians)?;
        Ok(self.stream.progress())
    }

    fn finish(self) -> io::Result<Self::Model> {
        if !self.stream.progress().done {
            return Err(crate::source_format::batch::incomplete());
        }
        Ok(PlyGaussians(self.gaussians))
    }
}

/// PLY writer for an existing collection, writing at most one batch per step.
pub struct PlyBatchWriter<'a, W: Write> {
    writer: W,
    gaussians: &'a PlyGaussians,
    written: usize,
}

impl<'a, W: Write> PlyBatchWriter<'a, W> {
    pub fn new(mut writer: W, gaussians: &'a PlyGaussians) -> io::Result<Self> {
        write_header(&mut writer, gaussians.len())?;
        Ok(Self {
            writer,
            gaussians,
            written: 0,
        })
    }
}

impl<W: Write> BatchWrite for PlyBatchWriter<'_, W> {
    type Writer = W;

    fn progress(&self) -> BatchProgress {
        let total = self.gaussians.len();
        BatchProgress {
            phase: "vertices",
            completed_in_phase: self.written,
            total_in_phase: total,
            completed_units: self.written,
            total_units: total,
            done: self.written == total,
        }
    }

    fn step(&mut self, max_items: NonZeroUsize) -> io::Result<BatchProgress> {
        let end = self.written + max_items.get().min(self.gaussians.len() - self.written);
        for gaussian in &self.gaussians.0[self.written..end] {
            self.writer.write_all(bytemuck::bytes_of(gaussian))?;
            self.written += 1;
        }
        Ok(self.progress())
    }

    fn finish(self) -> io::Result<W> {
        if !self.progress().done {
            return Err(crate::source_format::batch::incomplete());
        }
        Ok(self.writer)
    }
}

impl From<Vec<PlyGaussianPod>> for PlyGaussians {
    fn from(gaussians: Vec<PlyGaussianPod>) -> Self {
        Self(gaussians)
    }
}

impl<G: AsRef<Gaussian>> FromIterator<G> for PlyGaussians {
    fn from_iter<T: IntoIterator<Item = G>>(iter: T) -> Self {
        Self(iter.into_iter().map(|g| g.as_ref().to_ply()).collect())
    }
}

impl FromIterator<PlyGaussianPod> for PlyGaussians {
    fn from_iter<T: IntoIterator<Item = PlyGaussianPod>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

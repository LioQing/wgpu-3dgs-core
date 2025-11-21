use std::{
    io::{Read, Write},
    ops::RangeInclusive,
};

use flate2::{read::GzDecoder, write::GzEncoder};
use itertools::Itertools;

use crate::{
    Gaussian, GaussianToSpzOptions, IterGaussian, SpzGaussiansFromGaussianSliceError,
    SpzGaussiansFromIterError,
};

macro_rules! gaussian_field {
    (
        #[docname = $docname:literal]
        $name:ident {
            $(
                $(#[doc = $doc:literal])?
                $variant:ident $(($ty:ty))?
            ),+ $(,)?
        }
    ) => {
        paste::paste! {
            macro_rules! noop {
                ($tt:tt) => {};
                ($tt:tt _) => {_};
            }

            #[doc = "A single SPZ Gaussian "]
            #[doc = $docname]
            #[doc = " field."]
            #[derive(Debug, Clone)]
            pub enum [< SpzGaussian $name >]  {
                $(
                    $(#[doc = $doc])?
                    $variant $(($ty))?,
                )+
            }

            #[doc = "Reference to SPZ Gaussian "]
            #[doc = $docname]
            #[doc = " field."]
            #[derive(Debug, Clone)]
            pub enum [< SpzGaussian $name Ref>]<'a> {
                $(
                    $(#[doc = $doc])?
                    $variant $((&'a $ty))?,
                )+
            }

            #[doc = "Iterator over SPZ Gaussian "]
            #[doc = $docname]
            #[doc = " references."]
            pub enum [< SpzGaussian $name Iter >]<'a> {
                $(
                    $(#[doc = $doc])?
                    $variant $((std::slice::Iter<'a, $ty>))?,
                )+
            }

            impl<'a> Iterator for [< SpzGaussian $name Iter >]<'a> {
                type Item = [< SpzGaussian $name Ref >]<'a>;

                fn next(&mut self) -> Option<Self::Item> {
                    macro_rules! body {
                        ($variant_:ident, $ty_:ty, $iter:expr) => {
                            $iter.next().map(|v| [< SpzGaussian $name Ref >]:: $variant_ (v))
                        };
                        ($variant_:ident) => {
                            Some([< SpzGaussian $name Ref >]:: $variant_)
                        };
                    }

                    match self {
                        $(
                            #[allow(clippy::redundant_pattern)]
                            [< SpzGaussian $name Iter >]:: $variant $( (iter @ noop!($ty _)) )? => {
                                body!($variant $(, $ty, iter )?)
                            }
                        )+
                    }
                }
            }

            #[doc = "Representation of SPZ Gaussians "]
            #[doc = $docname]
            #[doc = "s."]
            #[derive(Debug, Clone)]
            pub enum [< SpzGaussians $name s>] {
                $(
                    $(#[doc = $doc])?
                    $variant $((Vec<$ty>))?,
                )+
            }

            impl [< SpzGaussians $name s>] {
                /// Get the number of elements.
                pub fn len(&self) -> usize {
                    match self {
                        $(
                            #[allow(clippy::redundant_pattern)]
                            [< SpzGaussians $name s>]:: $variant $( (vec @ noop!($ty _)) )? => {
                                #[allow(unused_variables)]
                                let len = 0;
                                $(
                                    noop!($ty);
                                    let len = vec.len();
                                )?
                                len
                            }
                        )+
                    }
                }

                /// Check if empty.
                pub fn is_empty(&self) -> bool {
                    self.len() == 0
                }

                /// Get an iterator over references.
                pub fn iter<'a>(&'a self) -> [< SpzGaussian $name Iter >]<'a> {
                    macro_rules! body {
                        ($variant_:ident, $ty_:ty, $vec:expr) => {
                            [< SpzGaussian $name Iter >]:: $variant_ ( $vec.iter() )
                        };
                        ($variant_:ident) => {
                            [< SpzGaussian $name Iter >]:: $variant_
                        };
                    }

                    match self {
                        $(
                            #[allow(clippy::redundant_pattern)]
                            [< SpzGaussians $name s>]:: $variant $( (vec @ noop!($ty _)) )? => {
                                body!($variant $(, $ty, vec )?)
                            }
                        )+
                    }
                }
            }

            impl FromIterator<[< SpzGaussian $name >]> for Result<
                [< SpzGaussians $name s>],
                $crate::error::SpzGaussiansCollectError<[< SpzGaussian $name >]>
            > {
                fn from_iter<I: IntoIterator<Item = [< SpzGaussian $name >]>>(iter: I) -> Self {
                    let mut iter = iter.into_iter();
                    let Some(first) = iter.next() else {
                        return Err($crate::error::SpzGaussiansCollectError::EmptyIterator);
                    };

                    match first {
                        $(
                            #[allow(clippy::redundant_pattern)]
                            [< SpzGaussian $name >]:: $variant $( (first_value @ noop!($ty _)) )? => {
                                $(
                                    noop!($ty);
                                    let vec = std::iter::once(Ok(first_value))
                                        .chain(
                                            iter.map(|v| {
                                                match v {
                                                    [< SpzGaussian $name >]:: $variant (value) => Ok(value),
                                                    other => Err(
                                                        $crate::error::SpzGaussiansCollectError::InvalidMixedVariant {
                                                            first_variant: [< SpzGaussian $name >]:: $variant (first_value),
                                                            current_variant: other,
                                                        }
                                                    ),
                                                }
                                            })
                                        )
                                        .collect::<Result<Vec<_>, _>>()?;
                                )?
                                Ok([< SpzGaussians $name s>]:: $variant $( ({ noop!($ty); vec }) )?)
                            }
                        )+
                    }
                }
            }
        }
    }
}

gaussian_field! {
    #[docname = "position"]
    Position {
        #[doc = "(x, y, z) each as 16-bit floating point."]
        Float16([u16; 3]),
        #[doc = "(x, y, z) each as 24-bit fixed point signed integer."]
        FixedPoint24([[u8; 3]; 3]),
    }
}

gaussian_field! {
    #[docname = "rotation"]
    Rotation {
        #[doc = "(x, y, z) each as 8-bit signed integer."]
        QuatFirstThree([u8; 3]),
        #[doc = "Smallest 3 components each as 10-bit signed integer. 2 bits for index of omitted component."]
        QuatSmallestThree([u8; 4]),
    }
}

gaussian_field! {
    #[docname = "SH coefficients"]
    Sh {
        Zero,
        One([[i8; 3]; 3]),
        Two([[i8; 3]; 8]),
        Three([[i8; 3]; 15]),
    }
}

/// A single SPZ Gaussian.
///
/// This is usually only used for [`SpzGaussians::from_iter`].
#[derive(Debug, Clone)]
pub struct SpzGaussian {
    pub position: SpzGaussianPosition,
    pub scale: [u8; 3],
    pub rotation: SpzGaussianRotation,
    pub alpha: u8,
    pub color: [u8; 3],
    pub sh: SpzGaussianSh,
}

impl SpzGaussianSh {
    /// Get an iterator over SH coefficients.
    pub fn iter(&self) -> impl Iterator<Item = &[i8; 3]> {
        match self {
            SpzGaussianSh::Zero => [].iter(),
            SpzGaussianSh::One(sh) => sh.iter(),
            SpzGaussianSh::Two(sh) => sh.iter(),
            SpzGaussianSh::Three(sh) => sh.iter(),
        }
    }

    /// Get an iterator over mutable SH coefficients.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [i8; 3]> {
        match self {
            SpzGaussianSh::Zero => [].iter_mut(),
            SpzGaussianSh::One(sh) => sh.iter_mut(),
            SpzGaussianSh::Two(sh) => sh.iter_mut(),
            SpzGaussianSh::Three(sh) => sh.iter_mut(),
        }
    }
}

/// Reference to a SPZ Gaussian.
#[derive(Debug)]
pub struct SpzGaussianRef<'a> {
    pub position: SpzGaussianPositionRef<'a>,
    pub scale: &'a [u8; 3],
    pub rotation: SpzGaussianRotationRef<'a>,
    pub alpha: &'a u8,
    pub color: &'a [u8; 3],
    pub sh: SpzGaussianShRef<'a>,
}

impl SpzGaussianShRef<'_> {
    /// Get an iterator over SH coefficients.
    pub fn iter(&self) -> impl Iterator<Item = &[i8; 3]> + '_ {
        match self {
            SpzGaussianShRef::Zero => [].iter(),
            SpzGaussianShRef::One(sh) => sh.iter(),
            SpzGaussianShRef::Two(sh) => sh.iter(),
            SpzGaussianShRef::Three(sh) => sh.iter(),
        }
    }
}

/// Header of SPZ Gaussians file.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpzGaussiansHeaderPod {
    pub magic: u32,
    pub version: u32,
    pub num_points: u32,
    pub sh_degree: u8,
    pub fractional_bits: u8,
    pub flags: u8,
    pub reserved: u8,
}

/// Header of SPZ Gaussians file.
///
/// This is the validated version of [`SpzGaussiansHeaderPod`]. This is simply a wrapper around
/// [`SpzGaussiansHeaderPod`] that ensures the values are valid, we could also implement
/// specialized structs for each field but it would be overkill for now.
#[derive(Debug, Clone)]
pub struct SpzGaussiansHeader(SpzGaussiansHeaderPod);

impl SpzGaussiansHeader {
    /// The magic number for SPZ Gaussians files.
    pub const MAGIC: u32 = 0x5053474e; // "NGSP"

    /// The supported SPZ versions.
    pub const SUPPORTED_VERSIONS: RangeInclusive<u32> = 1..=3;

    /// The supported SH degrees.
    pub const SUPPORTED_SH_DEGREES: RangeInclusive<u8> = 0..=3;

    /// Create a [`SpzGaussiansHeader`].
    ///
    /// Returns an error if the header is invalid.
    pub fn new(
        version: u32,
        num_points: u32,
        sh_degree: u8,
        fractional_bits: u8,
        antialiased: bool,
    ) -> Result<Self, std::io::Error> {
        Self::try_from_pod(SpzGaussiansHeaderPod {
            magic: Self::MAGIC,
            version,
            num_points,
            sh_degree,
            fractional_bits,
            flags: if antialiased { 0x1 } else { 0x0 },
            reserved: 0,
        })
    }

    /// Validate and create a validated SPZ Gaussians header.
    pub fn try_from_pod(pod: SpzGaussiansHeaderPod) -> Result<Self, std::io::Error> {
        if pod.magic != Self::MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid SPZ magic number: {:X}, expected {:X}",
                    pod.magic,
                    Self::MAGIC
                ),
            ));
        }

        if !Self::SUPPORTED_VERSIONS.contains(&pod.version) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Unsupported SPZ version: {}, expected one of {:?}",
                    pod.version,
                    Self::SUPPORTED_VERSIONS
                ),
            ));
        }

        if !Self::SUPPORTED_SH_DEGREES.contains(&pod.sh_degree) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Unsupported SH degree: {}, expected one of {:?}",
                    pod.sh_degree,
                    Self::SUPPORTED_SH_DEGREES
                ),
            ));
        }

        Ok(Self(pod))
    }

    /// Create a default [`SpzGaussiansHeader`] from number of points and SH degree.
    pub fn default(num_points: u32, sh_degree: u8) -> Result<Self, std::io::Error> {
        Self::new(
            Self::SUPPORTED_VERSIONS
                .last()
                .expect("at least one supported version"),
            num_points,
            sh_degree,
            12,
            false,
        )
    }

    /// Get the [`SpzGaussiansHeaderPod`].
    pub fn as_pod(&self) -> &SpzGaussiansHeaderPod {
        &self.0
    }

    /// Get the version of the SPZ file.
    pub fn version(&self) -> u32 {
        self.0.version
    }

    /// Get the number of points in the SPZ file.
    pub fn num_points(&self) -> usize {
        self.0.num_points as usize
    }

    /// Get the SH degree of the SPZ file.
    pub fn sh_degree(&self) -> u8 {
        self.0.sh_degree
    }

    /// Get the number of SH coefficients.
    pub fn sh_num_coefficients(&self) -> usize {
        match self.0.sh_degree {
            0 => 0,
            1 => 3,
            2 => 8,
            3 => 15,
            _ => unreachable!(),
        }
    }

    /// Get the number of fractional bits.
    pub fn fractional_bits(&self) -> usize {
        self.0.fractional_bits as usize
    }

    /// Check if the antialiased flag is set.
    pub fn is_antialiased(&self) -> bool {
        (self.0.flags & 0x1) != 0
    }

    /// Check if float16 encoding is used.
    pub fn uses_float16(&self) -> bool {
        self.version() == 1
    }

    /// Check if quaternion smallest three encoding is used.
    pub fn uses_quat_smallest_three(&self) -> bool {
        self.version() >= 3
    }
}

impl SpzGaussiansPositions {
    /// Read positions from reader.
    pub fn read_from(
        reader: &mut impl Read,
        count: usize,
        uses_float16: bool,
    ) -> Result<Self, std::io::Error> {
        if uses_float16 {
            let mut positions = vec![[0u16; 3]; count];
            reader.read_exact(bytemuck::cast_slice_mut(&mut positions))?;
            Ok(SpzGaussiansPositions::Float16(positions))
        } else {
            let mut positions = vec![[[0u8; 3]; 3]; count];
            reader.read_exact(bytemuck::cast_slice_mut(&mut positions))?;
            Ok(SpzGaussiansPositions::FixedPoint24(positions))
        }
    }

    /// Write positions to writer.
    pub fn write_to(&self, writer: &mut impl Write) -> Result<(), std::io::Error> {
        match self {
            SpzGaussiansPositions::Float16(positions) => {
                writer.write_all(bytemuck::cast_slice(positions))
            }
            SpzGaussiansPositions::FixedPoint24(positions) => {
                writer.write_all(bytemuck::cast_slice(positions))
            }
        }
    }
}

impl SpzGaussiansRotations {
    /// Read rotations from reader.
    pub fn read_from(
        reader: &mut impl Read,
        count: usize,
        uses_quat_smallest_three: bool,
    ) -> Result<Self, std::io::Error> {
        if !uses_quat_smallest_three {
            let mut rots = vec![[0u8; 3]; count];
            reader.read_exact(bytemuck::cast_slice_mut(&mut rots))?;
            Ok(SpzGaussiansRotations::QuatFirstThree(rots))
        } else {
            let mut rots = vec![[0u8; 4]; count];
            reader.read_exact(bytemuck::cast_slice_mut(&mut rots))?;
            Ok(SpzGaussiansRotations::QuatSmallestThree(rots))
        }
    }

    /// Write rotations to writer.
    pub fn write_to(&self, writer: &mut impl Write) -> Result<(), std::io::Error> {
        match self {
            SpzGaussiansRotations::QuatFirstThree(rots) => {
                writer.write_all(bytemuck::cast_slice(rots))
            }
            SpzGaussiansRotations::QuatSmallestThree(rots) => {
                writer.write_all(bytemuck::cast_slice(rots))
            }
        }
    }
}

impl SpzGaussiansShs {
    /// Read SH coefficients from reader.
    pub fn read_from(
        reader: &mut impl Read,
        count: usize,
        sh_degree: u8,
    ) -> Result<Self, std::io::Error> {
        match sh_degree {
            0 => Ok(SpzGaussiansShs::Zero),
            1 => {
                let mut sh_coeffs = vec![[[0i8; 3]; 3]; count];
                reader.read_exact(bytemuck::cast_slice_mut(&mut sh_coeffs))?;
                Ok(SpzGaussiansShs::One(sh_coeffs))
            }
            2 => {
                let mut sh_coeffs = vec![[[0i8; 3]; 8]; count];
                reader.read_exact(bytemuck::cast_slice_mut(&mut sh_coeffs))?;
                Ok(SpzGaussiansShs::Two(sh_coeffs))
            }
            3 => {
                let mut sh_coeffs = vec![[[0i8; 3]; 15]; count];
                reader.read_exact(bytemuck::cast_slice_mut(&mut sh_coeffs))?;
                Ok(SpzGaussiansShs::Three(sh_coeffs))
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported SH degree: {}", sh_degree),
            )),
        }
    }

    /// Write SH coefficients to writer.
    pub fn write_to(&self, writer: &mut impl Write) -> Result<(), std::io::Error> {
        match self {
            SpzGaussiansShs::Zero => Ok(()),
            SpzGaussiansShs::One(sh_coeffs) => writer.write_all(bytemuck::cast_slice(sh_coeffs)),
            SpzGaussiansShs::Two(sh_coeffs) => writer.write_all(bytemuck::cast_slice(sh_coeffs)),
            SpzGaussiansShs::Three(sh_coeffs) => writer.write_all(bytemuck::cast_slice(sh_coeffs)),
        }
    }
}

/// A collection of Gaussians in SPZ format.
#[derive(Debug, Clone)]
pub struct SpzGaussians {
    pub header: SpzGaussiansHeader,

    pub positions: SpzGaussiansPositions,

    /// `(x, y, z)` each as 8-bit log-encoded integer.
    pub scales: Vec<[u8; 3]>,

    pub rotations: SpzGaussiansRotations,

    /// 8-bit unsigned integer.
    pub alphas: Vec<u8>,

    /// `(r, g, b)` each as 8-bit unsigned integer.
    pub colors: Vec<[u8; 3]>,

    pub shs: SpzGaussiansShs,
}

impl SpzGaussians {
    /// Read a SPZ from buffer.
    ///
    /// `reader` should be a gzip compressed SPZ buffer.
    pub fn read_spz(reader: &mut impl Read) -> Result<Self, std::io::Error> {
        let mut decoder = GzDecoder::new(reader);
        Self::read_spz_decompressed(&mut decoder)
    }

    /// Read a SPZ from a decompressed buffer.
    ///
    /// `reader` should be decompressed SPZ buffer.
    pub fn read_spz_decompressed(reader: &mut impl Read) -> Result<Self, std::io::Error> {
        let header = Self::read_spz_header(reader)?;
        Self::read_spz_guassians(reader, header)
    }

    /// Read a SPZ header.
    ///
    /// `reader` should be decompressed SPZ buffer.
    pub fn read_spz_header(reader: &mut impl Read) -> Result<SpzGaussiansHeader, std::io::Error> {
        let mut header_bytes = [0u8; std::mem::size_of::<SpzGaussiansHeaderPod>()];
        reader.read_exact(&mut header_bytes)?;
        let header: SpzGaussiansHeaderPod = bytemuck::cast(header_bytes);
        SpzGaussiansHeader::try_from_pod(header)
    }

    /// Read the SPZ Gaussians.
    ///
    /// `reader` should be decompressed SPZ buffer positioned after the header.
    ///
    /// `header` may be parsed by calling [`SpzGaussians::read_spz_header`].
    pub fn read_spz_guassians(
        reader: &mut impl Read,
        header: SpzGaussiansHeader,
    ) -> Result<Self, std::io::Error> {
        let count = header.num_points();
        let uses_float16 = header.uses_float16();
        let uses_quat_smallest_three = header.uses_quat_smallest_three();

        let positions = SpzGaussiansPositions::read_from(reader, count, uses_float16)?;

        let mut scales = vec![[0u8; 3]; count];
        reader.read_exact(bytemuck::cast_slice_mut(&mut scales))?;

        let rotations = SpzGaussiansRotations::read_from(reader, count, uses_quat_smallest_three)?;

        let mut alphas = vec![0u8; count];
        reader.read_exact(bytemuck::cast_slice_mut(&mut alphas))?;

        let mut colors = vec![[0u8; 3]; count];
        reader.read_exact(bytemuck::cast_slice_mut(&mut colors))?;

        let shs = SpzGaussiansShs::read_from(reader, count, header.sh_degree())?;

        Ok(SpzGaussians {
            header,
            positions,
            scales,
            rotations,
            alphas,
            colors,
            shs,
        })
    }

    /// Write the Gaussians to a SPZ buffer.
    ///
    /// `writer` should receive the gzip compressed SPZ buffer.
    pub fn write_spz(&self, writer: &mut impl Write) -> Result<(), std::io::Error> {
        let mut encoder = GzEncoder::new(writer, flate2::Compression::default());
        self.write_spz_decompressed(&mut encoder)?;
        encoder.finish()?;
        Ok(())
    }

    /// Write the Gaussians to a SPZ buffer.
    ///
    /// `writer` will receive the decompressed SPZ buffer.
    pub fn write_spz_decompressed(&self, writer: &mut impl Write) -> Result<(), std::io::Error> {
        writer.write_all(bytemuck::cast_slice(std::slice::from_ref(
            self.header.as_pod(),
        )))?;

        self.positions.write_to(writer)?;

        writer.write_all(bytemuck::cast_slice(&self.scales))?;

        self.rotations.write_to(writer)?;

        writer.write_all(bytemuck::cast_slice(&self.alphas))?;

        writer.write_all(bytemuck::cast_slice(&self.colors))?;

        self.shs.write_to(writer)?;

        Ok(())
    }

    /// Convert from a slice of [`Gaussian`]s.
    pub fn from_gaussian_slice(gaussians: &[Gaussian]) -> Self {
        Self::from_gaussian_slice_with_options(
            gaussians,
            &SpzGaussiansFromGaussianSliceOptions::default(),
        )
        .expect("valid default options")
    }

    /// Convert from a slice of [`Gaussian`]s with options.
    pub fn from_gaussian_slice_with_options(
        gaussians: &[Gaussian],
        options: &SpzGaussiansFromGaussianSliceOptions,
    ) -> Result<Self, SpzGaussiansFromGaussianSliceError> {
        let header = SpzGaussiansHeader::new(
            options.version,
            gaussians.len() as u32,
            options.sh_degree,
            options.fractional_bits as u8,
            options.antialiased,
        )?;

        let gaussians = gaussians
            .iter()
            .map(|g| {
                g.to_spz(
                    &header,
                    &GaussianToSpzOptions {
                        sh_quantize_bits: options.sh_quantize_bits,
                    },
                )
            })
            .collect::<Vec<_>>();

        Ok(Self::from_iter(header, gaussians)?)
    }

    /// Convert from an [`IntoIterator`] of [`SpzGaussian`]s.
    pub fn from_iter(
        header: SpzGaussiansHeader,
        iter: impl IntoIterator<Item = SpzGaussian>,
    ) -> Result<Self, SpzGaussiansFromIterError> {
        let (positions, scales, rotations, alphas, colors, shs) = iter
            .into_iter()
            .map(|spz| {
                (
                    spz.position,
                    spz.scale,
                    spz.rotation,
                    spz.alpha,
                    spz.color,
                    spz.sh,
                )
            })
            .multiunzip::<(Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>, Vec<_>)>();

        let positions = positions
            .into_iter()
            .collect::<Result<_, _>>()
            .map_err(SpzGaussiansFromIterError::InvalidMixedPositionVariant)?;

        let rotations = rotations
            .into_iter()
            .collect::<Result<_, _>>()
            .map_err(SpzGaussiansFromIterError::InvalidMixedRotationVariant)?;

        let shs = shs
            .into_iter()
            .collect::<Result<_, _>>()
            .map_err(SpzGaussiansFromIterError::InvalidMixedShVariant)?;

        if positions.len() != header.num_points() {
            return Err(SpzGaussiansFromIterError::CountMismatch {
                actual_count: positions.len(),
                header_count: header.num_points(),
            });
        }

        if matches!(positions, SpzGaussiansPositions::Float16(_)) != header.uses_float16() {
            return Err(SpzGaussiansFromIterError::PositionFloat16Mismatch {
                is_float16: matches!(positions, SpzGaussiansPositions::Float16(_)),
                header_uses_float16: header.uses_float16(),
            });
        }

        if matches!(rotations, SpzGaussiansRotations::QuatSmallestThree(_))
            != header.uses_quat_smallest_three()
        {
            return Err(
                SpzGaussiansFromIterError::RotationQuatSmallestThreeMismatch {
                    is_quat_smallest_three: matches!(
                        rotations,
                        SpzGaussiansRotations::QuatSmallestThree(_)
                    ),
                    header_uses_quat_smallest_three: header.uses_quat_smallest_three(),
                },
            );
        }

        if !matches!(
            (&shs, header.sh_degree()),
            (SpzGaussiansShs::Zero, 0)
                | (SpzGaussiansShs::One(_), 1)
                | (SpzGaussiansShs::Two(_), 2)
                | (SpzGaussiansShs::Three(_), 3)
        ) {
            return Err(SpzGaussiansFromIterError::ShDegreeMismatch {
                sh_degree: match &shs {
                    SpzGaussiansShs::Zero => 0,
                    SpzGaussiansShs::One(_) => 1,
                    SpzGaussiansShs::Two(_) => 2,
                    SpzGaussiansShs::Three(_) => 3,
                },
                header_sh_degree: header.sh_degree(),
            });
        }

        Ok(SpzGaussians {
            header,
            positions,
            scales,
            rotations,
            alphas,
            colors,
            shs,
        })
    }

    /// Get an iterator over Gaussian references.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = SpzGaussianRef<'a>> + 'a {
        itertools::izip!(
            self.positions.iter(),
            self.scales.iter(),
            self.rotations.iter(),
            self.alphas.iter(),
            self.colors.iter(),
            self.shs.iter()
        )
        .map(
            |(position, scale, rotation, alpha, color, sh)| SpzGaussianRef {
                position,
                scale,
                rotation,
                alpha,
                color,
                sh,
            },
        )
    }
}

impl IterGaussian for SpzGaussians {
    fn iter_gaussian(&self) -> impl Iterator<Item = Gaussian> + '_ {
        self.iter()
            .map(|spz| Gaussian::from_spz(&spz, &self.header))
    }
}

impl From<&[Gaussian]> for SpzGaussians {
    fn from(gaussians: &[Gaussian]) -> Self {
        Self::from_gaussian_slice(gaussians)
    }
}

/// Options for [`SpzGaussians::from_gaussian_slice_with_options`].
///
/// The fields are not validated.
#[derive(Debug, Clone)]
pub struct SpzGaussiansFromGaussianSliceOptions {
    /// Version to use.
    pub version: u32,

    /// SH degree to use.
    pub sh_degree: u8,

    /// Number of fractional bits to use for position fixed point encoding.
    pub fractional_bits: usize,

    /// Whether to use antialiased encoding.
    pub antialiased: bool,

    /// The quantization bits for each SH degree.
    pub sh_quantize_bits: [u32; 3],
}

impl Default for SpzGaussiansFromGaussianSliceOptions {
    fn default() -> Self {
        let default_header = SpzGaussiansHeader::default(0, 3).expect("default header");
        let default_gaussian_to_spz_options = GaussianToSpzOptions::default();
        Self {
            version: default_header.version(),
            sh_degree: default_header.sh_degree(),
            fractional_bits: default_header.fractional_bits(),
            antialiased: default_header.is_antialiased(),
            sh_quantize_bits: default_gaussian_to_spz_options.sh_quantize_bits,
        }
    }
}

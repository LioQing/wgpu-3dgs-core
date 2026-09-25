use std::{
    io::{Read, Write},
    num::NonZeroUsize,
    ops::RangeInclusive,
};

use bytemuck::Zeroable;
use flate2::{read::GzDecoder, write::GzEncoder};
use itertools::Itertools;

use crate::{
    BatchProgress, BatchRead, BatchWrite, Gaussian, GaussianToSpzOptions, IterGaussian,
    ReadIterGaussian, SpzGaussiansFromIterError, SpzPhaseFromStrError, WriteIterGaussian,
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
            #[derive(Debug, Clone, PartialEq)]
            pub enum [< SpzGaussian $name >]  {
                $(
                    $(#[doc = $doc])?
                    $variant $(($ty))?,
                )+
            }

            #[doc = "Reference to SPZ Gaussian "]
            #[doc = $docname]
            #[doc = " field."]
            #[derive(Debug, Clone, Copy, PartialEq)]
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

                fn size_hint(&self) -> (usize, Option<usize>) {
                    match self {
                        $(
                            #[allow(clippy::redundant_pattern)]
                            [< SpzGaussian $name Iter >]:: $variant $( (iter @ noop!($ty _)) )? => {
                                #[allow(unused_variables)]
                                let len = 0;
                                $(
                                    noop!($ty);
                                    let len = iter.len();
                                )?
                                (len, Some(len))
                            }
                        )+
                    }
                }
            }

            impl<'a> ExactSizeIterator for [< SpzGaussian $name Iter >]<'a> {}

            #[doc = "Representation of SPZ Gaussians "]
            #[doc = $docname]
            #[doc = "s."]
            #[derive(Debug, Clone, PartialEq)]
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

                    #[allow(unused_variables)]
                    let first_value = ();
                    match first {
                        $(
                            #[allow(clippy::redundant_pattern)]
                            [< SpzGaussian $name >]:: $variant $( (first_value @ noop!($ty _)) )? => {
                                #[allow(unused_variables)]
                                let value = ();
                                #[allow(unused_variables)]
                                let vec = std::iter::once(Ok(first_value))
                                    .chain(
                                        iter.map(|v| {
                                            match v {
                                                [< SpzGaussian $name >]:: $variant $( (
                                                    value @ noop!($ty _)
                                                ) )? => Ok(value),
                                                other => Err(
                                                    $crate::error::SpzGaussiansCollectError::InvalidMixedVariant {
                                                        first_variant: [< SpzGaussian $name >]:: $variant $( (
                                                            { noop!($ty); first_value }
                                                        ) )?,
                                                        current_variant: other,
                                                    }
                                                ),
                                            }
                                        })
                                    )
                                    .collect::<Result<Vec<_>, _>>()?;
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
        One([[u8; 3]; 3]),
        Two([[u8; 3]; 8]),
        Three([[u8; 3]; 15]),
    }
}

impl SpzGaussianSh {
    /// Get the SH degree.
    pub fn degree(&self) -> SpzGaussianShDegree {
        match self {
            SpzGaussianSh::Zero => unsafe { SpzGaussianShDegree::new_unchecked(0) },
            SpzGaussianSh::One(_) => unsafe { SpzGaussianShDegree::new_unchecked(1) },
            SpzGaussianSh::Two(_) => unsafe { SpzGaussianShDegree::new_unchecked(2) },
            SpzGaussianSh::Three(_) => unsafe { SpzGaussianShDegree::new_unchecked(3) },
        }
    }

    /// Get an iterator over SH coefficients.
    pub fn iter(&self) -> impl Iterator<Item = &[u8; 3]> {
        match self {
            SpzGaussianSh::Zero => [].iter(),
            SpzGaussianSh::One(sh) => sh.iter(),
            SpzGaussianSh::Two(sh) => sh.iter(),
            SpzGaussianSh::Three(sh) => sh.iter(),
        }
    }

    /// Get an iterator over mutable SH coefficients.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [u8; 3]> {
        match self {
            SpzGaussianSh::Zero => [].iter_mut(),
            SpzGaussianSh::One(sh) => sh.iter_mut(),
            SpzGaussianSh::Two(sh) => sh.iter_mut(),
            SpzGaussianSh::Three(sh) => sh.iter_mut(),
        }
    }
}

impl SpzGaussianShRef<'_> {
    /// Get the SH degree.
    pub fn degree(&self) -> SpzGaussianShDegree {
        match self {
            SpzGaussianShRef::Zero => unsafe { SpzGaussianShDegree::new_unchecked(0) },
            SpzGaussianShRef::One(_) => unsafe { SpzGaussianShDegree::new_unchecked(1) },
            SpzGaussianShRef::Two(_) => unsafe { SpzGaussianShDegree::new_unchecked(2) },
            SpzGaussianShRef::Three(_) => unsafe { SpzGaussianShDegree::new_unchecked(3) },
        }
    }

    /// Get an iterator over SH coefficients.
    pub fn iter(&self) -> impl Iterator<Item = &[u8; 3]> + '_ {
        match self {
            SpzGaussianShRef::Zero => [].iter(),
            SpzGaussianShRef::One(sh) => sh.iter(),
            SpzGaussianShRef::Two(sh) => sh.iter(),
            SpzGaussianShRef::Three(sh) => sh.iter(),
        }
    }
}

impl SpzGaussiansShs {
    /// Get the SH degree.
    pub fn degree(&self) -> SpzGaussianShDegree {
        match self {
            SpzGaussiansShs::Zero => unsafe { SpzGaussianShDegree::new_unchecked(0) },
            SpzGaussiansShs::One(_) => unsafe { SpzGaussianShDegree::new_unchecked(1) },
            SpzGaussiansShs::Two(_) => unsafe { SpzGaussianShDegree::new_unchecked(2) },
            SpzGaussiansShs::Three(_) => unsafe { SpzGaussianShDegree::new_unchecked(3) },
        }
    }
}

/// The SPZ Gaussian spherical harmonics degrees.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpzGaussianShDegree(u8);

impl SpzGaussianShDegree {
    /// Create a new SPZ Gaussian SH degree.
    ///
    /// Returns [`None`] if the degree is not in the range of [`SpzGaussiansHeader::SUPPORTED_SH_DEGREES`].
    pub const fn new(sh_deg: u8) -> Option<Self> {
        match sh_deg {
            0..=3 => Some(Self(sh_deg)),
            _ => None,
        }
    }

    /// Create a new SPZ Gaussian SH degree without checking.
    ///
    /// # Safety
    ///
    /// The degree must be in the range of [`SpzGaussiansHeader::SUPPORTED_SH_DEGREES`].
    pub const unsafe fn new_unchecked(sh_deg: u8) -> Self {
        Self(sh_deg)
    }

    /// Get the degree.
    pub const fn get(&self) -> u8 {
        self.0
    }

    /// Get the number of SH coefficients.
    pub const fn num_coefficients(&self) -> usize {
        match self.0 {
            0 => 0,
            1 => 3,
            2 => 8,
            3 => 15,
            _ => unreachable!(),
        }
    }
}

impl Default for SpzGaussianShDegree {
    fn default() -> Self {
        // SAFETY: 3 is in the range of [0, 3].
        unsafe { Self::new_unchecked(3) }
    }
}

/// A single SPZ Gaussian.
///
/// This is usually only used for [`SpzGaussians::from_iter`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpzGaussian {
    pub position: SpzGaussianPosition,
    pub scale: [u8; 3],
    pub rotation: SpzGaussianRotation,
    pub alpha: u8,
    pub color: [u8; 3],
    pub sh: SpzGaussianSh,
}

impl SpzGaussian {
    /// Get a [`SpzGaussianRef`] reference to this Gaussian.
    pub fn as_ref(&self) -> SpzGaussianRef<'_> {
        SpzGaussianRef {
            position: match &self.position {
                SpzGaussianPosition::Float16(v) => SpzGaussianPositionRef::Float16(v),
                SpzGaussianPosition::FixedPoint24(v) => SpzGaussianPositionRef::FixedPoint24(v),
            },
            scale: &self.scale,
            rotation: match &self.rotation {
                SpzGaussianRotation::QuatFirstThree(v) => SpzGaussianRotationRef::QuatFirstThree(v),
                SpzGaussianRotation::QuatSmallestThree(v) => {
                    SpzGaussianRotationRef::QuatSmallestThree(v)
                }
            },
            alpha: &self.alpha,
            color: &self.color,
            sh: match &self.sh {
                SpzGaussianSh::Zero => SpzGaussianShRef::Zero,
                SpzGaussianSh::One(v) => SpzGaussianShRef::One(v),
                SpzGaussianSh::Two(v) => SpzGaussianShRef::Two(v),
                SpzGaussianSh::Three(v) => SpzGaussianShRef::Three(v),
            },
        }
    }
}

/// Reference to a SPZ Gaussian.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpzGaussianRef<'a> {
    pub position: SpzGaussianPositionRef<'a>,
    pub scale: &'a [u8; 3],
    pub rotation: SpzGaussianRotationRef<'a>,
    pub alpha: &'a u8,
    pub color: &'a [u8; 3],
    pub sh: SpzGaussianShRef<'a>,
}

impl SpzGaussianRef<'_> {
    /// Convert to [`SpzGaussian`].
    pub fn to_inner_owned(&self) -> SpzGaussian {
        SpzGaussian {
            position: match self.position {
                SpzGaussianPositionRef::Float16(v) => SpzGaussianPosition::Float16(*v),
                SpzGaussianPositionRef::FixedPoint24(v) => SpzGaussianPosition::FixedPoint24(*v),
            },
            scale: *self.scale,
            rotation: match self.rotation {
                SpzGaussianRotationRef::QuatFirstThree(v) => {
                    SpzGaussianRotation::QuatFirstThree(*v)
                }
                SpzGaussianRotationRef::QuatSmallestThree(v) => {
                    SpzGaussianRotation::QuatSmallestThree(*v)
                }
            },
            alpha: *self.alpha,
            color: *self.color,
            sh: match self.sh {
                SpzGaussianShRef::Zero => SpzGaussianSh::Zero,
                SpzGaussianShRef::One(v) => SpzGaussianSh::One(*v),
                SpzGaussianShRef::Two(v) => SpzGaussianSh::Two(*v),
                SpzGaussianShRef::Three(v) => SpzGaussianSh::Three(*v),
            },
        }
    }
}

/// Header of SPZ Gaussians file.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpzGaussiansHeaderPod {
    pub magic: u32,
    pub version: u32,
    pub num_points: u32,
    pub sh_degree: SpzGaussianShDegree,
    pub fractional_bits: u8,
    pub flags: u8,
    pub reserved: u8,
}

/// Header of SPZ Gaussians file.
///
/// This is the validated version of [`SpzGaussiansHeaderPod`]. This is simply a wrapper around
/// [`SpzGaussiansHeaderPod`] that ensures the values are valid, we could also implement
/// specialized structs for each field but it would be overkill for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        sh_degree: SpzGaussianShDegree,
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

        Ok(Self(pod))
    }

    /// Create a default [`SpzGaussiansHeader`] from number of points and SH degree.
    pub fn default(num_points: u32) -> Result<Self, std::io::Error> {
        Self::new(
            Self::SUPPORTED_VERSIONS
                .last()
                .expect("at least one supported version"),
            num_points,
            SpzGaussianShDegree::default(),
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

    /// Set the number of points.
    ///
    /// Setting the number of points does not invalidate the header.
    pub fn set_num_points(&mut self, num_points: u32) {
        self.0.num_points = num_points;
    }

    /// Get the number of points in the SPZ file.
    pub fn num_points(&self) -> usize {
        self.0.num_points as usize
    }

    /// Get the SH degree of the SPZ file.
    pub fn sh_degree(&self) -> SpzGaussianShDegree {
        self.0.sh_degree
    }

    /// Get the number of SH coefficients.
    pub fn sh_num_coefficients(&self) -> usize {
        self.0.sh_degree.num_coefficients()
    }

    /// Get the number of fractional bits.
    pub fn fractional_bits(&self) -> u8 {
        self.0.fractional_bits
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
        sh_degree: SpzGaussianShDegree,
    ) -> Result<Self, std::io::Error> {
        match sh_degree.get() {
            0 => Ok(SpzGaussiansShs::Zero),
            1 => {
                let mut sh_coeffs = vec![[[0u8; 3]; 3]; count];
                reader.read_exact(bytemuck::cast_slice_mut(&mut sh_coeffs))?;
                Ok(SpzGaussiansShs::One(sh_coeffs))
            }
            2 => {
                let mut sh_coeffs = vec![[[0u8; 3]; 8]; count];
                reader.read_exact(bytemuck::cast_slice_mut(&mut sh_coeffs))?;
                Ok(SpzGaussiansShs::Two(sh_coeffs))
            }
            3 => {
                let mut sh_coeffs = vec![[[0u8; 3]; 15]; count];
                reader.read_exact(bytemuck::cast_slice_mut(&mut sh_coeffs))?;
                Ok(SpzGaussiansShs::Three(sh_coeffs))
            }
            _ => {
                // SAFETY: SpzGaussianShDegree guarantees the degree is in [0, 3].
                unreachable!()
            }
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
#[derive(Debug, Clone, PartialEq)]
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
    /// Get the number of Gaussians.
    pub fn len(&self) -> usize {
        self.header.num_points()
    }

    /// Check if there are no Gaussians.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read a SPZ from a decompressed buffer.
    ///
    /// `reader` should be decompressed SPZ buffer.
    pub fn read_decompressed(reader: &mut impl Read) -> Result<Self, std::io::Error> {
        let header = Self::read_header(reader)?;
        Self::read_gaussians(reader, header)
    }

    /// Read a SPZ header.
    ///
    /// `reader` should be decompressed SPZ buffer.
    pub fn read_header(reader: &mut impl Read) -> Result<SpzGaussiansHeader, std::io::Error> {
        let mut header_bytes = [0u8; std::mem::size_of::<SpzGaussiansHeaderPod>()];
        reader.read_exact(&mut header_bytes)?;
        let header: SpzGaussiansHeaderPod = bytemuck::cast(header_bytes);
        SpzGaussiansHeader::try_from_pod(header)
    }

    /// Read the SPZ Gaussians.
    ///
    /// `reader` should be decompressed SPZ buffer positioned after the header.
    ///
    /// `header` may be parsed by calling [`SpzGaussians::read_header`].
    pub fn read_gaussians(
        reader: &mut impl Read,
        header: SpzGaussiansHeader,
    ) -> Result<Self, std::io::Error> {
        let count = header.num_points();
        let uses_float16 = header.uses_float16();
        let uses_quat_smallest_three = header.uses_quat_smallest_three();

        let positions = SpzGaussiansPositions::read_from(reader, count, uses_float16)?;

        let mut alphas = vec![0u8; count];
        reader.read_exact(bytemuck::cast_slice_mut(&mut alphas))?;

        let mut colors = vec![[0u8; 3]; count];
        reader.read_exact(bytemuck::cast_slice_mut(&mut colors))?;

        let mut scales = vec![[0u8; 3]; count];
        reader.read_exact(bytemuck::cast_slice_mut(&mut scales))?;

        let rotations = SpzGaussiansRotations::read_from(reader, count, uses_quat_smallest_three)?;

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
    /// `writer` will receive the decompressed SPZ buffer.
    pub fn write_decompressed(&self, writer: &mut impl Write) -> Result<(), std::io::Error> {
        writer.write_all(bytemuck::cast_slice(std::slice::from_ref(
            self.header.as_pod(),
        )))?;

        self.positions.write_to(writer)?;

        writer.write_all(bytemuck::cast_slice(&self.alphas))?;

        writer.write_all(bytemuck::cast_slice(&self.colors))?;

        writer.write_all(bytemuck::cast_slice(&self.scales))?;

        self.rotations.write_to(writer)?;

        self.shs.write_to(writer)?;

        Ok(())
    }

    /// Convert from a slice of [`Gaussian`]s.
    pub fn from_gaussians(gaussians: impl IntoIterator<Item = impl AsRef<Gaussian>>) -> Self {
        Self::from_gaussians_with_options(
            gaussians,
            &SpzGaussiansFromGaussianSliceOptions::default(),
        )
        .expect("valid default options")
    }

    /// Convert from a slice of [`Gaussian`]s with options.
    pub fn from_gaussians_with_options(
        gaussians: impl IntoIterator<Item = impl AsRef<Gaussian>>,
        options: &SpzGaussiansFromGaussianSliceOptions,
    ) -> Result<Self, std::io::Error> {
        let mut header = SpzGaussiansHeader::new(
            options.version,
            0,
            options.sh_degree,
            options.fractional_bits,
            options.antialiased,
        )?;

        let gaussians = gaussians
            .into_iter()
            .map(|g| {
                g.as_ref().to_spz(
                    &header,
                    &GaussianToSpzOptions {
                        sh_quantize_bits: options.sh_quantize_bits,
                    },
                )
            })
            .collect::<Vec<_>>();

        header.set_num_points(gaussians.len() as u32);

        Ok(Self::from_iter(header, gaussians)
            .expect("gaussians from valid Gaussians with valid header are valid"))
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

        if shs.degree() != header.sh_degree() {
            return Err(SpzGaussiansFromIterError::ShDegreeMismatch {
                sh_degree: shs.degree(),
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
    pub fn iter<'a>(&'a self) -> impl ExactSizeIterator<Item = SpzGaussianRef<'a>> + 'a {
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
    fn iter_gaussian(&self) -> impl ExactSizeIterator<Item = Gaussian> + '_ {
        self.iter().map(|spz| Gaussian::from_spz(spz, &self.header))
    }
}

impl ReadIterGaussian for SpzGaussians {
    fn read_from(reader: &mut impl std::io::BufRead) -> std::io::Result<Self> {
        let mut reader = SpzBatchReader::new(reader)?;
        while !BatchRead::progress(&reader).done {
            reader.step(NonZeroUsize::new(4096).unwrap())?;
        }
        reader.finish()
    }
}

impl WriteIterGaussian for SpzGaussians {
    fn write_to(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        let mut writer = SpzBatchWriter::new(writer, self)?;
        while !BatchWrite::progress(&writer).done {
            writer.step(NonZeroUsize::new(4096).unwrap())?;
        }
        writer.finish()?;
        Ok(())
    }
}

/// Fields appear consecutively in the decompressed SPZ stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpzPhase {
    Positions,
    Alphas,
    Colors,
    Scales,
    Rotations,
    Shs,
    Done,
}

impl SpzPhase {
    fn next(self) -> Self {
        match self {
            Self::Positions => Self::Alphas,
            Self::Alphas => Self::Colors,
            Self::Colors => Self::Scales,
            Self::Scales => Self::Rotations,
            Self::Rotations => Self::Shs,
            Self::Shs | Self::Done => Self::Done,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Positions => "positions",
            Self::Alphas => "alphas",
            Self::Colors => "colors",
            Self::Scales => "scales",
            Self::Rotations => "rotations",
            Self::Shs => "shs",
            Self::Done => "done",
        }
    }

    fn ordinal(self) -> usize {
        match self {
            Self::Positions => 0,
            Self::Alphas => 1,
            Self::Colors => 2,
            Self::Scales => 3,
            Self::Rotations => 4,
            Self::Shs => 5,
            Self::Done => 6,
        }
    }
}

impl std::str::FromStr for SpzPhase {
    type Err = SpzPhaseFromStrError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "positions" => Ok(Self::Positions),
            "alphas" => Ok(Self::Alphas),
            "colors" => Ok(Self::Colors),
            "scales" => Ok(Self::Scales),
            "rotations" => Ok(Self::Rotations),
            "shs" => Ok(Self::Shs),
            "done" => Ok(Self::Done),
            _ => Err(SpzPhaseFromStrError::InvalidSpzPhase(s.to_string())),
        }
    }
}

fn spz_progress(header: &SpzGaussiansHeader, phase: SpzPhase, completed: usize) -> BatchProgress {
    let count = header.num_points();
    let has_sh = header.sh_num_coefficients() != 0;
    let total_units = count.saturating_mul(5 + usize::from(has_sh));
    let prior = count.saturating_mul(phase.ordinal().min(5));
    let completed_units = if phase == SpzPhase::Done {
        total_units
    } else {
        prior.saturating_add(completed)
    };
    BatchProgress {
        phase: phase.name(),
        completed_in_phase: completed,
        total_in_phase: if phase == SpzPhase::Done { 0 } else { count },
        completed_units,
        total_units,
        done: phase == SpzPhase::Done,
    }
}

fn advance_spz_phase(phase: &mut SpzPhase, completed: &mut usize, header: &SpzGaussiansHeader) {
    if *completed == header.num_points() {
        *phase = phase.next();
        *completed = 0;
    }
    if *phase == SpzPhase::Shs && header.sh_num_coefficients() == 0 {
        *phase = SpzPhase::Done;
    }
    if header.num_points() == 0 {
        *phase = SpzPhase::Done;
    }
}

fn read_extend<T: bytemuck::Pod + Zeroable>(
    reader: &mut impl Read,
    values: &mut Vec<T>,
    count: usize,
) -> std::io::Result<()> {
    let start = values.len();
    values.resize(start + count, T::zeroed());
    reader.read_exact(bytemuck::cast_slice_mut(&mut values[start..]))
}

/// Bounded-work reader for a gzip-compressed SPZ model.
///
/// SPZ stores field arrays rather than consecutive Gaussians, so individual Gaussians cannot
/// be delivered early. After an I/O error, discard the reader.
pub struct SpzBatchReader<R: Read> {
    decoder: GzDecoder<R>,
    gaussians: SpzGaussians,
    phase: SpzPhase,
    completed: usize,
}

impl<R: Read> SpzBatchReader<R> {
    pub fn new(reader: R) -> std::io::Result<Self> {
        let mut decoder = GzDecoder::new(reader);
        let header = SpzGaussians::read_header(&mut decoder)?;
        let mut phase = SpzPhase::Positions;
        let mut completed = 0;

        advance_spz_phase(&mut phase, &mut completed, &header);

        Ok(Self {
            decoder,
            gaussians: SpzGaussians {
                header,
                positions: if header.uses_float16() {
                    SpzGaussiansPositions::Float16(Vec::new())
                } else {
                    SpzGaussiansPositions::FixedPoint24(Vec::new())
                },
                alphas: Vec::new(),
                colors: Vec::new(),
                scales: Vec::new(),
                rotations: if header.uses_quat_smallest_three() {
                    SpzGaussiansRotations::QuatSmallestThree(Vec::new())
                } else {
                    SpzGaussiansRotations::QuatFirstThree(Vec::new())
                },
                shs: match header.sh_degree().get() {
                    0 => SpzGaussiansShs::Zero,
                    1 => SpzGaussiansShs::One(Vec::new()),
                    2 => SpzGaussiansShs::Two(Vec::new()),
                    3 => SpzGaussiansShs::Three(Vec::new()),
                    _ => unreachable!(),
                },
            },
            phase,
            completed,
        })
    }

    pub fn header(&self) -> &SpzGaussiansHeader {
        &self.gaussians.header
    }
    pub fn phase(&self) -> SpzPhase {
        self.phase
    }
}

impl<R: Read> BatchRead for SpzBatchReader<R> {
    type Model = SpzGaussians;

    fn progress(&self) -> BatchProgress {
        spz_progress(&self.gaussians.header, self.phase, self.completed)
    }

    fn step(&mut self, max_items: NonZeroUsize) -> std::io::Result<BatchProgress> {
        if self.phase == SpzPhase::Done {
            return Ok(self.progress());
        }

        let count = max_items.get().min(self.gaussians.len() - self.completed);
        let decoder = &mut self.decoder;

        match self.phase {
            SpzPhase::Positions => match &mut self.gaussians.positions {
                SpzGaussiansPositions::Float16(v) => read_extend(decoder, v, count)?,
                SpzGaussiansPositions::FixedPoint24(v) => read_extend(decoder, v, count)?,
            },
            SpzPhase::Alphas => read_extend(decoder, &mut self.gaussians.alphas, count)?,
            SpzPhase::Colors => read_extend(decoder, &mut self.gaussians.colors, count)?,
            SpzPhase::Scales => read_extend(decoder, &mut self.gaussians.scales, count)?,
            SpzPhase::Rotations => match &mut self.gaussians.rotations {
                SpzGaussiansRotations::QuatFirstThree(v) => read_extend(decoder, v, count)?,
                SpzGaussiansRotations::QuatSmallestThree(v) => read_extend(decoder, v, count)?,
            },
            SpzPhase::Shs => match &mut self.gaussians.shs {
                SpzGaussiansShs::Zero => unreachable!(),
                SpzGaussiansShs::One(v) => read_extend(decoder, v, count)?,
                SpzGaussiansShs::Two(v) => read_extend(decoder, v, count)?,
                SpzGaussiansShs::Three(v) => read_extend(decoder, v, count)?,
            },
            SpzPhase::Done => unreachable!(),
        }

        self.completed += count;
        advance_spz_phase(&mut self.phase, &mut self.completed, &self.gaussians.header);
        Ok(self.progress())
    }

    fn finish(mut self) -> std::io::Result<Self::Model> {
        if !self.progress().done {
            return Err(crate::source_format::batch::incomplete());
        }
        // Force the decoder to consume the gzip trailer (and validate its checksum).
        let mut extra = [0u8; 1];
        if self.decoder.read(&mut extra)? != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "extra SPZ data",
            ));
        }
        Ok(self.gaussians)
    }
}

/// Bounded-work writer for an existing SPZ model. After an I/O error, discard the writer.
pub struct SpzBatchWriter<'a, W: Write> {
    encoder: GzEncoder<W>,
    gaussians: &'a SpzGaussians,
    phase: SpzPhase,
    completed: usize,
}

impl<'a, W: Write> SpzBatchWriter<'a, W> {
    pub fn new(writer: W, gaussians: &'a SpzGaussians) -> std::io::Result<Self> {
        let count = gaussians.len();
        if gaussians.positions.len() != count
            || gaussians.rotations.len() != count
            || gaussians.alphas.len() != count
            || gaussians.colors.len() != count
            || gaussians.scales.len() != count
            || (gaussians.header.sh_num_coefficients() != 0 && gaussians.shs.len() != count)
            || matches!(&gaussians.positions, SpzGaussiansPositions::Float16(_))
                != gaussians.header.uses_float16()
            || matches!(
                &gaussians.rotations,
                SpzGaussiansRotations::QuatSmallestThree(_)
            ) != gaussians.header.uses_quat_smallest_three()
            || gaussians.shs.degree() != gaussians.header.sh_degree()
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "SPZ field lengths or variants do not match header",
            ));
        }

        let mut encoder = GzEncoder::new(writer, flate2::Compression::default());
        encoder.write_all(bytemuck::bytes_of(gaussians.header.as_pod()))?;

        let mut phase = SpzPhase::Positions;
        let mut completed = 0;

        advance_spz_phase(&mut phase, &mut completed, &gaussians.header);

        Ok(Self {
            encoder,
            gaussians,
            phase,
            completed,
        })
    }

    pub fn phase(&self) -> SpzPhase {
        self.phase
    }
}

impl<W: Write> BatchWrite for SpzBatchWriter<'_, W> {
    type Writer = W;

    fn progress(&self) -> BatchProgress {
        spz_progress(&self.gaussians.header, self.phase, self.completed)
    }

    fn step(&mut self, max_items: NonZeroUsize) -> std::io::Result<BatchProgress> {
        if self.phase == SpzPhase::Done {
            return Ok(self.progress());
        }

        let end = self.completed + max_items.get().min(self.gaussians.len() - self.completed);
        let start = self.completed;
        let encoder = &mut self.encoder;

        match self.phase {
            SpzPhase::Positions => match &self.gaussians.positions {
                SpzGaussiansPositions::Float16(v) => {
                    encoder.write_all(bytemuck::cast_slice(&v[start..end]))?
                }
                SpzGaussiansPositions::FixedPoint24(v) => {
                    encoder.write_all(bytemuck::cast_slice(&v[start..end]))?
                }
            },
            SpzPhase::Alphas => encoder.write_all(&self.gaussians.alphas[start..end])?,
            SpzPhase::Colors => {
                encoder.write_all(bytemuck::cast_slice(&self.gaussians.colors[start..end]))?
            }
            SpzPhase::Scales => {
                encoder.write_all(bytemuck::cast_slice(&self.gaussians.scales[start..end]))?
            }
            SpzPhase::Rotations => match &self.gaussians.rotations {
                SpzGaussiansRotations::QuatFirstThree(v) => {
                    encoder.write_all(bytemuck::cast_slice(&v[start..end]))?
                }
                SpzGaussiansRotations::QuatSmallestThree(v) => {
                    encoder.write_all(bytemuck::cast_slice(&v[start..end]))?
                }
            },
            SpzPhase::Shs => match &self.gaussians.shs {
                SpzGaussiansShs::Zero => unreachable!(),
                SpzGaussiansShs::One(v) => {
                    encoder.write_all(bytemuck::cast_slice(&v[start..end]))?
                }
                SpzGaussiansShs::Two(v) => {
                    encoder.write_all(bytemuck::cast_slice(&v[start..end]))?
                }
                SpzGaussiansShs::Three(v) => {
                    encoder.write_all(bytemuck::cast_slice(&v[start..end]))?
                }
            },
            SpzPhase::Done => unreachable!(),
        }

        self.completed = end;
        advance_spz_phase(&mut self.phase, &mut self.completed, &self.gaussians.header);
        Ok(self.progress())
    }

    fn finish(self) -> std::io::Result<W> {
        if !self.progress().done {
            return Err(crate::source_format::batch::incomplete());
        }
        self.encoder.finish()
    }
}

impl<G: AsRef<Gaussian>> FromIterator<G> for SpzGaussians {
    fn from_iter<T: IntoIterator<Item = G>>(iter: T) -> Self {
        Self::from_gaussians(iter)
    }
}

/// Options for [`SpzGaussians::from_gaussians_with_options`].
///
/// The fields are not validated.
#[derive(Debug, Clone)]
pub struct SpzGaussiansFromGaussianSliceOptions {
    /// Version to use.
    pub version: u32,

    /// SH degree to use.
    pub sh_degree: SpzGaussianShDegree,

    /// Number of fractional bits to use for position fixed point encoding.
    pub fractional_bits: u8,

    /// Whether to use antialiased encoding.
    pub antialiased: bool,

    /// The quantization bits for each SH degree.
    pub sh_quantize_bits: [u32; 3],
}

impl Default for SpzGaussiansFromGaussianSliceOptions {
    fn default() -> Self {
        let default_header = SpzGaussiansHeader::default(0).expect("default header");
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

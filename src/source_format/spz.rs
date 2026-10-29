use std::{
    io::{self, Read, Take, Write},
    num::NonZeroUsize,
    ops::{Range, RangeInclusive},
};

use bytemuck::Zeroable;
use flate2::{read::GzDecoder, write::GzEncoder};
use itertools::Itertools;
use ruzstd::decoding::{FrameDecoder, StreamingDecoder};
use structured_zstd::encoding::{CompressionLevel, StreamingEncoder};

use crate::{
    AnyGaussian, BatchProgress, BatchRead, BatchWrite, GaussianShDegree, GaussianToSpzError,
    GaussianToSpzOptions, IterGaussian, ReadIterGaussian, ShDegreeMismatchError,
    SpzGaussiansFromGaussiansError, SpzGaussiansFromIterError, SpzPhaseFromStrError, ToAnyGaussian,
    WriteIterGaussian, source_format,
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
                // Preserve the public by-value diagnostics, including inline SH4 fields.
                #[allow(clippy::result_large_err)]
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
        Four([[u8; 3]; 24]),
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
            SpzGaussianSh::Four(_) => unsafe { SpzGaussianShDegree::new_unchecked(4) },
        }
    }

    /// Get an iterator over SH coefficients.
    pub fn iter(&self) -> impl Iterator<Item = &[u8; 3]> {
        match self {
            SpzGaussianSh::Zero => [].iter(),
            SpzGaussianSh::One(sh) => sh.iter(),
            SpzGaussianSh::Two(sh) => sh.iter(),
            SpzGaussianSh::Three(sh) => sh.iter(),
            SpzGaussianSh::Four(sh) => sh.iter(),
        }
    }

    /// Get an iterator over mutable SH coefficients.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [u8; 3]> {
        match self {
            SpzGaussianSh::Zero => [].iter_mut(),
            SpzGaussianSh::One(sh) => sh.iter_mut(),
            SpzGaussianSh::Two(sh) => sh.iter_mut(),
            SpzGaussianSh::Three(sh) => sh.iter_mut(),
            SpzGaussianSh::Four(sh) => sh.iter_mut(),
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
            SpzGaussianShRef::Four(_) => unsafe { SpzGaussianShDegree::new_unchecked(4) },
        }
    }

    /// Get an iterator over SH coefficients.
    pub fn iter(&self) -> impl Iterator<Item = &[u8; 3]> + '_ {
        match self {
            SpzGaussianShRef::Zero => [].iter(),
            SpzGaussianShRef::One(sh) => sh.iter(),
            SpzGaussianShRef::Two(sh) => sh.iter(),
            SpzGaussianShRef::Three(sh) => sh.iter(),
            SpzGaussianShRef::Four(sh) => sh.iter(),
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
            SpzGaussiansShs::Four(_) => unsafe { SpzGaussianShDegree::new_unchecked(4) },
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
    /// Returns [`None`] if the degree is not in the range of [`SpzGaussiansHeader::SUPPORTED_READ_SH_DEGREES`].
    pub const fn new(sh_deg: u8) -> Option<Self> {
        match sh_deg {
            0..=4 => Some(Self(sh_deg)),
            _ => None,
        }
    }

    /// Create a new SPZ Gaussian SH degree without checking.
    ///
    /// # Safety
    ///
    /// The degree must be in the range of [`SpzGaussiansHeader::SUPPORTED_READ_SH_DEGREES`].
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
            4 => 24,
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
                SpzGaussianSh::Four(v) => SpzGaussianShRef::Four(v),
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
                SpzGaussianShRef::Four(v) => SpzGaussianSh::Four(*v),
            },
        }
    }
}

/// Legacy 16-byte SPZ wire header, also used as common in-memory metadata.
///
/// For version 4 this is not the on-disk header. Use [`SpzGaussians::read_from`] and
/// [`SpzGaussians::write_to`] for v4 containers, whose 32-byte wire header is handled separately.
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

    /// The SPZ versions supported by constructors and writers.
    pub const SUPPORTED_VERSIONS: RangeInclusive<u32> = 1..=4;

    /// The SPZ versions supported by file readers.
    pub const SUPPORTED_READ_VERSIONS: RangeInclusive<u32> = 1..=4;

    /// The SH degrees supported by constructors and writers (degree 4 requires version 4).
    pub const SUPPORTED_SH_DEGREES: RangeInclusive<u8> = 0..=4;

    /// The SH degrees supported by readers (degree 4 requires a v4 container).
    pub const SUPPORTED_READ_SH_DEGREES: RangeInclusive<u8> = 0..=4;

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

        let supported_sh_degrees = if pod.version == 4 {
            Self::SUPPORTED_SH_DEGREES
        } else {
            0..=3
        };

        if !supported_sh_degrees.contains(&pod.sh_degree.get()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Unsupported SPZ SH degree: {}, expected one of {:?}",
                    pod.sh_degree.get(),
                    supported_sh_degrees
                ),
            ));
        }

        if pod.version == 4
            && (pod.fractional_bits > 30 || pod.flags & !0x3 != 0 || pod.reserved != 0)
        {
            return Err(invalid_data(
                "invalid SPZ v4 fractional bits, flags or reserved byte",
            ));
        }

        Ok(Self(pod))
    }

    /// Create a default version-3 [`SpzGaussiansHeader`] from number of points.
    pub fn default(num_points: u32) -> Result<Self, std::io::Error> {
        Self::new(3, num_points, SpzGaussianShDegree::default(), 12, false)
    }

    /// Get common metadata as a [`SpzGaussiansHeaderPod`].
    ///
    /// This is a wire header only for versions 1 through 3.
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

    fn ensure_legacy_layout(&self) -> std::io::Result<()> {
        if self.version() == 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "SPZ v4 does not use the legacy decompressed layout",
            ));
        }
        Ok(())
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
            4 => {
                let mut sh_coeffs = vec![[[0u8; 3]; 24]; count];
                reader.read_exact(bytemuck::cast_slice_mut(&mut sh_coeffs))?;
                Ok(SpzGaussiansShs::Four(sh_coeffs))
            }
            _ => {
                // SAFETY: SpzGaussianShDegree guarantees the degree is in [0, 4].
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
            SpzGaussiansShs::Four(sh_coeffs) => writer.write_all(bytemuck::cast_slice(sh_coeffs)),
        }
    }
}

/// A collection of Gaussians in SPZ format.
#[derive(Debug, Clone, PartialEq)]
pub struct SpzGaussians {
    /// Changing this to a new header with [`SpzGaussiansHeader::sh_degree`] not matching
    /// [`SpzGaussians::shs`] will cause [`IterGaussian::iter_gaussian`] to panic.
    pub header: SpzGaussiansHeader,

    pub positions: SpzGaussiansPositions,

    /// `(x, y, z)` each as 8-bit log-encoded integer.
    pub scales: Vec<[u8; 3]>,

    pub rotations: SpzGaussiansRotations,

    /// 8-bit unsigned integer.
    pub alphas: Vec<u8>,

    /// `(r, g, b)` each as 8-bit unsigned integer.
    pub colors: Vec<[u8; 3]>,

    /// Changing this to a new variant not matching [`SpzGaussiansHeader::sh_degree`] will cause
    /// [`IterGaussian::iter_gaussian`] to panic.
    pub shs: SpzGaussiansShs,
}

impl SpzGaussians {
    fn empty(header: SpzGaussiansHeader) -> Self {
        Self {
            header,
            positions: if header.uses_float16() {
                SpzGaussiansPositions::Float16(Vec::new())
            } else {
                SpzGaussiansPositions::FixedPoint24(Vec::new())
            },
            scales: Vec::new(),
            rotations: if header.uses_quat_smallest_three() {
                SpzGaussiansRotations::QuatSmallestThree(Vec::new())
            } else {
                SpzGaussiansRotations::QuatFirstThree(Vec::new())
            },
            alphas: Vec::new(),
            colors: Vec::new(),
            shs: match header.sh_degree().get() {
                0 => SpzGaussiansShs::Zero,
                1 => SpzGaussiansShs::One(Vec::new()),
                2 => SpzGaussiansShs::Two(Vec::new()),
                3 => SpzGaussiansShs::Three(Vec::new()),
                4 => SpzGaussiansShs::Four(Vec::new()),
                _ => unreachable!(),
            },
        }
    }

    /// Get the number of Gaussians.
    pub fn len(&self) -> usize {
        self.header.num_points()
    }

    /// Check if there are no Gaussians.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read a legacy (v1-3) SPZ from a decompressed buffer.
    ///
    /// `reader` should be decompressed SPZ buffer.
    pub fn read_decompressed(reader: &mut impl Read) -> Result<Self, std::io::Error> {
        let header = Self::read_header(reader)?;
        Self::read_gaussians(reader, header)
    }

    /// Read a legacy (v1-3) SPZ header.
    ///
    /// `reader` should be decompressed SPZ buffer.
    pub fn read_header(reader: &mut impl Read) -> Result<SpzGaussiansHeader, std::io::Error> {
        let mut header_bytes = [0u8; std::mem::size_of::<SpzGaussiansHeaderPod>()];
        reader.read_exact(&mut header_bytes)?;

        let header = SpzGaussiansHeaderPod {
            magic: u32::from_le_bytes(header_bytes[0..4].try_into().unwrap()),
            version: u32::from_le_bytes(header_bytes[4..8].try_into().unwrap()),
            num_points: u32::from_le_bytes(header_bytes[8..12].try_into().unwrap()),
            sh_degree: SpzGaussianShDegree(header_bytes[12]),
            fractional_bits: header_bytes[13],
            flags: header_bytes[14],
            reserved: header_bytes[15],
        };

        let header = SpzGaussiansHeader::try_from_pod(header)?;
        header.ensure_legacy_layout()?;
        Ok(header)
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
        header.ensure_legacy_layout()?;

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

    /// Write the Gaussians to a legacy (v1-3) SPZ buffer.
    ///
    /// `writer` will receive the decompressed SPZ buffer.
    pub fn write_decompressed(&self, writer: &mut impl Write) -> Result<(), std::io::Error> {
        self.header.ensure_legacy_layout()?;

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

    /// Convert from an iterator of [`ToAnyGaussian`]s using version 3 and SH degrees 0 to 3.
    ///
    /// Use [`Self::from_gaussians_with_options`] with version 4 for SH degree 4.
    pub fn from_gaussians<T: ToAnyGaussian>(
        gaussians: impl IntoIterator<Item = T>,
    ) -> Result<Self, SpzGaussiansFromGaussiansError> {
        let mut gaussians = gaussians.into_iter().peekable();
        let degree = gaussians
            .peek()
            .map(|g| g.to_any_gaussian().sh_degree().get())
            .unwrap_or(T::SH_DEGREE.unwrap_or(3));

        if degree > 3 {
            return Err(SpzGaussiansFromGaussiansError::UnsupportedShDegree { degree });
        }

        let sh_degree = SpzGaussianShDegree::new(degree)
            .ok_or(SpzGaussiansFromGaussiansError::UnsupportedShDegree { degree })?;

        Self::from_gaussians_with_options(
            gaussians,
            &SpzGaussiansFromGaussianSliceOptions {
                sh_degree,
                ..Default::default()
            },
        )
    }

    /// Convert from an iterator of [`ToAnyGaussian`]s with options.
    pub fn from_gaussians_with_options<T: ToAnyGaussian>(
        gaussians: impl IntoIterator<Item = T>,
        options: &SpzGaussiansFromGaussianSliceOptions,
    ) -> Result<Self, SpzGaussiansFromGaussiansError> {
        if let Some(actual_degree) = T::SH_DEGREE
            && actual_degree != options.sh_degree.get()
        {
            return Err(ShDegreeMismatchError {
                actual_degree,
                expected_degree: options.sh_degree.get(),
            }
            .into());
        }

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
                g.to_any_gaussian().to_spz(
                    &header,
                    &GaussianToSpzOptions {
                        sh_quantize_bits: options.sh_quantize_bits,
                    },
                )
            })
            .collect::<Result<Vec<_>, GaussianToSpzError>>()?;

        header.set_num_points(gaussians.len() as u32);

        Ok(Self::from_iter(header, gaussians)
            .expect("gaussians from valid Gaussians with valid header are valid"))
    }

    /// Convert from an [`IntoIterator`] of [`SpzGaussian`]s.
    ///
    /// An empty iterator produces empty fields with variants determined by the header,
    /// provided the header's point count is zero.
    #[allow(clippy::result_large_err)] // Preserve existing by-value diagnostics for SH4.
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

        if positions.is_empty() {
            return if header.num_points() == 0 {
                Ok(Self::empty(header))
            } else {
                Err(SpzGaussiansFromIterError::CountMismatch {
                    actual_count: 0,
                    header_count: header.num_points(),
                })
            };
        }

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

    fn field_bytes(&self, phase: SpzPhase, range: Range<usize>) -> &[u8] {
        match phase {
            SpzPhase::Positions => match &self.positions {
                SpzGaussiansPositions::Float16(v) => bytemuck::cast_slice(&v[range]),
                SpzGaussiansPositions::FixedPoint24(v) => bytemuck::cast_slice(&v[range]),
            },
            SpzPhase::Alphas => &self.alphas[range],
            SpzPhase::Colors => bytemuck::cast_slice(&self.colors[range]),
            SpzPhase::Scales => bytemuck::cast_slice(&self.scales[range]),
            SpzPhase::Rotations => match &self.rotations {
                SpzGaussiansRotations::QuatFirstThree(v) => bytemuck::cast_slice(&v[range]),
                SpzGaussiansRotations::QuatSmallestThree(v) => bytemuck::cast_slice(&v[range]),
            },
            SpzPhase::Shs => match &self.shs {
                SpzGaussiansShs::Zero => &[],
                SpzGaussiansShs::One(v) => bytemuck::cast_slice(&v[range]),
                SpzGaussiansShs::Two(v) => bytemuck::cast_slice(&v[range]),
                SpzGaussiansShs::Three(v) => bytemuck::cast_slice(&v[range]),
                SpzGaussiansShs::Four(v) => bytemuck::cast_slice(&v[range]),
            },
            SpzPhase::Done => &[],
        }
    }
}

impl IterGaussian for SpzGaussians {
    type Gaussian = AnyGaussian;

    fn sh_degree(&self) -> GaussianShDegree {
        GaussianShDegree::new(self.header.sh_degree().get()).unwrap()
    }

    fn iter_gaussian(&self) -> impl ExactSizeIterator<Item = AnyGaussian> + '_ {
        self.iter()
            .map(|spz| AnyGaussian::from_spz(spz, &self.header).expect("validated SPZ degree"))
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
    /// Get the next phase after the current one.
    pub fn next(self) -> Self {
        match self {
            Self::Positions => Self::Alphas,
            Self::Alphas => Self::Colors,
            Self::Colors => Self::Scales,
            Self::Scales => Self::Rotations,
            Self::Rotations => Self::Shs,
            Self::Shs | Self::Done => Self::Done,
        }
    }

    /// Get the phase name used in progress reports.
    pub fn name(self) -> &'static str {
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

    fn advance_spz_phase(&mut self, completed: &mut usize, header: &SpzGaussiansHeader) {
        if *completed == header.num_points() {
            *self = self.next();
            *completed = 0;
        }
        if *self == SpzPhase::Shs && header.sh_num_coefficients() == 0 {
            *self = SpzPhase::Done;
        }
        if header.num_points() == 0 {
            *self = SpzPhase::Done;
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

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[derive(Clone, Copy)]
struct StreamInfo {
    compressed_size: u64,
    uncompressed_size: u64,
}

/// SPZ v4 container reader, following nianticlabs/spz v3.0.0.
///
/// The six logical attributes retain the legacy field order. Empty attributes
/// have no TOC entry or frame. Decoding is sequential and needs no Seek support.
struct V4Reader<R: Read> {
    source: Option<R>,
    decoder: Option<StreamingDecoder<Take<R>, FrameDecoder>>,
    streams: Vec<StreamInfo>,
    next_stream: usize,
    remaining: u64,
    finished: bool,
}

impl<R: Read> V4Reader<R> {
    fn new(mut source: R) -> io::Result<(Self, SpzGaussiansHeader)> {
        let mut bytes = [0; 32];
        source.read_exact(&mut bytes)?;

        let u32_at = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());

        if u32_at(0) != SpzGaussiansHeader::MAGIC || u32_at(4) != 4 {
            return Err(invalid_data("unsupported SPZ plaintext header version"));
        }

        let sh_degree = SpzGaussianShDegree::new(bytes[12])
            .ok_or_else(|| invalid_data("unsupported SPZ v4 SH degree"))?;

        // All metadata is normalized into the existing 16-byte in-memory type.
        // It must never be serialized as a legacy header for version 4.
        let header = SpzGaussiansHeader(SpzGaussiansHeaderPod {
            magic: SpzGaussiansHeader::MAGIC,
            version: 4,
            num_points: u32_at(8),
            sh_degree,
            fractional_bits: bytes[13],
            flags: bytes[14],
            reserved: 0,
        });

        // Conversion uses a signed 32-bit shift when unpacking fixed-point data.
        if header.fractional_bits() > 30 {
            return Err(invalid_data("SPZ v4 fractional bits must not exceed 30"));
        }

        if bytes[14] & !0x3 != 0 || bytes[20..].iter().any(|&byte| byte != 0) {
            return Err(invalid_data("unsupported SPZ v4 flags or reserved bytes"));
        }

        let toc_offset = u32_at(16);

        if toc_offset < 32 {
            return Err(invalid_data("SPZ v4 TOC overlaps the header"));
        }

        let extension_size = u64::from(toc_offset - 32);

        if bytes[14] & 0x2 == 0 && extension_size != 0 {
            return Err(invalid_data("SPZ v4 extension zone without extension flag"));
        }

        Self::skip_extensions(&mut source, extension_size)?;

        let count = header.num_points();
        let field_widths = [9, 1, 3, 3, 4, header.sh_num_coefficients() * 3];
        let mut expected_sizes = Vec::new();
        let mut total_size = 0usize;

        for width in field_widths {
            let size = count
                .checked_mul(width)
                .filter(|&size| size <= isize::MAX as usize)
                .ok_or_else(|| invalid_data("SPZ v4 attribute size overflow"))?;

            total_size = total_size
                .checked_add(size)
                .ok_or_else(|| invalid_data("SPZ v4 decoded size overflow"))?;

            if size != 0 {
                expected_sizes.push(size as u64);
            }
        }

        if usize::from(bytes[15]) != expected_sizes.len() {
            return Err(invalid_data(
                "SPZ v4 stream count does not match attributes",
            ));
        }

        let mut streams = Vec::with_capacity(expected_sizes.len());
        let mut compressed_end = u64::from(toc_offset) + expected_sizes.len() as u64 * 16;

        for expected in expected_sizes {
            let mut entry = [0; 16];
            source.read_exact(&mut entry)?;

            let compressed_size = u64::from_le_bytes(entry[..8].try_into().unwrap());
            let uncompressed_size = u64::from_le_bytes(entry[8..].try_into().unwrap());

            if compressed_size == 0 || uncompressed_size != expected {
                return Err(invalid_data("SPZ v4 TOC attribute size mismatch"));
            }

            compressed_end = compressed_end
                .checked_add(compressed_size)
                .ok_or_else(|| invalid_data("SPZ v4 compressed size overflow"))?;

            streams.push(StreamInfo {
                compressed_size,
                uncompressed_size,
            });
        }

        Ok((
            Self {
                source: Some(source),
                decoder: None,
                streams,
                next_stream: 0,
                remaining: 0,
                finished: false,
            },
            header,
        ))
    }

    fn skip_extensions(source: &mut R, size: u64) -> io::Result<()> {
        let mut zone = source.take(size);
        let mut buffer = [0; 8192];

        while zone.limit() != 0 {
            if zone.limit() < 8 {
                return Err(invalid_data("truncated SPZ v4 extension record"));
            }

            let mut record = [0; 8];
            zone.read_exact(&mut record)?;

            let length = u64::from(u32::from_le_bytes(record[4..].try_into().unwrap()));

            if length > zone.limit() {
                return Err(invalid_data("SPZ v4 extension exceeds header zone"));
            }

            let mut remaining = length;

            while remaining != 0 {
                let count = remaining.min(buffer.len() as u64) as usize;
                zone.read_exact(&mut buffer[..count])?;
                remaining -= count as u64;
            }
        }

        Ok(())
    }

    fn finish_stream(&mut self) -> io::Result<()> {
        let Some(mut decoder) = self.decoder.take() else {
            return Ok(());
        };

        let mut extra = [0];

        if decoder
            .read(&mut extra)
            .map_err(|err| invalid_data(err.to_string()))?
            != 0
        {
            return Err(invalid_data("SPZ v4 attribute exceeds its declared size"));
        }

        if let Some(checksum) = decoder.decoder.get_checksum_from_data()
            && Some(checksum) != decoder.decoder.get_calculated_checksum()
        {
            return Err(invalid_data("SPZ v4 ZSTD checksum mismatch"));
        }

        let input = decoder.into_inner();

        if input.limit() != 0 {
            return Err(invalid_data("SPZ v4 frame does not consume its TOC entry"));
        }

        self.source = Some(input.into_inner());

        Ok(())
    }
}

impl<R: Read> Read for V4Reader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() || self.finished {
            return Ok(0);
        }

        if self.remaining == 0 {
            self.finish_stream()?;

            let Some(info) = self.streams.get(self.next_stream).copied() else {
                let mut extra = [0];

                if self.source.as_mut().unwrap().read(&mut extra)? != 0 {
                    return Err(invalid_data("extra SPZ v4 container data"));
                }

                self.finished = true;

                return Ok(0);
            };

            let source = self.source.take().unwrap().take(info.compressed_size);
            let decoder = StreamingDecoder::new_with_max_window_size(source, 128 * 1024 * 1024)
                .map_err(|err| invalid_data(err.to_string()))?;
            let content_size = decoder.decoder.content_size();

            if content_size != 0 && content_size != info.uncompressed_size {
                return Err(invalid_data("SPZ v4 ZSTD content size mismatch"));
            }

            self.decoder = Some(decoder);
            self.remaining = info.uncompressed_size;
            self.next_stream += 1;
        }

        let count = self.remaining.min(buf.len() as u64) as usize;
        let read = self
            .decoder
            .as_mut()
            .unwrap()
            .read(&mut buf[..count])
            .map_err(|err| invalid_data(err.to_string()))?;

        if read == 0 {
            return Err(invalid_data(
                "SPZ v4 attribute shorter than its declared size",
            ));
        }

        self.remaining -= read as u64;

        Ok(read)
    }
}

type SpzInput<R> = std::io::Chain<std::io::Cursor<[u8; 4]>, R>;

enum SpzDecoder<R: Read> {
    Gzip(Box<GzDecoder<SpzInput<R>>>),
    V4(Box<V4Reader<SpzInput<R>>>),
}

impl<R: Read> SpzDecoder<R> {
    fn new(mut reader: R) -> std::io::Result<(Self, SpzGaussiansHeader)> {
        let mut prefix = [0; 4];
        reader.read_exact(&mut prefix)?;

        let input = std::io::Cursor::new(prefix).chain(reader);

        if prefix == *b"NGSP" {
            let (reader, header) = V4Reader::new(input)?;

            Ok((Self::V4(Box::new(reader)), header))
        } else if prefix[..2] == [0x1f, 0x8b] {
            let mut decoder = GzDecoder::new(input);
            let header = SpzGaussians::read_header(&mut decoder)?;

            Ok((Self::Gzip(Box::new(decoder)), header))
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unrecognized SPZ container",
            ))
        }
    }
}

impl<R: Read> Read for SpzDecoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Gzip(reader) => reader.read(buf),
            Self::V4(reader) => reader.read(buf),
        }
    }
}

/// Bounded-work reader for a legacy gzip or v4 ZSTD SPZ model.
///
/// SPZ stores field arrays rather than consecutive Gaussians, so individual Gaussians cannot
/// be delivered early. V4 extensions are validated and skipped, not interpreted or retained.
/// V4 uses a 128 MiB ZSTD window limit. After an I/O error, discard the reader.
pub struct SpzBatchReader<R: Read> {
    decoder: SpzDecoder<R>,
    gaussians: SpzGaussians,
    phase: SpzPhase,
    completed: usize,
}

impl<R: Read> SpzBatchReader<R> {
    pub fn new(reader: R) -> std::io::Result<Self> {
        let (decoder, header) = SpzDecoder::new(reader)?;
        let mut phase = SpzPhase::Positions;
        let mut completed = 0;

        phase.advance_spz_phase(&mut completed, &header);

        Ok(Self {
            decoder,
            gaussians: SpzGaussians::empty(header),
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

        fn read_extend<T: bytemuck::Pod + Zeroable>(
            reader: &mut impl Read,
            values: &mut Vec<T>,
            count: usize,
        ) -> std::io::Result<()> {
            let start = values.len();
            values.try_reserve(count).map_err(std::io::Error::other)?;
            values.resize(start + count, T::zeroed());

            reader.read_exact(bytemuck::cast_slice_mut(&mut values[start..]))
        }

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
                SpzGaussiansShs::Four(v) => read_extend(decoder, v, count)?,
            },
            SpzPhase::Done => unreachable!(),
        }

        self.completed += count;
        self.phase
            .advance_spz_phase(&mut self.completed, &self.gaussians.header);

        Ok(self.progress())
    }

    fn finish(mut self) -> std::io::Result<Self::Model> {
        if !self.progress().done {
            return Err(source_format::batch::incomplete());
        }

        // Consume and validate the final gzip trailer or v4 frame and container boundary.
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

// V4 puts compressed sizes before the frames, so buffer compressed output until
// finish rather than requiring Seek or compressing the input a second time.
struct V4Writer<W: Write> {
    writer: W,
    header: SpzGaussiansHeader,
    encoder: Option<Box<StreamingEncoder<Vec<u8>>>>,
    frames: Vec<(Vec<u8>, u64)>,
}

impl<W: Write> V4Writer<W> {
    fn write_field(&mut self, bytes: &[u8], field_size: usize, last: bool) -> io::Result<()> {
        if self.encoder.is_none() {
            let mut encoder = Box::new(StreamingEncoder::new(
                Vec::new(),
                CompressionLevel::from_level(12),
            ));
            encoder.set_pledged_content_size(field_size as u64)?;
            self.encoder = Some(encoder);
        }

        self.encoder.as_mut().unwrap().write_all(bytes)?;

        if last {
            let frame = self.encoder.take().unwrap().finish()?;
            self.frames.push((frame, field_size as u64));
        }

        Ok(())
    }

    fn finish(mut self) -> io::Result<W> {
        let mut header = [0u8; 32];
        header[..4].copy_from_slice(&SpzGaussiansHeader::MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&4u32.to_le_bytes());
        header[8..12].copy_from_slice(&self.header.as_pod().num_points.to_le_bytes());
        header[12] = self.header.sh_degree().get();
        header[13] = self.header.fractional_bits();
        // Extension records are not retained by the reader, so do not advertise them.
        header[14] = u8::from(self.header.is_antialiased());
        header[15] = self.frames.len() as u8;
        header[16..20].copy_from_slice(&32u32.to_le_bytes());
        self.writer.write_all(&header)?;

        for (frame, uncompressed_size) in &self.frames {
            self.writer.write_all(&(frame.len() as u64).to_le_bytes())?;
            self.writer.write_all(&uncompressed_size.to_le_bytes())?;
        }

        for (frame, _) in self.frames {
            self.writer.write_all(&frame)?;
        }

        Ok(self.writer)
    }
}

enum SpzEncoder<W: Write> {
    Gzip(GzEncoder<W>),
    V4(V4Writer<W>),
}

impl<W: Write> SpzEncoder<W> {
    fn new(writer: W, header: SpzGaussiansHeader) -> io::Result<Self> {
        if header.version() == 4 {
            Ok(Self::V4(V4Writer {
                writer,
                header,
                encoder: None,
                frames: Vec::new(),
            }))
        } else {
            let mut encoder = GzEncoder::new(writer, flate2::Compression::default());
            encoder.write_all(bytemuck::bytes_of(header.as_pod()))?;
            Ok(Self::Gzip(encoder))
        }
    }

    fn write_field(&mut self, bytes: &[u8], field_size: usize, last: bool) -> io::Result<()> {
        match self {
            Self::Gzip(encoder) => encoder.write_all(bytes),
            Self::V4(writer) => writer.write_field(bytes, field_size, last),
        }
    }

    fn finish(self) -> io::Result<W> {
        match self {
            Self::Gzip(encoder) => encoder.finish(),
            Self::V4(writer) => writer.finish(),
        }
    }
}

/// Bounded-work writer for an existing legacy gzip or v4 ZSTD SPZ model.
///
/// V4 compression is incremental, but compressed frames are buffered in memory
/// until [`BatchWrite::finish`] writes the header, TOC and frames to the output.
/// No Seek support is required. After an I/O error, discard the writer.
pub struct SpzBatchWriter<'a, W: Write> {
    encoder: SpzEncoder<W>,
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

        let encoder = SpzEncoder::new(writer, gaussians.header)?;

        let mut phase = SpzPhase::Positions;
        let mut completed = 0;

        phase.advance_spz_phase(&mut completed, &gaussians.header);

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
        let bytes = self.gaussians.field_bytes(self.phase, self.completed..end);
        let field_size = self
            .gaussians
            .field_bytes(self.phase, 0..self.gaussians.len())
            .len();
        self.encoder
            .write_field(bytes, field_size, end == self.gaussians.len())?;

        self.completed = end;
        self.phase
            .advance_spz_phase(&mut self.completed, &self.gaussians.header);
        Ok(self.progress())
    }

    fn finish(self) -> std::io::Result<W> {
        if !self.progress().done {
            return Err(source_format::batch::incomplete());
        }
        self.encoder.finish()
    }
}

/// Options for [`SpzGaussians::from_gaussians_with_options`].
///
/// The fields are not validated.
#[derive(Debug, Clone)]
pub struct SpzGaussiansFromGaussianSliceOptions {
    /// Version to use. Defaults to 3; version 4 supports SH degree 4.
    pub version: u32,

    /// SH degree to use.
    pub sh_degree: SpzGaussianShDegree,

    /// Number of fractional bits to use for position fixed point encoding.
    pub fractional_bits: u8,

    /// Whether to use antialiased encoding.
    pub antialiased: bool,

    /// The quantization bits for SH degrees 1 through 3. Degree 4 reuses the degree-3 setting.
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

use std::{io::Read, ops::RangeInclusive};

use flate2::read::GzDecoder;

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

    /// Validate and create a validated SPZ Gaussians header.
    pub fn from_pod(pod: SpzGaussiansHeaderPod) -> Result<Self, std::io::Error> {
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

/// Packed representation of SPZ Gaussians positions.
#[derive(Debug, Clone)]
pub enum SpzGaussiansPackedPositions {
    /// `(x, y, z)` each as 16-bit floating point.
    Float16(Vec<[u16; 3]>),
    /// `(x, y, z)` each as 24-bit fixed point signed integer.
    FixedPoint24(Vec<[[u8; 3]; 3]>),
}

impl SpzGaussiansPackedPositions {
    /// Read positions from reader.
    pub fn read_from(
        reader: &mut impl Read,
        count: usize,
        uses_float16: bool,
    ) -> Result<Self, std::io::Error> {
        if uses_float16 {
            let mut positions = vec![[0u16; 3]; count];
            reader.read_exact(bytemuck::cast_slice_mut(&mut positions))?;
            Ok(SpzGaussiansPackedPositions::Float16(positions))
        } else {
            let mut positions = vec![[[0u8; 3]; 3]; count];
            reader.read_exact(bytemuck::cast_slice_mut(&mut positions))?;
            Ok(SpzGaussiansPackedPositions::FixedPoint24(positions))
        }
    }
}

/// Packed representation of SPZ Gaussians rotations.
#[derive(Debug, Clone)]
pub enum SpzGaussiansPackedRotations {
    /// `(x, y, z)` each as 8-bit signed integer.
    QuatFirstThree(Vec<[u8; 3]>),
    /// Smallest 3 components each as 10-bit signed integer. 2 bits for index of omitted component.
    QuatSmallestThree(Vec<[u8; 4]>),
}

impl SpzGaussiansPackedRotations {
    /// Read rotations from reader.
    pub fn read_from(
        reader: &mut impl Read,
        count: usize,
        uses_quat_smallest_three: bool,
    ) -> Result<Self, std::io::Error> {
        if !uses_quat_smallest_three {
            let mut rots = vec![[0u8; 3]; count];
            reader.read_exact(bytemuck::cast_slice_mut(&mut rots))?;
            Ok(SpzGaussiansPackedRotations::QuatFirstThree(rots))
        } else {
            let mut rots = vec![[0u8; 4]; count];
            reader.read_exact(bytemuck::cast_slice_mut(&mut rots))?;
            Ok(SpzGaussiansPackedRotations::QuatSmallestThree(rots))
        }
    }
}

/// Packed representation of SPZ Gaussians SH coefficients.
#[derive(Debug, Clone)]
pub enum SpzGaussiansPackedSh {
    Zero,
    One(Vec<[[i8; 3]; 3]>),
    Two(Vec<[[i8; 3]; 8]>),
    Three(Vec<[[i8; 3]; 15]>),
}

impl SpzGaussiansPackedSh {
    /// Read SH coefficients from reader.
    pub fn read_from(
        reader: &mut impl Read,
        count: usize,
        sh_degree: u8,
    ) -> Result<Self, std::io::Error> {
        match sh_degree {
            0 => Ok(SpzGaussiansPackedSh::Zero),
            1 => {
                let mut sh_coeffs = vec![[[0i8; 3]; 3]; count];
                reader.read_exact(bytemuck::cast_slice_mut(&mut sh_coeffs))?;
                Ok(SpzGaussiansPackedSh::One(sh_coeffs))
            }
            2 => {
                let mut sh_coeffs = vec![[[0i8; 3]; 8]; count];
                reader.read_exact(bytemuck::cast_slice_mut(&mut sh_coeffs))?;
                Ok(SpzGaussiansPackedSh::Two(sh_coeffs))
            }
            3 => {
                let mut sh_coeffs = vec![[[0i8; 3]; 15]; count];
                reader.read_exact(bytemuck::cast_slice_mut(&mut sh_coeffs))?;
                Ok(SpzGaussiansPackedSh::Three(sh_coeffs))
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported SH degree: {}", sh_degree),
            )),
        }
    }
}

/// Packed representation of SPZ Gaussians.
#[derive(Debug, Clone)]
pub struct SpzGaussiansPacked {
    pub count: usize,
    pub frac_bits: usize,
    pub uses_float16: bool,
    pub antialiased: bool,
    pub uses_quat_smallest_three: bool,

    pub positions: SpzGaussiansPackedPositions,

    /// `(x, y, z)` each as 8-bit log-encoded integer.
    pub scales: Vec<[u8; 3]>,

    pub rotations: SpzGaussiansPackedRotations,

    /// 8-bit unsigned integer.
    pub alphas: Vec<u8>,

    /// `(r, g, b)` each as 8-bit unsigned integer.
    pub colors: Vec<[u8; 3]>,

    pub sh: SpzGaussiansPackedSh,
}

impl SpzGaussiansPacked {
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
        Self::read_spz_guassians(reader, &header)
    }

    /// Read a SPZ header.
    ///
    /// `reader` should be decompressed SPZ buffer.
    pub fn read_spz_header(reader: &mut impl Read) -> Result<SpzGaussiansHeader, std::io::Error> {
        let mut header_bytes = [0u8; std::mem::size_of::<SpzGaussiansHeaderPod>()];
        reader.read_exact(&mut header_bytes)?;
        let header: SpzGaussiansHeaderPod = bytemuck::cast(header_bytes);
        SpzGaussiansHeader::from_pod(header)
    }

    /// Read the SPZ Gaussians.
    ///
    /// `reader` should be decompressed SPZ buffer positioned after the header.
    ///
    /// `header` may be parsed by calling [`SpzGaussiansPacked::read_spz_header`].
    pub fn read_spz_guassians(
        reader: &mut impl Read,
        header: &SpzGaussiansHeader,
    ) -> Result<Self, std::io::Error> {
        let count = header.num_points();
        let frac_bits = header.fractional_bits();
        let uses_float16 = header.uses_float16();
        let antialiased = header.is_antialiased();
        let uses_quat_smallest_three = header.uses_quat_smallest_three();

        let positions = SpzGaussiansPackedPositions::read_from(reader, count, uses_float16)?;

        let mut scales = vec![[0u8; 3]; count];
        reader.read_exact(bytemuck::cast_slice_mut(&mut scales))?;

        let rotations =
            SpzGaussiansPackedRotations::read_from(reader, count, uses_quat_smallest_three)?;

        let mut alphas = vec![0u8; count];
        reader.read_exact(bytemuck::cast_slice_mut(&mut alphas))?;

        let mut colors = vec![[0u8; 3]; count];
        reader.read_exact(bytemuck::cast_slice_mut(&mut colors))?;

        let sh = SpzGaussiansPackedSh::read_from(reader, count, header.sh_degree())?;

        Ok(SpzGaussiansPacked {
            count,
            frac_bits,
            uses_float16,
            antialiased,
            uses_quat_smallest_three,
            positions,
            scales,
            rotations,
            alphas,
            colors,
            sh,
        })
    }
}

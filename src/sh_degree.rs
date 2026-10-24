use std::fmt::Debug;

use glam::Vec3;
use half::f16;

use crate::{AnyGaussian, Gaussian};

/// A supported Gaussian spherical harmonics degree for storage and rendering.
///
/// Supports degrees 0 through 4. SPZ uses [`crate::SpzGaussianShDegree`] instead,
/// which supports degrees 0 through 3.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GaussianShDegree(u8);

impl GaussianShDegree {
    /// Create a new Gaussian SH degree.
    ///
    /// Returns [`None`] if the degree is not in the range of \[0, 4\].
    pub const fn new(sh_deg: u8) -> Option<Self> {
        match sh_deg {
            0..=4 => Some(Self(sh_deg)),
            _ => None,
        }
    }

    /// Create a new Gaussian SH degree without checking.
    ///
    /// # Safety
    ///
    /// The degree must be in the range of \[0, 4\].
    pub const unsafe fn new_unchecked(sh_deg: u8) -> Self {
        Self(sh_deg)
    }

    /// Get the degree.
    pub const fn get(&self) -> u8 {
        self.0
    }

    /// Number of RGB coefficient vectors, excluding SH0.
    pub const fn num_coefficients(&self) -> usize {
        (self.0 as usize + 1) * (self.0 as usize + 1) - 1
    }

    /// Infer a supported degree from its non-DC coefficient count.
    pub const fn from_coefficient_count(count: usize) -> Option<Self> {
        match count {
            0 => Self::new(0),
            3 => Self::new(1),
            8 => Self::new(2),
            15 => Self::new(3),
            24 => Self::new(4),
            _ => None,
        }
    }
}

impl Default for GaussianShDegree {
    fn default() -> Self {
        // SAFETY: 3 is in the range of [0, 4].
        unsafe { Self::new_unchecked(3) }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A supported, compile-time SH storage degree. SH0 is stored separately in color.
///
/// Sealed to guarantee that coefficient counts and GPU encodings agree.
pub trait ShDegree: sealed::Sealed + Debug + Copy + PartialEq + Eq + Send + Sync + 'static {
    const DEGREE: u8;
    const COEFFICIENT_COUNT: usize;
    const FEATURE: &'static str;

    type Coefficients: bytemuck::Pod
        + Debug
        + Copy
        + PartialEq
        + Send
        + Sync
        + AsRef<[Vec3]>
        + AsMut<[Vec3]>;
    type Half: bytemuck::Pod + Debug + Copy + PartialEq + Send + Sync + AsRef<[f16]> + AsMut<[f16]>;
    type Norm8: bytemuck::Pod + Debug + Copy + PartialEq + Send + Sync + AsRef<[i8]> + AsMut<[i8]>;

    fn into_any(gaussian: Gaussian<Self>) -> AnyGaussian;
}

macro_rules! sh_degree {
    ($name:ident, $degree:literal, $count:literal, $half:literal, $norm8:literal, $variant:ident) => {
        #[doc = concat!("SH storage degree ", stringify!($degree), ".")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name;

        impl sealed::Sealed for $name {}

        impl ShDegree for $name {
            const DEGREE: u8 = $degree;
            const COEFFICIENT_COUNT: usize = $count;
            const FEATURE: &'static str = concat!("sh_degree_", stringify!($degree));

            type Coefficients = [Vec3; $count];
            type Half = [f16; $half];
            type Norm8 = [i8; $norm8];

            fn into_any(gaussian: Gaussian<Self>) -> AnyGaussian {
                AnyGaussian::$variant(gaussian)
            }
        }
    };
}

sh_degree!(ShDegree0, 0, 0, 0, 0, Zero);
sh_degree!(ShDegree1, 1, 3, 10, 12, One);
sh_degree!(ShDegree2, 2, 8, 24, 24, Two);
sh_degree!(ShDegree3, 3, 15, 46, 48, Three);
sh_degree!(ShDegree4, 4, 24, 72, 72, Four);

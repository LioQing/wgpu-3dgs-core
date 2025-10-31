use bytemuck::Zeroable;

use crate::{Gaussian, ReadPlyError};

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
/// This represents the header parsed by [`Gaussians::read_ply_header`](crate::Gaussians::read_ply_header).
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
    I: Iterator<Item = Result<PlyGaussianPod, ReadPlyError>>,
    C: Iterator<Item = Result<PlyGaussianPod, ReadPlyError>>,
> {
    /// The Inria PLY format.
    Inria(I),

    /// Custom PLY format.
    ///
    /// This still is the same properties as Inria format, but may have different order.
    Custom(C),
}

impl<
    I: Iterator<Item = Result<PlyGaussianPod, ReadPlyError>>,
    C: Iterator<Item = Result<PlyGaussianPod, ReadPlyError>>,
> Iterator for PlyGaussianIter<I, C>
{
    type Item = Result<PlyGaussianPod, ReadPlyError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inria(iter) => iter.next(),
            Self::Custom(iter) => iter.next(),
        }
    }
}

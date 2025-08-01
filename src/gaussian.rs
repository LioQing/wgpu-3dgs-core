use std::io::{BufRead, Write};

use bytemuck::Zeroable;
use glam::*;

use crate::{Error, PlyGaussianIter, PlyGaussianPod, PlyHeader};

/// A scene containing Gaussians.
#[derive(Debug, Clone)]
pub struct Gaussians {
    /// The Gaussians.
    pub gaussians: Vec<Gaussian>,
}

impl Gaussians {
    /// Read a splat PLY file.
    pub fn read_ply(reader: &mut impl BufRead) -> Result<Self, Error> {
        let ply_header = Self::read_ply_header(reader)?;

        let count = ply_header.count()?;
        let mut gaussians = Vec::with_capacity(count);

        for gaussian in Self::read_ply_gaussians(reader, ply_header)? {
            gaussians.push(gaussian?.into());
        }

        Ok(Self { gaussians })
    }

    /// Read a splat PLY header.
    pub fn read_ply_header(reader: &mut impl BufRead) -> Result<PlyHeader, Error> {
        let parser = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
        let header = parser.read_header(reader)?;
        let vertex = header
            .elements
            .get("vertex")
            .ok_or(Error::PlyVertexNotFound)?;

        const SYSTEM_ENDIANNESS: ply_rs::ply::Encoding = match cfg!(target_endian = "little") {
            true => ply_rs::ply::Encoding::BinaryLittleEndian,
            false => ply_rs::ply::Encoding::BinaryBigEndian,
        };

        let ply_header =
            match vertex
                .properties
                .iter()
                .zip(PLY_PROPERTIES.iter())
                .all(|((a, property), b)| {
                    a == *b
                        && property.data_type
                            == ply_rs::ply::PropertyType::Scalar(ply_rs::ply::ScalarType::Float)
                })
                && header.encoding == SYSTEM_ENDIANNESS
            {
                true => PlyHeader::Inria(vertex.count),
                false => PlyHeader::Custom(header),
            };

        Ok(ply_header)
    }

    /// Read the splat PLY Gaussians into [`PlyGaussianPod`].
    pub fn read_ply_gaussians(
        reader: &mut impl BufRead,
        ply_header: PlyHeader,
    ) -> Result<impl Iterator<Item = Result<PlyGaussianPod, Error>>, Error> {
        let count = ply_header.count()?;
        log::info!("Reading PLY format with {count} Gaussians");

        Ok(match ply_header {
            PlyHeader::Inria(..) => PlyGaussianIter::Inria((0..count).map(|_| {
                let mut gaussian = PlyGaussianPod::zeroed();
                reader.read_exact(bytemuck::bytes_of_mut(&mut gaussian))?;
                Ok(gaussian)
            })),
            PlyHeader::Custom(header) => {
                let parser = ply_rs::parser::Parser::<PlyGaussianPod>::new();

                PlyGaussianIter::Custom((0..count).map(move |_| {
                    let Some(vertex) = header.elements.get("vertex") else {
                        return Err(Error::PlyVertexNotFound);
                    };
                    Ok(match header.encoding {
                        ply_rs::ply::Encoding::Ascii => {
                            let mut line = String::new();
                            reader.read_line(&mut line)?;

                            let mut gaussian = PlyGaussianPod::zeroed();
                            line.split(' ')
                                .map(|s| s.parse::<f32>())
                                .zip(vertex.properties.keys())
                                .try_for_each(|(value, name)| match value {
                                    Ok(value) => {
                                        gaussian.set_value(name, value);
                                        Ok(())
                                    }
                                    Err(_) => Err(Error::PlyVertexPropertyNotFound(name.clone())),
                                })?;

                            gaussian
                        }
                        ply_rs::ply::Encoding::BinaryLittleEndian => {
                            parser.read_little_endian_element(reader, vertex)?
                        }
                        ply_rs::ply::Encoding::BinaryBigEndian => {
                            parser.read_big_endian_element(reader, vertex)?
                        }
                    })
                }))
            }
        })
    }

    /// Write the Gaussians to a PLY file.
    pub fn write_ply(&self, writer: &mut impl Write) -> Result<(), Error> {
        writeln!(writer, "ply")?;
        writeln!(writer, "format binary_little_endian 1.0")?;
        writeln!(writer, "element vertex {}", self.gaussians.len())?;
        for property in PLY_PROPERTIES {
            writeln!(writer, "property float {property}")?;
        }
        writeln!(writer, "end_header")?;

        self.gaussians
            .iter()
            .map(|gaussian| gaussian.to_ply())
            .try_for_each(|gaussian| writer.write_all(bytemuck::bytes_of(&gaussian)))?;

        Ok(())
    }
}

/// The Gaussian.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    pub rot: Quat,
    pub pos: Vec3,
    pub color: U8Vec4,
    pub sh: [Vec3; 15],
    pub scale: Vec3,
}

impl Gaussian {
    /// Convert from PLY Gaussian to Gaussian.
    pub fn from_ply(ply: &PlyGaussianPod) -> Self {
        // Position
        let pos = Vec3::from_array(ply.pos);

        // Rotation
        let rot = Quat::from_xyzw(ply.rot[1], ply.rot[2], ply.rot[3], ply.rot[0]).normalize();

        // Scale
        let scale = Vec3::from_array(ply.scale).exp();

        // Color
        const SH_C0: f32 = 0.2820948;
        let color = ((Vec3::splat(0.5) + Vec3::from_array(ply.color) * SH_C0) * 255.0)
            .extend((1.0 / (1.0 + (-ply.alpha).exp())) * 255.0)
            .clamp(Vec4::splat(0.0), Vec4::splat(255.0))
            .as_u8vec4();

        // Spherical harmonics
        let sh = std::array::from_fn(|i| Vec3::new(ply.sh[i], ply.sh[i + 15], ply.sh[i + 30]));

        Self {
            rot,
            pos,
            color,
            sh,
            scale,
        }
    }

    /// Convert to PLY Gaussian.
    pub fn to_ply(&self) -> PlyGaussianPod {
        // Position
        let pos = self.pos.to_array();

        // Rotation
        let rot = [self.rot.w, self.rot.x, self.rot.y, self.rot.z];

        // Scale
        let scale = self.scale.map(|x| x.ln()).to_array();

        // Color
        const SH_C0: f32 = 0.2820948;
        let rgba = self.color.as_vec4() / 255.0;
        let color = ((rgba.xyz() / SH_C0) - Vec3::splat(0.5 / SH_C0)).to_array();

        // Alpha
        let alpha = -(1.0 / rgba.w - 1.0).ln();

        // Spherical harmonics
        let mut sh = [0.0; 3 * 15];
        for i in 0..15 {
            sh[i] = self.sh[i].x;
            sh[i + 15] = self.sh[i].y;
            sh[i + 30] = self.sh[i].z;
        }

        let normal = [0.0, 0.0, 1.0];

        PlyGaussianPod {
            pos,
            normal,
            color,
            sh,
            alpha,
            scale,
            rot,
        }
    }
}

impl From<PlyGaussianPod> for Gaussian {
    fn from(ply: PlyGaussianPod) -> Self {
        Self::from_ply(&ply)
    }
}

impl From<&PlyGaussianPod> for Gaussian {
    fn from(ply: &PlyGaussianPod) -> Self {
        Self::from_ply(ply)
    }
}

const PLY_PROPERTIES: &[&str] = &[
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

use std::io::Write;

use assert_matches::assert_matches;
use wgpu_3dgs_core::{Gaussian, Gaussians, PLY_PROPERTIES, PlyGaussianPod, ReadPlyError, glam::*};

use crate::common::given;

fn given_custom_gaussians_ply_buffer(
    gaussians: &[Gaussian],
    endianness: ply_rs::ply::Encoding,
) -> Vec<u8> {
    let mut buffer = Vec::new();

    writeln!(buffer, "ply").unwrap();
    writeln!(buffer, "format {} 1.0", endianness).unwrap();
    writeln!(buffer, "element vertex {}", gaussians.len()).unwrap();

    let mut properties = PLY_PROPERTIES.to_vec();
    properties.swap(1, 2); // Swap y and z to be different from Inria format.

    for property in properties {
        writeln!(buffer, "property float {property}").unwrap();
    }
    writeln!(buffer, "end_header").unwrap();

    for gaussian in gaussians.iter() {
        let mut ply = gaussian.to_ply();

        ply.pos.swap(1, 2); // Swap y and z to be different from Inria format.

        match endianness {
            ply_rs::ply::Encoding::Ascii => {
                fn to_string<'a>(v: impl Iterator<Item = &'a (impl ToString + 'a)>) -> String {
                    v.map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
                }

                writeln!(
                    buffer,
                    "{} {} {} {} {} {} {}",
                    to_string(ply.pos.iter()),
                    to_string(ply.normal.iter()),
                    to_string(ply.color.iter()),
                    to_string(ply.sh.iter()),
                    ply.alpha,
                    to_string(ply.scale.iter()),
                    to_string(ply.rot.iter()),
                )
                .unwrap();
            }
            ply_rs::ply::Encoding::BinaryLittleEndian => {
                buffer.extend_from_slice(bytemuck::bytes_of(&ply));
            }
            ply_rs::ply::Encoding::BinaryBigEndian => {
                const SIZE: usize = std::mem::size_of::<PlyGaussianPod>();
                let mut bytes: [u8; SIZE] = bytemuck::cast(ply);
                bytes.chunks_exact_mut(4).for_each(|chunk| chunk.reverse());
                buffer.extend_from_slice(&bytes);
            }
        }

        buffer.flush().unwrap();
    }

    buffer
}

fn assert_gaussian(a: &Gaussian, b: &Gaussian) {
    const EPSILON: f32 = 1e-4;

    assert!(
        a.rot.abs_diff_eq(b.rot, EPSILON),
        " left: {:?}\nright: {:?}",
        a.rot,
        b.rot
    );
    assert!(
        a.pos.abs_diff_eq(b.pos, EPSILON),
        " left: {:?}\nright: {:?}",
        a.pos,
        b.pos
    );

    // TODO(#2): Lost of precision is caused by conversion from file format to memory format.
    // assert_eq!(a.color, b.color);

    for i in 0..15 {
        assert!(
            a.sh[i].abs_diff_eq(b.sh[i], EPSILON),
            " left: {:?}\nright: {:?}",
            a.sh[i],
            b.sh[i]
        );
    }

    assert!(
        a.scale.abs_diff_eq(b.scale, EPSILON),
        " left: {:?}\nright: {:?}",
        a.scale,
        b.scale
    );
}

#[test]
fn test_gaussians_write_ply_and_read_ply_should_be_equal() {
    let gaussians = given::gaussians();

    let mut buffer = Vec::new();
    gaussians.write_ply(&mut buffer).unwrap();
    let gaussians_read = Gaussians::read_ply(&mut buffer.as_slice()).unwrap();

    assert_eq!(gaussians.gaussians.len(), gaussians_read.gaussians.len());

    for (a, b) in gaussians
        .gaussians
        .iter()
        .zip(gaussians_read.gaussians.iter())
    {
        assert_gaussian(a, b);
    }
}

#[test]
fn test_gaussians_read_ply_when_format_is_custom_and_ascii_should_match_original_gaussian() {
    let gaussians = given::gaussians();
    let buffer =
        given_custom_gaussians_ply_buffer(&gaussians.gaussians, ply_rs::ply::Encoding::Ascii);

    let gaussians_read = Gaussians::read_ply(&mut buffer.as_slice()).unwrap();
    assert_eq!(gaussians_read.gaussians.len(), 2);
    assert_gaussian(&gaussians.gaussians[0], &gaussians_read.gaussians[0]);
    assert_gaussian(&gaussians.gaussians[1], &gaussians_read.gaussians[1]);
}

#[test]
fn test_gaussians_read_ply_when_format_is_custom_and_be_should_match_original_gaussian() {
    let gaussians = given::gaussians();
    let buffer = given_custom_gaussians_ply_buffer(
        &gaussians.gaussians,
        ply_rs::ply::Encoding::BinaryBigEndian,
    );

    let gaussians_read = Gaussians::read_ply(&mut buffer.as_slice()).unwrap();
    assert_eq!(gaussians_read.gaussians.len(), 2);
    assert_gaussian(&gaussians.gaussians[0], &gaussians_read.gaussians[0]);
    assert_gaussian(&gaussians.gaussians[1], &gaussians_read.gaussians[1]);
}

#[test]
fn test_gaussians_read_ply_when_format_is_custom_and_le_should_match_original_gaussian() {
    let gaussians = given::gaussians();
    let buffer = given_custom_gaussians_ply_buffer(
        &gaussians.gaussians,
        ply_rs::ply::Encoding::BinaryLittleEndian,
    );

    let gaussians_read = Gaussians::read_ply(&mut buffer.as_slice()).unwrap();
    assert_eq!(gaussians_read.gaussians.len(), 2);
    assert_gaussian(&gaussians.gaussians[0], &gaussians_read.gaussians[0]);
    assert_gaussian(&gaussians.gaussians[1], &gaussians_read.gaussians[1]);
}

#[test]
fn test_gaussians_read_ply_when_missing_vertex_should_return_error() {
    let gaussian = given::gaussian();
    let ply = gaussian.to_ply();

    let mut buffer = Vec::new();

    writeln!(buffer, "ply").unwrap();
    writeln!(buffer, "format ascii 1.0").unwrap();
    writeln!(buffer, "element fragment 1").unwrap();
    for property in PLY_PROPERTIES {
        writeln!(buffer, "property float {property}").unwrap();
    }
    writeln!(buffer, "end_header").unwrap();

    fn to_string<'a>(v: impl Iterator<Item = &'a (impl ToString + 'a)>) -> String {
        v.map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
    }

    writeln!(
        buffer,
        "{} {} {} {} {} {} {}",
        to_string([&ply.pos[0], &ply.pos[2], &ply.pos[1]].iter()),
        to_string(ply.normal.iter()),
        to_string(ply.color.iter()),
        to_string(ply.sh.iter()),
        ply.alpha,
        to_string(ply.scale.iter()),
        to_string(ply.rot.iter()),
    )
    .unwrap();

    let result = Gaussians::read_ply(&mut buffer.as_slice());
    assert_matches!(result, Err(ReadPlyError::VertexNotFound));
}

#[test]
fn test_gaussians_read_ply_when_missing_value_should_return_error() {
    let gaussian = given::gaussian();
    let ply = gaussian.to_ply();

    let mut buffer = Vec::new();

    writeln!(buffer, "ply").unwrap();
    writeln!(buffer, "format ascii 1.0").unwrap();
    writeln!(buffer, "element vertex 1").unwrap();
    for property in PLY_PROPERTIES {
        writeln!(buffer, "property float {property}").unwrap();
    }
    writeln!(buffer, "end_header").unwrap();

    fn to_string<'a>(v: impl Iterator<Item = &'a (impl ToString + 'a)>) -> String {
        v.map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
    }

    writeln!(
        buffer,
        "{} {} {} {} {} {} {}",
        to_string([&ply.pos[0], &ply.pos[2], &ply.pos[1]].iter()),
        to_string(ply.normal.iter()),
        to_string(ply.color.iter()),
        to_string(ply.sh.iter()),
        ply.alpha,
        to_string(ply.scale.iter().take(2)),
        to_string(ply.rot.iter()),
    )
    .unwrap();

    let result = Gaussians::read_ply(&mut buffer.as_slice());

    // TODO(#3): Improve error message to just say missing value instead of guessing property name.
    assert_matches!(result, Err(ReadPlyError::VertexPropertyNotFound(property)) if property == "rot_3");
}

#[test]
fn test_gaussian_to_ply_and_from_ply_should_be_equal() {
    let gaussian = given::gaussian();

    let ply = gaussian.to_ply();
    let gaussian_from_ply = Gaussian::from_ply(&ply);

    assert_gaussian(&gaussian, &gaussian_from_ply);
}

#[test]
fn test_gaussian_from_and_from_ply_should_be_equal() {
    let ply = PlyGaussianPod {
        pos: [1.0, 2.0, 3.0],
        normal: [0.0, 0.0, 1.0],
        color: [0.5, 0.25, 0.125],
        sh: [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4,
            3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5,
        ],
        alpha: 0.5,
        scale: [0.1, 0.2, 0.3],
        rot: [0.4, 0.1, 0.2, 0.3],
    };

    let gaussian_from_ply = Gaussian::from_ply(&ply);
    let gaussian_from_ref = Gaussian::from(&ply);
    let gaussian_from = Gaussian::from(ply);

    assert_gaussian(&gaussian_from_ply, &gaussian_from_ref);
    assert_gaussian(&gaussian_from_ref, &gaussian_from);
}

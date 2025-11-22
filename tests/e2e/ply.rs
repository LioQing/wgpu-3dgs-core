use std::io::Write;

use assert_matches::assert_matches;
use wgpu_3dgs_core::{IterGaussian, PlyGaussianPod, PlyGaussians, glam::*};

use crate::common::{assert, given};

fn given_custom_gaussians_ply_buffer(
    plys: &[PlyGaussianPod],
    endianness: ply_rs::ply::Encoding,
) -> Vec<u8> {
    let mut buffer = Vec::new();

    writeln!(buffer, "ply").unwrap();
    writeln!(buffer, "format {} 1.0", endianness).unwrap();
    writeln!(buffer, "element vertex {}", plys.len()).unwrap();

    let mut properties = PlyGaussians::PLY_PROPERTIES.to_vec();
    properties.swap(1, 2); // Swap y and z to be different from Inria format.

    for property in properties {
        writeln!(buffer, "property float {property}").unwrap();
    }
    writeln!(buffer, "end_header").unwrap();

    for mut ply in plys.iter().copied() {
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

#[test]
fn test_ply_gaussian_pod_from_and_gaussian_to_ply_should_be_equal() {
    let gaussian = given::gaussian();

    let gaussian_to_ply = gaussian.to_ply();
    let ply_from_ref = PlyGaussianPod::from(&gaussian);
    let ply_from = PlyGaussianPod::from(gaussian);

    assert::ply_gaussian_pod(&gaussian_to_ply, &ply_from_ref);
    assert::ply_gaussian_pod(&gaussian_to_ply, &ply_from);
}

#[test]
fn test_ply_gaussian_pod_len_and_is_empty_should_be_correct() {
    let gaussians = given::ply_gaussians();

    assert_eq!(gaussians.len(), 2);
    assert!(!gaussians.is_empty());
}

#[test]
fn test_ply_gaussians_write_ply_file_and_read_ply_file_should_be_equal() {
    let gaussians = given::ply_gaussians();
    let path = given::temp_file_path(".ply");

    gaussians.write_ply_file(&path).unwrap();
    let gaussians_read = PlyGaussians::read_ply_file(&path).unwrap();

    assert_eq!(gaussians.len(), gaussians_read.len());

    for (a, b) in gaussians.iter().zip(gaussians_read.iter()) {
        assert::ply_gaussian_pod(a, b);
    }
}

#[test]
fn test_ply_gaussians_write_ply_and_read_ply_should_be_equal() {
    let gaussians = given::ply_gaussians();

    let mut buffer = Vec::new();
    gaussians.write_ply(&mut buffer).unwrap();
    let gaussians_read = PlyGaussians::read_ply(&mut buffer.as_slice()).unwrap();

    assert_eq!(gaussians.len(), gaussians_read.len());

    for (a, b) in gaussians.iter().zip(gaussians_read.iter()) {
        assert::ply_gaussian_pod(a, b);
    }
}

#[test]
fn test_ply_gaussians_from_vec_from_iter_and_iter_iter_mut_iter_gaussian_should_be_equal() {
    let original = given::ply_gaussians();
    let original_vec = original.0.clone();

    let from_vec = PlyGaussians::from(original_vec.clone());
    let from_iter: PlyGaussians = original_vec.clone().into_iter().collect();
    let mut from_iter_mut = from_iter.clone();

    for (original, vec, iter, iter_mut, iter_gaussian) in itertools::izip!(
        original_vec.iter(),
        from_vec.iter(),
        from_iter.iter(),
        from_iter_mut.iter_mut(),
        from_iter.iter_gaussian(),
    ) {
        assert::ply_gaussian_pod(original, vec);
        assert::ply_gaussian_pod(original, iter);
        assert::ply_gaussian_pod(original, iter_mut);
        assert::ply_gaussian_pod(original, &iter_gaussian.to_ply());
    }
}

#[test]
fn test_ply_gaussians_read_ply_when_format_is_custom_and_ascii_should_match_original_gaussian() {
    let gaussians = given::ply_gaussians();
    let buffer = given_custom_gaussians_ply_buffer(&gaussians.0, ply_rs::ply::Encoding::Ascii);

    let gaussians_read = PlyGaussians::read_ply(&mut buffer.as_slice()).unwrap();
    assert_eq!(gaussians_read.len(), 2);
    assert::ply_gaussian_pod(&gaussians.0[0], &gaussians_read.0[0]);
    assert::ply_gaussian_pod(&gaussians.0[1], &gaussians_read.0[1]);
}

#[test]
fn test_ply_gaussians_read_ply_when_format_is_custom_and_be_should_match_original_gaussian() {
    let gaussians = given::ply_gaussians();
    let buffer =
        given_custom_gaussians_ply_buffer(&gaussians.0, ply_rs::ply::Encoding::BinaryBigEndian);

    let gaussians_read = PlyGaussians::read_ply(&mut buffer.as_slice()).unwrap();
    assert_eq!(gaussians_read.len(), 2);
    assert::ply_gaussian_pod(&gaussians.0[0], &gaussians_read.0[0]);
    assert::ply_gaussian_pod(&gaussians.0[1], &gaussians_read.0[1]);
}

#[test]
fn test_ply_gaussians_read_ply_when_format_is_custom_and_le_should_match_original_gaussian() {
    let gaussians = given::ply_gaussians();
    let buffer =
        given_custom_gaussians_ply_buffer(&gaussians.0, ply_rs::ply::Encoding::BinaryLittleEndian);

    let gaussians_read = PlyGaussians::read_ply(&mut buffer.as_slice()).unwrap();
    assert_eq!(gaussians_read.len(), 2);
    assert::ply_gaussian_pod(&gaussians.0[0], &gaussians_read.0[0]);
    assert::ply_gaussian_pod(&gaussians.0[1], &gaussians_read.0[1]);
}

#[test]
fn test_ply_gaussians_read_ply_when_missing_vertex_should_return_error() {
    let gaussian = given::gaussian();
    let ply = gaussian.to_ply();

    let mut buffer = Vec::new();

    writeln!(buffer, "ply").unwrap();
    writeln!(buffer, "format ascii 1.0").unwrap();
    writeln!(buffer, "element fragment 1").unwrap();
    for property in PlyGaussians::PLY_PROPERTIES {
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

    let result = PlyGaussians::read_ply(&mut buffer.as_slice());
    assert_matches!(
        result,
        Err(e) if e.kind() == std::io::ErrorKind::InvalidData &&
            e.to_string() == "Gaussian vertex element not found in PLY header"
    );
}

#[test]
fn test_ply_gaussians_read_ply_when_missing_value_should_return_error() {
    let gaussian = given::gaussian();
    let ply = gaussian.to_ply();

    let mut buffer = Vec::new();

    writeln!(buffer, "ply").unwrap();
    writeln!(buffer, "format ascii 1.0").unwrap();
    writeln!(buffer, "element vertex 1").unwrap();
    for property in PlyGaussians::PLY_PROPERTIES {
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

    let result = PlyGaussians::read_ply(&mut buffer.as_slice());

    assert_matches!(
        result,
        Err(e) if e.kind() == std::io::ErrorKind::InvalidData &&
            e.to_string() == "Gaussian element property invalid or missing in PLY"
    );
}

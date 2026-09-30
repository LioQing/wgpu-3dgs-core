use std::io::Write;

use assert_matches::assert_matches;
use wgpu_3dgs_core::{
    BatchRead, BatchWrite, Gaussian, GaussianStream, IterGaussian, PlyBatchReader, PlyBatchWriter,
    PlyGaussian, PlyGaussianStream, PlyGaussians, PlyGaussiansFromIterError, ReadIterGaussian,
    WriteIterGaussian, glam::*,
};

use crate::common::{assert, given};

fn degree_gaussians(degree: u8) -> PlyGaussians {
    let count = (degree as usize + 1).pow(2) - 1;
    let mut gaussian = given::gaussian().to_ply();
    gaussian.sh = (0..count * 3).map(|i| i as f32 * 0.125 - 1.0).collect();

    PlyGaussians::new(vec![gaussian], degree).unwrap()
}

fn reordered_buffer(ply: &PlyGaussian, degree: u8, encoding: ply_rs::ply::Encoding) -> Vec<u8> {
    let mut buffer = Vec::new();

    writeln!(buffer, "ply").unwrap();
    writeln!(buffer, "format {encoding} 1.0").unwrap();
    writeln!(buffer, "element vertex 1").unwrap();

    let mut properties: Vec<(String, f32)> = [
        ("x".to_string(), ply.pos.x),
        ("z".to_string(), ply.pos.z),
        ("y".to_string(), ply.pos.y),
        ("nx".to_string(), ply.normal.x),
        ("ny".to_string(), ply.normal.y),
        ("nz".to_string(), ply.normal.z),
        ("f_dc_0".to_string(), ply.color.x),
        ("f_dc_1".to_string(), ply.color.y),
        ("f_dc_2".to_string(), ply.color.z),
    ]
    .into();
    properties.extend(
        ply.sh
            .iter()
            .enumerate()
            .map(|(i, &v)| (format!("f_rest_{i}"), v)),
    );
    properties.extend([
        ("opacity".to_string(), ply.alpha),
        ("scale_0".to_string(), ply.scale.x),
        ("scale_1".to_string(), ply.scale.y),
        ("scale_2".to_string(), ply.scale.z),
        ("rot_0".to_string(), ply.rot.w),
        ("rot_1".to_string(), ply.rot.x),
        ("rot_2".to_string(), ply.rot.y),
        ("rot_3".to_string(), ply.rot.z),
    ]);

    assert_eq!(ply.sh.len(), ((degree as usize + 1).pow(2) - 1) * 3);

    for (name, _) in &properties {
        writeln!(buffer, "property float {name}").unwrap();
    }

    writeln!(buffer, "end_header").unwrap();

    match encoding {
        ply_rs::ply::Encoding::Ascii => {
            writeln!(
                buffer,
                "{}",
                properties
                    .iter()
                    .map(|(_, v)| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
            .unwrap();
        }
        ply_rs::ply::Encoding::BinaryLittleEndian => {
            for (_, v) in properties {
                buffer.extend(v.to_le_bytes());
            }
        }
        ply_rs::ply::Encoding::BinaryBigEndian => {
            for (_, v) in properties {
                buffer.extend(v.to_be_bytes());
            }
        }
    }

    buffer
}

#[test]
fn test_ply_color_round_trip_should_preserve_float_precision() {
    for color in [
        Vec3::new(0.1234567, -0.2345678, 0.3456789),
        Vec3::new(-3.1234567, 2.2345678, 4.345679),
    ] {
        let ply = PlyGaussian {
            color,
            alpha: 0.1234567,
            ..given::gaussian().to_ply()
        };

        let round_trip = Gaussian::from_ply(&ply).to_ply();

        assert!(round_trip.color.abs_diff_eq(color, 1e-6));
        assert!((round_trip.alpha - ply.alpha).abs() < 1e-6);
    }
}

#[test]
fn test_ply_gaussian_from_gaussian() {
    let gaussian = given::gaussian();

    assert::ply_gaussian_pod(&gaussian.to_ply(), &PlyGaussian::from(&gaussian));
    assert::ply_gaussian_pod(&gaussian.to_ply(), &PlyGaussian::from(gaussian));
}

#[test]
fn test_ply_degrees_round_trip_and_convert_to_degree_three() {
    for degree in 0..=4 {
        let original = degree_gaussians(degree);
        assert_eq!(
            PlyGaussians::try_from(original.gaussians.clone())
                .unwrap()
                .sh_degree,
            degree
        );

        let mut bytes = Vec::new();
        original.write_to(&mut bytes).unwrap();

        let mut input = bytes.as_slice();
        let header = PlyGaussians::read_header(&mut input).unwrap();

        assert_eq!(header.sh_degree, degree);
        assert_eq!(header.count(), 1);

        let actual = PlyGaussians::read_gaussians(&mut input, header)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(actual, original.gaussians);
        assert!(input.is_empty());
        assert_eq!(
            PlyGaussians::read_from(&mut bytes.as_slice()).unwrap(),
            original
        );

        let converted = Gaussian::from_ply(&actual[0]);
        let count = ((degree as usize + 1).pow(2) - 1).min(15);

        for i in 0..15 {
            let expected = if i < count {
                Vec3::new(
                    actual[0].sh[i],
                    actual[0].sh[i + actual[0].sh.len() / 3],
                    actual[0].sh[i + 2 * actual[0].sh.len() / 3],
                )
            } else {
                Vec3::ZERO
            };

            assert_eq!(converted.sh[i], expected);
        }
    }
}

#[test]
fn test_empty_degree_four_ply_preserves_header() {
    let original = PlyGaussians::new(Vec::new(), 4).unwrap();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();

    assert_eq!(
        PlyGaussians::read_from(&mut bytes.as_slice()).unwrap(),
        original
    );
}

#[test]
fn test_ply_reordered_ascii_and_binary_should_preserve_all_degrees() {
    for degree in [1, 2, 3, 4] {
        let original = degree_gaussians(degree);

        for encoding in [
            ply_rs::ply::Encoding::Ascii,
            ply_rs::ply::Encoding::BinaryLittleEndian,
            ply_rs::ply::Encoding::BinaryBigEndian,
        ] {
            let bytes = reordered_buffer(&original.gaussians[0], degree, encoding);

            assert_eq!(
                PlyGaussians::read_from(&mut bytes.as_slice()).unwrap(),
                original
            );
        }
    }
}

#[test]
fn test_ply_stream_and_batch_writer_preserve_variable_degree() {
    for degree in [1, 2, 4] {
        let original = degree_gaussians(degree);
        let mut input = Vec::new();
        original.write_to(&mut input).unwrap();

        let stream = PlyGaussianStream::new(input.as_slice()).unwrap();
        assert_eq!(stream.header().sh_degree, degree);

        let mut writer = PlyBatchWriter::from_iter(
            Vec::new(),
            stream.total_gaussians(),
            stream.header().sh_degree,
            stream,
        )
        .unwrap();

        writer
            .step(std::num::NonZeroUsize::new(1).unwrap())
            .unwrap();

        let output = writer.finish().unwrap();
        let mut reader = PlyBatchReader::new(output.as_slice()).unwrap();

        reader
            .step(std::num::NonZeroUsize::new(1).unwrap())
            .unwrap();

        assert_eq!(reader.finish().unwrap(), original);
    }
}

#[test]
fn test_ply_rejects_missing_vertex_and_malformed_sh() {
    let missing_vertex =
        b"ply\nformat ascii 1.0\nelement fragment 1\nproperty float x\nend_header\n";
    let error = PlyGaussians::read_header(&mut missing_vertex.as_slice()).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Gaussian vertex element not found in PLY header"
    );

    let original = degree_gaussians(1);
    let bytes = reordered_buffer(&original.gaussians[0], 1, ply_rs::ply::Encoding::Ascii);
    let text = String::from_utf8(bytes).unwrap();

    for bad in [
        text.replace("property float f_rest_0", "property float f_rest_9"),
        text.replace("property float f_rest_0", "property int f_rest_0"),
        text.replace("property float f_rest_0\n", ""),
        text.replace(
            "property float f_rest_0",
            "property list uchar float f_rest_0",
        ),
    ] {
        assert_eq!(
            PlyGaussians::read_header(&mut bad.as_bytes())
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidData
        );
    }
}

#[test]
fn test_ply_read_when_missing_value_or_truncated_record_should_error() {
    let original = degree_gaussians(2);
    let bytes = reordered_buffer(&original.gaussians[0], 2, ply_rs::ply::Encoding::Ascii);
    let mut text = String::from_utf8(bytes).unwrap();
    text.pop();

    let end = text.rfind(' ').unwrap();
    text.truncate(end);

    let result = PlyGaussians::read_from(&mut text.as_bytes());

    assert_matches!(result, Err(e) if e.kind() == std::io::ErrorKind::InvalidData);

    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();
    bytes.pop();

    assert_matches!(PlyGaussians::read_from(&mut bytes.as_slice()), Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof);
}

#[test]
fn test_ply_writer_rejects_mismatched_sh_and_invalid_degree() {
    assert!(PlyGaussians::new(vec![], 5).is_err());

    let mut original = degree_gaussians(4);
    original.gaussians[0].sh.pop();
    let mut bytes = Vec::new();

    assert_eq!(
        original.write_to(&mut bytes).unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput
    );
}

#[test]
fn test_ply_file_round_trip_and_collection_constructors() {
    let gaussians = given::ply_gaussians();
    let path = given::temp_file_path(".ply");

    gaussians.write_to_file(&path).unwrap();

    assert_eq!(PlyGaussians::read_from_file(&path).unwrap(), gaussians);
    assert_eq!(gaussians.len(), 2);
    assert!(!gaussians.is_empty());

    let from_vec = PlyGaussians::try_from(gaussians.gaussians.clone()).unwrap();
    let mut from_iter: PlyGaussians = given::gaussians().iter().collect();

    assert_eq!(from_vec, gaussians);
    assert_eq!(
        PlyGaussians::try_from_iter(gaussians.gaussians.clone()).unwrap(),
        gaussians
    );
    assert_eq!(from_iter, gaussians);

    for (ply, gaussian) in from_iter.iter_mut().zip(gaussians.iter_gaussian()) {
        assert::ply_gaussian_pod(ply, &gaussian.to_ply());
    }
}

#[test]
fn test_ply_collection_conversions_reject_invalid_sh_lengths() {
    for degree in 0..=4 {
        let original = degree_gaussians(degree);
        assert_eq!(
            PlyGaussians::try_from_iter(original.gaussians.clone()).unwrap(),
            original
        );
    }

    let empty = PlyGaussians::try_from(Vec::new()).unwrap();

    assert_eq!(empty.sh_degree, 3);
    assert_eq!(PlyGaussians::try_from_iter(Vec::new()).unwrap(), empty);

    let valid = degree_gaussians(2).gaussians.remove(0);
    let mut invalid = valid.clone();
    invalid.sh.pop();

    let mut unsupported = valid.clone();
    unsupported.sh = vec![0.0; 3 * ((5_usize + 1).pow(2) - 1)];

    let mixed = vec![valid, degree_gaussians(1).gaussians.remove(0)];

    assert_matches!(
        PlyGaussians::try_from(vec![invalid.clone()]),
        Err(PlyGaussiansFromIterError::UnsupportedShCount { count }) if count == invalid.sh.len()
    );
    assert_matches!(
        PlyGaussians::try_from_iter(vec![unsupported.clone()]),
        Err(PlyGaussiansFromIterError::UnsupportedShCount { count }) if count == unsupported.sh.len()
    );

    for records in [vec![invalid], vec![unsupported], mixed.clone()] {
        assert_eq!(
            PlyGaussians::new(records, 2).unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }

    assert_matches!(
        PlyGaussians::try_from(mixed.clone()),
        Err(PlyGaussiansFromIterError::ShCountMismatch {
            actual_count: 9,
            expected_count: 24
        })
    );
    assert_matches!(
        PlyGaussians::try_from_iter(mixed),
        Err(PlyGaussiansFromIterError::ShCountMismatch {
            actual_count: 9,
            expected_count: 24
        })
    );
}

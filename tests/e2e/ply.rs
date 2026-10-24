use std::io::Write;

use assert_matches::assert_matches;
use wgpu_3dgs_core::{
    BatchRead, BatchWrite, Gaussian, GaussianStream, IterGaussian, PlyBatchReader, PlyBatchWriter,
    PlyGaussian, PlyGaussianStream, PlyGaussians, PlyGaussiansFromIterError, ReadIterGaussian,
    WriteIterGaussian, glam::*,
};

use crate::common::{assert, given};

fn sh_degree_gaussians(degree: u8) -> PlyGaussians {
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

fn record_values(ply: &PlyGaussian) -> Vec<f32> {
    ply.pos
        .to_array()
        .into_iter()
        .chain(ply.normal.to_array())
        .chain(ply.color.to_array())
        .chain(ply.sh.iter().copied())
        .chain([ply.alpha])
        .chain(ply.scale.to_array())
        .chain([ply.rot.w, ply.rot.x, ply.rot.y, ply.rot.z])
        .collect()
}

/// Encode test records with a selectable property layout and normal components.
///
/// `custom` reverses known properties and interleaves unknown properties of every
/// scalar type. Bits 0, 1, and 2 of `normals` include nx, ny, and nz, respectively.
fn layout_buffer(
    original: &PlyGaussians,
    encoding: ply_rs::ply::Encoding,
    custom: bool,
    normals: u8,
) -> Vec<u8> {
    use ply_rs::ply::Encoding;

    let rest = ((original.sh_degree as usize + 1).pow(2) - 1) * 3;
    let names = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    .into_iter()
    .map(str::to_string)
    .chain((0..rest).map(|i| format!("f_rest_{i}")))
    .chain(
        [
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ]
        .map(str::to_string),
    );

    let mut properties: Vec<_> = names
        .enumerate()
        .filter(|(index, _)| !(3..6).contains(index) || normals & (1 << (index - 3)) != 0)
        .map(|(index, name)| (name, "float", Some(index)))
        .collect();

    let extras = [
        ("char", vec![(-7i8) as u8], "-7"),
        ("uchar", vec![250], "250"),
        ("short", (-1234i16).to_le_bytes().to_vec(), "-1234"),
        ("ushort", 60000u16.to_le_bytes().to_vec(), "60000"),
        ("int", (-123456i32).to_le_bytes().to_vec(), "-123456"),
        ("uint", 4000000000u32.to_le_bytes().to_vec(), "4000000000"),
        ("float", 123.25f32.to_le_bytes().to_vec(), "123.25"),
        ("double", (-987.125f64).to_le_bytes().to_vec(), "-987.125"),
    ];

    if custom {
        properties.reverse();
        for (index, (ty, _, _)) in extras.iter().enumerate() {
            properties.insert(index * 2, (format!("extra_{ty}"), ty, None));
        }
    }

    let mut bytes = Vec::new();
    writeln!(
        bytes,
        "ply\nformat {encoding} 1.0\nelement vertex {}",
        original.len()
    )
    .unwrap();
    for (name, ty, _) in &properties {
        writeln!(bytes, "property {ty} {name}").unwrap();
    }
    writeln!(bytes, "end_header").unwrap();

    for gaussian in original.iter() {
        let values = record_values(gaussian);
        for (_, ty, index) in &properties {
            if let Some(index) = index {
                match encoding {
                    Encoding::Ascii => write!(bytes, "{} \t", values[*index]).unwrap(),
                    Encoding::BinaryLittleEndian => bytes.extend(values[*index].to_le_bytes()),
                    Encoding::BinaryBigEndian => bytes.extend(values[*index].to_be_bytes()),
                }
            } else {
                let (_, value, token) = extras
                    .iter()
                    .find(|(extra_ty, _, _)| extra_ty == ty)
                    .unwrap();
                match encoding {
                    Encoding::Ascii => write!(bytes, "{token}\t").unwrap(),
                    Encoding::BinaryLittleEndian => bytes.extend(value),
                    Encoding::BinaryBigEndian => bytes.extend(value.iter().rev()),
                }
            }
        }
        if encoding == Encoding::Ascii {
            bytes.extend_from_slice(b"\r\n");
        }
    }

    bytes
}

#[test]
fn test_ply_read_layouts_with_small_buffers_should_preserve_records_and_trailing_data() {
    use ply_rs::ply::Encoding;
    use std::{io::BufReader, num::NonZeroUsize};

    for degree in 0..=4 {
        for (custom, normals) in [(false, 7), (true, 7), (true, 0), (true, 1)] {
            let first = sh_degree_gaussians(degree).gaussians.remove(0);
            let mut second = first.clone();
            second.pos += Vec3::new(4.0, -2.0, 0.5);
            second.color += Vec3::splat(0.25);
            second.alpha = -0.75;
            second.sh.iter_mut().for_each(|value| *value += 0.5);

            let mut original = PlyGaussians::new(vec![first, second], degree).unwrap();
            for gaussian in original.iter_mut() {
                gaussian.normal = Vec3::new(
                    if normals & 1 != 0 { 0.25 } else { 0.0 },
                    if normals & 2 != 0 { -0.5 } else { 0.0 },
                    if normals & 4 != 0 { 0.75 } else { 0.0 },
                );
            }

            for encoding in [
                Encoding::Ascii,
                Encoding::BinaryLittleEndian,
                Encoding::BinaryBigEndian,
            ] {
                let mut bytes = layout_buffer(&original, encoding, custom, normals);
                bytes.extend_from_slice(b"trailing data");

                // A slice exercises the direct fill_buf path for custom layouts.
                let mut input = bytes.as_slice();
                assert_eq!(PlyGaussians::read_from(&mut input).unwrap(), original);
                assert_eq!(input, b"trailing data");

                for capacity in [1, 7, 256, 4096] {
                    let mut input = BufReader::with_capacity(capacity, bytes.as_slice());
                    assert_eq!(PlyGaussians::read_from(&mut input).unwrap(), original);
                    let mut trailing = String::new();
                    std::io::Read::read_to_string(&mut input, &mut trailing).unwrap();
                    assert_eq!(trailing, "trailing data");
                }

                let mut input = bytes.as_slice();
                let mut stream = PlyGaussianStream::new(&mut input).unwrap();
                let mut out = Vec::new();
                assert_eq!(
                    stream
                        .next_batch(NonZeroUsize::new(1).unwrap(), &mut out)
                        .unwrap(),
                    1
                );
                assert_eq!(stream.progress().completed_units, 1);
                assert!(!stream.progress().done);
                assert_eq!(
                    stream
                        .next_batch(NonZeroUsize::new(3).unwrap(), &mut out)
                        .unwrap(),
                    1
                );
                assert!(stream.progress().done);
                assert!(stream.next().is_none());
                assert_eq!(out, original.gaussians);
                drop(stream);
                assert_eq!(input, b"trailing data");

                let mut batch = PlyBatchReader::new(bytes.as_slice()).unwrap();
                assert_eq!(
                    batch
                        .step(NonZeroUsize::new(1).unwrap())
                        .unwrap()
                        .completed_units,
                    1
                );
                assert!(batch.step(NonZeroUsize::new(3).unwrap()).unwrap().done);
                assert_eq!(batch.finish().unwrap(), original);
            }
        }
    }
}

#[test]
fn test_ply_binary_read_when_custom_or_canonical_should_preserve_float_bits() {
    use ply_rs::ply::Encoding;

    let mut original = sh_degree_gaussians(4);
    original.gaussians[0].pos.x = -0.0;
    original.gaussians[0].sh[..4].copy_from_slice(&[
        f32::from_bits(0x7fc01234),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
    ]);

    for encoding in [Encoding::BinaryLittleEndian, Encoding::BinaryBigEndian] {
        for custom in [false, true] {
            let bytes = layout_buffer(&original, encoding, custom, 7);
            let actual = PlyGaussians::read_from(&mut bytes.as_slice()).unwrap();
            assert_eq!(
                record_values(&actual.gaussians[0])
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                record_values(&original.gaussians[0])
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
            );
        }
    }
}

#[test]
fn test_ply_binary_read_when_standard_layout_has_extra_property_should_use_full_stride() {
    for degree in 0..=4 {
        let original = sh_degree_gaussians(degree);
        let mut bytes = Vec::new();
        original.write_to(&mut bytes).unwrap();
        let marker = b"end_header\n";
        let end = bytes
            .windows(marker.len())
            .position(|window| window == marker)
            .unwrap();

        let mut input = bytes[..end].to_vec();
        input.extend_from_slice(b"property uchar extra\nend_header\n");
        input.extend_from_slice(&bytes[end + marker.len()..]);
        input.push(0xab);
        input.extend_from_slice(b"trailing data");

        let mut reader = input.as_slice();
        assert_eq!(PlyGaussians::read_from(&mut reader).unwrap(), original);
        assert_eq!(reader, b"trailing data");
    }
}

#[test]
fn test_ply_whole_model_read_when_count_overflows_allocation_should_return_error() {
    let original = sh_degree_gaussians(0);
    let bytes = layout_buffer(&original, ply_rs::ply::Encoding::Ascii, false, 7);
    let text = String::from_utf8(bytes).unwrap();
    let text = text.replace(
        "element vertex 1",
        &format!("element vertex {}", usize::MAX),
    );

    assert!(PlyGaussians::read_from(&mut text.as_bytes()).is_err());

    // Stream construction still does not allocate storage for the declared model.
    let stream = PlyGaussianStream::new(text.as_bytes()).unwrap();
    assert_eq!(stream.total_gaussians(), usize::MAX);
    assert_eq!(stream.progress().completed_units, 0);
}

#[test]
fn test_ply_stream_when_record_is_truncated_should_fail_once_and_keep_completed_records() {
    use ply_rs::ply::Encoding;
    use std::{io::BufReader, num::NonZeroUsize};

    for degree in 0..=4 {
        let record = sh_degree_gaussians(degree).gaussians.remove(0);
        let original = PlyGaussians::new(vec![record; 2], degree).unwrap();
        for encoding in [
            Encoding::Ascii,
            Encoding::BinaryLittleEndian,
            Encoding::BinaryBigEndian,
        ] {
            for custom in [false, true] {
                let mut bytes = layout_buffer(&original, encoding, custom, 7);

                if encoding == Encoding::Ascii {
                    // Remove the last token as well as the line ending.
                    let text = std::str::from_utf8(&bytes).unwrap();
                    let end = text.trim_end().rfind(char::is_whitespace).unwrap();
                    bytes.truncate(end);
                } else {
                    bytes.pop();
                }

                let reader = BufReader::with_capacity(7, bytes.as_slice());
                let mut stream = PlyGaussianStream::new(reader).unwrap();
                let mut out = Vec::new();
                let error = stream
                    .next_batch(NonZeroUsize::new(3).unwrap(), &mut out)
                    .unwrap_err();

                assert_eq!(
                    error.kind(),
                    if encoding == Encoding::Ascii {
                        std::io::ErrorKind::InvalidData
                    } else {
                        std::io::ErrorKind::UnexpectedEof
                    }
                );
                assert_eq!(out, original.gaussians[..1]);
                assert_eq!(stream.progress().completed_units, 1);
                assert!(!stream.progress().done);
                assert!(stream.next().is_none());
            }
        }
    }
}

#[test]
fn test_ply_ascii_read_should_check_known_values_and_token_counts_but_skip_unknown_values() {
    use ply_rs::ply::Encoding;

    let original = sh_degree_gaussians(0);
    let bytes = layout_buffer(&original, Encoding::Ascii, true, 7);
    let text = String::from_utf8(bytes).unwrap();
    let (header, body) = text.split_once("end_header\n").unwrap();
    let mut tokens: Vec<_> = body.split_whitespace().collect();

    // Unknown properties are skipped, including float properties.
    tokens[0] = "ignored";
    tokens[12] = "ignored_float";

    let input = format!("{header}end_header\n{}", tokens.join(" "));
    assert_eq!(
        PlyGaussians::read_from(&mut input.as_bytes()).unwrap(),
        original
    );
    tokens[1] = "invalid_float";

    let input = format!("{header}end_header\n{}", tokens.join(" "));
    assert_eq!(
        PlyGaussians::read_from(&mut input.as_bytes())
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::InvalidData
    );

    let extra = format!("{} extra\n", text.trim_end());
    assert_eq!(
        PlyGaussians::read_from(&mut extra.as_bytes())
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::InvalidData
    );
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

        let round_trip = Gaussian::<wgpu_3dgs_core::ShDegree3>::from_ply(&ply)
            .unwrap()
            .to_ply();

        assert!(round_trip.color.abs_diff_eq(color, 1e-6));
        assert!((round_trip.alpha - ply.alpha).abs() < 1e-6);
    }
}

#[test]
fn test_ply_gaussian_from_gaussian_should_equal_to_ply() {
    let gaussian = given::gaussian();

    assert::ply_gaussian_pod(&gaussian.to_ply(), &PlyGaussian::from(&gaussian));
    assert::ply_gaussian_pod(&gaussian.to_ply(), &PlyGaussian::from(gaussian));
}

#[test]
fn test_ply_round_trip_should_preserve_all_native_degrees() {
    for degree in 0..=4 {
        let original = sh_degree_gaussians(degree);
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

        let actual = PlyGaussians::read_gaussians(&mut input, header).unwrap();

        assert_eq!(actual, original);
        assert!(input.is_empty());
        assert_eq!(
            PlyGaussians::read_from(&mut bytes.as_slice()).unwrap(),
            original
        );

        let converted = wgpu_3dgs_core::AnyGaussian::from_ply(&actual.gaussians[0]).unwrap();

        assert_eq!(converted.sh_degree().get(), degree);
        assert::ply_gaussian_pod(&converted.to_ply(), &actual.gaussians[0]);
    }
}

#[test]
fn test_ply_round_trip_when_degree_four_model_is_empty_should_preserve_header() {
    let original = PlyGaussians::new(Vec::new(), 4).unwrap();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();

    assert_eq!(
        PlyGaussians::read_from(&mut bytes.as_slice()).unwrap(),
        original
    );
}

#[test]
fn test_ply_read_gaussians_when_model_is_empty_should_preserve_header_degree() {
    for degree in 0..=4 {
        let original = PlyGaussians::new(Vec::new(), degree).unwrap();
        let mut bytes = Vec::new();
        original.write_to(&mut bytes).unwrap();

        let mut input = bytes.as_slice();
        let header = PlyGaussians::read_header(&mut input).unwrap();

        assert_eq!(
            PlyGaussians::read_gaussians(&mut input, header).unwrap(),
            original
        );
        assert!(input.is_empty());
    }
}

#[test]
fn test_ply_stream_from_header_should_preserve_metadata_and_deliver_records() {
    for degree in 0..=4 {
        for count in [0, 3] {
            let record = sh_degree_gaussians(degree).gaussians.remove(0);
            let original = PlyGaussians::new(vec![record; count], degree).unwrap();
            let mut bytes = Vec::new();
            original.write_to(&mut bytes).unwrap();
            bytes.extend_from_slice(b"trailing data");

            let mut input = bytes.as_slice();
            let header = PlyGaussians::read_header(&mut input).unwrap();
            let remaining = input;

            let mut body = remaining;
            assert_eq!(
                PlyGaussians::read_gaussians(&mut body, header.clone()).unwrap(),
                original
            );
            assert_eq!(body, b"trailing data");

            // Preparing a stream must not consume any body data.
            let stream = PlyGaussianStream::from_header(&mut input, header.clone());
            assert_eq!(stream.header().sh_degree, degree);
            assert_eq!(stream.total_gaussians(), count);
            assert_eq!(stream.progress().completed_units, 0);
            assert_eq!(stream.progress().done, count == 0);
            drop(stream);
            assert_eq!(input, remaining);

            let mut stream = PlyGaussianStream::from_header(&mut input, header);
            let mut records = Vec::new();
            for expected in &original.gaussians {
                records.push(stream.next().unwrap().unwrap());
                assert_eq!(records.last().unwrap(), expected);
                assert_eq!(stream.progress().completed_units, records.len());
            }
            assert!(stream.progress().done);
            assert!(stream.next().is_none());
            drop(stream);
            assert_eq!(input, b"trailing data");
        }
    }
}

#[test]
fn test_ply_read_when_ascii_or_binary_properties_are_reordered_should_preserve_all_degrees() {
    for degree in [1, 2, 3, 4] {
        let original = sh_degree_gaussians(degree);

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
fn test_ply_stream_and_batch_writer_should_preserve_variable_degree() {
    for degree in [1, 2, 4] {
        let original = sh_degree_gaussians(degree);
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
fn test_ply_read_header_when_vertex_is_missing_or_sh_is_malformed_should_return_error() {
    let missing_vertex =
        b"ply\nformat ascii 1.0\nelement fragment 1\nproperty float x\nend_header\n";
    let error = PlyGaussians::read_header(&mut missing_vertex.as_slice()).unwrap_err();

    assert_eq!(
        error.to_string(),
        "Gaussian vertex element not found in PLY header"
    );

    let original = sh_degree_gaussians(1);
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
    let original = sh_degree_gaussians(2);
    let bytes = reordered_buffer(&original.gaussians[0], 2, ply_rs::ply::Encoding::Ascii);
    let mut text = String::from_utf8(bytes).unwrap();
    text.pop();

    let end = text.rfind(' ').unwrap();
    text.truncate(end);

    let result = PlyGaussians::read_from(&mut text.as_bytes());

    assert_matches!(result, Err(e) if e.kind() == std::io::ErrorKind::InvalidData);

    let mut input = text.as_bytes();
    let header = PlyGaussians::read_header(&mut input).unwrap();
    assert_matches!(PlyGaussians::read_gaussians(&mut input, header), Err(e) if e.kind() == std::io::ErrorKind::InvalidData);

    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();
    bytes.pop();

    assert_matches!(PlyGaussians::read_from(&mut bytes.as_slice()), Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof);

    let mut input = bytes.as_slice();
    let header = PlyGaussians::read_header(&mut input).unwrap();
    assert_matches!(PlyGaussians::read_gaussians(&mut input, header.clone()), Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof);

    let mut input = bytes.as_slice();
    let header = PlyGaussians::read_header(&mut input).unwrap();
    let mut stream = PlyGaussianStream::from_header(input, header);
    assert_matches!(stream.next(), Some(Err(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof);
    assert!(stream.next().is_none());
    assert!(!stream.progress().done);
}

#[test]
fn test_ply_writer_when_sh_mismatches_or_degree_is_invalid_should_return_error() {
    assert!(PlyGaussians::new(vec![], 5).is_err());

    let mut original = sh_degree_gaussians(4);
    original.gaussians[0].sh.pop();
    let mut bytes = Vec::new();

    assert_eq!(
        original.write_to(&mut bytes).unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput
    );
}

#[test]
fn test_ply_file_round_trip_and_collection_constructors_should_preserve_gaussians() {
    let gaussians = given::ply_gaussians();
    let path = given::temp_file_path(".ply");

    gaussians.write_to_file(&path).unwrap();

    assert_eq!(PlyGaussians::read_from_file(&path).unwrap(), gaussians);
    assert_eq!(gaussians.len(), 2);
    assert!(!gaussians.is_empty());

    let from_vec = PlyGaussians::try_from(gaussians.gaussians.clone()).unwrap();
    let mut from_iter = PlyGaussians::from_gaussians(given::gaussians()).unwrap();

    assert_eq!(from_vec, gaussians);
    assert_eq!(
        PlyGaussians::from_iter(gaussians.gaussians.clone(), gaussians.sh_degree).unwrap(),
        gaussians
    );
    assert_eq!(from_iter, gaussians);

    for (ply, gaussian) in from_iter.iter_mut().zip(gaussians.iter_gaussian()) {
        assert::ply_gaussian_pod(ply, &gaussian.to_ply());
    }
}

#[test]
fn test_ply_from_iter_should_require_matching_explicit_degree_including_empty_input() {
    for degree in 0..=4 {
        let empty = PlyGaussians::from_iter(std::iter::empty(), degree).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.sh_degree, degree);

        let original = sh_degree_gaussians(degree);
        assert_eq!(
            PlyGaussians::from_iter(original.gaussians.clone().into_iter(), degree).unwrap(),
            original
        );
    }

    assert_matches!(
        PlyGaussians::from_iter(Vec::new(), 5),
        Err(PlyGaussiansFromIterError::UnsupportedShDegree { degree: 5 })
    );
    let records = sh_degree_gaussians(1).gaussians;
    assert_matches!(
        PlyGaussians::from_iter(records, 2),
        Err(PlyGaussiansFromIterError::ShCountMismatch {
            actual_count: 9,
            expected_count: 24,
        })
    );
}

#[test]
fn test_ply_collection_conversions_when_sh_lengths_are_invalid_should_return_error() {
    for degree in 0..=4 {
        let original = sh_degree_gaussians(degree);
        assert_eq!(
            PlyGaussians::from_iter(original.gaussians.clone(), degree).unwrap(),
            original
        );
    }

    let empty = PlyGaussians::try_from(Vec::new()).unwrap();

    assert_eq!(empty.sh_degree, 3);
    assert_eq!(PlyGaussians::from_iter(Vec::new(), 3).unwrap(), empty);

    let valid = sh_degree_gaussians(2).gaussians.remove(0);
    let mut invalid = valid.clone();
    invalid.sh.pop();

    let mut unsupported = valid.clone();
    unsupported.sh = vec![0.0; 3 * ((5_usize + 1).pow(2) - 1)];

    let mixed = vec![valid, sh_degree_gaussians(1).gaussians.remove(0)];

    assert_matches!(
        PlyGaussians::try_from(vec![invalid.clone()]),
        Err(PlyGaussiansFromIterError::UnsupportedShCount { count }) if count == invalid.sh.len()
    );
    assert_matches!(
        PlyGaussians::from_iter(vec![unsupported.clone()], 2),
        Err(PlyGaussiansFromIterError::ShCountMismatch {
            actual_count,
            expected_count: 24,
        }) if actual_count == unsupported.sh.len()
    );

    for records in [vec![invalid], vec![unsupported], mixed.clone()] {
        let actual_count = records.iter().find(|g| g.sh.len() != 24).unwrap().sh.len();
        assert_matches!(
            PlyGaussians::new(records, 2),
            Err(PlyGaussiansFromIterError::ShCountMismatch {
                actual_count: count,
                expected_count: 24,
            }) if count == actual_count
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
        PlyGaussians::from_iter(mixed, 2),
        Err(PlyGaussiansFromIterError::ShCountMismatch {
            actual_count: 9,
            expected_count: 24
        })
    );
}

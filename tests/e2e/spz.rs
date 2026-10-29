use assert_matches::assert_matches;
use glam::*;
use std::{
    io::{self, ErrorKind, Read},
    num::NonZeroUsize,
};
use wgpu_3dgs_core::{
    AnyGaussian, BatchRead, Gaussian, GaussianToSpzOptions, Gaussians, GaussiansBatchReader,
    GaussiansSource, IterGaussian, ReadIterGaussian, SpzBatchReader, SpzBatchWriter, SpzGaussian,
    SpzGaussianPosition, SpzGaussianRef, SpzGaussianRotation, SpzGaussianSh, SpzGaussianShDegree,
    SpzGaussianShRef, SpzGaussians, SpzGaussiansCollectError, SpzGaussiansFromGaussianSliceOptions,
    SpzGaussiansFromGaussiansError, SpzGaussiansFromIterError, SpzGaussiansHeader,
    SpzGaussiansHeaderPod, SpzGaussiansPositions, SpzGaussiansRotations, SpzGaussiansShs, SpzPhase,
    WriteIterGaussian,
};

use crate::common::{assert, given};

// SPZ has relatively loose precision requirements
const ASSERT_GAUSSIAN_OPTIONS: assert::GaussianOptions = assert::GaussianOptions {
    pos_epsilon: 1.0,
    rot_epsilon: 1e-1,
    color_tolerance: 2,
    sh_epsilon: 1e-1,
    scale_epsilon: 1.0,
};

fn given_spz_gaussian_and_header(
    num_point: u32,
    options: &SpzGaussiansFromGaussianSliceOptions,
) -> (SpzGaussian, SpzGaussiansHeader) {
    (
        SpzGaussian {
            position: match options.version {
                1 => SpzGaussianPosition::Float16([0; 3]),
                _ => SpzGaussianPosition::FixedPoint24([[0; 3]; 3]),
            },
            scale: [0; 3],
            rotation: match options.version {
                1..=2 => SpzGaussianRotation::QuatFirstThree([0; 3]),
                _ => SpzGaussianRotation::QuatSmallestThree([0; 4]),
            },
            alpha: 0,
            color: [0; 3],
            sh: match options.sh_degree.get() {
                0 => SpzGaussianSh::Zero,
                1 => SpzGaussianSh::One([[0; 3]; 3]),
                2 => SpzGaussianSh::Two([[0; 3]; 8]),
                3 => SpzGaussianSh::Three([[0; 3]; 15]),
                _ => panic!("invalid SH degree"),
            },
        },
        SpzGaussiansHeader::new(
            options.version,
            num_point,
            options.sh_degree,
            options.fractional_bits,
            options.antialiased,
        )
        .expect("valid header"),
    )
}

fn test_spz_gaussians_buffer_round_trip_with_options_should_preserve_count(
    options: &SpzGaussiansFromGaussianSliceOptions,
) {
    let from_options = SpzGaussians::from_gaussians_with_options(
        given::runtime_gaussians(options.sh_degree.get()),
        options,
    )
    .unwrap();

    let mut buffer = Vec::new();
    from_options.write_to(&mut buffer).unwrap();
    let spz_gaussians_read = SpzGaussians::read_from(&mut buffer.as_slice()).unwrap();
    assert_eq!(from_options.len(), spz_gaussians_read.len());
}

fn test_spz_gaussians_from_gaussians_with_options_and_iter_should_be_equal(
    options: &SpzGaussiansFromGaussianSliceOptions,
    mut assertion: impl FnMut(SpzGaussianRef, &Gaussian, &SpzGaussiansHeader),
) {
    let gaussians = given::gaussians();
    let from_options = SpzGaussians::from_gaussians_with_options(
        given::runtime_gaussians(options.sh_degree.get()),
        options,
    )
    .unwrap();

    for (a, b) in from_options.iter().zip(gaussians.iter()) {
        assertion(a, b, &from_options.header);
    }
}

fn fixture(degree: usize) -> Vec<u8> {
    let hex = include_str!("../fixtures/spz-v4/upstream.hex")
        .lines()
        .nth(degree)
        .unwrap();

    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

fn read(bytes: &[u8]) -> io::Result<SpzGaussians> {
    SpzGaussians::read_from(&mut &*bytes)
}

fn set_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn set_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn replace_first_frame(mut bytes: Vec<u8>, frame: &[u8]) -> Vec<u8> {
    let start = 32 + usize::from(bytes[15]) * 16;
    let old_size = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;

    set_u64(&mut bytes, 32, frame.len() as u64);
    bytes.splice(start..start + old_size, frame.iter().copied());

    bytes
}

// A standard ZSTD frame without a content-size field, using raw blocks and a
// 128 KiB window. This is independent of the production reader and its decoder.
fn raw_frame(data: &[u8]) -> Vec<u8> {
    let mut frame = vec![0x28, 0xb5, 0x2f, 0xfd, 0, 0x38];
    let blocks = data.chunks(128 * 1024);
    let block_count = blocks.len();

    for (index, block) in blocks.enumerate() {
        let header = (block.len() as u32) << 3 | u32::from(index + 1 == block_count);
        frame.extend_from_slice(&header.to_le_bytes()[..3]);
        frame.extend_from_slice(block);
    }

    frame
}

fn with_extension(mut bytes: Vec<u8>, extension: &[u8]) -> Vec<u8> {
    bytes[14] |= 0x2;
    set_u32(&mut bytes, 16, 32 + extension.len() as u32);
    bytes.splice(32..32, extension.iter().copied());

    bytes
}

// Deliberately neither BufRead nor Seek, and short reads split the magic/header/TOC.
struct ShortRead<'a>(&'a [u8]);

impl Read for ShortRead<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let count = out.len().min(3);

        self.0.read(&mut out[..count])
    }
}

#[test]
fn test_gaussian_to_spz_options_when_degree_is_zero_should_return_none() {
    let degree = SpzGaussianShDegree::new(0).unwrap();
    let options = GaussianToSpzOptions::default();

    assert_eq!(options.sh_bits(degree), None);
    assert_eq!(options.sh_bucket_size(degree), None);
}

#[test]
fn test_gaussian_to_spz_options_when_degree_is_nonzero_should_return_quantization_values() {
    for (options, expected_bits, expected_bucket_sizes) in [
        (GaussianToSpzOptions::default(), [5, 4, 4], [8, 16, 16]),
        (
            GaussianToSpzOptions {
                sh_quantize_bits: [0, 6, 8],
            },
            [0, 6, 8],
            [256, 4, 1],
        ),
    ] {
        for degree in 1..=3 {
            let index = degree as usize - 1;
            let degree = SpzGaussianShDegree::new(degree).unwrap();

            assert_eq!(options.sh_bits(degree), Some(expected_bits[index]));
            assert_eq!(
                options.sh_bucket_size(degree),
                Some(expected_bucket_sizes[index])
            );
        }
    }
}

#[test]
fn test_spz_phase_name_should_round_trip_through_from_str() {
    for phase in [
        SpzPhase::Positions,
        SpzPhase::Alphas,
        SpzPhase::Colors,
        SpzPhase::Scales,
        SpzPhase::Rotations,
        SpzPhase::Shs,
        SpzPhase::Done,
    ] {
        assert_eq!(phase.name().parse::<SpzPhase>().unwrap(), phase);
    }
}

#[test]
fn test_spz_color_round_trip_should_preserve_all_byte_values() {
    let (mut spz, header) = given_spz_gaussian_and_header(1, &Default::default());
    for value in 0..=255 {
        spz.color = [value; 3];
        spz.alpha = value;

        let gaussian = wgpu_3dgs_core::AnyGaussian::from_spz(spz.as_ref(), &header).unwrap();
        let round_trip = gaussian.to_spz(&header, &Default::default()).unwrap();

        assert_eq!(round_trip.color, spz.color);
        assert_eq!(round_trip.alpha, spz.alpha);
    }
}

#[test]
fn test_spz_gaussian_pod_len_and_is_empty_should_be_correct() {
    let spz_gaussians = given::spz_gaussians();

    assert_eq!(spz_gaussians.len(), 2);
    assert!(!spz_gaussians.is_empty());
}

#[test]
fn test_spz_gaussians_file_round_trip_should_be_equal() {
    let spz_gaussians = given::spz_gaussians();
    let path = given::temp_file_path(".spz");

    spz_gaussians.write_to_file(&path).unwrap();
    let spz_gaussians_read = SpzGaussians::read_from_file(&path).unwrap();

    assert_eq!(spz_gaussians.len(), spz_gaussians_read.len());

    for (a, b) in spz_gaussians.iter().zip(spz_gaussians_read.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn test_spz_gaussians_buffer_round_trip_should_be_equal() {
    let spz_gaussians = given::spz_gaussians();

    let mut buffer = Vec::new();
    spz_gaussians.write_to(&mut buffer).unwrap();
    let spz_gaussians_read = SpzGaussians::read_from(&mut buffer.as_slice()).unwrap();

    assert_eq!(spz_gaussians.len(), spz_gaussians_read.len());

    for (a, b) in spz_gaussians.iter().zip(spz_gaussians_read.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn test_spz_decompressed_write_and_reads_should_preserve_field_layout() {
    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        for degree in SpzGaussiansHeader::SUPPORTED_SH_DEGREES {
            let original = SpzGaussians::from_gaussians_with_options(
                given::runtime_gaussians(degree),
                &SpzGaussiansFromGaussianSliceOptions {
                    version,
                    sh_degree: SpzGaussianShDegree::new(degree).unwrap(),
                    ..Default::default()
                },
            )
            .unwrap();

            let mut expected = bytemuck::bytes_of(original.header.as_pod()).to_vec();
            match &original.positions {
                SpzGaussiansPositions::Float16(values) => {
                    expected.extend_from_slice(bytemuck::cast_slice(values));
                }
                SpzGaussiansPositions::FixedPoint24(values) => {
                    expected.extend_from_slice(bytemuck::cast_slice(values));
                }
            }
            expected.extend_from_slice(&original.alphas);
            expected.extend_from_slice(bytemuck::cast_slice(&original.colors));
            expected.extend_from_slice(bytemuck::cast_slice(&original.scales));
            match &original.rotations {
                SpzGaussiansRotations::QuatFirstThree(values) => {
                    expected.extend_from_slice(bytemuck::cast_slice(values));
                }
                SpzGaussiansRotations::QuatSmallestThree(values) => {
                    expected.extend_from_slice(bytemuck::cast_slice(values));
                }
            }
            match &original.shs {
                SpzGaussiansShs::Zero => {}
                SpzGaussiansShs::One(values) => {
                    expected.extend_from_slice(bytemuck::cast_slice(values));
                }
                SpzGaussiansShs::Two(values) => {
                    expected.extend_from_slice(bytemuck::cast_slice(values));
                }
                SpzGaussiansShs::Three(values) => {
                    expected.extend_from_slice(bytemuck::cast_slice(values));
                }
                SpzGaussiansShs::Four(_) => unreachable!("legacy fixture"),
            }

            let mut bytes = Vec::new();
            original.write_decompressed(&mut bytes).unwrap();
            assert_eq!(bytes, expected, "version {version}, SH degree {degree}");

            let mut input = bytes.as_slice();
            let header = SpzGaussians::read_header(&mut input).unwrap();
            assert_eq!(header, original.header);
            assert_eq!(
                SpzGaussians::read_gaussians(&mut input, header).unwrap(),
                original
            );
            assert!(input.is_empty());

            let mut input = bytes.as_slice();
            assert_eq!(
                SpzGaussians::read_decompressed(&mut input).unwrap(),
                original
            );
            assert!(input.is_empty());
        }
    }
}

#[test]
fn test_spz_decompressed_reads_when_header_or_fields_are_truncated_should_return_error() {
    let original = given::spz_gaussians();
    let mut bytes = Vec::new();
    original.write_decompressed(&mut bytes).unwrap();

    let header_size = std::mem::size_of::<SpzGaussiansHeaderPod>();
    assert_eq!(
        SpzGaussians::read_decompressed(&mut &bytes[..header_size - 1])
            .unwrap_err()
            .kind(),
        ErrorKind::UnexpectedEof
    );

    let field_lengths = [
        original.len() * 9, // version 3: fixed-point positions
        original.alphas.len(),
        original.colors.len() * 3,
        original.scales.len() * 3,
        original.len() * 4,      // version 3: smallest-three rotations
        original.len() * 15 * 3, // default SH degree 3
    ];
    let mut end = header_size;
    for length in field_lengths {
        end += length;
        let truncated = &bytes[..end - 1];
        let mut input = &truncated[header_size..];
        assert_eq!(
            SpzGaussians::read_gaussians(&mut input, original.header)
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
        let mut input = truncated;
        assert_eq!(
            SpzGaussians::read_decompressed(&mut input)
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
    }
    assert_eq!(end, bytes.len());
}

#[test]
fn test_spz_gaussians_buffer_round_trip_when_versions_vary_should_preserve_count() {
    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        println!("Version: {version}");
        test_spz_gaussians_buffer_round_trip_with_options_should_preserve_count(
            &SpzGaussiansFromGaussianSliceOptions {
                version,
                ..Default::default()
            },
        );
    }
}

#[test]
fn test_spz_gaussians_buffer_round_trip_when_sh_degrees_vary_should_preserve_count() {
    for sh_degree in SpzGaussiansHeader::SUPPORTED_SH_DEGREES {
        println!("SH Degree: {sh_degree}");
        test_spz_gaussians_buffer_round_trip_with_options_should_preserve_count(
            &SpzGaussiansFromGaussianSliceOptions {
                sh_degree: SpzGaussianShDegree::new(sh_degree).expect("valid SH degree"),
                ..Default::default()
            },
        );
    }
}

#[test]
fn test_spz_gaussians_buffer_round_trip_when_fractional_bits_vary_should_preserve_count() {
    for fractional_bits in [8, 12, 16] {
        println!("Fractional Bits: {fractional_bits}");
        test_spz_gaussians_buffer_round_trip_with_options_should_preserve_count(
            &SpzGaussiansFromGaussianSliceOptions {
                fractional_bits,
                ..Default::default()
            },
        );
    }
}

#[test]
fn test_spz_gaussians_from_iter_and_iter_iter_gaussian_should_be_equal() {
    let gaussians = given::gaussians();
    let original_spz = given::spz_gaussians(); // `spz_gaussians` uses the same as `gaussians`

    let from_iter = SpzGaussians::from_gaussians(gaussians.iter()).unwrap();

    assert_eq!(original_spz, from_iter);

    for (original_ref, iter_ref, slice_gaussian) in itertools::izip!(
        original_spz.iter(),
        from_iter.iter(),
        from_iter.iter_gaussian(),
    ) {
        assert_eq!(original_ref, iter_ref);
        assert::gaussian(
            &slice_gaussian,
            &wgpu_3dgs_core::AnyGaussian::from_spz(iter_ref, &from_iter.header).unwrap(),
            &ASSERT_GAUSSIAN_OPTIONS,
        );
    }
}

#[test]
fn test_spz_gaussians_from_gaussians_with_options_and_iter_when_versions_should_be_equal() {
    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        println!("Version: {version}");
        test_spz_gaussians_from_gaussians_with_options_and_iter_should_be_equal(
            &SpzGaussiansFromGaussianSliceOptions {
                version,
                ..Default::default()
            },
            |spz_gaussian_ref, gaussian, header| {
                let gaussian_from_spz =
                    wgpu_3dgs_core::AnyGaussian::from_spz(spz_gaussian_ref, header)
                        .unwrap()
                        .convert_sh_degree::<wgpu_3dgs_core::ShDegree3>();
                assert::gaussian(&gaussian_from_spz, gaussian, &ASSERT_GAUSSIAN_OPTIONS);
            },
        );
    }
}

#[test]
fn test_spz_gaussians_from_gaussians_with_options_and_iter_when_sh_degrees_should_be_equal() {
    for sh_degree in SpzGaussiansHeader::SUPPORTED_SH_DEGREES {
        println!("SH Degree: {sh_degree}");
        test_spz_gaussians_from_gaussians_with_options_and_iter_should_be_equal(
            &SpzGaussiansFromGaussianSliceOptions {
                sh_degree: SpzGaussianShDegree::new(sh_degree).expect("valid SH degree"),
                ..Default::default()
            },
            |spz_gaussian_ref, gaussian, header| {
                let gaussian_from_spz =
                    wgpu_3dgs_core::AnyGaussian::from_spz(spz_gaussian_ref, header)
                        .unwrap()
                        .convert_sh_degree::<wgpu_3dgs_core::ShDegree3>();
                assert::gaussian(
                    &gaussian_from_spz,
                    &Gaussian {
                        sh: gaussian
                            .sh
                            .iter()
                            .take(header.sh_num_coefficients())
                            .cloned()
                            .chain(std::iter::repeat(Vec3::ZERO))
                            .take(15)
                            .collect::<Vec<_>>()
                            .try_into()
                            .expect("SH coefficient with 15 elements"),
                        ..*gaussian
                    },
                    &ASSERT_GAUSSIAN_OPTIONS,
                );
            },
        );
    }
}

#[test]
fn test_spz_gaussians_from_gaussians_and_with_options_iter_when_fractional_bits_should_be_equal() {
    for fractional_bits in [8, 12, 16] {
        println!("Fractional Bits: {fractional_bits}");
        test_spz_gaussians_from_gaussians_with_options_and_iter_should_be_equal(
            &SpzGaussiansFromGaussianSliceOptions {
                fractional_bits,
                ..Default::default()
            },
            |spz_gaussian_ref, gaussian, header| {
                let gaussian_from_spz =
                    wgpu_3dgs_core::AnyGaussian::from_spz(spz_gaussian_ref, header)
                        .unwrap()
                        .convert_sh_degree::<wgpu_3dgs_core::ShDegree3>();
                assert::gaussian(&gaussian_from_spz, gaussian, &ASSERT_GAUSSIAN_OPTIONS);
            },
        );
    }
}

#[test]
fn test_spz_gaussians_from_gaussians_with_options_and_iter_when_sh_quantize_bits_should_be_equal() {
    for sh_quantize_bits in [
        [0; 3],
        [4; 3],
        [5; 3],
        [8; 3],
        [0, 1, 2],
        [2, 4, 6],
        [4, 5, 5],
    ] {
        println!("SH Quantize Bits: {:?}", sh_quantize_bits);
        test_spz_gaussians_from_gaussians_with_options_and_iter_should_be_equal(
            &SpzGaussiansFromGaussianSliceOptions {
                sh_quantize_bits,
                ..Default::default()
            },
            |spz_gaussian_ref, gaussian, header| {
                let gaussian_from_spz =
                    wgpu_3dgs_core::AnyGaussian::from_spz(spz_gaussian_ref, header)
                        .unwrap()
                        .convert_sh_degree::<wgpu_3dgs_core::ShDegree3>();
                assert::gaussian(&gaussian_from_spz, gaussian, &ASSERT_GAUSSIAN_OPTIONS);
            },
        );
    }
}

#[test]
fn test_spz_gaussians_from_gaussians_with_options_when_header_version_is_invalid_should_return_error()
 {
    let gaussians = given::gaussians();
    let options = wgpu_3dgs_core::SpzGaussiansFromGaussianSliceOptions {
        version: 999,
        ..Default::default()
    };

    let result = SpzGaussians::from_gaussians_with_options(gaussians.iter(), &options);

    assert_matches!(
        result,
        Err(SpzGaussiansFromGaussiansError::Header(e))
        if e.kind() == std::io::ErrorKind::InvalidData &&
            e.to_string() == "Unsupported SPZ version: 999, expected one of 1..=3"
    );
}

#[test]
fn test_spz_gaussians_from_empty_iter_should_use_header_variants() {
    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        for degree in SpzGaussiansHeader::SUPPORTED_SH_DEGREES {
            let options = SpzGaussiansFromGaussianSliceOptions {
                version,
                sh_degree: SpzGaussianShDegree::new(degree).unwrap(),
                ..Default::default()
            };
            let (_, header) = given_spz_gaussian_and_header(0, &options);
            let spz = SpzGaussians::from_iter(header, std::iter::empty::<SpzGaussian>()).unwrap();

            assert_eq!(spz.header, header);
            assert!(spz.is_empty());
            assert_eq!(spz.iter().count(), 0);
            assert_eq!(spz.iter_gaussian().count(), 0);
            assert!(spz.scales.is_empty());
            assert!(spz.alphas.is_empty());
            assert!(spz.colors.is_empty());
            assert_eq!(spz.positions.len(), 0);
            assert_eq!(spz.rotations.len(), 0);
            assert_eq!(spz.shs.degree(), header.sh_degree());
            assert_eq!(
                matches!(spz.positions, SpzGaussiansPositions::Float16(_)),
                header.uses_float16()
            );
            assert_eq!(
                matches!(spz.rotations, SpzGaussiansRotations::QuatSmallestThree(_)),
                header.uses_quat_smallest_three()
            );

            let from_gaussians = SpzGaussians::from_gaussians_with_options(
                Vec::<wgpu_3dgs_core::AnyGaussian>::new(),
                &options,
            )
            .unwrap();
            assert_eq!(from_gaussians, spz);

            let mut decompressed = Vec::new();
            spz.write_decompressed(&mut decompressed).unwrap();
            assert_eq!(
                SpzGaussians::read_decompressed(&mut decompressed.as_slice()).unwrap(),
                spz
            );

            let mut compressed = Vec::new();
            spz.write_to(&mut compressed).unwrap();
            assert_eq!(
                SpzGaussians::read_from(&mut compressed.as_slice()).unwrap(),
                spz
            );
        }
    }

    let collected = SpzGaussians::from_gaussians(std::iter::empty::<Gaussian>()).unwrap();
    assert!(collected.is_empty());
    assert_eq!(collected.shs.degree(), SpzGaussianShDegree::default());
}

#[test]
fn test_spz_gaussians_from_empty_iter_when_header_count_is_nonzero_should_return_count_mismatch() {
    let (_, header) = given_spz_gaussian_and_header(2, &Default::default());

    assert_matches!(
        SpzGaussians::from_iter(header, std::iter::empty::<SpzGaussian>()),
        Err(SpzGaussiansFromIterError::CountMismatch {
            actual_count: 0,
            header_count: 2
        })
    );
}

#[test]
fn test_spz_gaussians_from_iter_when_invalid_mixed_position_variant_should_return_error() {
    let (gaussian, header) = given_spz_gaussian_and_header(2, &Default::default());
    let gaussians = vec![
        SpzGaussian {
            position: SpzGaussianPosition::Float16([0; 3]),
            ..gaussian.clone()
        },
        SpzGaussian {
            position: SpzGaussianPosition::FixedPoint24([[0; 3]; 3]),
            ..gaussian.clone()
        },
    ];

    let result = SpzGaussians::from_iter(header, gaussians);

    assert_matches!(
        result,
        Err(SpzGaussiansFromIterError::InvalidMixedPositionVariant(
            SpzGaussiansCollectError::InvalidMixedVariant { .. }
        ))
    );
}

#[test]
fn test_spz_gaussians_from_iter_when_invalid_mixed_rotation_variant_should_return_error() {
    let (gaussian, header) = given_spz_gaussian_and_header(2, &Default::default());
    let gaussians = vec![
        SpzGaussian {
            rotation: SpzGaussianRotation::QuatFirstThree([0; 3]),
            ..gaussian.clone()
        },
        SpzGaussian {
            rotation: SpzGaussianRotation::QuatSmallestThree([0; 4]),
            ..gaussian.clone()
        },
    ];

    let result = SpzGaussians::from_iter(header, gaussians);

    assert_matches!(
        result,
        Err(SpzGaussiansFromIterError::InvalidMixedRotationVariant(
            SpzGaussiansCollectError::InvalidMixedVariant { .. }
        ))
    );
}

#[test]
fn test_spz_gaussians_from_iter_when_invalid_mixed_sh_variant_should_return_error() {
    let (gaussian, header) = given_spz_gaussian_and_header(2, &Default::default());
    let gaussians = vec![
        SpzGaussian {
            sh: SpzGaussianSh::Zero,
            ..gaussian.clone()
        },
        SpzGaussian {
            sh: SpzGaussianSh::One([[0; 3]; 3]),
            ..gaussian.clone()
        },
    ];

    let result = SpzGaussians::from_iter(header, gaussians);

    assert_matches!(
        result,
        Err(SpzGaussiansFromIterError::InvalidMixedShVariant(
            SpzGaussiansCollectError::InvalidMixedVariant { .. }
        ))
    );
}

#[test]
fn test_spz_gaussians_from_iter_when_header_count_mismatched_should_return_error() {
    let (gaussian, header) = given_spz_gaussian_and_header(2, &Default::default());
    let gaussians = vec![gaussian.clone()];

    let result = SpzGaussians::from_iter(header, gaussians);

    assert_matches!(
        result,
        Err(SpzGaussiansFromIterError::CountMismatch {
            actual_count: 1,
            header_count: 2
        })
    );
}

#[test]
fn test_spz_gaussians_from_iter_when_header_position_float16_mismatched_should_return_error() {
    // Version 1 uses Float16
    let (gaussian, header) = given_spz_gaussian_and_header(1, &Default::default());
    let gaussians = vec![SpzGaussian {
        position: SpzGaussianPosition::Float16([0; 3]),
        ..gaussian.clone()
    }];

    let result = SpzGaussians::from_iter(header, gaussians);

    assert_matches!(
        result,
        Err(SpzGaussiansFromIterError::PositionFloat16Mismatch {
            is_float16: true,
            header_uses_float16: false
        })
    );
}

#[test]
fn test_spz_gaussians_from_iter_when_header_rotation_quat_smallest_three_mismatched_should_return_error()
 {
    // Version 3 uses QuatSmallestThree
    let (gaussian, header) = given_spz_gaussian_and_header(1, &Default::default());
    let gaussians = vec![SpzGaussian {
        rotation: SpzGaussianRotation::QuatFirstThree([0; 3]),
        ..gaussian.clone()
    }];

    let result = SpzGaussians::from_iter(header, gaussians);

    assert_matches!(
        result,
        Err(
            SpzGaussiansFromIterError::RotationQuatSmallestThreeMismatch {
                is_quat_smallest_three: false,
                header_uses_quat_smallest_three: true
            }
        )
    );
}

#[test]
fn test_spz_gaussians_from_iter_when_header_sh_degree_mismatched_should_return_error() {
    for (sh, header_sh_degree) in [
        (
            SpzGaussianSh::Zero,
            SpzGaussianShDegree::new(1).expect("valid SH degree"),
        ),
        (
            SpzGaussianSh::One([[0; 3]; 3]),
            SpzGaussianShDegree::new(2).expect("valid SH degree"),
        ),
        (
            SpzGaussianSh::Two([[0; 3]; 8]),
            SpzGaussianShDegree::new(3).expect("valid SH degree"),
        ),
        (
            SpzGaussianSh::Three([[0; 3]; 15]),
            SpzGaussianShDegree::new(0).expect("valid SH degree"),
        ),
    ] {
        let (gaussian, header) = given_spz_gaussian_and_header(
            1,
            &SpzGaussiansFromGaussianSliceOptions {
                sh_degree: header_sh_degree,
                ..Default::default()
            },
        );
        let gaussians = vec![SpzGaussian {
            sh: sh.clone(),
            ..gaussian.clone()
        }];

        let result = SpzGaussians::from_iter(header, gaussians);

        assert_matches!(
            result,
            Err(SpzGaussiansFromIterError::ShDegreeMismatch {
                sh_degree,
                header_sh_degree,
            })
            if sh_degree == sh.degree() && header_sh_degree == header_sh_degree
        );
    }
}

#[test]
fn test_sh_gaussians_header_try_from_pod_when_magic_is_incorrect_should_return_error() {
    let pod = SpzGaussiansHeaderPod {
        magic: 0,
        version: 1,
        num_points: 0,
        sh_degree: SpzGaussianShDegree::default(),
        fractional_bits: 0,
        flags: 0,
        reserved: 0,
    };

    let result = SpzGaussiansHeader::try_from_pod(pod);

    assert_matches!(
        result,
        Err(e) if e.kind() == std::io::ErrorKind::InvalidData &&
            e.to_string() == "Invalid SPZ magic number: 0, expected 5053474E"
    );
}

#[test]
fn test_spz_gaussians_header_when_sh_degree_is_invalid_should_return_error() {
    let invalid_degree: SpzGaussianShDegree = bytemuck::cast(4u8);
    let pod = SpzGaussiansHeaderPod {
        magic: SpzGaussiansHeader::MAGIC,
        version: 3,
        num_points: 0,
        sh_degree: invalid_degree,
        fractional_bits: 12,
        flags: 0,
        reserved: 0,
    };

    assert_matches!(
        SpzGaussiansHeader::try_from_pod(pod),
        Err(e) if e.kind() == ErrorKind::InvalidData
    );

    let bytes = bytemuck::bytes_of(&pod);
    let mut input = bytes;
    assert_matches!(
        SpzGaussians::read_header(&mut input),
        Err(e) if e.kind() == ErrorKind::InvalidData
    );
}

#[test]
fn test_spz_gaussian_sh_degree_should_be_correct() {
    assert_eq!(SpzGaussianSh::Zero.degree().get(), 0);
    assert_eq!(SpzGaussianSh::One([[0; 3]; 3]).degree().get(), 1);
    assert_eq!(SpzGaussianSh::Two([[0; 3]; 8]).degree().get(), 2);
    assert_eq!(SpzGaussianSh::Three([[0; 3]; 15]).degree().get(), 3);
}

#[test]
fn test_spz_gaussian_sh_ref_degree_should_be_correct() {
    assert_eq!(SpzGaussianShRef::Zero.degree().get(), 0);
    assert_eq!(SpzGaussianShRef::One(&[[0; 3]; 3]).degree().get(), 1);
    assert_eq!(SpzGaussianShRef::Two(&[[0; 3]; 8]).degree().get(), 2);
    assert_eq!(SpzGaussianShRef::Three(&[[0; 3]; 15]).degree().get(), 3);
}

#[test]
fn test_spz_gaussians_shs_degree_should_be_correct() {
    assert_eq!(SpzGaussiansShs::Zero.degree().get(), 0);
    assert_eq!(SpzGaussiansShs::One(vec![]).degree().get(), 1);
    assert_eq!(SpzGaussiansShs::Two(vec![]).degree().get(), 2);
    assert_eq!(SpzGaussiansShs::Three(vec![]).degree().get(), 3);
}

#[test]
fn test_spz_gaussian_sh_iter_should_be_correct() {
    let sh = SpzGaussianSh::One([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let coeffs: Vec<_> = sh.iter().collect();
    assert_eq!(coeffs, vec![&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]);
}

#[test]
fn test_spz_gaussian_sh_iter_mut_should_be_correct() {
    let mut sh = SpzGaussianSh::One([[0; 3]; 3]);
    for coeff in sh.iter_mut() {
        *coeff = [1, 1, 1];
    }
    assert_eq!(sh, SpzGaussianSh::One([[1, 1, 1]; 3]));
}

#[test]
fn test_spz_gaussians_positions_len_and_is_empty_should_be_correct() {
    assert_eq!(SpzGaussiansPositions::Float16(vec![]).len(), 0);
    assert!(SpzGaussiansPositions::Float16(vec![]).is_empty());

    assert_eq!(SpzGaussiansPositions::Float16(vec![[0; 3]]).len(), 1);
    assert!(!SpzGaussiansPositions::Float16(vec![[0; 3]]).is_empty());

    assert_eq!(SpzGaussiansPositions::FixedPoint24(vec![]).len(), 0);
    assert!(SpzGaussiansPositions::FixedPoint24(vec![]).is_empty());

    assert_eq!(
        SpzGaussiansPositions::FixedPoint24(vec![[[0; 3]; 3]]).len(),
        1
    );
    assert!(!SpzGaussiansPositions::FixedPoint24(vec![[[0; 3]; 3]]).is_empty());
}

#[test]
fn test_spz_gaussians_rotations_len_and_is_empty_should_be_correct() {
    assert_eq!(SpzGaussiansRotations::QuatFirstThree(vec![]).len(), 0);
    assert!(SpzGaussiansRotations::QuatFirstThree(vec![]).is_empty());

    assert_eq!(SpzGaussiansRotations::QuatFirstThree(vec![[0; 3]]).len(), 1);
    assert!(!SpzGaussiansRotations::QuatFirstThree(vec![[0; 3]]).is_empty());

    assert_eq!(SpzGaussiansRotations::QuatSmallestThree(vec![]).len(), 0);
    assert!(SpzGaussiansRotations::QuatSmallestThree(vec![]).is_empty());

    assert_eq!(
        SpzGaussiansRotations::QuatSmallestThree(vec![[0; 4]]).len(),
        1
    );
    assert!(!SpzGaussiansRotations::QuatSmallestThree(vec![[0; 4]]).is_empty());
}

#[test]
fn test_spz_gaussians_shs_len_and_is_empty_should_be_correct() {
    assert_eq!(SpzGaussiansShs::Zero.len(), 0);
    assert!(SpzGaussiansShs::Zero.is_empty());

    assert_eq!(SpzGaussiansShs::One(vec![]).len(), 0);
    assert!(SpzGaussiansShs::One(vec![]).is_empty());

    assert_eq!(SpzGaussiansShs::One(vec![[[0; 3]; 3]]).len(), 1);
    assert!(!SpzGaussiansShs::One(vec![[[0; 3]; 3]]).is_empty());

    assert_eq!(SpzGaussiansShs::Two(vec![]).len(), 0);
    assert!(SpzGaussiansShs::Two(vec![]).is_empty());

    assert_eq!(SpzGaussiansShs::Two(vec![[[0; 3]; 8]]).len(), 1);
    assert!(!SpzGaussiansShs::Two(vec![[[0; 3]; 8]]).is_empty());

    assert_eq!(SpzGaussiansShs::Three(vec![]).len(), 0);
    assert!(SpzGaussiansShs::Three(vec![]).is_empty());

    assert_eq!(SpzGaussiansShs::Three(vec![[[0; 3]; 15]]).len(), 1);
    assert!(!SpzGaussiansShs::Three(vec![[[0; 3]; 15]]).is_empty());
}

#[test]
fn test_spz_gaussian_as_ref_and_ref_to_inner_owned_should_be_equal() {
    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        for sh_degree in SpzGaussiansHeader::SUPPORTED_SH_DEGREES {
            let (gaussian, _) = given_spz_gaussian_and_header(
                1,
                &SpzGaussiansFromGaussianSliceOptions {
                    version,
                    sh_degree: SpzGaussianShDegree::new(sh_degree).expect("valid SH degree"),
                    ..Default::default()
                },
            );

            assert_eq!(gaussian.as_ref().to_inner_owned(), gaussian);
        }
    }
}

#[test]
fn test_spz_v4_upstream_files_should_preserve_attributes_and_all_sh_degrees() {
    for degree in 0..=4 {
        let model = read(&fixture(degree)).unwrap();

        assert_eq!(model.header.version(), 4);
        assert_eq!(model.header.num_points(), 2);
        assert_eq!(model.header.fractional_bits(), 12);
        assert!(model.header.is_antialiased());
        assert_eq!(model.sh_degree().get(), degree as u8);
        assert_eq!(model.alphas, [128, 186]);
        assert_eq!(model.colors, [[128, 166, 89], [147, 108, 128]]);
        assert_eq!(model.scales, [[160, 168, 152], [176, 144, 160]]);

        let coefficients = (degree + 1) * (degree + 1) - 1;

        for (point, (packed, gaussian)) in model.iter().zip(model.iter_gaussian()).enumerate() {
            let gaussian = gaussian.convert_sh_degree::<wgpu_3dgs_core::ShDegree4>();

            assert_eq!(
                gaussian.pos,
                [Vec3::new(1.0, -2.0, 0.5), Vec3::new(-4.0, 8.0, -0.25)][point]
            );

            let expected_rotation = [
                Quat::from_xyzw(1.0, 2.0, 3.0, 4.0).normalize(),
                Quat::from_xyzw(4.0, -3.0, 2.0, -1.0).normalize(),
            ][point];

            assert!(gaussian.rot.abs_diff_eq(expected_rotation, 0.003));

            let logs = [[0.0, 0.5, -0.5], [1.0, -1.0, 0.0]][point];

            assert!((gaussian.scale - Vec3::from_array(logs).exp()).length() < 1e-5);
            assert_eq!(packed.sh.iter().count(), coefficients);

            for (coefficient, rgb) in packed.sh.iter().enumerate() {
                let bucket = if coefficient < 3 { 8 } else { 16 };

                for (channel, &value) in rgb.iter().enumerate() {
                    let index = point * coefficients * 3 + coefficient * 3 + channel;
                    let raw = (index as i32 % 17 - 8) * 8 + 128;
                    let expected = ((raw + bucket / 2) / bucket * bucket) as u8;

                    assert_eq!(value, expected);
                    assert_eq!(
                        gaussian.sh[coefficient][channel],
                        (expected as f32 - 128.0) / 128.0
                    );
                }
            }

            assert_eq!(packed.to_inner_owned().as_ref(), packed);
        }

        if degree == 4 {
            assert!(matches!(
                model.iter_gaussian().next(),
                Some(AnyGaussian::Four(_))
            ));
            assert!(matches!(model.shs, SpzGaussiansShs::Four(_)));
        }
    }
}

#[test]
fn test_spz_v4_batch_reads_should_work_without_seek_and_respect_item_budget() {
    for degree in 0..=4 {
        let bytes = fixture(degree);
        let expected = read(&bytes).unwrap();

        for budget in [1, 2, 4096] {
            let mut reader = SpzBatchReader::new(ShortRead(&bytes)).unwrap();

            assert_eq!(reader.header(), &expected.header);

            let mut completed = 0;

            while !reader.progress().done {
                let progress = reader.step(NonZeroUsize::new(budget).unwrap()).unwrap();
                assert!(progress.completed_units - completed <= budget);
                completed = progress.completed_units;
            }

            assert_eq!(reader.finish().unwrap(), expected);
        }

        let mut reader = GaussiansBatchReader::new(bytes.as_slice(), GaussiansSource::Spz).unwrap();

        while !reader.progress().done {
            reader.step(NonZeroUsize::new(1).unwrap()).unwrap();
        }

        assert!(matches!(reader.finish().unwrap(), Gaussians::Spz(model) if model == expected));
    }
}

#[test]
fn test_spz_v4_unified_and_file_readers_should_preserve_source_conversions() {
    for degree in 0..=4 {
        let bytes = fixture(degree);
        let original = read(&bytes).unwrap();
        let unified = Gaussians::read_from(&mut bytes.as_slice(), GaussiansSource::Spz).unwrap();

        assert_eq!(unified.sh_degree().get(), degree as u8);

        let expected = original.iter_gaussian().collect::<Vec<_>>();
        let native = unified.clone().into_internal().unwrap();

        assert_eq!(native.iter_gaussian().collect::<Vec<_>>(), expected);

        let ply = unified.clone().into_ply().unwrap();

        assert_eq!(ply.sh_degree().get(), degree as u8);
        assert_eq!(ply.iter_gaussian().count(), 2);
        assert_eq!(unified.clone().into_spz().unwrap(), original);

        let options = wgpu_3dgs_core::SpzGaussiansFromGaussianSliceOptions {
            sh_degree: SpzGaussianShDegree::new(degree as u8).unwrap(),
            ..Default::default()
        };

        if degree == 4 {
            assert!(unified.into_spz_with_options(&options).is_err());
        } else {
            let legacy = unified.into_spz_with_options(&options).unwrap();

            assert_eq!(legacy.header.version(), 3);
            assert!(legacy.write_to(&mut Vec::new()).is_ok());
        }

        let path = crate::common::given::temp_file_path(".spz");
        std::fs::write(&path, &bytes).unwrap();

        assert_eq!(SpzGaussians::read_from_file(&path).unwrap(), original);
    }
}

#[test]
fn test_spz_v4_extensions_should_be_skipped_without_changing_gaussians() {
    let original = fixture(4);
    let extension = [0xfe, 0xca, 0xad, 0xde, 3, 0, 0, 0, 10, 20, 30];
    let model = read(&with_extension(original.clone(), &extension)).unwrap();

    assert_eq!(
        model.iter().collect::<Vec<_>>(),
        read(&original).unwrap().iter().collect::<Vec<_>>()
    );

    let mut two_records = extension.to_vec();
    two_records.extend_from_slice(&[1, 2, 3, 4, 0, 0, 0, 0]);

    assert!(read(&with_extension(original, &two_records)).is_ok());
}

#[test]
fn test_spz_v4_malformed_headers_toc_and_extensions_should_return_errors() {
    let original = fixture(3);
    let mut cases = Vec::new();

    for (offset, value) in [(0, 0), (4, 3), (4, 5), (16, 0), (16, 31)] {
        let mut bytes = original.clone();
        set_u32(&mut bytes, offset, value);
        cases.push(bytes);
    }

    for (offset, value) in [
        (12, 5),
        (13, 31),
        (14, 0x80),
        (15, 5),
        (15, 7),
        (20, 1),
        (31, 1),
    ] {
        let mut bytes = original.clone();
        bytes[offset] = value;
        cases.push(bytes);
    }

    for (offset, value) in [(32, 0), (32, u64::MAX), (40, 1), (40, u64::MAX)] {
        let mut bytes = original.clone();
        set_u64(&mut bytes, offset, value);
        cases.push(bytes);
    }

    cases.push(with_extension(original.clone(), &[1, 2, 3]));
    cases.push(with_extension(original.clone(), &[1, 0, 0, 0, 9, 0, 0, 0]));

    let mut no_flag = with_extension(original, &[1, 0, 0, 0, 0, 0, 0, 0]);
    no_flag[14] &= !0x2;
    cases.push(no_flag);

    for (index, bytes) in cases.iter().enumerate() {
        assert!(read(bytes).is_err(), "malformed case {index}");
    }
}

#[test]
fn test_spz_v4_every_truncation_and_trailing_data_should_return_errors() {
    for degree in [0, 3, 4] {
        let original = fixture(degree);

        for length in 0..original.len() {
            assert!(
                read(&original[..length]).is_err(),
                "SH{degree}, truncation {length}"
            );
        }

        let mut extra = original.clone();
        extra.push(0);

        assert!(read(&extra).is_err());
    }
}

#[test]
fn test_spz_v4_frame_boundaries_and_sizes_should_be_validated() {
    let original = fixture(3);

    // First stream begins at 32 + 6 * 16 and is a raw frame with a content size.
    let start = 128;

    for (offset, value) in [(start, 0), (start + 5, 17), (start + 5, 19)] {
        let mut bytes = original.clone();
        bytes[offset] = value;
        assert!(read(&bytes).is_err());
    }

    // Extend the first TOC range with bytes that are not part of its frame.
    let mut padded = original.clone();
    set_u64(&mut padded, 32, 28);
    padded.insert(start + 27, 0);

    assert!(read(&padded).is_err());

    // Shorten its TOC range, preventing the decoder from reading the next stream.
    let mut short = original;
    set_u64(&mut short, 32, 26);

    assert!(read(&short).is_err());
}

#[test]
fn test_spz_v4_checksums_should_be_validated() {
    // Produced with libzstd level 12 and write_checksum=true, for the first
    // upstream fixture's 18 position bytes. Trailing XXH64 low bits: c417e404.
    let frame = [
        0x28, 0xb5, 0x2f, 0xfd, 0x24, 0x12, 0x91, 0, 0, 0, 0x10, 0, 0, 0xe0, 0xff, 0, 8, 0, 0,
        0xc0, 0xff, 0, 0x80, 0, 0, 0xfc, 0xff, 0xc4, 0x17, 0xe4, 4,
    ];
    let bytes = replace_first_frame(fixture(3), &frame);

    assert_eq!(read(&bytes).unwrap(), read(&fixture(3)).unwrap());

    let mut corrupt_frame = frame;
    corrupt_frame[30] ^= 1;

    let corrupt = replace_first_frame(fixture(3), &corrupt_frame);
    let error = read(&corrupt).unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("checksum"));
}

#[test]
fn test_spz_v4_finish_should_validate_the_final_frame_checksum() {
    let mut bytes = fixture(0);
    let last_start = bytes.len() - 17;

    // Original libzstd frame plus checksum e1584c9e for the rotation bytes.
    bytes[last_start + 4] |= 0x4;
    bytes.extend_from_slice(&[0xe1, 0x58, 0x4c, 0x9e]);
    set_u64(&mut bytes, 32 + 4 * 16, 21);

    assert_eq!(read(&bytes).unwrap(), read(&fixture(0)).unwrap());

    *bytes.last_mut().unwrap() ^= 1;

    let mut reader = SpzBatchReader::new(bytes.as_slice()).unwrap();

    while !reader.progress().done {
        reader.step(NonZeroUsize::new(1).unwrap()).unwrap();
    }

    let error = reader.finish().unwrap_err();

    assert!(error.to_string().contains("checksum"));
    assert!(
        SpzBatchReader::new(fixture(0).as_slice())
            .unwrap()
            .finish()
            .is_err()
    );
}

#[test]
fn test_spz_v3_should_decode_upstream_smallest_three_rotations_in_the_same_order() {
    let mut model = read(&fixture(3)).unwrap();
    let expected = model.iter_gaussian().collect::<Vec<_>>();

    model.header = SpzGaussiansHeader::new(
        3,
        2,
        model.header.sh_degree(),
        model.header.fractional_bits(),
        true,
    )
    .unwrap();

    let mut gzip = Vec::new();
    model.write_to(&mut gzip).unwrap();

    assert_eq!(
        read(&gzip).unwrap().iter_gaussian().collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn test_spz_v4_unknown_frame_content_size_should_still_require_exact_attribute_sizes() {
    let positions = [
        0, 0x10, 0, 0, 0xe0, 0xff, 0, 8, 0, 0, 0xc0, 0xff, 0, 0x80, 0, 0, 0xfc, 0xff,
    ];
    let original = fixture(3);
    let valid = replace_first_frame(original.clone(), &raw_frame(&positions));

    assert_eq!(read(&valid).unwrap(), read(&original).unwrap());

    for length in [17, 19] {
        let mut payload = positions.to_vec();
        payload.resize(length, 0);

        let bytes = replace_first_frame(original.clone(), &raw_frame(&payload));

        assert!(read(&bytes).is_err());
    }

    let mut oversized_window = raw_frame(&positions);
    oversized_window[5] = 0xff;

    assert!(read(&replace_first_frame(original, &oversized_window)).is_err());
}

#[test]
fn test_spz_v4_multiple_zstd_blocks_should_decode_in_batches() {
    let count = 20_000u32;
    let mut bytes = fixture(0);
    bytes.truncate(32);
    set_u32(&mut bytes, 8, count);

    let frames = [9, 1, 3, 3, 4].map(|width| {
        let size = count as usize * width;
        let frame = raw_frame(&vec![0; size]);
        bytes.extend_from_slice(&(frame.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(size as u64).to_le_bytes());

        frame
    });

    for frame in frames {
        bytes.extend_from_slice(&frame);
    }

    let mut reader = SpzBatchReader::new(ShortRead(&bytes)).unwrap();

    while !reader.progress().done {
        reader.step(NonZeroUsize::new(127).unwrap()).unwrap();
    }

    let model = reader.finish().unwrap();

    assert_eq!(model.len(), count as usize);
    assert!(model.alphas.iter().all(|&value| value == 0));
    assert_eq!(model.iter_gaussian().count(), count as usize);
}

#[test]
fn test_spz_v4_empty_models_should_retain_degree_without_frames() {
    for degree in 0..=4 {
        let mut bytes = fixture(degree);
        bytes.truncate(32);
        set_u32(&mut bytes, 8, 0);
        bytes[15] = 0;

        let reader = SpzBatchReader::new(ShortRead(&bytes)).unwrap();

        assert!(reader.progress().done);

        let model = reader.finish().unwrap();

        assert!(model.is_empty());
        assert_eq!(model.sh_degree().get(), degree as u8);
        assert_eq!(model.iter_gaussian().count(), 0);
    }
}

#[test]
fn test_spz_v4_writing_and_legacy_layout_access_should_fail_without_emitting_bytes() {
    let model = read(&fixture(4)).unwrap();
    let mut output = Vec::new();

    assert!(model.write_to(&mut output).is_err());
    assert!(output.is_empty());
    assert!(model.write_decompressed(&mut output).is_err());
    assert!(output.is_empty());
    assert!(SpzBatchWriter::new(&mut output, &model).is_err());
    assert!(SpzGaussians::read_gaussians(&mut &[][..], model.header).is_err());
    assert!(SpzGaussians::read_decompressed(&mut fixture(0).as_slice()).is_err());

    assert_eq!(SpzGaussiansHeader::default(0).unwrap().version(), 3);
    assert_eq!(std::mem::size_of::<SpzGaussiansHeaderPod>(), 16);
    assert_eq!(SpzGaussiansHeader::SUPPORTED_VERSIONS, 1..=3);
    assert_eq!(SpzGaussiansHeader::SUPPORTED_READ_VERSIONS, 1..=4);
}

#[test]
fn test_spz_v4_degree_four_fields_should_preserve_borrowed_and_owned_values() {
    let mut sh = SpzGaussianSh::Four([[128; 3]; 24]);

    assert_eq!(sh.degree().get(), 4);
    assert_eq!(sh.degree().num_coefficients(), 24);

    sh.iter_mut().last().unwrap()[2] = 255;

    assert_eq!(sh.iter().last().unwrap(), &[128, 128, 255]);
    assert!(SpzGaussianShDegree::new(5).is_none());
    assert!(SpzGaussiansShs::Four(Vec::new()).is_empty());
}

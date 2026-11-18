use assert_matches::assert_matches;
use std::{
    io::{ErrorKind, Read, Write},
    num::NonZeroUsize,
};
use wgpu_3dgs_core::{
    AnyGaussian, BatchRead, BatchWrite, Gaussian, GaussianShDegree, GaussianToSpzOptions,
    Gaussians, GaussiansBatchReader, GaussiansBatchWriter, GaussiansSource, IterGaussian,
    ReadIterGaussian, SpzBatchReader, SpzBatchWriter, SpzGaussian, SpzGaussianPosition,
    SpzGaussianRef, SpzGaussianRotation, SpzGaussianSh, SpzGaussianShDegree, SpzGaussianShRef,
    SpzGaussians, SpzGaussiansCollectError, SpzGaussiansFromGaussianSliceOptions,
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

// Wire-layout constants are independent of the production container implementation.
const V4_HEADER_SIZE: usize = 32;
const V4_TOC_ENTRY_SIZE: usize = 16;
const V4_MAGIC_OFFSET: usize = 0;
const V4_VERSION_OFFSET: usize = 4;
const V4_NUM_POINTS_OFFSET: usize = 8;
const V4_SH_DEGREE_OFFSET: usize = 12;
const V4_FRACTIONAL_BITS_OFFSET: usize = 13;
const V4_FLAGS_OFFSET: usize = 14;
const V4_STREAM_COUNT_OFFSET: usize = 15;
const V4_TOC_OFFSET: usize = 16;
const V4_RESERVED_OFFSET: usize = 20;
const V4_EXTENSION_FLAG: u8 = 0x2;
const ZSTD_BLOCK_SIZE: usize = 128 * 1024;

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
                4 => SpzGaussianSh::Four([[0; 3]; 24]),
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

fn assert_spz_buffer_round_trip(options: &SpzGaussiansFromGaussianSliceOptions) {
    let from_options = SpzGaussians::from_gaussians_with_options(
        given::runtime_gaussians(options.sh_degree.get()),
        options,
    )
    .unwrap();

    let mut buffer = Vec::new();
    from_options.write_to(&mut buffer).unwrap();
    let spz_gaussians_read = SpzGaussians::read_from(&mut buffer.as_slice()).unwrap();

    assert_eq!(spz_gaussians_read, from_options);
}

fn assert_spz_native_round_trip(options: &SpzGaussiansFromGaussianSliceOptions) {
    let gaussians = given::runtime_gaussians(options.sh_degree.get());
    let from_options = SpzGaussians::from_gaussians_with_options(&gaussians, options).unwrap();

    assert_eq!(from_options.len(), gaussians.len());

    for (packed, expected) in from_options.iter().zip(&gaussians) {
        let restored = AnyGaussian::from_spz(packed, &from_options.header).unwrap();

        assert::gaussian(&restored, expected, &ASSERT_GAUSSIAN_OPTIONS);
    }
}

fn given_spz_v4_gaussians(degree: u8) -> SpzGaussians {
    SpzGaussians::from_gaussians_with_options(
        given::runtime_gaussians(degree),
        &SpzGaussiansFromGaussianSliceOptions {
            version: 4,
            sh_degree: SpzGaussianShDegree::new(degree).unwrap(),
            antialiased: true,
            ..Default::default()
        },
    )
    .unwrap()
}

fn given_spz_v4_buffer(degree: u8) -> Vec<u8> {
    let mut bytes = Vec::new();
    given_spz_v4_gaussians(degree).write_to(&mut bytes).unwrap();

    bytes
}

fn set_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn set_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn replace_frame(mut bytes: Vec<u8>, index: usize, frame: &[u8]) -> Vec<u8> {
    let mut start = V4_HEADER_SIZE + usize::from(bytes[V4_STREAM_COUNT_OFFSET]) * V4_TOC_ENTRY_SIZE;

    for stream in 0..index {
        let toc = V4_HEADER_SIZE + stream * V4_TOC_ENTRY_SIZE;
        start += u64::from_le_bytes(bytes[toc..toc + 8].try_into().unwrap()) as usize;
    }

    let toc = V4_HEADER_SIZE + index * V4_TOC_ENTRY_SIZE;
    let old_size = u64::from_le_bytes(bytes[toc..toc + 8].try_into().unwrap()) as usize;

    set_u64(&mut bytes, toc, frame.len() as u64);
    bytes.splice(start..start + old_size, frame.iter().copied());

    bytes
}

// A standard ZSTD frame without a content-size field, using raw blocks and a
// 128 KiB window. This is independent of the production reader and its decoder.
fn raw_frame(data: &[u8]) -> Vec<u8> {
    let mut frame = vec![0x28, 0xb5, 0x2f, 0xfd, 0, 0x38];
    let blocks = data.chunks(ZSTD_BLOCK_SIZE);
    let block_count = blocks.len();

    for (index, block) in blocks.enumerate() {
        let header = (block.len() as u32) << 3 | u32::from(index + 1 == block_count);
        frame.extend_from_slice(&header.to_le_bytes()[..3]);
        frame.extend_from_slice(block);
    }

    frame
}

fn with_extension(mut bytes: Vec<u8>, extension: &[u8]) -> Vec<u8> {
    bytes[V4_FLAGS_OFFSET] |= V4_EXTENSION_FLAG;
    set_u32(
        &mut bytes,
        V4_TOC_OFFSET,
        (V4_HEADER_SIZE + extension.len()) as u32,
    );
    bytes.splice(V4_HEADER_SIZE..V4_HEADER_SIZE, extension.iter().copied());

    bytes
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
        for degree in 1u8..=4 {
            let index = degree.min(3) as usize - 1;
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
fn test_spz_phase_from_str_when_phase_is_unknown_should_return_error() {
    assert_matches!(
        "unknown".parse::<SpzPhase>(),
        Err(wgpu_3dgs_core::SpzPhaseFromStrError::InvalidSpzPhase(name)) if name == "unknown"
    );
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
fn test_bundled_spz_model_should_use_version_four_and_support_whole_model_and_batch_reads() {
    let bytes = include_bytes!("../../examples/model.spz");
    let model = SpzGaussians::read_from(&mut bytes.as_slice()).unwrap();

    assert_eq!(model.header.version(), 4);
    assert_eq!(model.len(), 9);
    assert_eq!(model.sh_degree().get(), 3);

    let mut reader = SpzBatchReader::new(given::ShortRead(bytes)).unwrap();

    while !reader.progress().done {
        reader.step(NonZeroUsize::new(1).unwrap()).unwrap();
    }

    assert_eq!(reader.finish().unwrap(), model);
}

#[test]
fn test_spz_gaussians_file_round_trip_should_be_equal() {
    let spz_gaussians = given::spz_gaussians();
    let path = given::temp_file_path(".spz");

    spz_gaussians.write_to_file(&path).unwrap();
    let spz_gaussians_read = SpzGaussians::read_from_file(&path).unwrap();

    assert_eq!(spz_gaussians_read, spz_gaussians);
}

#[test]
fn test_spz_gaussians_buffer_round_trip_should_be_equal() {
    let spz_gaussians = given::spz_gaussians();

    let mut buffer = Vec::new();
    spz_gaussians.write_to(&mut buffer).unwrap();
    let spz_gaussians_read = SpzGaussians::read_from(&mut buffer.as_slice()).unwrap();

    assert_eq!(spz_gaussians_read, spz_gaussians);
}

#[test]
fn test_spz_decompressed_write_and_reads_should_preserve_field_layout() {
    for version in 1..=3 {
        for degree in 0..=3 {
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
    let original = SpzGaussians::from_gaussians_with_options(
        given::gaussians(),
        &SpzGaussiansFromGaussianSliceOptions {
            version: 3,
            sh_degree: SpzGaussianShDegree::new(3).expect("valid SH degree"),
            ..Default::default()
        },
    )
    .unwrap();
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
        original.len() * 15 * 3, // SH degree 3
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
fn test_spz_default_constructors_should_use_version_four_and_preserve_all_sh_degrees() {
    for degree in SpzGaussiansHeader::SUPPORTED_SH_DEGREES {
        for count in [0, 2] {
            let gaussians = given::runtime_gaussians(degree)
                .into_iter()
                .take(count)
                .collect::<Vec<_>>();
            let options = SpzGaussiansFromGaussianSliceOptions {
                sh_degree: SpzGaussianShDegree::new(degree).unwrap(),
                ..Default::default()
            };
            let expected = SpzGaussians::from_gaussians_with_options(&gaussians, &options).unwrap();
            let unified = Gaussians::from_gaussians_iter_with_sh_degree(
                gaussians.clone().into_iter(),
                GaussiansSource::Spz,
                GaussianShDegree::new(degree).unwrap(),
            )
            .unwrap()
            .into_spz()
            .unwrap();

            assert_eq!(expected.header.version(), 4);
            assert_eq!(expected.sh_degree().get(), degree);
            assert_eq!(unified, expected);

            if count != 0 {
                assert_eq!(SpzGaussians::from_gaussians(&gaussians).unwrap(), expected);
                assert_eq!(
                    Gaussians::from_gaussians_iter(gaussians.into_iter(), GaussiansSource::Spz)
                        .unwrap()
                        .into_spz()
                        .unwrap(),
                    expected
                );
            }

            let mut bytes = Vec::new();
            expected.write_to(&mut bytes).unwrap();

            assert_eq!(&bytes[..4], &SpzGaussiansHeader::MAGIC.to_le_bytes());
            assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 4);
            assert_eq!(
                SpzGaussians::read_from(&mut bytes.as_slice()).unwrap(),
                expected
            );
        }
    }
}

#[test]
fn test_spz_gaussians_buffer_round_trip_when_versions_vary_should_preserve_model() {
    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        assert_spz_buffer_round_trip(&SpzGaussiansFromGaussianSliceOptions {
            version,
            sh_degree: SpzGaussianShDegree::new(3).expect("valid SH degree"),
            ..Default::default()
        });
    }
}

#[test]
fn test_spz_gaussians_buffer_round_trip_when_sh_degrees_vary_should_preserve_model() {
    for sh_degree in SpzGaussiansHeader::SUPPORTED_SH_DEGREES {
        assert_spz_buffer_round_trip(&SpzGaussiansFromGaussianSliceOptions {
            sh_degree: SpzGaussianShDegree::new(sh_degree).expect("valid SH degree"),
            ..Default::default()
        });
    }
}

#[test]
fn test_spz_gaussians_buffer_round_trip_when_fractional_bits_vary_should_preserve_model() {
    for fractional_bits in [8, 12, 16] {
        assert_spz_buffer_round_trip(&SpzGaussiansFromGaussianSliceOptions {
            fractional_bits,
            ..Default::default()
        });
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
fn test_spz_gaussians_iter_and_iter_gaussian_when_sh_degree_is_zero_should_report_remaining_count()
{
    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        for count in [0, 2] {
            let gaussians = given::runtime_gaussians(0)
                .into_iter()
                .take(count)
                .collect::<Vec<_>>();
            let model = SpzGaussians::from_gaussians_with_options(
                &gaussians,
                &SpzGaussiansFromGaussianSliceOptions {
                    version,
                    sh_degree: SpzGaussianShDegree::new(0).expect("valid SH degree"),
                    ..Default::default()
                },
            )
            .unwrap();
            let mut packed = model.iter();
            let mut native = model.iter_gaussian();

            assert_eq!(packed.len(), count);
            assert_eq!(native.len(), count);

            for remaining in (0..count).rev() {
                assert_matches!(
                    packed.next(),
                    Some(SpzGaussianRef {
                        sh: SpzGaussianShRef::Zero,
                        ..
                    })
                );
                assert_matches!(native.next(), Some(AnyGaussian::Zero(_)));
                assert_eq!(packed.size_hint(), (remaining, Some(remaining)));
                assert_eq!(native.size_hint(), (remaining, Some(remaining)));
                assert_eq!(packed.len(), remaining);
                assert_eq!(native.len(), remaining);
            }

            assert!(packed.next().is_none());
            assert!(native.next().is_none());
            assert!(packed.next().is_none());
            assert!(native.next().is_none());
        }
    }
}

#[test]
fn test_spz_gaussians_from_gaussians_with_options_when_versions_vary_should_preserve_attributes() {
    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        assert_spz_native_round_trip(&SpzGaussiansFromGaussianSliceOptions {
            version,
            sh_degree: SpzGaussianShDegree::new(3).expect("valid SH degree"),
            ..Default::default()
        });
    }
}

#[test]
fn test_spz_gaussians_from_gaussians_with_options_when_sh_degrees_vary_should_preserve_attributes()
{
    for sh_degree in SpzGaussiansHeader::SUPPORTED_SH_DEGREES {
        assert_spz_native_round_trip(&SpzGaussiansFromGaussianSliceOptions {
            sh_degree: SpzGaussianShDegree::new(sh_degree).expect("valid SH degree"),
            ..Default::default()
        });
    }
}

#[test]
fn test_spz_gaussians_from_gaussians_with_options_when_fractional_bits_vary_should_preserve_attributes()
 {
    for fractional_bits in [8, 12, 16] {
        assert_spz_native_round_trip(&SpzGaussiansFromGaussianSliceOptions {
            fractional_bits,
            ..Default::default()
        });
    }
}

#[test]
fn test_spz_gaussians_from_gaussians_with_options_when_sh_quantize_bits_vary_should_preserve_attributes()
 {
    for sh_quantize_bits in [
        [0; 3],
        [4; 3],
        [5; 3],
        [8; 3],
        [0, 1, 2],
        [2, 4, 6],
        [4, 5, 5],
    ] {
        assert_spz_native_round_trip(&SpzGaussiansFromGaussianSliceOptions {
            sh_quantize_bits,
            ..Default::default()
        });
    }
}

#[test]
fn test_spz_gaussians_from_gaussians_with_options_when_header_version_is_invalid_should_return_error()
 {
    let gaussians = given::gaussians();
    let options = wgpu_3dgs_core::SpzGaussiansFromGaussianSliceOptions {
        version: 999,
        sh_degree: SpzGaussianShDegree::new(3).expect("valid SH degree"),
        ..Default::default()
    };

    let result = SpzGaussians::from_gaussians_with_options(gaussians.iter(), &options);

    assert_matches!(
        result,
        Err(SpzGaussiansFromGaussiansError::Header(e))
        if e.kind() == std::io::ErrorKind::InvalidData &&
            e.to_string() == "Unsupported SPZ version: 999, expected one of 1..=4"
    );
}

#[test]
fn test_spz_gaussians_from_iter_when_input_is_empty_should_use_header_variants() {
    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        for degree in SpzGaussiansHeader::SUPPORTED_SH_DEGREES {
            if version < 4 && degree == 4 {
                continue;
            }

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

            if version < 4 {
                let mut decompressed = Vec::new();
                spz.write_decompressed(&mut decompressed).unwrap();

                assert_eq!(
                    SpzGaussians::read_decompressed(&mut decompressed.as_slice()).unwrap(),
                    spz
                );
            }

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
    assert_eq!(collected.shs.degree().get(), 3);
}

#[test]
fn test_spz_gaussians_from_iter_when_input_is_empty_and_header_count_is_nonzero_should_return_count_mismatch()
 {
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
    // Versions 3 and 4 use QuatSmallestThree
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
            SpzGaussianShDegree::new(4).expect("valid SH degree"),
        ),
        (
            SpzGaussianSh::Four([[0; 3]; 24]),
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
                header_sh_degree: actual_header_sh_degree,
            })
            if sh_degree == sh.degree() && actual_header_sh_degree == header_sh_degree
        );
    }
}

#[test]
fn test_spz_gaussians_header_try_from_pod_when_magic_is_incorrect_should_return_error() {
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
    let invalid_degree: SpzGaussianShDegree = bytemuck::cast(5u8);
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
    assert_eq!(SpzGaussianSh::Four([[0; 3]; 24]).degree().get(), 4);
}

#[test]
fn test_spz_gaussian_sh_degree_num_coefficients_should_be_correct() {
    for (degree, expected) in [(0, 0), (1, 3), (2, 8), (3, 15), (4, 24)] {
        let degree = SpzGaussianShDegree::new(degree).expect("valid SH degree");

        assert_eq!(degree.num_coefficients(), expected);
    }
}

#[test]
fn test_spz_gaussian_sh_degree_when_degree_is_invalid_should_return_none() {
    for degree in 5..=u8::MAX {
        assert_eq!(SpzGaussianShDegree::new(degree), None);
    }
}

#[test]
fn test_spz_gaussian_sh_ref_degree_should_be_correct() {
    assert_eq!(SpzGaussianShRef::Zero.degree().get(), 0);
    assert_eq!(SpzGaussianShRef::One(&[[0; 3]; 3]).degree().get(), 1);
    assert_eq!(SpzGaussianShRef::Two(&[[0; 3]; 8]).degree().get(), 2);
    assert_eq!(SpzGaussianShRef::Three(&[[0; 3]; 15]).degree().get(), 3);
    assert_eq!(SpzGaussianShRef::Four(&[[0; 3]; 24]).degree().get(), 4);
}

#[test]
fn test_spz_gaussians_shs_degree_should_be_correct() {
    assert_eq!(SpzGaussiansShs::Zero.degree().get(), 0);
    assert_eq!(SpzGaussiansShs::One(vec![]).degree().get(), 1);
    assert_eq!(SpzGaussiansShs::Two(vec![]).degree().get(), 2);
    assert_eq!(SpzGaussiansShs::Three(vec![]).degree().get(), 3);
    assert_eq!(SpzGaussiansShs::Four(vec![]).degree().get(), 4);
}

#[test]
fn test_spz_gaussian_sh_iter_should_be_correct() {
    for (sh, count) in [
        (SpzGaussianSh::Zero, 0),
        (SpzGaussianSh::One(std::array::from_fn(|i| [i as u8; 3])), 3),
        (SpzGaussianSh::Two(std::array::from_fn(|i| [i as u8; 3])), 8),
        (
            SpzGaussianSh::Three(std::array::from_fn(|i| [i as u8; 3])),
            15,
        ),
        (
            SpzGaussianSh::Four(std::array::from_fn(|i| [i as u8; 3])),
            24,
        ),
    ] {
        assert_eq!(
            sh.iter().copied().collect::<Vec<_>>(),
            (0..count).map(|i| [i as u8; 3]).collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_spz_gaussian_sh_iter_mut_should_be_correct() {
    for (mut sh, count) in [
        (SpzGaussianSh::Zero, 0),
        (SpzGaussianSh::One([[0; 3]; 3]), 3),
        (SpzGaussianSh::Two([[0; 3]; 8]), 8),
        (SpzGaussianSh::Three([[0; 3]; 15]), 15),
        (SpzGaussianSh::Four([[0; 3]; 24]), 24),
    ] {
        for coeff in sh.iter_mut() {
            *coeff = [1, 2, 3];
        }

        assert_eq!(
            sh.iter().copied().collect::<Vec<_>>(),
            vec![[1, 2, 3]; count]
        );
    }
}

#[test]
fn test_spz_gaussian_sh_when_degree_is_four_should_preserve_mutated_coefficients() {
    let mut sh = SpzGaussianSh::Four([[128; 3]; 24]);

    sh.iter_mut().last().unwrap()[2] = 255;

    let mut expected = [[128; 3]; 24];
    expected[23][2] = 255;

    assert_eq!(sh, SpzGaussianSh::Four(expected));
    assert_eq!(sh.iter().copied().collect::<Vec<_>>(), expected);
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

    assert_eq!(SpzGaussiansShs::Four(vec![]).len(), 0);
    assert!(SpzGaussiansShs::Four(vec![]).is_empty());

    assert_eq!(SpzGaussiansShs::Four(vec![[[0; 3]; 24]]).len(), 1);
    assert!(!SpzGaussiansShs::Four(vec![[[0; 3]; 24]]).is_empty());
}

#[test]
fn test_spz_gaussian_as_ref_and_ref_to_inner_owned_should_be_equal() {
    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        for sh_degree in SpzGaussiansHeader::SUPPORTED_SH_DEGREES {
            if version < 4 && sh_degree == 4 {
                continue;
            }

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
fn test_spz_v4_round_trip_should_preserve_attributes_and_all_sh_degrees() {
    for degree in 0..=4 {
        let original = given_spz_v4_gaussians(degree);
        let mut bytes = Vec::new();
        original.write_to(&mut bytes).unwrap();
        let model = SpzGaussians::read_from(&mut bytes.as_slice()).unwrap();

        assert_eq!(model, original);
        assert_eq!(model.header.version(), 4);
        assert_eq!(model.header.num_points(), 2);
        assert_eq!(model.header.fractional_bits(), 12);
        assert!(model.header.is_antialiased());
        assert_eq!(model.sh_degree().get(), degree);

        let expected = given::runtime_gaussians(degree);

        assert_eq!(model.iter_gaussian().len(), expected.len());

        for (gaussian, expected) in model.iter_gaussian().zip(&expected) {
            assert::gaussian(&gaussian, expected, &ASSERT_GAUSSIAN_OPTIONS);
        }

        for packed in model.iter() {
            assert_eq!(packed.sh.iter().count(), model.header.sh_num_coefficients());
            assert_eq!(packed.to_inner_owned().as_ref(), packed);
        }

        if degree == 4 {
            assert_matches!(model.iter_gaussian().next(), Some(AnyGaussian::Four(_)));
            assert_matches!(model.shs, SpzGaussiansShs::Four(_));
        }
    }
}

#[test]
fn test_spz_v4_batch_reads_when_input_has_short_reads_and_no_seek_should_respect_item_budget() {
    for degree in 0..=4 {
        let bytes = given_spz_v4_buffer(degree);
        let expected = given_spz_v4_gaussians(degree);

        for budget in [1, 2, 4096] {
            let mut reader = SpzBatchReader::new(given::ShortRead(&bytes)).unwrap();

            assert_eq!(reader.header(), &expected.header);

            let mut completed = 0;

            while !reader.progress().done {
                let progress = reader.step(NonZeroUsize::new(budget).unwrap()).unwrap();

                assert!(progress.completed_units > completed);
                assert!(progress.completed_units - completed <= budget);

                completed = progress.completed_units;
            }

            let progress = reader.progress();

            assert_eq!(completed, progress.total_units);
            assert_eq!(
                reader.step(NonZeroUsize::new(1).unwrap()).unwrap(),
                progress
            );
            assert_eq!(reader.finish().unwrap(), expected);
        }

        let mut reader = GaussiansBatchReader::new(bytes.as_slice(), GaussiansSource::Spz).unwrap();
        let mut completed = 0;

        while !reader.progress().done {
            let progress = reader.step(NonZeroUsize::new(1).unwrap()).unwrap();

            assert_eq!(progress.completed_units, completed + 1);

            completed = progress.completed_units;
        }

        assert_matches!(reader.finish().unwrap(), Gaussians::Spz(model) if model == expected);
    }
}

#[test]
fn test_spz_v4_unified_and_file_readers_should_preserve_source_conversions() {
    for degree in 0..=4 {
        let bytes = given_spz_v4_buffer(degree);
        let original = given_spz_v4_gaussians(degree);
        let unified = Gaussians::read_from(&mut bytes.as_slice(), GaussiansSource::Spz).unwrap();

        assert_eq!(unified.sh_degree().get(), degree);

        let expected = original.iter_gaussian().collect::<Vec<_>>();
        let native = unified.clone().into_internal().unwrap();

        assert_eq!(native.iter_gaussian().collect::<Vec<_>>(), expected);

        let ply = unified.clone().into_ply().unwrap();

        assert_eq!(ply.sh_degree().get(), degree);
        assert_eq!(ply.iter_gaussian().len(), expected.len());

        for (gaussian, expected) in ply.iter_gaussian().zip(&expected) {
            assert::gaussian(&gaussian, expected, &ASSERT_GAUSSIAN_OPTIONS);
        }

        assert_eq!(unified.clone().into_spz().unwrap(), original);

        let options = SpzGaussiansFromGaussianSliceOptions {
            version: 3,
            sh_degree: SpzGaussianShDegree::new(degree).unwrap(),
            ..Default::default()
        };

        if degree == 4 {
            assert_matches!(
                unified.into_spz_with_options(&options),
                Err(SpzGaussiansFromGaussiansError::UnsupportedShDegree { degree: 4 })
            );
        } else {
            let legacy = unified.into_spz_with_options(&options).unwrap();

            assert_eq!(legacy.header.version(), 3);
            let mut bytes = Vec::new();
            legacy.write_to(&mut bytes).unwrap();

            assert_eq!(
                SpzGaussians::read_from(&mut bytes.as_slice()).unwrap(),
                legacy
            );
        }

        let path = given::temp_file_path(".spz");
        std::fs::write(&path, &bytes).unwrap();

        assert_eq!(SpzGaussians::read_from_file(&path).unwrap(), original);
    }
}

#[test]
fn test_spz_v4_reads_when_extensions_are_present_should_skip_them_without_changing_gaussians() {
    let original = given_spz_v4_buffer(4);
    let extension = [0xfe, 0xca, 0xad, 0xde, 3, 0, 0, 0, 10, 20, 30];
    let mut two_records = extension.to_vec();
    two_records.extend_from_slice(&[1, 2, 3, 4, 0, 0, 0, 0]);

    for extension in [extension.as_slice(), two_records.as_slice()] {
        let bytes = with_extension(original.clone(), extension);
        let model = SpzGaussians::read_from(&mut bytes.as_slice()).unwrap();
        let mut expected = given_spz_v4_gaussians(4);
        let mut pod = *expected.header.as_pod();
        pod.flags |= V4_EXTENSION_FLAG;
        expected.header = SpzGaussiansHeader::try_from_pod(pod).unwrap();

        assert_eq!(model, expected);
    }
}

#[test]
fn test_spz_v4_reads_when_header_is_malformed_should_return_error() {
    let original = given_spz_v4_buffer(3);

    for (name, offset, value, expected) in [
        ("magic", V4_MAGIC_OFFSET, 0, "unrecognized SPZ container"),
        (
            "legacy version",
            V4_VERSION_OFFSET,
            3,
            "unsupported SPZ plaintext header version",
        ),
        (
            "unsupported version",
            V4_VERSION_OFFSET,
            5,
            "unsupported SPZ plaintext header version",
        ),
        (
            "zero TOC offset",
            V4_TOC_OFFSET,
            0,
            "SPZ v4 TOC overlaps the header",
        ),
        (
            "overlapping TOC",
            V4_TOC_OFFSET,
            V4_HEADER_SIZE as u32 - 1,
            "SPZ v4 TOC overlaps the header",
        ),
    ] {
        let mut bytes = original.clone();
        set_u32(&mut bytes, offset, value);

        let error = SpzGaussians::read_from(&mut bytes.as_slice()).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidData, "{name}");
        assert_eq!(error.to_string(), expected, "{name}");
    }

    for (name, offset, value, expected) in [
        (
            "SH degree",
            V4_SH_DEGREE_OFFSET,
            5,
            "unsupported SPZ v4 SH degree",
        ),
        (
            "fractional bits",
            V4_FRACTIONAL_BITS_OFFSET,
            31,
            "SPZ v4 fractional bits must not exceed 30",
        ),
        (
            "flags",
            V4_FLAGS_OFFSET,
            0x80,
            "unsupported SPZ v4 flags or reserved bytes",
        ),
        (
            "missing stream",
            V4_STREAM_COUNT_OFFSET,
            5,
            "SPZ v4 stream count does not match attributes",
        ),
        (
            "extra stream",
            V4_STREAM_COUNT_OFFSET,
            7,
            "SPZ v4 stream count does not match attributes",
        ),
        (
            "first reserved byte",
            V4_RESERVED_OFFSET,
            1,
            "unsupported SPZ v4 flags or reserved bytes",
        ),
        (
            "last reserved byte",
            V4_HEADER_SIZE - 1,
            1,
            "unsupported SPZ v4 flags or reserved bytes",
        ),
    ] {
        let mut bytes = original.clone();
        bytes[offset] = value;

        let error = SpzGaussians::read_from(&mut bytes.as_slice()).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidData, "{name}");
        assert_eq!(error.to_string(), expected, "{name}");
    }
}

#[test]
fn test_spz_v4_reads_when_toc_is_malformed_should_return_error() {
    let original = given_spz_v4_buffer(3);

    for (name, offset, value, expected) in [
        (
            "zero compressed size",
            V4_HEADER_SIZE,
            0,
            "SPZ v4 TOC attribute size mismatch",
        ),
        (
            "overflowing compressed size",
            V4_HEADER_SIZE,
            u64::MAX,
            "SPZ v4 compressed size overflow",
        ),
        (
            "incorrect uncompressed size",
            V4_HEADER_SIZE + 8,
            1,
            "SPZ v4 TOC attribute size mismatch",
        ),
        (
            "oversized uncompressed size",
            V4_HEADER_SIZE + 8,
            u64::MAX,
            "SPZ v4 TOC attribute size mismatch",
        ),
    ] {
        let mut bytes = original.clone();
        set_u64(&mut bytes, offset, value);

        let error = SpzGaussians::read_from(&mut bytes.as_slice()).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidData, "{name}");
        assert_eq!(error.to_string(), expected, "{name}");
    }
}

#[test]
fn test_spz_v4_reads_when_extensions_are_malformed_should_return_error() {
    let original = given_spz_v4_buffer(3);

    let mut no_flag = with_extension(original.clone(), &[1, 0, 0, 0, 0, 0, 0, 0]);
    no_flag[V4_FLAGS_OFFSET] &= !V4_EXTENSION_FLAG;

    for (name, bytes, expected) in [
        (
            "truncated record",
            with_extension(original.clone(), &[1, 2, 3]),
            "truncated SPZ v4 extension record",
        ),
        (
            "oversized record",
            with_extension(original, &[1, 0, 0, 0, 9, 0, 0, 0]),
            "SPZ v4 extension exceeds header zone",
        ),
        (
            "missing flag",
            no_flag,
            "SPZ v4 extension zone without extension flag",
        ),
    ] {
        let error = SpzGaussians::read_from(&mut bytes.as_slice()).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidData, "{name}");
        assert_eq!(error.to_string(), expected, "{name}");
    }
}

#[test]
fn test_spz_v4_reads_when_data_is_truncated_or_has_trailing_bytes_should_return_error() {
    for degree in [0, 3, 4] {
        let original = given_spz_v4_buffer(degree);
        let data_offset =
            V4_HEADER_SIZE + usize::from(original[V4_STREAM_COUNT_OFFSET]) * V4_TOC_ENTRY_SIZE;

        for length in 0..original.len() {
            let error = SpzGaussians::read_from(&mut &original[..length]).unwrap_err();

            assert_eq!(
                error.kind(),
                if length < data_offset {
                    ErrorKind::UnexpectedEof
                } else {
                    ErrorKind::InvalidData
                },
                "SH{degree}, truncation {length}"
            );
        }

        let mut extra = original.clone();
        extra.push(0);

        let error = SpzGaussians::read_from(&mut extra.as_slice()).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "extra SPZ v4 container data");
    }
}

#[test]
fn test_spz_v4_reads_when_frame_boundaries_or_sizes_are_invalid_should_return_error() {
    let original = given_spz_v4_buffer(3);
    let start = V4_HEADER_SIZE + usize::from(original[V4_STREAM_COUNT_OFFSET]) * V4_TOC_ENTRY_SIZE;
    let compressed_size = u64::from_le_bytes(
        original[V4_HEADER_SIZE..V4_HEADER_SIZE + 8]
            .try_into()
            .unwrap(),
    ) as usize;

    let mut corrupt = original.clone();
    corrupt[start] = 0;

    assert_eq!(
        SpzGaussians::read_from(&mut corrupt.as_slice())
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidData
    );

    // Extend the first TOC range with bytes that are not part of its frame.
    let mut padded = original.clone();
    set_u64(&mut padded, V4_HEADER_SIZE, compressed_size as u64 + 1);
    padded.insert(start + compressed_size, 0);

    let error = SpzGaussians::read_from(&mut padded.as_slice()).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidData);
    assert_eq!(
        error.to_string(),
        "SPZ v4 frame does not consume its TOC entry"
    );

    // Shorten its TOC range, preventing the decoder from reading the next stream.
    let mut short = original;
    set_u64(&mut short, V4_HEADER_SIZE, compressed_size as u64 - 1);

    assert_eq!(
        SpzGaussians::read_from(&mut short.as_slice())
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidData
    );
}

#[test]
fn test_spz_v4_reads_when_checksum_is_invalid_should_return_error() {
    let original = given_spz_v4_gaussians(3);
    let mut positions = Vec::new();
    original.positions.write_to(&mut positions).unwrap();

    // Ruzstd's encoder emits a content checksum with its default hash feature.
    let mut frame = ruzstd::encoding::compress_to_vec(
        positions.as_slice(),
        ruzstd::encoding::CompressionLevel::Fastest,
    );
    let bytes = replace_frame(given_spz_v4_buffer(3), 0, &frame);

    assert_eq!(
        SpzGaussians::read_from(&mut bytes.as_slice()).unwrap(),
        original
    );

    *frame.last_mut().unwrap() ^= 1;
    let corrupt = replace_frame(given_spz_v4_buffer(3), 0, &frame);
    let error = SpzGaussians::read_from(&mut corrupt.as_slice()).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidData);
    assert!(error.to_string().contains("checksum"));
}

#[test]
fn test_spz_v4_finish_when_final_frame_checksum_is_invalid_should_return_error() {
    let original = given_spz_v4_gaussians(0);
    let mut rotations = Vec::new();
    original.rotations.write_to(&mut rotations).unwrap();

    let frame = ruzstd::encoding::compress_to_vec(
        rotations.as_slice(),
        ruzstd::encoding::CompressionLevel::Fastest,
    );
    let mut bytes = replace_frame(given_spz_v4_buffer(0), 4, &frame);

    assert_eq!(
        SpzGaussians::read_from(&mut bytes.as_slice()).unwrap(),
        original
    );

    *bytes.last_mut().unwrap() ^= 1;

    let mut reader = SpzBatchReader::new(bytes.as_slice()).unwrap();

    while !reader.progress().done {
        reader.step(NonZeroUsize::new(1).unwrap()).unwrap();
    }

    let error = reader.finish().unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidData);
    assert!(error.to_string().contains("checksum"));
}

#[test]
fn test_spz_v4_finish_when_batch_is_incomplete_should_return_error() {
    assert_eq!(
        SpzBatchReader::new(given_spz_v4_buffer(0).as_slice())
            .unwrap()
            .finish()
            .unwrap_err()
            .kind(),
        ErrorKind::UnexpectedEof
    );
}

#[test]
fn test_spz_smallest_three_rotation_should_match_wire_component_order() {
    use wgpu_3dgs_core::glam::Quat;

    // Largest component is w. Stored x, y, z have magnitudes 181, 90, 45,
    // with the sign bit set only for y. x occupies the highest stored ten bits.
    let rotation = SpzGaussianRotation::QuatSmallestThree(
        ((3u32 << 30) | (181 << 20) | (602 << 10) | 45).to_le_bytes(),
    );
    let expected = Quat::from_xyzw(
        0.25,
        -0.125,
        0.0625,
        (1.0f32 - 0.25f32.powi(2) - 0.125f32.powi(2) - 0.0625f32.powi(2)).sqrt(),
    );

    for version in [3, 4] {
        let (mut packed, header) = given_spz_gaussian_and_header(
            1,
            &SpzGaussiansFromGaussianSliceOptions {
                version,
                sh_degree: SpzGaussianShDegree::new(0).unwrap(),
                ..Default::default()
            },
        );
        packed.rotation = rotation.clone();
        let restored =
            Gaussian::<wgpu_3dgs_core::ShDegree0>::from_spz(packed.as_ref(), &header).unwrap();

        assert!(restored.rot.abs_diff_eq(expected, 1e-3));
        assert_eq!(
            restored
                .to_spz(&header, &Default::default())
                .unwrap()
                .rotation,
            rotation
        );
    }
}

#[test]
fn test_gaussian_spz_round_trip_when_positions_and_rotations_are_negative_should_preserve_geometry()
{
    use wgpu_3dgs_core::{
        ShDegree0,
        glam::{Quat, Vec3},
    };

    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        let (_, header) = given_spz_gaussian_and_header(
            1,
            &SpzGaussiansFromGaussianSliceOptions {
                version,
                sh_degree: SpzGaussianShDegree::new(0).unwrap(),
                ..Default::default()
            },
        );

        for rotation in [
            [-0.9, 0.1, -0.2, 0.3],
            [0.1, -0.9, 0.2, -0.3],
            [-0.2, 0.1, -0.9, 0.3],
            [0.1, -0.2, 0.3, -0.9],
        ] {
            let original = Gaussian::<ShDegree0> {
                pos: Vec3::new(-1.25, -0.0625, 2.5),
                rot: Quat::from_array(rotation).normalize(),
                ..given::gaussian_for_sh_degree(0)
            };
            let packed = original.to_spz(&header, &Default::default()).unwrap();
            let restored = Gaussian::<ShDegree0>::from_spz(packed.as_ref(), &header).unwrap();

            assert_eq!(restored.pos, original.pos);
            assert!((restored.rot.dot(original.rot).abs() - 1.0).abs() < 1e-3);
        }
    }
}

#[test]
fn test_spz_reads_when_gzip_contains_extra_model_data_should_return_error() {
    let original = SpzGaussians::from_gaussians_with_options(
        given::gaussians(),
        &SpzGaussiansFromGaussianSliceOptions {
            version: 3,
            sh_degree: SpzGaussianShDegree::new(3).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    let mut decompressed = Vec::new();
    original.write_decompressed(&mut decompressed).unwrap();
    decompressed.push(0);
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&decompressed).unwrap();
    let bytes = encoder.finish().unwrap();

    let error = SpzGaussians::read_from(&mut bytes.as_slice()).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidData);
    assert_eq!(error.to_string(), "extra SPZ data");
}

#[test]
#[should_panic(expected = "validated SPZ degree")]
fn test_spz_iter_gaussian_when_sh_degree_disagrees_with_header_should_panic() {
    let mut model = given::spz_gaussians();
    model.shs = SpzGaussiansShs::One(vec![[[128; 3]; 3]; model.len()]);

    model.iter_gaussian().next();
}

#[test]
fn test_spz_v4_reads_when_frame_content_size_is_unknown_should_require_exact_attribute_sizes() {
    let expected = given_spz_v4_gaussians(3);
    let mut positions = Vec::new();
    expected.positions.write_to(&mut positions).unwrap();
    let original = given_spz_v4_buffer(3);
    let valid = replace_frame(original.clone(), 0, &raw_frame(&positions));

    assert_eq!(
        SpzGaussians::read_from(&mut valid.as_slice()).unwrap(),
        expected
    );

    for length in [positions.len() - 1, positions.len() + 1] {
        let mut payload = positions.to_vec();
        payload.resize(length, 0);

        let bytes = replace_frame(original.clone(), 0, &raw_frame(&payload));

        let error = SpzGaussians::read_from(&mut bytes.as_slice()).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert_eq!(
            error.to_string(),
            if length < positions.len() {
                "SPZ v4 attribute shorter than its declared size"
            } else {
                "SPZ v4 attribute exceeds its declared size"
            }
        );
    }

    let mut oversized_window = raw_frame(&positions);
    oversized_window[5] = 0xff;

    let bytes = replace_frame(original, 0, &oversized_window);

    assert_eq!(
        SpzGaussians::read_from(&mut bytes.as_slice())
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidData
    );
}

#[test]
fn test_spz_v4_batch_reads_when_frames_have_multiple_zstd_blocks_should_decode_all_gaussians() {
    let count = 20_000u32;
    let mut bytes = given_spz_v4_buffer(0);
    bytes.truncate(V4_HEADER_SIZE);
    set_u32(&mut bytes, V4_NUM_POINTS_OFFSET, count);

    let header = SpzGaussiansHeader::new(
        4,
        count,
        SpzGaussianShDegree::new(0).expect("valid SH degree"),
        12,
        true,
    )
    .unwrap();
    let points = (0..count).map(|index| SpzGaussian {
        position: SpzGaussianPosition::FixedPoint24(std::array::from_fn(|axis| {
            (index * 3 + axis as u32).to_le_bytes()[..3]
                .try_into()
                .unwrap()
        })),
        alpha: index.wrapping_mul(7) as u8,
        color: [index as u8, (index >> 8) as u8, 37],
        scale: [53, index.wrapping_mul(11) as u8, 97],
        rotation: SpzGaussianRotation::QuatSmallestThree([
            index as u8,
            (index >> 8) as u8,
            0x20,
            0x40,
        ]),
        sh: SpzGaussianSh::Zero,
    });
    let expected = SpzGaussians::from_iter(header, points).unwrap();
    let SpzGaussiansPositions::FixedPoint24(positions) = &expected.positions else {
        panic!("v4 positions");
    };
    let SpzGaussiansRotations::QuatSmallestThree(rotations) = &expected.rotations else {
        panic!("v4 rotations");
    };
    let fields: [&[u8]; 5] = [
        bytemuck::cast_slice(positions),
        &expected.alphas,
        bytemuck::cast_slice(&expected.colors),
        bytemuck::cast_slice(&expected.scales),
        bytemuck::cast_slice(rotations),
    ];

    assert!(fields[0].len() > ZSTD_BLOCK_SIZE);

    let frames = fields.map(|field| {
        let frame = raw_frame(field);
        bytes.extend_from_slice(&(frame.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(field.len() as u64).to_le_bytes());

        frame
    });

    for frame in frames {
        bytes.extend_from_slice(&frame);
    }

    let mut reader = SpzBatchReader::new(given::ShortRead(&bytes)).unwrap();
    let mut completed = 0;

    while !reader.progress().done {
        let progress = reader.step(NonZeroUsize::new(127).unwrap()).unwrap();

        assert!(progress.completed_units > completed);
        assert!(progress.completed_units - completed <= 127);

        completed = progress.completed_units;
    }

    let model = reader.finish().unwrap();

    assert_eq!(model, expected);
    assert_eq!(model.iter_gaussian().count(), count as usize);
}

#[test]
fn test_spz_v4_batch_reads_when_model_is_empty_should_retain_degree_without_frames() {
    for degree in 0..=4 {
        let mut bytes = given_spz_v4_buffer(degree);
        bytes.truncate(V4_HEADER_SIZE);
        set_u32(&mut bytes, V4_NUM_POINTS_OFFSET, 0);
        bytes[V4_STREAM_COUNT_OFFSET] = 0;

        let reader = SpzBatchReader::new(given::ShortRead(&bytes)).unwrap();

        assert!(reader.progress().done);

        let model = reader.finish().unwrap();

        assert!(model.is_empty());
        assert_eq!(model.sh_degree().get(), degree);
        assert_eq!(model.iter_gaussian().count(), 0);
    }
}

#[test]
fn test_spz_decompressed_io_when_version_is_four_should_fail_without_emitting_bytes() {
    let model = given_spz_v4_gaussians(4);
    let mut output = Vec::new();

    let error = model.write_decompressed(&mut output).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(
        error.to_string(),
        "SPZ v4 does not use the legacy decompressed layout"
    );
    assert!(output.is_empty());

    let mut input = b"unread body".as_slice();
    let error = SpzGaussians::read_gaussians(&mut input, model.header).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(
        error.to_string(),
        "SPZ v4 does not use the legacy decompressed layout"
    );
    assert_eq!(input, b"unread body");

    // A valid in-memory header isolates layout rejection from wire-header validation.
    let bytes = bytemuck::bytes_of(model.header.as_pod());

    for result in [
        SpzGaussians::read_header(&mut &bytes[..]).map(|_| ()),
        SpzGaussians::read_decompressed(&mut &bytes[..]).map(|_| ()),
    ] {
        let error = result.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "SPZ v4 does not use the legacy decompressed layout"
        );
    }
}

#[test]
fn test_spz_gaussians_header_should_default_to_version_four_and_support_legacy_metadata() {
    assert_eq!(SpzGaussiansHeader::default(0).unwrap().version(), 4);
    assert_eq!(SpzGaussiansFromGaussianSliceOptions::default().version, 4);

    assert_eq!(std::mem::size_of::<SpzGaussiansHeaderPod>(), 16);
    assert_eq!(SpzGaussiansHeader::SUPPORTED_VERSIONS, 1..=4);
    assert_eq!(SpzGaussiansHeader::SUPPORTED_SH_DEGREES, 0..=4);
}

#[test]
fn test_spz_v4_writes_should_preserve_field_layout_and_emit_exact_toc_ranges() {
    for degree in 0..=4 {
        let original = given_spz_v4_gaussians(degree);
        let mut bytes = Vec::new();
        original.write_to(&mut bytes).unwrap();

        assert_eq!(
            SpzGaussians::read_from(&mut bytes.as_slice()).unwrap(),
            original
        );
        assert_eq!(
            &bytes[..V4_STREAM_COUNT_OFFSET],
            &bytemuck::bytes_of(original.header.as_pod())[..V4_STREAM_COUNT_OFFSET]
        );
        assert_eq!(
            bytes[V4_STREAM_COUNT_OFFSET],
            if degree == 0 { 5 } else { 6 }
        );
        assert_eq!(
            &bytes[V4_TOC_OFFSET..V4_RESERVED_OFFSET],
            &(V4_HEADER_SIZE as u32).to_le_bytes()
        );
        assert_eq!(
            &bytes[V4_RESERVED_OFFSET..V4_HEADER_SIZE],
            &[0; V4_HEADER_SIZE - V4_RESERVED_OFFSET]
        );

        let SpzGaussiansPositions::FixedPoint24(positions) = &original.positions else {
            panic!("v4 positions");
        };
        let SpzGaussiansRotations::QuatSmallestThree(rotations) = &original.rotations else {
            panic!("v4 rotations");
        };
        let shs = match &original.shs {
            SpzGaussiansShs::Zero => &[][..],
            SpzGaussiansShs::One(v) => bytemuck::cast_slice(v),
            SpzGaussiansShs::Two(v) => bytemuck::cast_slice(v),
            SpzGaussiansShs::Three(v) => bytemuck::cast_slice(v),
            SpzGaussiansShs::Four(v) => bytemuck::cast_slice(v),
        };
        let fields = [
            bytemuck::cast_slice(positions),
            original.alphas.as_slice(),
            bytemuck::cast_slice(&original.colors),
            bytemuck::cast_slice(&original.scales),
            bytemuck::cast_slice(rotations),
            shs,
        ];
        let mut offset =
            V4_HEADER_SIZE + usize::from(bytes[V4_STREAM_COUNT_OFFSET]) * V4_TOC_ENTRY_SIZE;

        // Parse the wire layout separately from the production container reader.
        for (index, field) in fields
            .into_iter()
            .filter(|field| !field.is_empty())
            .enumerate()
        {
            let toc = V4_HEADER_SIZE + index * V4_TOC_ENTRY_SIZE;
            let compressed_size =
                u64::from_le_bytes(bytes[toc..toc + 8].try_into().unwrap()) as usize;
            let uncompressed_size =
                u64::from_le_bytes(bytes[toc + 8..toc + V4_TOC_ENTRY_SIZE].try_into().unwrap());

            assert_eq!(uncompressed_size, field.len() as u64);

            let frame = &bytes[offset..offset + compressed_size];

            assert_eq!(&frame[..4], &[0x28, 0xb5, 0x2f, 0xfd]);

            let mut decoder = ruzstd::decoding::StreamingDecoder::new(frame).unwrap();
            let mut decoded = Vec::new();
            decoder.read_to_end(&mut decoded).unwrap();

            assert_eq!(decoded, field);
            assert!(decoder.into_inner().is_empty());

            offset += compressed_size;
        }

        assert_eq!(offset, bytes.len());
    }
}

#[test]
fn test_spz_v4_batch_writes_when_output_has_short_writes_and_no_seek_should_respect_item_budgets() {
    for degree in 0..=4 {
        let original = given_spz_v4_gaussians(degree);
        let mut expected = Vec::new();
        original.write_to(&mut expected).unwrap();

        for budget in [1, 2, 4096] {
            let mut writer = SpzBatchWriter::new(given::ShortWrite::default(), &original).unwrap();
            let mut completed = 0;

            while !writer.progress().done {
                let progress = writer.step(NonZeroUsize::new(budget).unwrap()).unwrap();

                assert!(progress.completed_units > completed);
                assert!(progress.completed_units - completed <= budget);

                completed = progress.completed_units;
            }

            let progress = writer.progress();

            assert_eq!(
                writer.step(NonZeroUsize::new(1).unwrap()).unwrap(),
                progress
            );
            assert_eq!(writer.finish().unwrap().0, expected);
        }

        let unified = Gaussians::from(original.clone());
        let mut writer = GaussiansBatchWriter::new(given::ShortWrite::default(), &unified).unwrap();

        while !writer.progress().done {
            writer.step(NonZeroUsize::new(1).unwrap()).unwrap();
        }

        assert_eq!(writer.finish().unwrap().0, expected);
    }
}

#[test]
fn test_spz_v4_native_and_unified_conversions_should_write_all_sh_degrees() {
    for degree in 0..=4 {
        let options = SpzGaussiansFromGaussianSliceOptions {
            version: 4,
            sh_degree: SpzGaussianShDegree::new(degree).unwrap(),
            fractional_bits: 8,
            antialiased: true,
            ..Default::default()
        };
        let native = given::runtime_gaussians(degree);
        let original = SpzGaussians::from_gaussians_with_options(&native, &options).unwrap();
        let path = given::temp_file_path(".spz");
        original.write_to_file(&path).unwrap();

        assert_eq!(SpzGaussians::read_from_file(&path).unwrap(), original);

        let source =
            Gaussians::from_gaussians_iter(native.into_iter(), GaussiansSource::Internal).unwrap();

        for source in [source.clone(), Gaussians::from(source.into_ply().unwrap())] {
            let converted = source.into_spz_with_options(&options).unwrap();

            assert_eq!(converted, original);

            let mut bytes = Vec::new();
            converted.write_to(&mut bytes).unwrap();

            assert_eq!(
                SpzGaussians::read_from(&mut bytes.as_slice()).unwrap(),
                original
            );
        }
    }
}

#[test]
fn test_spz_v4_batch_writes_when_model_is_empty_should_emit_only_the_header_for_all_sh_degrees() {
    for degree in 0..=4 {
        let original = SpzGaussians::from_gaussians_with_options(
            Vec::<AnyGaussian>::new(),
            &SpzGaussiansFromGaussianSliceOptions {
                version: 4,
                sh_degree: SpzGaussianShDegree::new(degree).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
        let writer = SpzBatchWriter::new(given::ShortWrite::default(), &original).unwrap();

        assert!(writer.progress().done);

        let bytes = writer.finish().unwrap().0;

        assert_eq!(bytes.len(), V4_HEADER_SIZE);
        assert_eq!(bytes[V4_STREAM_COUNT_OFFSET], 0);
        assert_eq!(
            SpzGaussians::read_from(&mut bytes.as_slice()).unwrap(),
            original
        );
    }
}

#[test]
fn test_spz_v4_writes_when_extensions_were_discarded_should_clear_extension_flag() {
    let extended = with_extension(given_spz_v4_buffer(4), &[1, 0, 0, 0, 0, 0, 0, 0]);
    let original = SpzGaussians::read_from(&mut extended.as_slice()).unwrap();
    let mut bytes = Vec::new();
    original.write_to(&mut bytes).unwrap();

    assert_eq!(bytes[V4_FLAGS_OFFSET] & V4_EXTENSION_FLAG, 0);
    assert_eq!(bytes[V4_FLAGS_OFFSET], 1);
    assert_eq!(
        &bytes[V4_TOC_OFFSET..V4_RESERVED_OFFSET],
        &(V4_HEADER_SIZE as u32).to_le_bytes()
    );

    assert_eq!(
        SpzGaussians::read_from(&mut bytes.as_slice()).unwrap(),
        given_spz_v4_gaussians(4)
    );
}

#[test]
fn test_spz_v4_writes_when_fields_are_mismatched_should_return_error_without_emitting_bytes() {
    let original = given_spz_v4_gaussians(4);
    let mut cases = Vec::new();

    let mut model = original.clone();
    model.header.set_num_points(1);
    cases.push(("header count", model));

    let mut model = original.clone();
    model.alphas.pop();
    cases.push(("alpha count", model));

    let mut model = original.clone();
    model.colors.push([0; 3]);
    cases.push(("color count", model));

    let mut model = original.clone();
    model.scales.clear();
    cases.push(("scale count", model));

    let mut model = original.clone();
    model.positions = SpzGaussiansPositions::FixedPoint24(Vec::new());
    cases.push(("position count", model));

    let mut model = original.clone();
    model.positions = SpzGaussiansPositions::Float16(vec![[0; 3]; 2]);
    cases.push(("position variant", model));

    let mut model = original.clone();
    model.rotations = SpzGaussiansRotations::QuatSmallestThree(Vec::new());
    cases.push(("rotation count", model));

    let mut model = original.clone();
    model.rotations = SpzGaussiansRotations::QuatFirstThree(vec![[0; 3]; 2]);
    cases.push(("rotation variant", model));

    let mut model = original.clone();
    model.shs = SpzGaussiansShs::Four(vec![[[0; 3]; 24]]);
    cases.push(("SH count", model));

    let mut model = original;
    model.shs = SpzGaussiansShs::Three(vec![[[0; 3]; 15]; 2]);
    cases.push(("SH degree", model));

    for (name, model) in cases {
        let mut bytes = Vec::new();
        let error = model.write_to(&mut bytes).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidInput, "{name}");
        assert_eq!(
            error.to_string(),
            "SPZ field lengths or variants do not match header",
            "{name}"
        );
        assert!(bytes.is_empty(), "{name}");
        let error = SpzBatchWriter::new(&mut bytes, &model)
            .err()
            .expect("mismatched fields should fail");

        assert_eq!(error.kind(), ErrorKind::InvalidInput, "{name}");
        assert_eq!(
            error.to_string(),
            "SPZ field lengths or variants do not match header",
            "{name}"
        );
        assert!(bytes.is_empty(), "{name}");
    }
}

#[test]
fn test_spz_gaussians_header_when_degree_is_four_and_version_is_legacy_should_return_error() {
    let degree = SpzGaussianShDegree::new(4).unwrap();

    for version in 1..=3 {
        let error = SpzGaussiansHeader::new(version, 0, degree, 12, false).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert_eq!(
            error.to_string(),
            "SPZ versions 1 to 3 don't support SH degree 4"
        );
    }
}

#[test]
fn test_spz_v4_header_when_fractional_bits_are_at_supported_limits_should_succeed() {
    let degree = SpzGaussianShDegree::new(4).unwrap();

    for fractional_bits in [0, 30] {
        let header = SpzGaussiansHeader::new(4, 0, degree, fractional_bits, false).unwrap();

        assert_eq!(header.fractional_bits(), fractional_bits);
    }
}

#[test]
fn test_spz_v4_header_when_fractional_bits_are_out_of_range_should_return_error() {
    let degree = SpzGaussianShDegree::new(4).unwrap();

    for fractional_bits in [31, 255] {
        let error = SpzGaussiansHeader::new(4, 0, degree, fractional_bits, false).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert_eq!(
            error.to_string(),
            "invalid SPZ v4 fractional bits, flags or reserved byte"
        );
    }
}

#[test]
fn test_spz_v4_header_when_flags_or_reserved_bytes_are_invalid_should_return_error() {
    let degree = SpzGaussianShDegree::new(4).unwrap();
    let pod = *SpzGaussiansHeader::new(4, 0, degree, 12, false)
        .unwrap()
        .as_pod();

    for (name, pod) in [
        ("flags", SpzGaussiansHeaderPod { flags: 0x80, ..pod }),
        (
            "reserved byte",
            SpzGaussiansHeaderPod { reserved: 1, ..pod },
        ),
    ] {
        let error = SpzGaussiansHeader::try_from_pod(pod).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidData, "{name}");
        assert_eq!(
            error.to_string(),
            "invalid SPZ v4 fractional bits, flags or reserved byte",
            "{name}"
        );
    }
}

#[test]
fn test_spz_v4_writes_when_output_fails_should_propagate_error() {
    let original = given_spz_v4_gaussians(4);
    let mut complete = Vec::new();
    original.write_to(&mut complete).unwrap();

    for limit in [
        0,
        V4_HEADER_SIZE - 1,
        V4_HEADER_SIZE,
        127,
        complete.len() - 1,
    ] {
        assert_eq!(
            original
                .write_to(&mut given::FailingWrite(limit))
                .unwrap_err()
                .kind(),
            ErrorKind::BrokenPipe
        );
    }
}

#[test]
fn test_spz_v4_batch_writer_finish_when_incomplete_should_return_error_without_writing() {
    let original = given_spz_v4_gaussians(4);
    let mut bytes = Vec::new();
    let mut writer = SpzBatchWriter::new(&mut bytes, &original).unwrap();
    writer.step(NonZeroUsize::new(1).unwrap()).unwrap();

    assert_eq!(
        writer.finish().unwrap_err().kind(),
        ErrorKind::UnexpectedEof
    );
    assert!(bytes.is_empty());
}

#[test]
fn test_spz_v4_writes_when_attributes_span_multiple_blocks_should_compress_and_round_trip() {
    let original = given_spz_v4_gaussians(4);
    let point = original.iter().next().unwrap().to_inner_owned();
    let mut header = original.header;
    header.set_num_points(20_000);
    let model = SpzGaussians::from_iter(header, std::iter::repeat_n(point, 20_000)).unwrap();
    let mut writer = SpzBatchWriter::new(Vec::new(), &model).unwrap();

    while !writer.progress().done {
        writer.step(NonZeroUsize::new(127).unwrap()).unwrap();
    }

    let bytes = writer.finish().unwrap();

    assert!(
        bytes.len() < 20_000,
        "repetitive fields should be compressed"
    );
    assert_eq!(
        SpzGaussians::read_from(&mut bytes.as_slice()).unwrap(),
        model
    );
}

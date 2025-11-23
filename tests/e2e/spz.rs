use assert_matches::assert_matches;
use glam::*;
use wgpu_3dgs_core::{
    Gaussian, IterGaussian, SpzGaussian, SpzGaussianPosition, SpzGaussianRef, SpzGaussianRotation,
    SpzGaussianSh, SpzGaussians, SpzGaussiansCollectError, SpzGaussiansFromGaussianSliceOptions,
    SpzGaussiansFromIterError, SpzGaussiansHeader,
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
            sh: match options.sh_degree {
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

#[test]
fn test_spz_gaussian_pod_len_and_is_empty_should_be_correct() {
    let spz_gaussians = given::spz_gaussians();

    assert_eq!(spz_gaussians.len(), 2);
    assert!(!spz_gaussians.is_empty());
}

#[test]
fn test_spz_gaussians_write_spz_file_and_read_spz_file_should_be_equal() {
    let spz_gaussians = given::spz_gaussians();
    let path = given::temp_file_path(".spz");

    spz_gaussians.write_spz_file(&path).unwrap();
    let spz_gaussians_read = SpzGaussians::read_spz_file(&path).unwrap();

    assert_eq!(spz_gaussians.len(), spz_gaussians_read.len());

    for (a, b) in spz_gaussians.iter().zip(spz_gaussians_read.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
fn test_spz_gaussians_write_spz_and_read_spz_should_be_equal() {
    let spz_gaussians = given::spz_gaussians();

    let mut buffer = Vec::new();
    spz_gaussians.write_spz(&mut buffer).unwrap();
    let spz_gaussians_read = SpzGaussians::read_spz(&mut buffer.as_slice()).unwrap();

    assert_eq!(spz_gaussians.len(), spz_gaussians_read.len());

    for (a, b) in spz_gaussians.iter().zip(spz_gaussians_read.iter()) {
        assert_eq!(a, b);
    }
}

fn test_spz_gaussians_write_spz_with_options_and_read_spz_should_be_equal(
    options: &SpzGaussiansFromGaussianSliceOptions,
) {
    let gaussians = given::gaussians();
    let from_options =
        SpzGaussians::from_gaussians_with_options(gaussians.as_slice(), options).unwrap();

    let mut buffer = Vec::new();
    from_options.write_spz(&mut buffer).unwrap();
    let spz_gaussians_read = SpzGaussians::read_spz(&mut buffer.as_slice()).unwrap();
    assert_eq!(from_options.len(), spz_gaussians_read.len());
}

#[test]
fn test_spz_gaussians_write_spz_with_options_when_versions_and_read_spz_should_be_equal() {
    for version in SpzGaussiansHeader::SUPPORTED_VERSIONS {
        println!("Version: {version}");
        test_spz_gaussians_write_spz_with_options_and_read_spz_should_be_equal(
            &SpzGaussiansFromGaussianSliceOptions {
                version,
                ..Default::default()
            },
        );
    }
}

#[test]
fn test_spz_gaussians_write_spz_with_options_when_sh_degrees_and_read_spz_should_be_equal() {
    for sh_degree in SpzGaussiansHeader::SUPPORTED_SH_DEGREES {
        println!("SH Degree: {sh_degree}");
        test_spz_gaussians_write_spz_with_options_and_read_spz_should_be_equal(
            &SpzGaussiansFromGaussianSliceOptions {
                sh_degree,
                ..Default::default()
            },
        );
    }
}

#[test]
fn test_spz_gaussians_write_spz_with_options_when_fractional_bits_and_read_spz_should_be_equal() {
    for fractional_bits in [8, 12, 16] {
        println!("Fractional Bits: {fractional_bits}");
        test_spz_gaussians_write_spz_with_options_and_read_spz_should_be_equal(
            &SpzGaussiansFromGaussianSliceOptions {
                fractional_bits,
                ..Default::default()
            },
        );
    }
}

#[test]
fn test_spz_gaussians_from_slice_and_iter_iter_gaussian_should_be_equal() {
    let gaussians = given::gaussians();
    let original_spz = given::spz_gaussians(); // `spz_gaussians` uses the same as `gaussians`

    let from_slice = SpzGaussians::from(gaussians.as_slice());

    assert_eq!(original_spz, from_slice);

    for (original_ref, slice_ref, slice_gaussian) in itertools::izip!(
        original_spz.iter(),
        from_slice.iter(),
        from_slice.iter_gaussian(),
    ) {
        assert_eq!(original_ref, slice_ref);
        assert::gaussian(
            &slice_gaussian,
            &Gaussian::from_spz(slice_ref, &from_slice.header),
            ASSERT_GAUSSIAN_OPTIONS,
        );
    }
}

fn test_spz_gaussians_from_gaussians_with_options_and_iter_should_be_equal(
    options: &SpzGaussiansFromGaussianSliceOptions,
    mut assertion: impl FnMut(SpzGaussianRef, &Gaussian, &SpzGaussiansHeader),
) {
    let gaussians = given::gaussians();
    let from_options =
        SpzGaussians::from_gaussians_with_options(gaussians.as_slice(), options).unwrap();

    for (a, b) in from_options.iter().zip(gaussians.iter()) {
        assertion(a, b, &from_options.header);
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
                let gaussian_from_spz = Gaussian::from_spz(spz_gaussian_ref, header);
                assert::gaussian(&gaussian_from_spz, gaussian, ASSERT_GAUSSIAN_OPTIONS);
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
                sh_degree,
                ..Default::default()
            },
            |spz_gaussian_ref, gaussian, header| {
                let gaussian_from_spz = Gaussian::from_spz(spz_gaussian_ref, header);
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
                    ASSERT_GAUSSIAN_OPTIONS,
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
                let gaussian_from_spz = Gaussian::from_spz(spz_gaussian_ref, header);
                assert::gaussian(&gaussian_from_spz, gaussian, ASSERT_GAUSSIAN_OPTIONS);
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
                let gaussian_from_spz = Gaussian::from_spz(spz_gaussian_ref, header);
                assert::gaussian(&gaussian_from_spz, gaussian, ASSERT_GAUSSIAN_OPTIONS);
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
        Err(e)
        if e.kind() == std::io::ErrorKind::InvalidData &&
            e.to_string() == "Unsupported SPZ version: 999, expected one of 1..=3"
    );
}

#[test]
fn test_spz_gaussians_from_gaussians_with_options_when_header_sh_degree_is_invalid_should_return_error()
 {
    let gaussians = given::gaussians();
    let options = wgpu_3dgs_core::SpzGaussiansFromGaussianSliceOptions {
        sh_degree: 99,
        ..Default::default()
    };

    let result = SpzGaussians::from_gaussians_with_options(gaussians.iter(), &options);

    assert_matches!(
        result,
        Err(e)
        if e.kind() == std::io::ErrorKind::InvalidData &&
            e.to_string() == "Unsupported SH degree: 99, expected one of 0..=3"
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
    let (gaussian, header) = given_spz_gaussian_and_header(1, &Default::default());
    let gaussians = vec![SpzGaussian {
        sh: SpzGaussianSh::One([[0; 3]; 3]),
        ..gaussian.clone()
    }];

    let result = SpzGaussians::from_iter(header, gaussians);

    assert_matches!(
        result,
        Err(SpzGaussiansFromIterError::ShDegreeMismatch {
            sh_degree: 1,
            header_sh_degree: 3
        })
    );
}

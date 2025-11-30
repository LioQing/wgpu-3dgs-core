use assert_matches::assert_matches;
use glam::*;
use wgpu_3dgs_core::{
    Gaussian, IterGaussian, SpzGaussian, SpzGaussianPosition, SpzGaussianRef, SpzGaussianRotation,
    SpzGaussianSh, SpzGaussianShDegree, SpzGaussianShRef, SpzGaussians, SpzGaussiansCollectError,
    SpzGaussiansFromGaussianSliceOptions, SpzGaussiansFromIterError, SpzGaussiansHeader,
    SpzGaussiansHeaderPod, SpzGaussiansPositions, SpzGaussiansRotations, SpzGaussiansShs,
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
                sh_degree: SpzGaussianShDegree::new(sh_degree).expect("valid SH degree"),
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
fn test_spz_gaussians_from_iter_and_iter_iter_gaussian_should_be_equal() {
    let gaussians = given::gaussians();
    let original_spz = given::spz_gaussians(); // `spz_gaussians` uses the same as `gaussians`

    let from_iter = gaussians.iter().collect::<SpzGaussians>();

    assert_eq!(original_spz, from_iter);

    for (original_ref, iter_ref, slice_gaussian) in itertools::izip!(
        original_spz.iter(),
        from_iter.iter(),
        from_iter.iter_gaussian(),
    ) {
        assert_eq!(original_ref, iter_ref);
        assert::gaussian(
            &slice_gaussian,
            &Gaussian::from_spz(iter_ref, &from_iter.header),
            &ASSERT_GAUSSIAN_OPTIONS,
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
                let gaussian_from_spz = Gaussian::from_spz(spz_gaussian_ref, header);
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
                let gaussian_from_spz = Gaussian::from_spz(spz_gaussian_ref, header);
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
        Err(e)
        if e.kind() == std::io::ErrorKind::InvalidData &&
            e.to_string() == "Unsupported SPZ version: 999, expected one of 1..=3"
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

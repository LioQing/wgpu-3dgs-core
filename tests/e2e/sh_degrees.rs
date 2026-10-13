use std::{mem::size_of, num::NonZeroUsize};

use assert_matches::assert_matches;

use wgpu_3dgs_core::*;

use crate::common::{assert, given};

fn round_trip<D: ShDegree>()
where
    InternalGaussians: From<Vec<Gaussian<D>>>,
{
    let original = (0..3)
        .map(given::gaussian_for_sh_degree::<D>)
        .collect::<Vec<_>>();
    let ply = PlyGaussians::from_gaussians(&original).unwrap();
    let empty_ply = PlyGaussians::new(Vec::new(), D::DEGREE).unwrap();

    assert_eq!(ply.sh_degree, D::DEGREE);
    assert_eq!(empty_ply.sh_degree, D::DEGREE);
    assert_eq!(size_of::<D::Coefficients>(), D::COEFFICIENT_COUNT * 12);

    let native = Gaussians::from(ply.clone()).into_internal().unwrap();

    let restored = ply
        .iter()
        .map(|g| Gaussian::<D>::from_ply(g).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(native, restored.into());
    assert_eq!(native.sh_degree().get(), D::DEGREE);

    for (native, ply) in original.iter().zip(ply.iter()) {
        let restored = Gaussian::<D>::from_ply(ply).unwrap();

        assert::gaussian(
            native,
            &restored,
            &assert::GaussianOptions {
                pos_epsilon: 1e-5,
                rot_epsilon: 1e-5,
                color_tolerance: 1,
                sh_epsilon: 1e-6,
                scale_epsilon: 1e-4,
            },
        );
    }

    for input in [ply, empty_ply] {
        let mut bytes = Vec::new();
        input.write_to(&mut bytes).unwrap();

        let stream = GaussiansStream::new(bytes.as_slice(), GaussiansSource::Ply).unwrap();

        assert_eq!(stream.sh_degree().get(), D::DEGREE);

        let mut stream = stream.into_typed::<D>().unwrap();
        let mut batch = Vec::new();

        while !stream.progress().done {
            stream
                .next_batch(NonZeroUsize::new(1).unwrap(), &mut batch)
                .unwrap();
        }

        assert_eq!(batch.len(), input.len());

        let converted = Gaussians::from_gaussians_iter_with_sh_degree(
            batch.into_iter().map(|g| g.to_any_gaussian()),
            GaussiansSource::Ply,
            GaussianShDegree::new(D::DEGREE).unwrap(),
        )
        .unwrap();

        assert_eq!(converted.sh_degree().get(), D::DEGREE);
        assert_eq!(converted.len(), input.len());
    }

    if D::DEGREE <= 3 {
        let spz = SpzGaussians::from_gaussians(&original).unwrap();
        let empty = SpzGaussians::from_gaussians(Vec::<Gaussian<D>>::new()).unwrap();

        assert_eq!(spz.header.sh_degree().get(), D::DEGREE);
        assert_eq!(empty.header.sh_degree().get(), D::DEGREE);

        for spz in spz.iter() {
            assert_eq!(
                Gaussian::<D>::from_spz(spz, &empty.header)
                    .unwrap()
                    .sh_degree()
                    .get(),
                D::DEGREE
            );
        }
    } else {
        assert!(SpzGaussians::from_gaussians(&original).is_err());
        assert!(SpzGaussians::from_gaussians(Vec::<Gaussian<D>>::new()).is_err());
    }
}

#[test]
fn test_native_degrees_should_preserve_storage_and_source_metadata() {
    round_trip::<ShDegree0>();
    round_trip::<ShDegree1>();
    round_trip::<ShDegree2>();
    round_trip::<ShDegree3>();
    round_trip::<ShDegree4>();
}

fn inferred_native_constructors<D: ShDegree>()
where
    InternalGaussians: From<Vec<Gaussian<D>>>,
{
    for count in [0, 1, 3] {
        let original = (0..count)
            .map(given::gaussian_for_sh_degree::<D>)
            .collect::<Vec<_>>();
        let expected_native = InternalGaussians::from(original.clone());
        let expected_ply =
            PlyGaussians::new(original.iter().map(Gaussian::to_ply).collect(), D::DEGREE).unwrap();

        assert_eq!(
            InternalGaussians::from_gaussians(original.clone()).unwrap(),
            expected_native
        );
        assert_eq!(
            InternalGaussians::from_gaussians(&original).unwrap(),
            expected_native
        );
        assert_eq!(
            PlyGaussians::from_gaussians(original.clone()).unwrap(),
            expected_ply
        );
        assert_eq!(
            PlyGaussians::from_gaussians(&original).unwrap(),
            expected_ply
        );

        if count != 0 {
            let runtime = original
                .iter()
                .map(ToAnyGaussian::to_any_gaussian)
                .collect::<Vec<_>>();
            assert_eq!(
                InternalGaussians::from_gaussians(runtime.clone()).unwrap(),
                expected_native
            );
            assert_eq!(
                InternalGaussians::from_gaussians(&runtime).unwrap(),
                expected_native
            );
            assert_eq!(
                PlyGaussians::from_gaussians(runtime.clone()).unwrap(),
                expected_ply
            );
            assert_eq!(
                PlyGaussians::from_gaussians(&runtime).unwrap(),
                expected_ply
            );
        }
    }

    // Zero-valued coefficients still carry the declared storage degree.
    let mut gaussian = given::gaussian_for_sh_degree::<D>(0);
    gaussian.sh.as_mut().fill(glam::Vec3::ZERO);
    assert_eq!(
        InternalGaussians::from_gaussians([gaussian.to_any_gaussian()])
            .unwrap()
            .sh_degree()
            .get(),
        D::DEGREE
    );
    assert_eq!(
        PlyGaussians::from_gaussians([gaussian.to_any_gaussian()])
            .unwrap()
            .sh_degree,
        D::DEGREE
    );
}

#[test]
fn test_inferred_native_constructors_should_preserve_all_degrees_and_typed_empty_input() {
    inferred_native_constructors::<ShDegree0>();
    inferred_native_constructors::<ShDegree1>();
    inferred_native_constructors::<ShDegree2>();
    inferred_native_constructors::<ShDegree3>();
    inferred_native_constructors::<ShDegree4>();
}

#[test]
fn test_inferred_native_constructors_when_runtime_input_is_empty_should_default_to_degree_three() {
    let runtime = Vec::<AnyGaussian>::new();
    let expected_native = InternalGaussians::Three(Vec::new());
    let expected_ply = PlyGaussians::new(Vec::new(), 3).unwrap();

    assert_eq!(
        InternalGaussians::from_gaussians(runtime.clone()).unwrap(),
        expected_native
    );
    assert_eq!(
        InternalGaussians::from_gaussians(&runtime).unwrap(),
        expected_native
    );
    assert_eq!(
        PlyGaussians::from_gaussians(runtime.clone()).unwrap(),
        expected_ply
    );
    assert_eq!(
        PlyGaussians::from_gaussians(&runtime).unwrap(),
        expected_ply
    );
}

#[test]
fn test_inferred_native_constructors_when_degrees_are_mixed_should_return_error() {
    let high = given::gaussian_for_sh_degree::<ShDegree4>(0).to_any_gaussian();
    let low = given::gaussian_for_sh_degree::<ShDegree1>(1).to_any_gaussian();

    for records in [[high, low], [low, high]] {
        let actual_degree = records[1].sh_degree();
        let expected_degree = records[0].sh_degree();
        assert_eq!(
            InternalGaussians::from_gaussians(records).unwrap_err(),
            ShDegreeMismatchError {
                actual_degree: actual_degree.get(),
                expected_degree: expected_degree.get(),
            }
        );
        assert_eq!(
            InternalGaussians::from_gaussians(records).unwrap_err(),
            ShDegreeMismatchError {
                actual_degree: actual_degree.get(),
                expected_degree: expected_degree.get(),
            }
        );
        for result in [
            PlyGaussians::from_gaussians(records),
            PlyGaussians::from_gaussians(records),
        ] {
            assert_matches!(
                result,
                Err(PlyGaussiansFromIterError::ShCountMismatch {
                    actual_count,
                    expected_count,
                }) if actual_count == actual_degree.num_coefficients() * 3
                    && expected_count == expected_degree.num_coefficients() * 3
            );
        }
    }
}

#[test]
fn test_internal_iterators_should_borrow_and_mutate_all_storage_degrees() {
    macro_rules! check {
        ($degree:ty, $variant:ident) => {
            for count in [0, 1, 3] {
                let mut expected = (0..count)
                    .map(given::gaussian_for_sh_degree::<$degree>)
                    .collect::<Vec<_>>();
                let mut model = InternalGaussians::from(expected.clone());
                let ptr = match &model {
                    InternalGaussians::$variant(stored) => stored.as_ptr(),
                    _ => panic!("unexpected storage degree"),
                };

                {
                    let mut iter = model.iter();
                    assert_eq!(iter.len(), expected.len());
                    for (index, expected) in expected.iter().enumerate() {
                        match iter.next().unwrap() {
                            AnyGaussianRef::$variant(actual) => {
                                assert!(std::ptr::eq(actual, ptr.wrapping_add(index)));
                                assert_eq!(actual, expected);
                            }
                            _ => panic!("unexpected borrowed degree"),
                        }
                        assert_eq!(iter.len(), count as usize - index - 1);
                    }
                    assert!(iter.next().is_none());
                }

                {
                    let mut iter = model.iter_mut();
                    assert_eq!(iter.len(), expected.len());
                    for (index, expected) in expected.iter_mut().enumerate() {
                        expected.pos += glam::Vec3::ONE;
                        expected.color.w = 0.25;
                        for sh in expected.sh.as_mut() {
                            *sh += glam::Vec3::ONE;
                        }
                        match iter.next().unwrap() {
                            AnyGaussianMut::$variant(actual) => {
                                assert!(std::ptr::eq(&*actual, ptr.wrapping_add(index)));
                                *actual = *expected;
                            }
                            _ => panic!("unexpected mutable degree"),
                        }
                        assert_eq!(iter.len(), count as usize - index - 1);
                    }
                    assert!(iter.next().is_none());
                }

                assert_eq!(model, InternalGaussians::from(expected));
                assert_eq!(model.sh_degree().get(), <$degree>::DEGREE);
            }
        };
    }

    check!(ShDegree0, Zero);
    check!(ShDegree1, One);
    check!(ShDegree2, Two);
    check!(ShDegree3, Three);
    check!(ShDegree4, Four);
}

fn source_conversions<D: ShDegree>()
where
    InternalGaussians: From<Vec<Gaussian<D>>>,
{
    for original in [
        (0..3)
            .map(given::gaussian_for_sh_degree::<D>)
            .collect::<Vec<_>>(),
        Vec::new(),
    ] {
        let native = Gaussians::from(original);
        let ply = native.clone().into_ply().unwrap();
        let mut sources = vec![native, Gaussians::from(ply)];

        if D::DEGREE <= 3 {
            let spz = sources[0].clone().into_spz().unwrap();
            sources.push(Gaussians::from(spz));
        }

        for source in sources {
            let ply = source.clone().into_ply().unwrap();
            let expected = match &source {
                Gaussians::Ply(ply) => ply.clone(),
                _ => PlyGaussians::new(
                    source.iter_gaussian().map(|g| g.to_ply()).collect(),
                    D::DEGREE,
                )
                .unwrap(),
            };

            assert_eq!(ply, expected);
            assert_eq!(ply.sh_degree, D::DEGREE);
            assert_eq!(ply.len(), source.len());

            if D::DEGREE <= 3 {
                let options = SpzGaussiansFromGaussianSliceOptions {
                    sh_degree: SpzGaussianShDegree::new(D::DEGREE).unwrap(),
                    ..Default::default()
                };
                let spz = source.clone().into_spz().unwrap();
                let expected = match &source {
                    Gaussians::Spz(spz) => spz.clone(),
                    _ => {
                        SpzGaussians::from_gaussians_with_options(source.iter_gaussian(), &options)
                            .unwrap()
                    }
                };

                assert_eq!(spz, expected);
                assert_eq!(spz.header.sh_degree().get(), D::DEGREE);
                assert_eq!(spz.len(), source.len());

                let expected =
                    SpzGaussians::from_gaussians_with_options(source.iter_gaussian(), &options)
                        .unwrap();
                let reencoded = source.into_spz_with_options(&options).unwrap();

                assert_eq!(reencoded, expected);
                assert_eq!(reencoded.header.sh_degree().get(), D::DEGREE);
            } else {
                assert_matches!(
                    source.clone().into_spz(),
                    Err(SpzGaussiansFromGaussiansError::UnsupportedShDegree { degree: 4 })
                );
                assert_matches!(
                    source.into_spz_with_options(&Default::default()),
                    Err(SpzGaussiansFromGaussiansError::UnsupportedShDegree { degree: 4 })
                );
            }
        }
    }
}

#[test]
fn test_source_conversions_should_preserve_degrees_including_empty_models() {
    source_conversions::<ShDegree0>();
    source_conversions::<ShDegree1>();
    source_conversions::<ShDegree2>();
    source_conversions::<ShDegree3>();
    source_conversions::<ShDegree4>();
}

#[test]
fn test_source_conversions_when_source_matches_target_should_return_existing_storage_unchanged() {
    let mut ply = given::ply_gaussians();
    ply.gaussians[0].normal = glam::Vec3::X;
    let expected = ply.clone();
    let ptr = ply.gaussians.as_ptr();
    let converted = Gaussians::from(ply).into_ply().unwrap();

    assert_eq!(converted, expected);
    assert_eq!(converted.gaussians.as_ptr(), ptr);

    let options = SpzGaussiansFromGaussianSliceOptions {
        version: 2,
        fractional_bits: 8,
        antialiased: true,
        sh_quantize_bits: [8; 3],
        ..Default::default()
    };
    let spz = SpzGaussians::from_gaussians_with_options(given::gaussians(), &options).unwrap();
    let expected = spz.clone();
    let ptr = spz.alphas.as_ptr();
    let converted = Gaussians::from(spz).into_spz().unwrap();

    assert_eq!(converted, expected);
    assert_eq!(converted.alphas.as_ptr(), ptr);
}

#[test]
fn test_explicit_spz_conversion_should_reencode_all_sources_with_requested_options() {
    let options = SpzGaussiansFromGaussianSliceOptions {
        version: 2,
        fractional_bits: 8,
        antialiased: true,
        sh_quantize_bits: [8; 3],
        ..Default::default()
    };

    for source in [
        Gaussians::from(given::gaussians()),
        Gaussians::from(given::ply_gaussians()),
        Gaussians::from(given::spz_gaussians()),
        Gaussians::from(Vec::<Gaussian>::new()),
    ] {
        let expected =
            SpzGaussians::from_gaussians_with_options(source.iter_gaussian(), &options).unwrap();
        let converted = source.into_spz_with_options(&options).unwrap();

        assert_eq!(converted, expected);
        assert_eq!(converted.header.version(), options.version);
        assert_eq!(converted.header.fractional_bits(), options.fractional_bits);
        assert_eq!(converted.header.is_antialiased(), options.antialiased);
    }
}

#[test]
fn test_explicit_spz_conversion_when_options_are_invalid_should_preserve_validation_errors() {
    for original in [
        vec![given::gaussian_for_sh_degree::<ShDegree1>(0)],
        Vec::new(),
    ] {
        let native = Gaussians::from(original);
        let ply = native.clone().into_ply().unwrap();
        let spz = native.clone().into_spz().unwrap();

        for source in [native, Gaussians::from(ply), Gaussians::from(spz)] {
            assert_matches!(
                source.into_spz_with_options(&Default::default()),
                Err(SpzGaussiansFromGaussiansError::ShDegreeMismatch(
                    ShDegreeMismatchError {
                        actual_degree: 1,
                        expected_degree: 3,
                    }
                ))
            );
        }
    }

    let native = Gaussians::from(given::gaussians());
    assert_matches!(
        native
            .clone()
            .into_spz_with_options(&SpzGaussiansFromGaussianSliceOptions {
                version: 0,
                ..Default::default()
            }),
        Err(SpzGaussiansFromGaussiansError::Header(_))
    );
    assert_matches!(
        native.into_spz_with_options(&SpzGaussiansFromGaussianSliceOptions {
            sh_quantize_bits: [5, 9, 4],
            ..Default::default()
        }),
        Err(SpzGaussiansFromGaussiansError::Gaussian(
            GaussianToSpzError::InvalidShQuantizeBits { degree: 2, bits: 9 }
        ))
    );
}

#[test]
fn test_gaussian_conversions_when_degrees_mismatch_should_require_explicit_conversion() {
    let high = given::gaussian_for_sh_degree::<ShDegree4>(0);
    let low = high.convert_sh_degree::<ShDegree1>();
    let promoted = low.convert_sh_degree::<ShDegree4>();

    assert_eq!(&promoted.sh[..3], &high.sh[..3]);
    assert!(promoted.sh[3..].iter().all(|v| *v == glam::Vec3::ZERO));

    assert_matches!(
        Gaussian::<ShDegree3>::from_ply(&high.to_ply()).unwrap_err(),
        PlyGaussiansFromIterError::ShCountMismatch {
            actual_count: 72,
            expected_count: 45,
        }
    );
    assert_eq!(
        AnyGaussian::Four(high)
            .try_typed::<ShDegree3>()
            .unwrap_err(),
        ShDegreeMismatchError {
            actual_degree: 4,
            expected_degree: 3
        },
    );
    assert!(
        InternalGaussians::from_iter(
            [AnyGaussian::Four(high), AnyGaussian::One(low)],
            GaussianShDegree::new(4).unwrap(),
        )
        .is_err()
    );

    assert!(
        Gaussians::from_gaussians_iter(
            [AnyGaussian::Four(high), AnyGaussian::One(low)].into_iter(),
            GaussiansSource::Ply,
        )
        .is_err()
    );

    assert!(SpzGaussians::from_gaussians_with_options([low], &Default::default()).is_err());
    assert!(
        SpzGaussians::from_gaussians_with_options(
            Vec::<Gaussian<ShDegree1>>::new(),
            &Default::default()
        )
        .is_err()
    );

    let spz = SpzGaussians::from_gaussians([low]).unwrap();

    assert!(Gaussian::<ShDegree3>::from_spz(spz.iter().next().unwrap(), &spz.header).is_err());
    assert!(high.to_spz(&spz.header, &Default::default()).is_err());
    assert!(SpzGaussians::from_gaussians([high.convert_sh_degree::<ShDegree3>()]).is_ok());

    let ply = PlyGaussians::from_iter([high.to_ply()], 4).unwrap();
    let mut bytes = Vec::new();
    ply.write_to(&mut bytes).unwrap();

    let stream = GaussiansStream::new(bytes.as_slice(), GaussiansSource::Ply).unwrap();

    assert_eq!(
        stream.into_typed::<ShDegree3>().err().unwrap(),
        ShDegreeMismatchError {
            actual_degree: 4,
            expected_degree: 3
        }
    );
}

#[test]
fn test_gaussian_conversions_when_input_is_invalid_should_preserve_validation_details() {
    let high = given::gaussian_for_sh_degree::<ShDegree4>(0);
    let low = high.convert_sh_degree::<ShDegree1>();
    let unsupported = SpzGaussians::from_gaussians([high]).unwrap_err();

    assert_matches!(
        unsupported,
        SpzGaussiansFromGaussiansError::UnsupportedShDegree { degree: 4 }
    );
    assert_matches!(
        SpzGaussians::from_gaussians([high].iter()),
        Err(SpzGaussiansFromGaussiansError::UnsupportedShDegree { degree: 4 })
    );
    assert_matches!(
        Gaussians::from_gaussians_iter([high].into_iter(), GaussiansSource::Spz),
        Err(GaussiansFromIterError::Spz(
            SpzGaussiansFromGaussiansError::UnsupportedShDegree { degree: 4 }
        ))
    );
    for source in [GaussiansSource::Internal, GaussiansSource::Ply] {
        assert_matches!(
            Gaussians::from_gaussians_iter(
                [high.to_any_gaussian(), low.to_any_gaussian()].into_iter(),
                source,
            ),
            Err(GaussiansFromIterError::ShDegreeMismatch(
                ShDegreeMismatchError {
                    actual_degree: 1,
                    expected_degree: 4,
                }
            ))
        );
    }

    assert_matches!(
        PlyGaussians::new(Vec::new(), 5),
        Err(PlyGaussiansFromIterError::UnsupportedShDegree { degree: 5 })
    );
    let mut invalid = low.to_ply();
    invalid.sh.pop();
    assert_matches!(
        AnyGaussian::from_ply(&invalid),
        Err(PlyGaussiansFromIterError::UnsupportedShCount { count: 8 })
    );

    let spz = SpzGaussians::from_gaussians([low]).unwrap();
    assert_matches!(
        high.to_spz(&spz.header, &Default::default()),
        Err(GaussianToSpzError::ShDegreeMismatch(
            ShDegreeMismatchError {
                actual_degree: 4,
                expected_degree: 1,
            }
        ))
    );
    assert_matches!(
        Gaussian::<ShDegree3>::from_spz(spz.iter().next().unwrap(), &spz.header),
        Err(ShDegreeMismatchError {
            actual_degree: 1,
            expected_degree: 3
        })
    );
    assert_matches!(
        SpzGaussians::from_gaussians_with_options(
            Vec::<Gaussian<ShDegree1>>::new(),
            &Default::default()
        ),
        Err(SpzGaussiansFromGaussiansError::ShDegreeMismatch(
            ShDegreeMismatchError {
                actual_degree: 1,
                expected_degree: 3,
            }
        ))
    );

    let options = GaussianToSpzOptions {
        sh_quantize_bits: [5, 9, 4],
    };
    assert_matches!(
        low.to_spz(&spz.header, &options),
        Err(GaussianToSpzError::InvalidShQuantizeBits { degree: 2, bits: 9 })
    );
    let error = SpzGaussians::from_gaussians_with_options(
        [low],
        &SpzGaussiansFromGaussianSliceOptions {
            sh_degree: SpzGaussianShDegree::new(1).unwrap(),
            sh_quantize_bits: options.sh_quantize_bits,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(std::error::Error::source(&error).is_some());
    assert_matches!(
        error,
        SpzGaussiansFromGaussiansError::Gaussian(GaussianToSpzError::InvalidShQuantizeBits {
            degree: 2,
            bits: 9,
        })
    );
}

#[test]
fn test_gaussians_when_coefficients_are_zero_or_model_is_empty_should_preserve_declared_degree() {
    let mut gaussian = given::gaussian_for_sh_degree::<ShDegree4>(0);
    gaussian.sh.fill(glam::Vec3::ZERO);

    assert_eq!(gaussian.to_ply().sh.len(), 72);
    assert_eq!(AnyGaussian::Four(gaussian).sh_degree().get(), 4);

    let model = Gaussians::from(Vec::<Gaussian<ShDegree4>>::new());

    assert_eq!(model.sh_degree().get(), 4);
    assert_eq!(model.into_internal().unwrap().sh_degree().get(), 4);
}

#[test]
fn test_degree_counts_and_compact_half_strides_should_be_correct() {
    for (degree, count) in [0, 3, 8, 15, 24].into_iter().enumerate() {
        let degree = GaussianShDegree::new(degree as u8).unwrap();

        assert_eq!(degree.num_coefficients(), count);
        assert_eq!(
            GaussianShDegree::from_coefficient_count(count),
            Some(degree)
        );
    }

    assert!(GaussianShDegree::from_coefficient_count(16).is_none());
    assert!(GaussianShDegree::new(5).is_none());

    assert_eq!(size_of::<PackedGaussian<ShHalf<ShDegree0>, CovHalf>>(), 32);
    assert_eq!(size_of::<PackedGaussian<ShHalf<ShDegree1>, CovHalf>>(), 48);
    assert_eq!(size_of::<PackedGaussian<ShHalf<ShDegree2>, CovHalf>>(), 80);
    assert_eq!(size_of::<PackedGaussian<ShHalf<ShDegree3>, CovHalf>>(), 128);
    assert_eq!(size_of::<PackedGaussian<ShHalf<ShDegree4>, CovHalf>>(), 176);
}

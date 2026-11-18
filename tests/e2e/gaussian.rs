use assert_matches::assert_matches;
use wgpu_3dgs_core::{
    AnyGaussian, Gaussian, Gaussians, GaussiansSource, IterGaussian, IteratorGaussianExt,
};

use crate::common::{assert, given};

#[test]
fn test_gaussians_collect_gaussians_from_and_iter_gaussian_when_source_is_internal_should_be_equal()
{
    let original = given::gaussians();
    let gaussians = original
        .clone()
        .into_iter()
        .collect_gaussians(GaussiansSource::Internal)
        .unwrap();
    let from = Gaussians::from(original.clone());

    let iter = gaussians.iter_gaussian();
    let from_iter = from.iter_gaussian();

    assert_eq!(iter.len(), original.len());
    assert_eq!(from_iter.len(), original.len());

    let iterated: Vec<Gaussian> = iter.map(|g| g.try_typed().unwrap()).collect();
    let from_iterated: Vec<Gaussian> = from_iter.map(|g| g.try_typed().unwrap()).collect();

    assert_eq!(original.len(), iterated.len());
    assert_eq!(original.len(), from_iterated.len());

    for (a, b, c) in itertools::izip!(original.iter(), iterated.iter(), from_iterated.iter()) {
        assert_eq!(a, b);
        assert_eq!(a, c);
    }
}

#[test]
fn test_gaussians_collect_gaussians_from_and_iter_gaussian_when_source_is_ply_should_be_equal() {
    let original = given::gaussians();
    let original_ply = given::ply_gaussians();
    let gaussians = original
        .clone()
        .into_iter()
        .collect_gaussians(GaussiansSource::Ply)
        .unwrap();
    let from = Gaussians::from(original_ply.clone());

    let iter = gaussians.iter_gaussian();
    let from_iter = from.iter_gaussian();

    assert_eq!(iter.len(), original.len());
    assert_eq!(from_iter.len(), original.len());

    let iterated: Vec<Gaussian> = iter.map(|g| g.try_typed().unwrap()).collect();
    let from_iterated: Vec<Gaussian> = from_iter.map(|g| g.try_typed().unwrap()).collect();

    assert_eq!(original.len(), iterated.len());
    assert_eq!(original.len(), from_iterated.len());

    let options = assert::GaussianOptions {
        pos_epsilon: 1e-5,
        rot_epsilon: 1e-5,
        color_tolerance: 1,
        sh_epsilon: 1e-5,
        scale_epsilon: 1e-4,
    };

    for (a, b, c) in itertools::izip!(original.iter(), iterated.iter(), from_iterated.iter()) {
        assert::gaussian(a, b, &options);
        assert::gaussian(a, c, &options);
    }
}

#[test]
fn test_gaussians_collect_gaussians_from_and_iter_gaussian_when_source_is_spz_should_be_equal() {
    let original = given::gaussians();
    let original_spz = given::spz_gaussians();
    let gaussians = original
        .clone()
        .into_iter()
        .collect_gaussians(GaussiansSource::Spz)
        .unwrap();
    let from = Gaussians::from(original_spz.clone());

    let iter = gaussians.iter_gaussian();
    let from_iter = from.iter_gaussian();

    assert_eq!(iter.len(), original.len());
    assert_eq!(from_iter.len(), original.len());

    let iterated: Vec<Gaussian> = iter.map(|g| g.try_typed().unwrap()).collect();
    let from_iterated: Vec<Gaussian> = from_iter.map(|g| g.try_typed().unwrap()).collect();

    assert_eq!(original.len(), iterated.len());
    assert_eq!(original.len(), from_iterated.len());

    let options = assert::GaussianOptions {
        pos_epsilon: 0.01,
        rot_epsilon: 0.05,
        color_tolerance: 5,
        sh_epsilon: 0.1,
        scale_epsilon: 2.0,
    };

    for (a, b, c) in itertools::izip!(original.iter(), iterated.iter(), from_iterated.iter()) {
        assert::gaussian(a, b, &options);
        assert::gaussian(a, c, &options);
    }
}

#[test]
fn test_gaussians_collect_gaussians_and_source_should_be_equal() {
    let original = given::gaussians();

    for source in [
        GaussiansSource::Internal,
        GaussiansSource::Ply,
        GaussiansSource::Spz,
    ] {
        println!("Source: {source:?}");

        let gaussians = original
            .clone()
            .into_iter()
            .collect_gaussians(source)
            .unwrap();

        assert_eq!(gaussians.source(), source);
    }
}

#[test]
fn test_gaussians_len_and_is_empty_should_be_correct() {
    let original = given::gaussians();

    for source in [
        GaussiansSource::Internal,
        GaussiansSource::Ply,
        GaussiansSource::Spz,
    ] {
        let gaussians = original
            .clone()
            .into_iter()
            .collect_gaussians(source)
            .unwrap();

        assert_eq!(gaussians.len(), original.len());
        assert_eq!(gaussians.is_empty(), original.is_empty());
    }
}

#[test]
fn test_gaussians_from_gaussians_iter_when_runtime_input_is_empty_should_default_to_degree_three() {
    for source in [
        GaussiansSource::Internal,
        GaussiansSource::Ply,
        GaussiansSource::Spz,
    ] {
        let model =
            Gaussians::from_gaussians_iter(std::iter::empty::<AnyGaussian>(), source).unwrap();

        assert_eq!(model.source(), source);
        assert_eq!(model.sh_degree().get(), 3);
        assert_eq!(model.len(), 0);
        assert!(model.is_empty());
    }
}

#[test]
fn test_gaussians_iter_gaussian_should_report_remaining_count_for_each_source() {
    for model in [
        Gaussians::from(given::gaussians()),
        Gaussians::from(given::ply_gaussians()),
        Gaussians::from(given::spz_gaussians()),
    ] {
        let expected = model.iter_gaussian().collect::<Vec<_>>();
        let mut iter = model.iter_gaussian();

        assert_eq!(iter.len(), expected.len());
        assert_eq!(iter.size_hint(), (expected.len(), Some(expected.len())));

        for (index, expected) in expected.into_iter().enumerate() {
            assert_eq!(iter.next(), Some(expected));
            let remaining = model.len() - index - 1;

            assert_eq!(iter.len(), remaining);
            assert_eq!(iter.size_hint(), (remaining, Some(remaining)));
        }

        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }
}

#[test]
fn test_gaussians_read_from_when_header_is_invalid_should_propagate_error() {
    for source in [GaussiansSource::Ply, GaussiansSource::Spz] {
        assert!(Gaussians::read_from(&mut b"invalid header".as_slice(), source).is_err());
    }
}

#[test]
fn test_gaussians_from_iter_should_have_internal_source() {
    let original = given::gaussians();
    let gaussians: Gaussians = original.into_iter().collect();

    assert_eq!(gaussians.source(), GaussiansSource::Internal);
}

#[test]
fn test_gaussians_file_round_trip_when_source_is_ply_should_be_equal() {
    let gaussians = Gaussians::from(given::ply_gaussians());
    let path = given::temp_file_path(".ply");

    gaussians.write_to_file(&path).unwrap();
    let gaussians_read = Gaussians::read_from_file(&path, GaussiansSource::Ply).unwrap();

    assert_eq!(gaussians.len(), gaussians_read.len());
    assert_eq!(gaussians, gaussians_read);
}

#[test]
fn test_gaussians_file_round_trip_when_source_is_spz_should_be_equal() {
    let gaussians = Gaussians::from(given::spz_gaussians());
    let path = given::temp_file_path(".spz");

    gaussians.write_to_file(&path).unwrap();
    let gaussians_read = Gaussians::read_from_file(&path, GaussiansSource::Spz).unwrap();

    assert_eq!(gaussians.len(), gaussians_read.len());
    assert_eq!(gaussians, gaussians_read);
}

#[test]
fn test_gaussians_write_to_file_when_source_is_internal_should_return_error() {
    let gaussians = Gaussians::from(given::gaussians());
    let path = given::temp_file_path(".bin");

    let result = gaussians.write_to_file(&path);

    assert_matches!(
        result,
        Err(e) if e.kind() == std::io::ErrorKind::InvalidInput &&
            e.to_string() == "cannot write Internal Gaussians to file"
    );
}

#[test]
fn test_gaussians_read_from_file_when_source_is_internal_should_return_error() {
    let path = given::temp_file_path(".bin");

    let result = Gaussians::read_from_file(&path, GaussiansSource::Internal);

    assert_matches!(
        result,
        Err(e) if e.kind() == std::io::ErrorKind::InvalidInput &&
            e.to_string() == "cannot read Internal Gaussians from file"
    );
}

#[test]
fn test_gaussians_buffer_round_trip_when_source_is_ply_should_be_equal() {
    let gaussians = Gaussians::from(given::ply_gaussians());

    let mut buffer = Vec::new();
    gaussians.write_to(&mut buffer).unwrap();
    let gaussians_read =
        Gaussians::read_from(&mut buffer.as_slice(), GaussiansSource::Ply).unwrap();

    assert_eq!(gaussians, gaussians_read);
}

#[test]
fn test_gaussians_buffer_round_trip_when_source_is_spz_should_be_equal() {
    let gaussians = Gaussians::from(given::spz_gaussians());

    let mut buffer = Vec::new();
    gaussians.write_to(&mut buffer).unwrap();
    let gaussians_read =
        Gaussians::read_from(&mut buffer.as_slice(), GaussiansSource::Spz).unwrap();

    assert_eq!(gaussians, gaussians_read);
}

#[test]
fn test_gaussians_write_to_when_source_is_internal_should_return_error() {
    let gaussians = Gaussians::from(given::gaussians());
    let mut buffer = Vec::new();

    let result = gaussians.write_to(&mut buffer);

    assert_matches!(
        result,
        Err(e) if e.kind() == std::io::ErrorKind::InvalidInput &&
            e.to_string() == "cannot write Internal Gaussians to buffer"
    );
}

#[test]
fn test_gaussians_read_from_when_source_is_internal_should_return_error() {
    let buffer = Vec::new();
    let result = Gaussians::read_from(&mut buffer.as_slice(), GaussiansSource::Internal);

    assert_matches!(
        result,
        Err(e) if e.kind() == std::io::ErrorKind::InvalidInput &&
            e.to_string() == "cannot read Internal Gaussians from buffer"
    );
}

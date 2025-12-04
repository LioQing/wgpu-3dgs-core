use assert_matches::assert_matches;
use wgpu_3dgs_core::{Gaussian, Gaussians, GaussiansSource, IterGaussian, IteratorGaussianExt};

use crate::common::{assert, given};

#[test]
fn test_gaussians_collect_gaussians_from_and_iter_gaussian_when_source_is_internal_should_be_equal()
{
    let original = given::gaussians();
    let gaussians = original
        .clone()
        .into_iter()
        .collect_gaussians(GaussiansSource::Internal);
    let from = Gaussians::from(original.clone());

    let iterated: Vec<Gaussian> = gaussians.iter_gaussian().collect();
    let from_iterated: Vec<Gaussian> = from.iter_gaussian().collect();

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
        .collect_gaussians(GaussiansSource::Ply);
    let from = Gaussians::from(original_ply.clone());

    let iterated: Vec<Gaussian> = gaussians.iter_gaussian().collect();
    let from_iterated: Vec<Gaussian> = from.iter_gaussian().collect();

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
        .collect_gaussians(GaussiansSource::Spz);
    let from = Gaussians::from(original_spz.clone());

    let iterated: Vec<Gaussian> = gaussians.iter_gaussian().collect();
    let from_iterated: Vec<Gaussian> = from.iter_gaussian().collect();

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

        let gaussians = original.clone().into_iter().collect_gaussians(source);

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
        let gaussians = original.clone().into_iter().collect_gaussians(source);

        assert_eq!(gaussians.len(), original.len());
        assert_eq!(gaussians.is_empty(), original.is_empty());
    }
}

#[test]
fn test_gaussians_from_iter_should_have_internal_source() {
    let original = given::gaussians();
    let gaussians: Gaussians = original.into_iter().collect();

    assert_eq!(gaussians.source(), GaussiansSource::Internal);
}

#[test]
fn test_gaussians_write_to_file_and_read_from_file_when_source_is_ply_should_be_equal() {
    let gaussians = Gaussians::from(given::ply_gaussians());
    let path = given::temp_file_path(".ply");

    gaussians.write_to_file(&path).unwrap();
    let gaussians_read = Gaussians::read_from_file(&path, GaussiansSource::Ply).unwrap();

    assert_eq!(gaussians.len(), gaussians_read.len());
    assert_eq!(gaussians, gaussians_read);
}

#[test]
fn test_gaussians_write_to_file_and_read_from_file_when_source_is_spz_should_be_equal() {
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
fn test_gaussians_write_to_and_read_from_when_source_is_ply_should_be_equal() {
    let gaussians = Gaussians::from(given::ply_gaussians());

    let mut buffer = Vec::new();
    gaussians.write_to(&mut buffer).unwrap();
    let gaussians_read =
        Gaussians::read_from(&mut buffer.as_slice(), GaussiansSource::Ply).unwrap();

    assert_eq!(gaussians, gaussians_read);
}

#[test]
fn test_gaussians_write_to_and_read_from_when_source_is_spz_should_be_equal() {
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

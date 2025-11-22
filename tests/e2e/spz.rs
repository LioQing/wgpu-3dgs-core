use wgpu_3dgs_core::SpzGaussians;

use crate::common::given;

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

use wgpu_3dgs_core::PlyGaussianPod;

use crate::common::{assert, given};

#[test]
fn test_ply_gaussian_pod_from_and_gaussian_to_ply_should_be_equal() {
    let gaussian = given::gaussian();

    let gaussian_to_ply = gaussian.to_ply();
    let ply_from_ref = PlyGaussianPod::from(&gaussian);
    let ply_from = PlyGaussianPod::from(gaussian);

    assert::ply_gaussian_pod(&gaussian_to_ply, &ply_from_ref);
    assert::ply_gaussian_pod(&gaussian_to_ply, &ply_from);
}

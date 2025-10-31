use glam::*;
use wgpu_3dgs_core::PlyGaussianPod;

use crate::common::given;

#[test]
fn test_ply_gaussian_pod_from_and_gaussian_to_ply_should_be_equal() {
    let gaussian = given::gaussian();

    let gaussian_to_ply = gaussian.to_ply();
    let ply_from_ref = PlyGaussianPod::from(&gaussian);
    let ply_from = PlyGaussianPod::from(gaussian);

    fn assert_ply_gaussian_pod(a: &PlyGaussianPod, b: &PlyGaussianPod) {
        const EPSILON: f32 = 1e-4;

        assert!(
            a.rot
                .into_iter()
                .zip(b.rot.into_iter())
                .all(|(x, y)| (x - y).abs() < EPSILON),
            " left: {:?}\nright: {:?}",
            a.rot,
            b.rot
        );
        assert!(
            a.pos
                .into_iter()
                .zip(b.pos.into_iter())
                .all(|(x, y)| (x - y).abs() < EPSILON),
            " left: {:?}\nright: {:?}",
            a.pos,
            b.pos
        );

        assert_eq!(a.color, b.color);

        assert!(
            a.sh.into_iter()
                .zip(b.sh.into_iter())
                .all(|(x, y)| (x - y).abs() < EPSILON),
            " left: {:?}\nright: {:?}",
            a.sh,
            b.sh
        );

        assert!(
            a.scale
                .into_iter()
                .zip(b.scale.into_iter())
                .all(|(x, y)| (x - y).abs() < EPSILON),
            " left: {:?}\nright: {:?}",
            a.scale,
            b.scale
        );
    }

    assert_ply_gaussian_pod(&gaussian_to_ply, &ply_from_ref);
    assert_ply_gaussian_pod(&gaussian_to_ply, &ply_from);
}

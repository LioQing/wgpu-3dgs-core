use glam::*;
use wgpu_3dgs_core::Gaussian;

use crate::common::given;

#[test]
fn test_gaussian_to_ply_and_from_ply_should_be_approximately_equal() {
    let gaussian = given::gaussian();

    let ply = gaussian.to_ply();
    let gaussian_from_ply = Gaussian::from_ply(&ply);

    const EPSILON: f32 = 1e-4;

    // Color are stored as exponential in PLY but linear in Gaussian
    // so there is expected lost of precision.
    const COLOR_EPSILON: u8 = 1;

    assert!(
        gaussian.rot.abs_diff_eq(gaussian_from_ply.rot, EPSILON),
        " left: {:?}\nright: {:?}",
        gaussian.rot,
        gaussian_from_ply.rot
    );

    assert!(
        gaussian.pos.abs_diff_eq(gaussian_from_ply.pos, EPSILON),
        " left: {:?}\nright: {:?}",
        gaussian.pos,
        gaussian_from_ply.pos
    );

    assert!(
        (gaussian.color - gaussian_from_ply.color)
            .cmple(U8Vec4::splat(COLOR_EPSILON))
            .all(),
        " left: {:?}\nright: {:?}",
        gaussian.color,
        gaussian_from_ply.color
    );

    for i in 0..15 {
        assert!(
            gaussian.sh[i].abs_diff_eq(gaussian_from_ply.sh[i], EPSILON),
            " left: {:?}\nright: {:?}",
            gaussian.sh[i],
            gaussian_from_ply.sh[i]
        );
    }

    assert!(
        gaussian.scale.abs_diff_eq(gaussian_from_ply.scale, EPSILON),
        " left: {:?}\nright: {:?}",
        gaussian.scale,
        gaussian_from_ply.scale
    );
}

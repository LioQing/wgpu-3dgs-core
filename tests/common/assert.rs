use wgpu_3dgs_core::{Gaussian, PlyGaussianPod};

pub fn ply_gaussian_pod(a: &PlyGaussianPod, b: &PlyGaussianPod) {
    const EPSILON: f32 = 1e-4;

    assert!(
        a.rot
            .into_iter()
            .zip(b.rot.into_iter())
            .all(|(x, y)| (x - y).abs() < EPSILON),
        "rotation assertion failed\n left: {:?}\nright: {:?}",
        a.rot,
        b.rot
    );

    assert!(
        a.pos
            .into_iter()
            .zip(b.pos.into_iter())
            .all(|(x, y)| (x - y).abs() < EPSILON),
        "position assertion failed\n left: {:?}\nright: {:?}",
        a.pos,
        b.pos
    );

    assert!(
        a.normal
            .into_iter()
            .zip(b.normal.into_iter())
            .all(|(x, y)| (x - y).abs() < EPSILON),
        "normal assertion failed\n left: {:?}\nright: {:?}",
        a.normal,
        b.normal
    );

    assert!(
        a.sh.into_iter()
            .zip(b.sh.into_iter())
            .all(|(x, y)| (x - y).abs() < EPSILON),
        "sh assertion failed\n left: {:?}\nright: {:?}",
        a.sh,
        b.sh
    );

    assert!(
        a.scale
            .into_iter()
            .zip(b.scale.into_iter())
            .all(|(x, y)| (x - y).abs() < EPSILON),
        "scale assertion failed\n left: {:?}\nright: {:?}",
        a.scale,
        b.scale
    );
}

pub struct GaussianOptions {
    pub pos_epsilon: f32,
    pub rot_epsilon: f32,
    pub color_tolerance: u8,
    pub sh_epsilon: f32,
    pub scale_epsilon: f32,
}

pub fn gaussian(
    a: &Gaussian,
    b: &Gaussian,
    GaussianOptions {
        pos_epsilon,
        rot_epsilon,
        color_tolerance,
        sh_epsilon,
        scale_epsilon,
    }: GaussianOptions,
) {
    assert!(
        a.rot.abs_diff_eq(b.rot, rot_epsilon),
        "rotation assertion failed\n left: {:?}\nright: {:?}",
        a.rot,
        b.rot
    );

    assert!(
        a.pos.abs_diff_eq(b.pos, pos_epsilon),
        "position assertion failed\n left: {:?}\nright: {:?}",
        a.pos,
        b.pos
    );

    assert!(
        (a.color.x as i16 - b.color.x as i16).unsigned_abs() as u8 <= color_tolerance
            && (a.color.y as i16 - b.color.y as i16).unsigned_abs() as u8 <= color_tolerance
            && (a.color.z as i16 - b.color.z as i16).unsigned_abs() as u8 <= color_tolerance
            && (a.color.w as i16 - b.color.w as i16).unsigned_abs() as u8 <= color_tolerance,
        "color assertion failed\n left: {:?}\nright: {:?}",
        a.color,
        b.color
    );

    for i in 0..15 {
        assert!(
            a.sh[i].abs_diff_eq(b.sh[i], sh_epsilon),
            "sh[{}] assertion failed\n left: {:?}\nright: {:?}",
            i,
            a.sh[i],
            b.sh[i]
        );
    }

    assert!(
        a.scale.abs_diff_eq(b.scale, scale_epsilon),
        "scale assertion failed\n left: {:?}\nright: {:?}",
        a.scale,
        b.scale
    );
}

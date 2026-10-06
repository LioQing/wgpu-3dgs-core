use wgpu_3dgs_core::{PlyGaussian, ShDegree4, ToAnyGaussian};

pub fn ply_gaussian_pod(a: &PlyGaussian, b: &PlyGaussian) {
    const EPSILON: f32 = 1e-4;

    assert_eq!(a.sh.len(), b.sh.len());
    assert!(a.color.abs_diff_eq(b.color, EPSILON));
    assert!((a.alpha - b.alpha).abs() < EPSILON);

    assert!(
        a.rot
            .to_array()
            .into_iter()
            .zip(b.rot.to_array())
            .all(|(x, y)| (x - y).abs() < EPSILON),
        "rotation assertion failed\n left: {:?}\nright: {:?}",
        a.rot,
        b.rot
    );

    assert!(
        a.pos
            .to_array()
            .into_iter()
            .zip(b.pos.to_array())
            .all(|(x, y)| (x - y).abs() < EPSILON),
        "position assertion failed\n left: {:?}\nright: {:?}",
        a.pos,
        b.pos
    );

    assert!(
        a.normal
            .to_array()
            .into_iter()
            .zip(b.normal.to_array())
            .all(|(x, y)| (x - y).abs() < EPSILON),
        "normal assertion failed\n left: {:?}\nright: {:?}",
        a.normal,
        b.normal
    );

    assert!(
        a.sh.iter()
            .zip(b.sh.iter())
            .all(|(x, y)| (x - y).abs() < EPSILON),
        "sh assertion failed\n left: {:?}\nright: {:?}",
        a.sh,
        b.sh
    );

    assert!(
        a.scale
            .to_array()
            .into_iter()
            .zip(b.scale.to_array())
            .all(|(x, y)| (x - y).abs() < EPSILON),
        "scale assertion failed\n left: {:?}\nright: {:?}",
        a.scale,
        b.scale
    );
}

#[derive(Debug, Clone)]
pub struct GaussianOptions {
    pub pos_epsilon: f32,
    pub rot_epsilon: f32,
    pub color_tolerance: u8,
    pub sh_epsilon: f32,
    pub scale_epsilon: f32,
}

pub fn gaussian(
    a: &impl ToAnyGaussian,
    b: &impl ToAnyGaussian,
    &GaussianOptions {
        pos_epsilon,
        rot_epsilon,
        color_tolerance,
        sh_epsilon,
        scale_epsilon,
    }: &GaussianOptions,
) {
    let a = a.to_any_gaussian();
    let b = b.to_any_gaussian();

    assert_eq!(a.sh_degree(), b.sh_degree());

    let a = a.convert_sh_degree::<ShDegree4>();
    let b = b.convert_sh_degree::<ShDegree4>();

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
        a.color.abs_diff_eq(b.color, color_tolerance as f32 / 255.0),
        "color assertion failed\n left: {:?}\nright: {:?}",
        a.color,
        b.color
    );

    for i in 0..24 {
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

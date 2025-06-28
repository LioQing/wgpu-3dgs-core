struct GaussianTransform {
    size: f32,
    flags: u32,
}

const gaussian_display_mode_splat = 0u;
const gaussian_display_mode_ellipse = 1u;
const gaussian_display_mode_point = 2u;

fn gaussian_transform_display_mode() -> u32 {
    return unpack4xU8(@var(gaussian_transform).flags).x;
}

fn gaussian_transform_sh_deg() -> u32 {
    return unpack4xU8(@var(gaussian_transform).flags).y;
}

fn gaussian_transform_no_sh0() -> bool {
    return unpack4xU8(@var(gaussian_transform).flags).z != 0u;
}
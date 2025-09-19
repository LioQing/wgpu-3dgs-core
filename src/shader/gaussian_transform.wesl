// Gaussian transform buffer structure.
//
// Corresponds to `GaussianTransformPod`.
struct GaussianTransform {
    size: f32,
    flags: u32,
}

const gaussian_display_mode_splat = 0u;
const gaussian_display_mode_ellipse = 1u;
const gaussian_display_mode_point = 2u;

// Get the display mode from `GaussianTransform.flags`.
fn gaussian_transform_display_mode(flags: u32) -> u32 {
    return unpack4xU8(flags).x;
}

// Get the SH degree from `GaussianTransform.flags`.
fn gaussian_transform_sh_deg(flags: u32) -> u32 {
    return unpack4xU8(flags).y;
}

// Check if the SH0 is disabled from `GaussianTransform.flags`.
fn gaussian_transform_no_sh0(flags: u32) -> bool {
    return unpack4xU8(flags).z != 0u;
}

// Get the standard deviation from `GaussianTransform.flags`.
fn gaussian_transform_std_dev(flags: u32) -> f32 {
    return f32(unpack4xU8(flags).w) / 255.0 * 3.0;
}
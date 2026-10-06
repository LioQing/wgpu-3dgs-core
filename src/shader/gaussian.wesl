// Compile-time storage degree. Exactly one degree feature must be enabled.
@if(sh_degree_0) const gaussian_sh_degree = 0u;
@if(sh_degree_1) const gaussian_sh_degree = 1u;
@if(sh_degree_2) const gaussian_sh_degree = 2u;
@if(sh_degree_3) const gaussian_sh_degree = 3u;
@if(sh_degree_4) const gaussian_sh_degree = 4u;

const gaussian_sh_count = (gaussian_sh_degree + 1u) * (gaussian_sh_degree + 1u) - 1u;

// Gaussian buffer structure.
//
// Corresponds to `GaussianPod`.
struct Gaussian {
    pos: vec3<f32>,
    color: u32,

    @if(sh_single && !sh_degree_0)
        sh: array<f32, (3u * gaussian_sh_count)>,
    @if(sh_half && !sh_degree_0)
        sh: array<u32, ((3u * gaussian_sh_count + 1u) / 2u)>,
    @if(sh_norm8 && !sh_degree_0)
        sh: array<u32, ((3u * gaussian_sh_count + 3u) / 4u)>,

    @if(cov3d_rot_scale)
        cov3d: array<f32, 7>,
    @if(cov3d_single)
        cov3d: array<f32, 6>,
    @if(cov3d_half)
        cov3d: array<u32, 3>,
}

// Unpack the color from `Gaussian.color`.
fn gaussian_unpack_color(gaussian: Gaussian) -> vec4<f32> {
    return unpack4x8unorm(gaussian.color);
}

// Unpack the SH coefficients from `Gaussian.sh`.
@if(sh_single && !sh_degree_0) fn gaussian_unpack_sh(gaussian: Gaussian, sh_index: u32) -> vec3<f32> {
    if sh_index >= gaussian_sh_count {
        return vec3<f32>(0.0);
    }

    return vec3<f32>(
        gaussian.sh[sh_index * 3],
        gaussian.sh[sh_index * 3 + 1],
        gaussian.sh[sh_index * 3 + 2],
    );
}

// Unpack the SH coefficients from `Gaussian.sh`.
@if(sh_half && !sh_degree_0) fn gaussian_unpack_sh(gaussian: Gaussian, sh_index: u32) -> vec3<f32> {
    if sh_index >= gaussian_sh_count {
        return vec3<f32>(0.0);
    }

    let i = sh_index * 3;
    let xi = i / 2;
    let yi = (i + 1) / 2;
    let zi = (i + 2) / 2;
    
    if xi == yi {
        return vec3<f32>(
            unpack2x16float(gaussian.sh[xi]),
            unpack2x16float(gaussian.sh[zi]).x,
        );
    } else {
        return vec3<f32>(
            unpack2x16float(gaussian.sh[xi]).y,
            unpack2x16float(gaussian.sh[yi]),
        );
    }
}

// Unpack the SH coefficients from `Gaussian.sh`.
@if(sh_norm8 && !sh_degree_0) fn gaussian_unpack_sh(gaussian: Gaussian, sh_index: u32) -> vec3<f32> {
    if sh_index >= gaussian_sh_count {
        return vec3<f32>(0.0);
    }

    let i = sh_index * 3;
    let xi = i / 4;
    let xj = i % 4;
    let yi = (i + 1) / 4;
    let yj = (i + 1) % 4;
    let zi = (i + 2) / 4;
    let zj = (i + 2) % 4;
    
    return vec3<f32>(
        unpack4x8snorm(gaussian.sh[xi])[xj],
        unpack4x8snorm(gaussian.sh[yi])[yj],
        unpack4x8snorm(gaussian.sh[zi])[zj],
    );
}

// Unpack the SH coefficients from `Gaussian.sh`.
@if(sh_degree_0) fn gaussian_unpack_sh(gaussian: Gaussian, sh_index: u32) -> vec3<f32> {
    return vec3<f32>(0.0);
}

// Unpack the 3D covariance matrix from `Gaussian.cov3d`.
@if(cov3d_rot_scale) fn gaussian_unpack_cov3d(gaussian: Gaussian) -> array<f32, 6> {
    let rot = vec4<f32>(gaussian.cov3d[0], gaussian.cov3d[1], gaussian.cov3d[2], gaussian.cov3d[3]);
    let scale = vec3<f32>(gaussian.cov3d[4], gaussian.cov3d[5], gaussian.cov3d[6]);

    let x2 = rot.x + rot.x;
    let y2 = rot.y + rot.y;
    let z2 = rot.z + rot.z;
    let xx = rot.x * x2;
    let xy = rot.x * y2;
    let xz = rot.x * z2;
    let yy = rot.y * y2;
    let yz = rot.y * z2;
    let zz = rot.z * z2;
    let wx = rot.w * x2;
    let wy = rot.w * y2;
    let wz = rot.w * z2;

    let sx = scale.x;
    let sy = scale.y;
    let sz = scale.z;

    let m = mat3x3<f32>(
        vec3<f32>(
            (1.0 - (yy + zz)) * sx,
            (xy + wz) * sx,
            (xz - wy) * sx,
        ),
        vec3<f32>(
            (xy - wz) * sy,
            (1.0 - (xx + zz)) * sy,
            (yz + wx) * sy,
        ),
        vec3<f32>(
            (xz + wy) * sz,
            (yz - wx) * sz,
            (1.0 - (xx + yy)) * sz,
        ),
    );

    let sigma = m * transpose(m);

    return array<f32, 6>(
        sigma[0][0],
        sigma[0][1],
        sigma[0][2],
        sigma[1][1],
        sigma[1][2],
        sigma[2][2],
    );
}

// Unpack the 3D covariance matrix from `Gaussian.cov3d`.
@if(cov3d_single) fn gaussian_unpack_cov3d(gaussian: Gaussian) -> array<f32, 6> {
    return gaussian.cov3d;
}

// Unpack the 3D covariance matrix from `Gaussian.cov3d`.
@if(cov3d_half) fn gaussian_unpack_cov3d(gaussian: Gaussian) -> array<f32, 6> {
    let x = unpack2x16float(gaussian.cov3d[0]);
    let y = unpack2x16float(gaussian.cov3d[1]);
    let z = unpack2x16float(gaussian.cov3d[2]);
    return array<f32, 6>(
        x.x,
        x.y,
        y.x,
        y.y,
        z.x,
        z.y,
    );
}

// Clamp the requested display degree to the available storage degree.
fn gaussian_effective_sh_degree(requested_degree: u32) -> u32 {
    return min(requested_degree, gaussian_sh_degree);
}

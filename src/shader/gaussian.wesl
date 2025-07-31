struct Gaussian {
    pos: vec3<f32>,
    color: u32,

    @if(sh_single)
        sh: array<f32, (3 * 15)>,
    @if(sh_half)
        sh: array<u32, ((3 * 15 + 1) / 2)>,
    @if(sh_norm8)
        sh: array<u32, (1 + (3 * 15 + 3) / 4)>,

    @if(cov3d_rot_scale)
        cov3d: array<f32, 7>,
    @if(cov3d_single)
        cov3d: array<f32, 6>,
    @if(cov3d_half)
        cov3d: array<u32, 3>,
}

fn gaussian_unpack_color(gaussian: Gaussian) -> vec4<f32> {
    return unpack4x8unorm(gaussian.color);
}

@if(sh_single) fn gaussian_unpack_sh(gaussian: Gaussian, sh_index: u32) -> vec3<f32> {
    return vec3<f32>(
        gaussian.sh[(sh_index - 1) * 3],
        gaussian.sh[(sh_index - 1) * 3 + 1],
        gaussian.sh[(sh_index - 1) * 3 + 2],
    );
}

@if(sh_half) fn gaussian_unpack_sh(gaussian: Gaussian, sh_index: u32) -> vec3<f32> {
    let i = (sh_index - 1) * 3;
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

@if(sh_norm8) fn gaussian_unpack_sh(gaussian: Gaussian, sh_index: u32) -> vec3<f32> {
    let minmax = unpack2x16float(gaussian.sh[0]);

    let i = (sh_index - 1) * 3;
    let xi = i / 4;
    let xj = i % 4;
    let yi = (i + 1) / 4;
    let yj = (i + 1) % 4;
    let zi = (i + 2) / 4;
    let zj = (i + 2) % 4;
    
    let norm = vec3<f32>(
        unpack4x8unorm(gaussian.sh[1 + xi])[xj],
        unpack4x8unorm(gaussian.sh[1 + yi])[yj],
        unpack4x8unorm(gaussian.sh[1 + zi])[zj],
    );

    return minmax.x + norm * (minmax.y - minmax.x);
}

@if(sh_none) fn gaussian_unpack_sh(gaussian: Gaussian, sh_index: u32) -> vec3<f32> {
    return vec3<f32>(0.0);
}

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

@if(cov3d_single) fn gaussian_unpack_cov3d(gaussian: Gaussian) -> array<f32, 6> {
    return gaussian.cov3d;
}

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
import package::gaussian::Gaussian;

// Model transform buffer structure.
//
// Corresponds to `ModelTransformPod`.
struct ModelTransform {
    pos: vec3<f32>,
    rot: vec4<f32>,
    scale: vec3<f32>,
}

// Transform a position from model space to world space.
fn model_to_world(model_transform: ModelTransform, pos: vec3<f32>) -> vec4<f32> {
    return model_transform_mat(model_transform) * vec4<f32>(pos, 1.0);
}

// Get the transformation matrix from `ModelTransform`.
fn model_transform_mat(model_transform: ModelTransform) -> mat4x4<f32> {
    let pos = model_transform.pos;
    let rot = model_transform.rot;
    let scale = model_transform.scale;

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

    return mat4x4<f32>(
        vec4<f32>(
            (1.0 - (yy + zz)) * sx,
            (xy + wz) * sx,
            (xz - wy) * sx,
            0.0,
        ),
        vec4<f32>(
            (xy - wz) * sy,
            (1.0 - (xx + zz)) * sy,
            (yz + wx) * sy,
            0.0,
        ),
        vec4<f32>(
            (xz + wy) * sz,
            (yz - wx) * sz,
            (1.0 - (xx + yy)) * sz,
            0.0,
        ),
        vec4<f32>(pos, 1.0),
    );
}

// Get the inverse scale-rotation matrix from `ModelTransform`.
fn model_transform_inv_sr_mat(model_transform: ModelTransform) -> mat3x3<f32> {
    let rot = model_transform.rot;
    let scale = model_transform.scale;

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

    return mat3x3<f32>(
        vec3<f32>(
            (1.0 - (yy + zz)) / sx,
            (xy - wz) / sy,
            (xz + wy) / sz,
        ),
        vec3<f32>(
            (xy + wz) / sx,
            (1.0 - (xx + zz)) / sy,
            (yz - wx) / sz,
        ),
        vec3<f32>(
            (xz - wy) / sx,
            (yz + wx) / sy,
            (1.0 - (xx + yy)) / sz,
        ),
    );
}

// Get the scale-rotation matrix from `ModelTransform`.
fn model_scale_rot_mat(model_transform: ModelTransform) -> mat3x3<f32> {
    let rot = model_transform.rot;
    let scale = model_transform.scale;

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

    return mat3x3<f32>(
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
}
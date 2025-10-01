use wgpu::util::DeviceExt;
use wgpu_3dgs_core::{
    BufferWrapper, ComputeBundleBuilder, DownloadableBufferWrapper, ModelTransformBuffer,
    ModelTransformPod, glam::*,
};

use crate::{common::TestContext, inline_wesl_pkg};

const TEST_PACKAGE: wesl::Pkg = inline_wesl_pkg!(
    use [&wgpu_3dgs_core::shader::PACKAGE],

    "test_model_transform":
    import wgpu_3dgs_core::model_transform::{
        ModelTransform,
        model_to_world,
        model_transform_mat,
        model_transform_inv_sr_mat,
        model_scale_rot_mat,
    };

    struct Output {
        transformed_pos: vec4<f32>,
        transform_mat: mat4x4<f32>,
        inv_sr_mat: mat3x3<f32>,
        scale_rot_mat: mat3x3<f32>,
    }

    @group(0) @binding(0)
    var<uniform> model_transform: ModelTransform;

    @group(0) @binding(1)
    var<uniform> test_pos: vec3<f32>;

    @group(0) @binding(2)
    var<storage, read_write> output: Output;

    override workgroup_size: u32;

    @compute @workgroup_size(workgroup_size)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x;

        if index >= 1 {
            return;
        }

        output.transformed_pos = model_to_world(model_transform, test_pos);
        output.transform_mat = model_transform_mat(model_transform);
        output.inv_sr_mat = model_transform_inv_sr_mat(model_transform);
        output.scale_rot_mat = model_scale_rot_mat(model_transform);
    }
);

const TEST_PACKAGE_BIND_GROUP_LAYOUTS: wgpu::BindGroupLayoutDescriptor<'static> =
    wgpu::BindGroupLayoutDescriptor {
        label: Some("Test Package Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    };

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct Output {
    transformed_pos: Vec4,
    transform_mat: Mat4,
    inv_sr_mat: Mat3A,
    scale_rot_mat: Mat3A,
}

#[test]
fn test_model_transform_wesl_functions_should_return_correct_values() {
    let ctx = TestContext::new();

    let pos = Vec3::new(5.0, 10.0, 15.0);
    let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4)
        * Quat::from_rotation_x(std::f32::consts::FRAC_PI_6);
    let scale = Vec3::new(2.0, 3.0, 4.0);
    let transform = ModelTransformPod::new(pos, rot, scale);

    let transform_buffer = ModelTransformBuffer::new(&ctx.device);
    transform_buffer.update_with_pod(&ctx.queue, &transform);

    let test_pos = Vec3::new(1.0, 2.0, 3.0);
    let test_pos_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Position Buffer"),
            contents: bytemuck::bytes_of(&test_pos),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: std::mem::size_of::<Output>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bundle = ComputeBundleBuilder::new()
        .bind_group_layout(&TEST_PACKAGE_BIND_GROUP_LAYOUTS)
        .resolver({
            let mut resolver = wesl::PkgResolver::new();
            resolver.add_package(&TEST_PACKAGE);
            resolver.add_package(&wgpu_3dgs_core::shader::PACKAGE);
            resolver
        })
        .main_shader("test_model_transform".parse().expect("parse"))
        .entry_point("main")
        .build(
            &ctx.device,
            [[
                transform_buffer.buffer().as_entire_binding(),
                test_pos_buffer.as_entire_binding(),
                output_buffer.as_entire_binding(),
            ]],
        )
        .expect("build_without_bind_groups");

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    bundle.dispatch(&mut encoder, 1);

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded = pollster::block_on(output_buffer.download::<Output>(&ctx.device, &ctx.queue))
        .expect("download")[0];

    let expected_transform_mat = Mat4::from_scale_rotation_translation(scale, rot, pos);
    assert!(
        downloaded
            .transform_mat
            .abs_diff_eq(expected_transform_mat, 1e-6),
        " left: {:?}\nright: {:?}",
        downloaded.transform_mat,
        expected_transform_mat,
    );

    let expected_transformed_pos = expected_transform_mat * test_pos.extend(1.0);
    assert!(
        downloaded
            .transformed_pos
            .abs_diff_eq(expected_transformed_pos, 1e-6),
        " left: {:?}\nright: {:?}",
        downloaded.transformed_pos,
        expected_transformed_pos,
    );

    let expected_scale_rot_mat = Mat3A::from_mat4(expected_transform_mat);
    assert!(
        downloaded
            .scale_rot_mat
            .abs_diff_eq(expected_scale_rot_mat, 1e-6),
        " left: {:?}\nright: {:?}",
        downloaded.scale_rot_mat,
        expected_scale_rot_mat,
    );

    let expected_inv_sr_mat = expected_scale_rot_mat.inverse();
    assert!(
        downloaded.inv_sr_mat.abs_diff_eq(expected_inv_sr_mat, 1e-6),
        " left: {:?}\nright: {:?}",
        downloaded.inv_sr_mat,
        expected_inv_sr_mat,
    );
}

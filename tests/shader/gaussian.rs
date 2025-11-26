use pollster::FutureExt;
use wgpu_3dgs_core::{
    BufferWrapper, ComputeBundleBuilder, GaussianCov3dConfig, GaussianPod,
    GaussianPodWithShHalfCov3dSingleConfigs, GaussianPodWithShNorm8Cov3dSingleConfigs,
    GaussianPodWithShSingleCov3dHalfConfigs, GaussianPodWithShSingleCov3dRotScaleConfigs,
    GaussianPodWithShSingleCov3dSingleConfigs, GaussiansBuffer, glam::*,
};

use crate::{
    common::{TestContext, given},
    inline_wesl_pkg,
};

const TEST_PACKAGE: wesl::Pkg = inline_wesl_pkg!(
    use [&wgpu_3dgs_core::shader::PACKAGE],

    "test_gaussian":
    import wgpu_3dgs_core::gaussian::{
        Gaussian,
        gaussian_unpack_color,
        gaussian_unpack_sh,
        gaussian_unpack_cov3d,
    };

    struct Output {
        color: vec4<f32>,
        sh: array<f32, 45>,
        cov3d: array<f32, 6>,
    }

    @group(0) @binding(0)
    var<storage> gaussians: array<Gaussian>;

    @group(0) @binding(1)
    var<storage, read_write> output: Output;

    override workgroup_size: u32;

    @compute @workgroup_size(workgroup_size)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x;

        if index >= 1 {
            return;
        }

        let gaussian = gaussians[index];

        output.color = gaussian_unpack_color(gaussian);

        for (var i: u32 = 0u; i < 15u; i = i + 1u) {
            let sh = gaussian_unpack_sh(gaussian, i);
            output.sh[i * 3u + 0u] = sh.x;
            output.sh[i * 3u + 1u] = sh.y;
            output.sh[i * 3u + 2u] = sh.z;
        }

        output.cov3d = gaussian_unpack_cov3d(gaussian);
    }
);

const TEST_PACKAGE_BIND_GROUP_LAYOUT: wgpu::BindGroupLayoutDescriptor<'static> =
    wgpu::BindGroupLayoutDescriptor {
        label: Some("Test Package Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
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
    color: [f32; 4],
    sh: [f32; 45],
    cov3d: [f32; 6],
    padding: u32,
}

impl Output {
    fn color(&self) -> Vec4 {
        Vec4::from(self.color)
    }

    fn sh(&self) -> &[Vec3] {
        bytemuck::cast_slice::<f32, Vec3>(&self.sh)
    }

    fn cov3d(&self) -> &[f32; 6] {
        &self.cov3d
    }
}

fn dispatch_test<G: GaussianPod>(ctx: &TestContext, buffer: &GaussiansBuffer<G>) -> Output {
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: std::mem::size_of::<Output>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bundle = ComputeBundleBuilder::new()
        .bind_group_layout(&TEST_PACKAGE_BIND_GROUP_LAYOUT)
        .resolver({
            let mut resolver = wesl::PkgResolver::new();
            resolver.add_package(&TEST_PACKAGE);
            resolver.add_package(&wgpu_3dgs_core::shader::PACKAGE);
            resolver
        })
        .wesl_compile_options(wesl::CompileOptions {
            features: G::wesl_features(),
            ..Default::default()
        })
        .main_shader("test_gaussian".parse().expect("parse"))
        .entry_point("main")
        .build(
            &ctx.device,
            [[
                buffer.buffer().as_entire_binding(),
                output_buffer.as_entire_binding(),
            ]],
        )
        .map_err(|e| println!("{e}"))
        .expect("build");

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    bundle.dispatch(&mut encoder, 1);

    ctx.queue.submit(Some(encoder.finish()));

    output_buffer
        .download::<Output>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download")[0]
}

#[test]
fn test_gaussian_unpack_color_should_return_correct_value() {
    let ctx = TestContext::new();

    type G = GaussianPodWithShSingleCov3dSingleConfigs;

    let gaussian = given::gaussian();
    let gaussians = vec![gaussian];
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let output = dispatch_test(&ctx, &buffer);

    let expected_color = gaussian.color.as_vec4().map(|v| v / 255.0);

    assert!(
        output.color().abs_diff_eq(expected_color, 1e-4),
        " left: {:?}\nright: {:?}",
        output.color(),
        expected_color,
    );
}

#[test]
fn test_gaussian_unpack_sh_when_config_is_single_should_return_correct_value() {
    let ctx = TestContext::new();

    type G = GaussianPodWithShSingleCov3dSingleConfigs;

    let gaussian = given::gaussian();
    let gaussians = vec![gaussian];
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let output = dispatch_test(&ctx, &buffer);

    let expected_sh = gaussian.sh;

    assert!(
        expected_sh
            .iter()
            .zip(output.sh().iter())
            .all(|(a, b)| a.abs_diff_eq(*b, 1e-2)),
        " left: {:?}\nright: {:?}",
        output.sh(),
        expected_sh,
    );
}

#[test]
fn test_gaussian_unpack_sh_when_config_is_half_should_return_correct_value() {
    let ctx = TestContext::new();

    type G = GaussianPodWithShHalfCov3dSingleConfigs;

    let gaussian = given::gaussian();
    let gaussians = vec![gaussian];
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let output = dispatch_test(&ctx, &buffer);

    let expected_sh = gaussian.sh;

    assert!(
        expected_sh
            .iter()
            .zip(output.sh().iter())
            .all(|(a, b)| a.abs_diff_eq(*b, 1e-1)),
        " left: {:?}\nright: {:?}",
        output.sh(),
        expected_sh,
    );
}

#[test]
fn test_gaussian_unpack_sh_when_config_is_norm_8_should_return_correct_value() {
    let ctx = TestContext::new();

    type G = GaussianPodWithShNorm8Cov3dSingleConfigs;

    let gaussian = given::gaussian();
    let gaussians = vec![gaussian];
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let output = dispatch_test(&ctx, &buffer);

    let expected_sh = gaussian.sh;

    assert!(
        expected_sh
            .iter()
            .zip(output.sh().iter())
            .all(|(a, b)| a.abs_diff_eq(*b, 1e-1)),
        " left: {:?}\nright: {:?}",
        output.sh(),
        expected_sh,
    );
}

#[test]
fn test_gaussian_unpack_cov3d_when_config_is_rot_scale_should_return_correct_value() {
    let ctx = TestContext::new();

    type G = GaussianPodWithShSingleCov3dRotScaleConfigs;

    let gaussian = given::gaussian();
    let gaussians = vec![gaussian];
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let output = dispatch_test(&ctx, &buffer);

    let expected_cov3d = <GaussianPodWithShSingleCov3dSingleConfigs as wgpu_3dgs_core::GaussianPod>::Cov3dConfig::from_rot_scale(
        gaussian.rot,
        gaussian.scale,
    );

    assert!(
        expected_cov3d
            .iter()
            .zip(output.cov3d().iter())
            .all(|(a, b)| (a - b).abs() < 1e-2),
        " left: {:?}\nright: {:?}",
        output.cov3d(),
        expected_cov3d,
    );
}

#[test]
fn test_gaussian_unpack_cov3d_when_config_is_single_should_return_correct_value() {
    let ctx = TestContext::new();

    type G = GaussianPodWithShSingleCov3dSingleConfigs;

    let gaussian = given::gaussian();
    let gaussians = vec![gaussian];
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let output = dispatch_test(&ctx, &buffer);

    let expected_cov3d = <GaussianPodWithShSingleCov3dSingleConfigs as wgpu_3dgs_core::GaussianPod>::Cov3dConfig::from_rot_scale(
        gaussian.rot,
        gaussian.scale,
    );

    assert!(
        expected_cov3d
            .iter()
            .zip(output.cov3d().iter())
            .all(|(a, b)| (a - b).abs() < 1e-2),
        " left: {:?}\nright: {:?}",
        output.cov3d(),
        expected_cov3d,
    );
}

#[test]
fn test_gaussian_unpack_cov3d_when_config_is_half_should_return_correct_value() {
    let ctx = TestContext::new();

    type G = GaussianPodWithShSingleCov3dHalfConfigs;

    let gaussian = given::gaussian();
    let gaussians = vec![gaussian];
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let output = dispatch_test(&ctx, &buffer);

    let expected_cov3d = <GaussianPodWithShSingleCov3dSingleConfigs as wgpu_3dgs_core::GaussianPod>::Cov3dConfig::from_rot_scale(
        gaussian.rot,
        gaussian.scale,
    );

    assert!(
        expected_cov3d
            .iter()
            .zip(output.cov3d().iter())
            .all(|(a, b)| (a - b).abs() < 1.0),
        " left: {:?}\nright: {:?}",
        output.cov3d(),
        expected_cov3d,
    );
}

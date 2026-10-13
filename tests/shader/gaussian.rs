use pollster::FutureExt;
use wgpu_3dgs_core::{
    BufferWrapper, ComputeBundleBuilder, CovHalf, CovRotScale, CovSingle, GaussianCov3dConfig,
    GaussianPod, GaussianShConfig, GaussiansBuffer, PackedGaussian, ShDegree, ShHalf, ShNorm8,
    ShSingle, glam::*,
};

use crate::{
    common::{TestContext, given},
    inline_wesl_pkg,
};

const TEST_PACKAGE: wesl::CodegenPkg = inline_wesl_pkg!(
    use [&wgpu_3dgs_core::shader::PACKAGE],

    "test_gaussian":
    import wgpu_3dgs_core::gaussian::{
        Gaussian,
        gaussian_unpack_color,
        gaussian_unpack_sh,
        gaussian_unpack_cov3d,
        gaussian_sh_degree,
        gaussian_effective_sh_degree,
    };

    struct Output {
        color: vec4<f32>,
        sh: array<f32, 72>,
        cov3d: array<f32, 6>,
        degree: u32,
        effective_degree: u32,
        pos: vec3<f32>,
    }

    @group(0) @binding(0)
    var<storage> gaussians: array<Gaussian>;

    @group(0) @binding(1)
    var<storage, read_write> output: array<Output>;

    override workgroup_size: u32;

    @compute @workgroup_size(workgroup_size)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x;

        if index >= arrayLength(&gaussians) {
            return;
        }

        let gaussian = gaussians[index];

        output[index].color = gaussian_unpack_color(gaussian);

        for (var i: u32 = 0u; i < 24u; i = i + 1u) {
            let sh = gaussian_unpack_sh(gaussian, i);
            output[index].sh[i * 3u + 0u] = sh.x;
            output[index].sh[i * 3u + 1u] = sh.y;
            output[index].sh[i * 3u + 2u] = sh.z;
        }

        output[index].cov3d = gaussian_unpack_cov3d(gaussian);
        output[index].degree = gaussian_sh_degree;
        output[index].effective_degree = gaussian_effective_sh_degree(99u);
        output[index].pos = gaussian.pos;
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
    sh: [f32; 72],
    cov3d: [f32; 6],
    degree: u32,
    effective_degree: u32,
    pos: [f32; 3],
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
    dispatch_outputs(ctx, buffer)[0]
}

fn dispatch_outputs<G: GaussianPod>(ctx: &TestContext, buffer: &GaussiansBuffer<G>) -> Vec<Output> {
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (std::mem::size_of::<Output>() * buffer.len()) as wgpu::BufferAddress,
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

    bundle.dispatch(&mut encoder, buffer.len() as u32);

    ctx.queue.submit(Some(encoder.finish()));

    output_buffer
        .download::<Output>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download")
}

#[test]
fn test_gaussian_unpack_should_use_correct_stride_for_multiple_gaussians_of_all_degrees() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3)
            .map(given::gaussian_for_sh_degree::<G::ShDegree>)
            .collect::<Vec<_>>();
        let buffer = GaussiansBuffer::<G>::new(&ctx.device, &gaussians);
        let output = dispatch_outputs(&ctx, &buffer);

        for (gaussian, output) in gaussians.iter().zip(output) {
            let field = G::ShConfig::from_sh(&gaussian.sh);
            let expected = G::ShConfig::to_sh(&field);

            assert_eq!(output.pos, gaussian.pos.to_array());
            assert_eq!(output.degree, G::ShDegree::DEGREE as u32);
            assert_eq!(output.effective_degree, output.degree);
            assert!(output.color().abs_diff_eq(gaussian.color, 1e-6));

            for (actual, expected) in output.sh().iter().zip(expected.as_ref()) {
                assert!(actual.abs_diff_eq(*expected, 1e-6));
            }

            assert!(
                output.sh()[G::ShDegree::COEFFICIENT_COUNT..]
                    .iter()
                    .all(|v| *v == Vec3::ZERO)
            );

            let expected_cov = CovSingle::from_rot_scale(gaussian.rot, gaussian.scale);

            assert!(
                output
                    .cov3d
                    .iter()
                    .zip(expected_cov)
                    .all(|(a, b)| (a - b).abs() < 0.01)
            );
        }
    }

    crate::for_each_sh_degree_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussian_unpack_color_should_return_correct_value() {
    let ctx = TestContext::new();

    type G = PackedGaussian<ShSingle, CovSingle>;

    let gaussian = given::gaussian();
    let gaussians = vec![gaussian];
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let output = dispatch_test(&ctx, &buffer);

    let expected_color = gaussian.color;

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

    type G = PackedGaussian<ShSingle, CovSingle>;

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

    type G = PackedGaussian<ShHalf, CovSingle>;

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

    type G = PackedGaussian<ShNorm8, CovSingle>;

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

    type G = PackedGaussian<ShSingle, CovRotScale>;

    let gaussian = given::gaussian();
    let gaussians = vec![gaussian];
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let output = dispatch_test(&ctx, &buffer);

    let expected_cov3d = <PackedGaussian<ShSingle, CovSingle> as wgpu_3dgs_core::GaussianPod>::Cov3dConfig::from_rot_scale(
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

    type G = PackedGaussian<ShSingle, CovSingle>;

    let gaussian = given::gaussian();
    let gaussians = vec![gaussian];
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let output = dispatch_test(&ctx, &buffer);

    let expected_cov3d = <PackedGaussian<ShSingle, CovSingle> as wgpu_3dgs_core::GaussianPod>::Cov3dConfig::from_rot_scale(
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

    type G = PackedGaussian<ShSingle, CovHalf>;

    let gaussian = given::gaussian();
    let gaussians = vec![gaussian];
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let output = dispatch_test(&ctx, &buffer);

    let expected_cov3d = <PackedGaussian<ShSingle, CovSingle> as wgpu_3dgs_core::GaussianPod>::Cov3dConfig::from_rot_scale(
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

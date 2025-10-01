use wgpu_3dgs_core::{
    BufferWrapper, ComputeBundleBuilder, DownloadableBufferWrapper, GaussianDisplayMode,
    GaussianShDegree, GaussianTransformBuffer, GaussianTransformPod,
};

use crate::{common::TestContext, inline_wesl_pkg};

const TEST_PACKAGE: wesl::Pkg = inline_wesl_pkg!(
    use [&wgpu_3dgs_core::shader::PACKAGE],

    "test_gaussian_transform":
    import wgpu_3dgs_core::gaussian_transform::{
        GaussianTransform,
        gaussian_transform_display_mode,
        gaussian_transform_sh_deg,
        gaussian_transform_no_sh0,
        gaussian_transform_std_dev,
    };

    struct Output {
        display_mode: u32,
        sh_deg: u32,
        no_sh0: u32,
        std_dev: f32,
    }

    @group(0) @binding(0)
    var<uniform> transform: GaussianTransform;

    @group(0) @binding(1)
    var<storage, read_write> output: Output;

    override workgroup_size: u32;

    @compute @workgroup_size(workgroup_size)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x;

        if index >= 1 {
            return;
        }

        output.display_mode = gaussian_transform_display_mode(transform.flags);
        output.sh_deg = gaussian_transform_sh_deg(transform.flags);
        output.no_sh0 = select(0u, 1u, gaussian_transform_no_sh0(transform.flags));
        output.std_dev = gaussian_transform_std_dev(transform.flags);
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
    display_mode: u32,
    sh_deg: u32,
    no_sh0: u32,
    std_dev: f32,
}

#[test]
fn test_gaussian_transform_wesl_functions_should_return_correct_values() {
    let ctx = TestContext::new();

    let display_mode = GaussianDisplayMode::Ellipse;
    let sh_deg = GaussianShDegree::new(2).expect("new");
    let no_sh0 = true;
    let std_dev = 3.0;
    let transform = GaussianTransformPod::new(1.0, display_mode, sh_deg, no_sh0, std_dev);

    let transform_buffer = GaussianTransformBuffer::new(&ctx.device);
    transform_buffer.update_with_pod(&ctx.queue, &transform);

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
        .main_shader("test_gaussian_transform".parse().expect("parse"))
        .entry_point("main")
        .build(
            &ctx.device,
            [[
                transform_buffer.buffer().as_entire_binding(),
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

    assert_eq!(downloaded.display_mode, display_mode as u32);
    assert_eq!(downloaded.sh_deg, sh_deg.degree() as u32);
    assert_eq!(downloaded.no_sh0, no_sh0 as u32);
    assert!(
        (downloaded.std_dev - std_dev).abs() < 1e-6,
        " left: {}\nright: {}",
        downloaded.std_dev,
        std_dev,
    );
}

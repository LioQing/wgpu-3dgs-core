use assert_matches::assert_matches;
use wgpu::util::DeviceExt;
use wgpu_3dgs_core::{
    BufferWrapper, ComputeBundleBuildError, ComputeBundleBuilder, ComputeBundleCreateError,
};

use crate::common::{TestContext, shader};

#[test]
fn test_compute_bundle_when_with_bind_group_should_run_correctly() {
    let ctx = TestContext::new();

    let data = shader::given::array_map_add_data(&ctx.device);
    let mut bundle = ComputeBundleBuilder::new()
        .bind_group_layout(&shader::ARRAY_MAP_ADD_BIND_GROUP_LAYOUT_DESCRIPTOR)
        .resolver(wesl::StandardResolver::new(shader::SHADER_DIR))
        .main_shader(shader::ARRAY_MAP_ADD_MODULE_PATH.parse().expect("parse"))
        .entry_point(shader::SHADER_ENTRY_POINT)
        .build(&ctx.device, [[data.as_entire_binding()]])
        .expect("build");

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Bundle Command Encoder"),
        });

    bundle.dispatch(
        &mut encoder,
        shader::ARRAY_MAP_ADD_DEFAULT_DATA.len() as u32,
    );

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded =
        pollster::block_on(data.download::<u32>(&ctx.device, &ctx.queue)).expect("download");

    assert_eq!(
        &downloaded,
        &shader::ARRAY_MAP_ADD_DEFAULT_DATA.map(|v| v + 1)
    );

    const NEW_DATA: [u32; 3] = [100, 200, 300];
    let new_data = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("New Data Buffer"),
            contents: bytemuck::cast_slice(&NEW_DATA),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

    bundle.update_bind_group_with_binding_resources(&ctx.device, 0, [new_data.as_entire_binding()]);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Bundle Command Encoder"),
        });

    bundle.dispatch(&mut encoder, NEW_DATA.len() as u32);

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded =
        pollster::block_on(new_data.download::<u32>(&ctx.device, &ctx.queue)).expect("download");

    assert_eq!(&downloaded, &NEW_DATA.map(|v| v + 1));
}

#[test]
fn test_compute_bundle_when_all_options_and_without_bind_group_should_run_correctly() {
    let ctx = TestContext::new();

    let data = shader::given::array_map_add_data(&ctx.device);
    let additional_value = shader::given::array_map_add_additional_uniform(&ctx.device);
    let bundle = ComputeBundleBuilder::default()
        .label("Compute Bundle")
        .bind_group_layouts([
            &shader::ARRAY_MAP_ADD_BIND_GROUP_LAYOUT_DESCRIPTOR,
            &shader::ARRAY_MAP_ADD_SECOND_BIND_GROUP_LAYOUT_DESCRIPTOR,
        ])
        .pipeline_compile_options(wgpu::PipelineCompilationOptions {
            constants: shader::ARRAY_MAP_ADD_ADDITIONAL_CONSTANTS,
            ..Default::default()
        })
        .wesl_compile_options(wesl::CompileOptions {
            features: wesl::Features {
                flags: shader::ARRAY_MAP_ADD_WESL_FEATURE_FLAGS
                    .iter()
                    .map(|(name, flag)| (name.to_string(), *flag))
                    .collect(),
                ..Default::default()
            },
            ..Default::default()
        })
        .resolver(wesl::StandardResolver::new(shader::SHADER_DIR))
        .mangler(wesl::NoMangler)
        .main_shader(shader::ARRAY_MAP_ADD_MODULE_PATH.parse().expect("parse"))
        .entry_point(shader::SHADER_ENTRY_POINT)
        .build_without_bind_groups(&ctx.device)
        .expect("build");

    let bind_groups = (0..2)
        .map(|i| {
            bundle
                .create_bind_group(
                    &ctx.device,
                    i,
                    match i {
                        0 => [data.as_entire_binding()],
                        1 => [additional_value.as_entire_binding()],
                        _ => unreachable!(),
                    },
                )
                .expect("create_bind_group")
        })
        .collect::<Vec<_>>();

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Bundle Command Encoder"),
        });

    bundle.dispatch(
        &mut encoder,
        shader::ARRAY_MAP_ADD_DEFAULT_DATA.len() as u32,
        &bind_groups,
    );

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded =
        pollster::block_on(data.download::<u32>(&ctx.device, &ctx.queue)).expect("download");

    assert_eq!(
        &downloaded,
        &shader::ARRAY_MAP_ADD_DEFAULT_DATA.map(|v| {
            v + 1
                + shader::ARRAY_MAP_ADD_DEFAULT_ADDITIONAL_UNIFORM
                + shader::ARRAY_MAP_ADD_ADDITIONAL_CONSTANT
        })
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Bundle Command Encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            ..Default::default()
        });

        pass.set_pipeline(bundle.pipeline());

        for (i, group) in bind_groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }

        pass.dispatch_workgroups(
            shader::ARRAY_MAP_ADD_DEFAULT_DATA
                .len()
                .div_ceil(bundle.workgroup_size() as usize) as u32,
            1,
            1,
        );
    }

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded =
        pollster::block_on(data.download::<u32>(&ctx.device, &ctx.queue)).expect("download");

    assert_eq!(
        &downloaded,
        &shader::ARRAY_MAP_ADD_DEFAULT_DATA.map(|v| {
            v + 2
                + 2 * shader::ARRAY_MAP_ADD_DEFAULT_ADDITIONAL_UNIFORM
                + 2 * shader::ARRAY_MAP_ADD_ADDITIONAL_CONSTANT
        })
    );
}

#[test]
fn test_compute_bundle_builder_build_without_bind_groups_when_missing_bind_group_layout_should_return_error()
 {
    let ctx = TestContext::new();

    let result = ComputeBundleBuilder::new()
        .resolver(wesl::StandardResolver::new(shader::SHADER_DIR))
        .main_shader(shader::ARRAY_MAP_ADD_MODULE_PATH.parse().expect("parse"))
        .entry_point(shader::SHADER_ENTRY_POINT)
        .build_without_bind_groups(&ctx.device);

    assert_matches!(result, Err(ComputeBundleBuildError::MissingBindGroupLayout));
}

#[test]
fn test_compute_bundle_builder_build_without_bind_groups_when_missing_resolver_should_return_error()
{
    let ctx = TestContext::new();

    let result = ComputeBundleBuilder::new()
        .bind_group_layout(&shader::ARRAY_MAP_ADD_BIND_GROUP_LAYOUT_DESCRIPTOR)
        .main_shader(shader::ARRAY_MAP_ADD_MODULE_PATH.parse().expect("parse"))
        .entry_point(shader::SHADER_ENTRY_POINT)
        .build_without_bind_groups(&ctx.device);

    assert_matches!(result, Err(ComputeBundleBuildError::MissingResolver));
}

#[test]
fn test_compute_bundle_builder_build_without_bind_groups_when_missing_entry_point_should_return_error()
 {
    let ctx = TestContext::new();

    let result = ComputeBundleBuilder::new()
        .bind_group_layout(&shader::ARRAY_MAP_ADD_BIND_GROUP_LAYOUT_DESCRIPTOR)
        .resolver(wesl::StandardResolver::new(shader::SHADER_DIR))
        .main_shader(shader::ARRAY_MAP_ADD_MODULE_PATH.parse().expect("parse"))
        .build_without_bind_groups(&ctx.device);

    assert_matches!(result, Err(ComputeBundleBuildError::MissingEntryPoint));
}

#[test]
fn test_compute_bundle_builder_build_without_bind_groups_when_missing_main_shader_should_return_error()
 {
    let ctx = TestContext::new();

    let result = ComputeBundleBuilder::new()
        .bind_group_layout(&shader::ARRAY_MAP_ADD_BIND_GROUP_LAYOUT_DESCRIPTOR)
        .resolver(wesl::StandardResolver::new(shader::SHADER_DIR))
        .entry_point(shader::SHADER_ENTRY_POINT)
        .build_without_bind_groups(&ctx.device);

    assert_matches!(result, Err(ComputeBundleBuildError::MissingMainShader));
}

#[test]
fn test_compute_bundle_builder_build_when_missing_bind_group_layout_should_return_error() {
    let ctx = TestContext::new();
    let data = shader::given::array_map_add_data(&ctx.device);

    let result = ComputeBundleBuilder::new()
        .resolver(wesl::StandardResolver::new(shader::SHADER_DIR))
        .main_shader(shader::ARRAY_MAP_ADD_MODULE_PATH.parse().expect("parse"))
        .entry_point(shader::SHADER_ENTRY_POINT)
        .build(&ctx.device, [[data.as_entire_binding()]]);

    assert_matches!(result, Err(ComputeBundleBuildError::MissingBindGroupLayout));
}

#[test]
fn test_compute_bundle_builder_build_when_missing_resolver_should_return_error() {
    let ctx = TestContext::new();
    let data = shader::given::array_map_add_data(&ctx.device);

    let result = ComputeBundleBuilder::new()
        .bind_group_layout(&shader::ARRAY_MAP_ADD_BIND_GROUP_LAYOUT_DESCRIPTOR)
        .main_shader(shader::ARRAY_MAP_ADD_MODULE_PATH.parse().expect("parse"))
        .entry_point(shader::SHADER_ENTRY_POINT)
        .build(&ctx.device, [[data.as_entire_binding()]]);

    assert_matches!(result, Err(ComputeBundleBuildError::MissingResolver));
}

#[test]
fn test_compute_bundle_builder_build_when_missing_entry_point_should_return_error() {
    let ctx = TestContext::new();
    let data = shader::given::array_map_add_data(&ctx.device);

    let result = ComputeBundleBuilder::new()
        .bind_group_layout(&shader::ARRAY_MAP_ADD_BIND_GROUP_LAYOUT_DESCRIPTOR)
        .resolver(wesl::StandardResolver::new(shader::SHADER_DIR))
        .main_shader(shader::ARRAY_MAP_ADD_MODULE_PATH.parse().expect("parse"))
        .build(&ctx.device, [[data.as_entire_binding()]]);

    assert_matches!(result, Err(ComputeBundleBuildError::MissingEntryPoint));
}

#[test]
fn test_compute_bundle_builder_build_when_missing_main_shader_should_return_error() {
    let ctx = TestContext::new();
    let data = shader::given::array_map_add_data(&ctx.device);

    let result = ComputeBundleBuilder::new()
        .bind_group_layout(&shader::ARRAY_MAP_ADD_BIND_GROUP_LAYOUT_DESCRIPTOR)
        .resolver(wesl::StandardResolver::new(shader::SHADER_DIR))
        .entry_point(shader::SHADER_ENTRY_POINT)
        .build(&ctx.device, [[data.as_entire_binding()]]);

    assert_matches!(result, Err(ComputeBundleBuildError::MissingMainShader));
}

#[test]
fn test_compute_bundle_new_when_resource_count_mismatched_should_return_error() {
    let ctx = TestContext::new();
    let data = shader::given::array_map_add_data(&ctx.device);

    let result = ComputeBundleBuilder::new()
        .bind_group_layout(&shader::ARRAY_MAP_ADD_BIND_GROUP_LAYOUT_DESCRIPTOR)
        .resolver(wesl::StandardResolver::new(shader::SHADER_DIR))
        .main_shader(shader::ARRAY_MAP_ADD_MODULE_PATH.parse().expect("parse"))
        .entry_point(shader::SHADER_ENTRY_POINT)
        .build(
            &ctx.device,
            [[data.as_entire_binding()], [data.as_entire_binding()]],
        );

    assert_matches!(
        result,
        Err(ComputeBundleBuildError::Create(
            ComputeBundleCreateError::ResourceCountMismatch {
                resource_count: 2,
                bind_group_layout_count: 1,
            }
        ))
    );
}

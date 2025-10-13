use glam::*;
use wgpu::util::DeviceExt;
use wgpu_3dgs_core::{
    BufferWrapper, DownloadableBufferWrapper, ModelTransformBuffer, ModelTransformPod,
};

use crate::common::TestContext;

#[test]
fn test_model_transform_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = ModelTransformBuffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<wgpu_3dgs_core::ModelTransformPod>() as wgpu::BufferAddress
    );
}

#[test]
fn test_model_transform_buffer_update_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer =
        ModelTransformBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Model Transform Buffer"),
            size: std::mem::size_of::<ModelTransformPod>() as wgpu::BufferAddress,
            usage: ModelTransformBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
        .expect("try_from");

    let pos = Vec3::new(1.0, 2.0, 3.0);
    let rot = Quat::from_rotation_y(std::f32::consts::PI / 4.0);
    let scale = Vec3::new(2.0, 3.0, 4.0);
    let pod = ModelTransformPod::new(pos, rot, scale);

    buffer.update(&ctx.queue, pos, rot, scale);

    let downloaded =
        pollster::block_on(buffer.download::<ModelTransformPod>(&ctx.device, &ctx.queue))
            .expect("download")[0];

    assert_eq!(downloaded, pod);
}

#[test]
fn test_model_transform_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let pod = ModelTransformPod::new(
        Vec3::new(1.0, 2.0, 3.0),
        Quat::from_rotation_y(std::f32::consts::PI / 4.0),
        Vec3::new(2.0, 3.0, 4.0),
    );
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Model Transform Buffer"),
            contents: bytemuck::bytes_of(&pod),
            usage: ModelTransformBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer = ModelTransformBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded = pollster::block_on(
        wgpu_converted_buffer.download::<ModelTransformPod>(&ctx.device, &ctx.queue),
    )
    .expect("download");
    let converted_downloaded =
        pollster::block_on(converted_buffer.download::<ModelTransformPod>(&ctx.device, &ctx.queue))
            .expect("download");
    let wgpu_converted_downloaded =
        pollster::block_on(wgpu_buffer.download::<ModelTransformPod>(&ctx.device, &ctx.queue))
            .expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

#[test]
fn test_model_transform_pod_new_should_return_correct_pod() {
    let pos = Vec3::new(1.0, 2.0, 3.0);
    let rot = Quat::from_rotation_y(std::f32::consts::PI / 4.0);
    let scale = Vec3::new(2.0, 3.0, 4.0);
    let pod = ModelTransformPod::new(pos, rot, scale);

    assert_eq!(pod.pos, pos);
    assert_eq!(pod.rot, rot);
    assert_eq!(pod.scale, scale);
}

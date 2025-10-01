use assert_matches::assert_matches;
use wgpu_3dgs_core::{
    BufferWrapper, DownloadableBufferWrapper, GaussianDisplayMode, GaussianShDegree,
    GaussianTransformBuffer, GaussianTransformPod,
};

use crate::common::TestContext;

#[test]
fn test_gaussian_sh_degree_new_when_sh_deg_is_valid_should_return_some() {
    for sh_deg in [0, 1, 2, 3] {
        let degree = GaussianShDegree::new(sh_deg);
        assert_matches!(degree, Some(d) if d.degree() == sh_deg);
    }
}

#[test]
fn test_gaussian_sh_degree_new_when_sh_deg_is_invalid_should_return_none() {
    for sh_deg in [4, 5, 6, 7, 8, 9, 10, 255] {
        let degree = GaussianShDegree::new(sh_deg);
        assert!(degree.is_none());
    }
}

#[test]
fn test_gaussian_sh_degree_new_unchecked_should_always_succeed() {
    for sh_deg in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255] {
        let degree = GaussianShDegree::new_unchecked(sh_deg);
        assert_eq!(degree.degree(), sh_deg);
    }
}

#[test]
fn test_gaussian_sh_degree_degree_should_return_correct_value() {
    for sh_deg in [0, 1, 2, 3] {
        let degree = GaussianShDegree::new(sh_deg).unwrap();
        assert_eq!(degree.degree(), sh_deg);
    }
}

#[test]
fn test_gaussian_transform_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = GaussianTransformBuffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<wgpu_3dgs_core::GaussianTransformPod>() as wgpu::BufferAddress
    );
}

#[test]
fn test_gaussian_transform_buffer_update_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer =
        GaussianTransformBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Gaussian Transform Buffer"),
            size: std::mem::size_of::<GaussianTransformPod>() as wgpu::BufferAddress,
            usage: GaussianTransformBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
        .expect("try_from");

    let size = 1.0;
    let display_mode = GaussianDisplayMode::Ellipse;
    let sh_deg = GaussianShDegree::new(2).unwrap();
    let no_sh0 = true;
    let std_dev = 2.0;
    let pod = GaussianTransformPod::new(size, display_mode, sh_deg, no_sh0, std_dev);

    buffer.update(&ctx.queue, size, display_mode, sh_deg, no_sh0, std_dev);

    let downloaded =
        pollster::block_on(buffer.download::<GaussianTransformPod>(&ctx.device, &ctx.queue))
            .expect("download")[0];

    assert_eq!(downloaded, pod);
}

#[test]
fn test_gaussian_transform_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Gaussian Transform Buffer"),
        size: std::mem::size_of::<GaussianTransformPod>() as wgpu::BufferAddress,
        usage: GaussianTransformBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let wgpu_buffer = buffer.buffer().clone();

    let converted_buffer = GaussianTransformBuffer::try_from(buffer).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded = pollster::block_on(
        wgpu_converted_buffer.download::<GaussianTransformPod>(&ctx.device, &ctx.queue),
    )
    .expect("download");
    let converted_downloaded = pollster::block_on(
        converted_buffer.download::<GaussianTransformPod>(&ctx.device, &ctx.queue),
    )
    .expect("download");
    let wgpu_converted_downloaded =
        pollster::block_on(wgpu_buffer.download::<GaussianTransformPod>(&ctx.device, &ctx.queue))
            .expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

#[test]
fn test_gaussian_transform_pod_new_should_return_correct_pod() {
    let size = 1.0;
    let display_mode = GaussianDisplayMode::Ellipse;
    let sh_deg = GaussianShDegree::new(2).unwrap();
    let no_sh0 = true;
    let std_dev = 2.0;
    let pod =
        wgpu_3dgs_core::GaussianTransformPod::new(size, display_mode, sh_deg, no_sh0, std_dev);

    assert_eq!(pod.size, size);
    assert_eq!(pod.flags.x, display_mode as u8);
    assert_eq!(pod.flags.y, sh_deg.degree());
    assert_eq!(pod.flags.z, no_sh0 as u8);
    assert_eq!(pod.flags.w, (std_dev / 3.0 * 255.0) as u8);
}

use assert_matches::assert_matches;
use pollster::FutureExt;
use wgpu::util::DeviceExt;
use wgpu_3dgs_core::{
    BufferWrapper, FixedSizeBufferWrapper, GaussianDisplayMode, GaussianMaxStdDev,
    GaussianShDegree, GaussianTransformBuffer, GaussianTransformPod,
};

use crate::common::TestContext;

#[test]
fn test_gaussian_sh_degree_new_when_sh_deg_is_valid_should_return_some() {
    for sh_deg in [0, 1, 2, 3] {
        let degree = GaussianShDegree::new(sh_deg);
        assert_matches!(degree, Some(d) if d.get() == sh_deg);
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
        let degree = unsafe { GaussianShDegree::new_unchecked(sh_deg) };
        assert_eq!(degree.get(), sh_deg);
    }
}

#[test]
fn test_gaussian_sh_degree_degree_should_return_correct_value() {
    for sh_deg in [0, 1, 2, 3] {
        let degree = GaussianShDegree::new(sh_deg).unwrap();
        assert_eq!(degree.get(), sh_deg);
    }
}

#[test]
fn test_gaussian_max_std_dev_new_when_value_is_valid_should_return_some() {
    for max_std_dev in [0.0f32, 1.5, 3.0] {
        let expected_u8 = (max_std_dev / 3.0 * 255.0) as u8;
        assert_matches!(
            GaussianMaxStdDev::new(max_std_dev),
            Some(std_dev) if std_dev.as_u8() == expected_u8
        );
    }
}

#[test]
fn test_gaussian_max_std_dev_new_when_value_is_invalid_should_return_none() {
    let invalid_values: [f32; 7] = [
        -0.001,
        -1.0,
        3.0001,
        4.0,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
    ];

    for &max_std_dev in &invalid_values {
        assert_matches!(GaussianMaxStdDev::new(max_std_dev), None);
    }
}

#[test]
fn test_gaussian_max_std_dev_new_unchecked_should_always_succeed() {
    for max_std_dev in [-1.0f32, 0.0, 0.5, 2.5, 3.0, 4.0] {
        let std_dev = unsafe { GaussianMaxStdDev::new_unchecked(max_std_dev) };
        let expected_u8 = (max_std_dev / 3.0 * 255.0) as u8;
        assert_eq!(std_dev.as_u8(), expected_u8);
    }
}

#[test]
fn test_gaussian_max_std_dev_get_should_return_value_within_tolerance() {
    const TOLERANCE: f32 = 3.0 / 255.0;

    for max_std_dev in [0.0f32, 0.5, 1.5, 3.0] {
        let std_dev = GaussianMaxStdDev::new(max_std_dev).unwrap();
        assert!(
            (std_dev.get() - max_std_dev).abs() <= TOLERANCE,
            " left: {}\nright: {}",
            std_dev.get(),
            max_std_dev
        );
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
    let max_std_dev = GaussianMaxStdDev::new(2.0).unwrap();
    let pod = GaussianTransformPod::new(size, display_mode, sh_deg, no_sh0, max_std_dev);

    buffer.update(&ctx.queue, size, display_mode, sh_deg, no_sh0, max_std_dev);

    let downloaded = buffer
        .download_single(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download single");

    assert_eq!(downloaded, pod);
}

#[test]
fn test_gaussian_transform_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let pod = GaussianTransformPod::new(
        1.0,
        GaussianDisplayMode::Ellipse,
        GaussianShDegree::new(2).unwrap(),
        true,
        GaussianMaxStdDev::new(2.0).unwrap(),
    );
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Gaussian Transform Buffer"),
            contents: bytemuck::bytes_of(&pod),
            usage: GaussianTransformBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer =
        GaussianTransformBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded = wgpu_converted_buffer
        .download::<GaussianTransformPod>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");
    let converted_downloaded = converted_buffer
        .download::<GaussianTransformPod>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");
    let wgpu_converted_downloaded = wgpu_buffer
        .download::<GaussianTransformPod>(&ctx.device, &ctx.queue)
        .block_on()
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
    let max_std_dev = GaussianMaxStdDev::new(2.0).unwrap();
    let pod =
        wgpu_3dgs_core::GaussianTransformPod::new(size, display_mode, sh_deg, no_sh0, max_std_dev);

    assert_eq!(pod.size, size);
    assert_eq!(pod.flags.x, display_mode as u8);
    assert_eq!(pod.flags.y, sh_deg.get());
    assert_eq!(pod.flags.z, no_sh0 as u8);
    assert_eq!(pod.flags.w, max_std_dev.as_u8());
}

use assert_matches::assert_matches;
use wgpu::util::DeviceExt;
use wgpu_3dgs_core::{BufferWrapper, FixedSizeBufferWrapper, FixedSizeBufferWrapperError};

use crate::common;

mod gaussian;
mod gaussian_transform;
mod model_transform;

#[test]
fn test_buffer_wrapper_buffer_when_struct_is_wgpu_buffer_should_return_itself() {
    let ctx = common::TestContext::new();
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    assert_eq!(buffer.buffer(), &buffer);
}

#[test]
fn test_downloadable_buffer_wrapper_download_should_download_buffer_data() {
    let ctx = common::TestContext::new();
    let buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Buffer"),
            contents: bytemuck::cast_slice(&[1u32, 2, 3, 4]),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

    {
        use wgpu_3dgs_core::DownloadableBufferWrapper;
        let downloaded = pollster::block_on(buffer.download::<u32>(&ctx.device, &ctx.queue));

        assert_matches!(downloaded, Ok(data) if data == vec![1u32, 2, 3, 4]);
    }
}

#[derive(Debug)]
struct TestBufferWrapper(wgpu::Buffer);

impl BufferWrapper for TestBufferWrapper {
    fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

impl From<TestBufferWrapper> for wgpu::Buffer {
    fn from(wrapper: TestBufferWrapper) -> Self {
        wrapper.0
    }
}

impl TryFrom<wgpu::Buffer> for TestBufferWrapper {
    type Error = FixedSizeBufferWrapperError;

    fn try_from(buffer: wgpu::Buffer) -> Result<Self, Self::Error> {
        Self::verify_buffer_size(&buffer).map(|()| Self(buffer))
    }
}

impl FixedSizeBufferWrapper for TestBufferWrapper {
    type Pod = u32;
}

#[test]
fn test_fixed_size_buffer_wrapper_pod_size_should_return_correct_size() {
    let ctx = common::TestContext::new();
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    assert_eq!(TestBufferWrapper::pod_size(), 4);
    assert_matches!(TestBufferWrapper::try_from(buffer), Ok(_));
}

#[test]
fn test_fixed_size_buffer_wrapper_verify_buffer_size_when_size_matched_should_return_ok() {
    let ctx = common::TestContext::new();
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Buffer"),
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    assert_matches!(TestBufferWrapper::verify_buffer_size(&buffer), Ok(()));
}

#[test]
fn test_fixed_size_buffer_wrapper_verify_buffer_size_when_size_mismatched_should_return_error() {
    let ctx = common::TestContext::new();
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Buffer"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    assert_matches!(
        TestBufferWrapper::verify_buffer_size(&buffer),
        Err(FixedSizeBufferWrapperError::BufferSizeMismatched {
            buffer_size: 8,
            expected_size: 4
        })
    );
}

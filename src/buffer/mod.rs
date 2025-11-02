mod gaussian;
mod gaussian_transform;
mod model_transform;

pub use gaussian::*;
pub use gaussian_transform::*;
pub use model_transform::*;

use crate::{DownloadBufferError, FixedSizeBufferWrapperError};

/// A trait to to enable any wrapper to act like a [`wgpu::Buffer`].
pub trait BufferWrapper: Into<wgpu::Buffer> {
    /// The default usages.
    const DEFAULT_USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_retain(
        wgpu::BufferUsages::UNIFORM.bits() | wgpu::BufferUsages::COPY_DST.bits(),
    );

    /// Returns a reference to the buffer data.
    fn buffer(&self) -> &wgpu::Buffer;

    /// Download the buffer data into a [`Vec`].
    fn download<T: bytemuck::NoUninit + bytemuck::AnyBitPattern>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> impl Future<Output = Result<Vec<T>, DownloadBufferError>> + Send
    where
        Self: Send + Sync,
    {
        async {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Buffer Wrapper Download Encoder"),
            });
            let download = self.prepare_download(device, &mut encoder);
            queue.submit(Some(encoder.finish()));

            Self::map_download(&download, device).await
        }
    }

    /// Prepare for downloading the buffer data.
    ///
    /// Returns the download buffer (with [`wgpu::BufferUsages::COPY_DST`] and
    /// [`wgpu::BufferUsages::MAP_READ`]) holding the selection buffer data.
    fn prepare_download(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Buffer {
        let download = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer Wrapper Prepare Download Buffer"),
            size: self.buffer().size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(self.buffer(), 0, &download, 0, download.size());

        download
    }

    /// Map the download buffer to read the buffer data.
    ///
    /// `download` should be created with [`wgpu::BufferUsages::MAP_READ`].
    fn map_download<T: bytemuck::NoUninit + bytemuck::AnyBitPattern>(
        download: &wgpu::Buffer,
        device: &wgpu::Device,
    ) -> impl Future<Output = Result<Vec<T>, DownloadBufferError>> + Send {
        async {
            let (tx, rx) = oneshot::channel();
            let buffer_slice = download.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                if let Err(e) = tx.send(result) {
                    log::error!("Error occurred while sending buffer download data: {e:?}");
                }
            });
            device.poll(wgpu::PollType::Wait)?;
            rx.await??;

            let edits = bytemuck::allocation::pod_collect_to_vec(&buffer_slice.get_mapped_range());
            download.unmap();

            Ok(edits)
        }
    }
}

impl BufferWrapper for wgpu::Buffer {
    fn buffer(&self) -> &wgpu::Buffer {
        self
    }
}

/// A [`BufferWrapper`] with a fixed size that can be validated from a [`wgpu::Buffer`].
pub trait FixedSizeBufferWrapper: BufferWrapper + TryFrom<wgpu::Buffer> {
    /// The POD element type that defines the layout/size.
    type Pod;

    /// Returns the size in bytes of the POD element.
    fn pod_size() -> wgpu::BufferAddress {
        std::mem::size_of::<Self::Pod>() as wgpu::BufferAddress
    }

    /// Check if the given buffer matches the expected size.
    ///
    /// This is a helper function for implementing [`TryFrom`].
    fn verify_buffer_size(buffer: &wgpu::Buffer) -> Result<(), FixedSizeBufferWrapperError> {
        let expected_size = Self::pod_size();
        let buffer_size = buffer.size();
        if buffer_size != expected_size {
            return Err(FixedSizeBufferWrapperError::BufferSizeMismatched {
                buffer_size,
                expected_size,
            });
        }
        Ok(())
    }

    /// Download a single [`FixedSizeBufferWrapper::Pod`].
    fn download_single(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> impl Future<Output = Result<Self::Pod, DownloadBufferError>> + Send
    where
        Self: Send + Sync,
        Self::Pod: bytemuck::NoUninit + bytemuck::AnyBitPattern,
    {
        async move {
            let vec = self.download::<Self::Pod>(device, queue).await?;
            Ok(vec.into_iter().next().expect("downloaded single element"))
        }
    }
}

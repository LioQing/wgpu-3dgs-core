mod gaussian;
mod gaussian_transform;
mod model_transform;

pub use gaussian::*;
pub use gaussian_transform::*;
pub use model_transform::*;

use crate::Error;

/// A trait to to enable any wrapper to act like a [`wgpu::Buffer`].
pub trait BufferWrapper {
    /// Returns a reference to the buffer data.
    fn buffer(&self) -> &wgpu::Buffer;
}

impl BufferWrapper for wgpu::Buffer {
    fn buffer(&self) -> &wgpu::Buffer {
        self
    }
}

/// A trait to enable any [`BufferWrapper`] to download the buffer data.
pub trait DownloadableBufferWrapper: BufferWrapper + Send + Sync {
    /// Download the buffer data.
    fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> impl Future<Output = Result<Vec<u32>, Error>> + Send {
        async {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Selection Download Encoder"),
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
            label: Some(format!("Selection Download Buffer").as_str()),
            size: self.buffer().size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(self.buffer(), 0, &download, 0, download.size());

        download
    }

    /// Map the download buffer to read the buffer data.
    fn map_download(
        download: &wgpu::Buffer,
        device: &wgpu::Device,
    ) -> impl Future<Output = Result<Vec<u32>, Error>> + Send {
        async {
            let (tx, rx) = oneshot::channel();
            let buffer_slice = download.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                if let Err(e) = tx.send(result) {
                    log::error!("Error occurred while sending Gaussian selection: {e:?}");
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

impl<T: BufferWrapper + Send + Sync> DownloadableBufferWrapper for T {}

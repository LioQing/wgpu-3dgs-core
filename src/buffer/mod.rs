mod gaussian;
mod gaussian_transform;
mod model_transform;

pub use gaussian::*;
pub use gaussian_transform::*;
pub use model_transform::*;

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

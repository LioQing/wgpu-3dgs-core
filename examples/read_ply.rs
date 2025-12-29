//! This example reads a PLY file containing Gaussians and uploads them to a GPU buffer.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example write_ply -- "path/to/input.ply"
//! ```

use glam::*;
use wgpu_3dgs_core::{self as gs, BufferWrapper, ReadIterGaussian};

type GaussianPod = gs::GaussianPodWithShHalfCov3dHalfConfigs;

#[pollster::main]
async fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/model.ply".to_string());

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("adapter");

    let (device, _) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("device");

    println!("Reading gaussians from {}", model_path);

    let gaussians = gs::PlyGaussians::read_from_file(&model_path).expect("gaussians");

    let gaussians_buffer = gs::GaussiansBuffer::<GaussianPod>::new(&device, &gaussians);

    println!(
        "Loaded {} gaussians ({:.3} KB) into GPU buffer.",
        gaussians_buffer.len(),
        gaussians_buffer.buffer().size() as f32 / 1024.0,
    );
}

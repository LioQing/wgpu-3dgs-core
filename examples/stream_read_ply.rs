//! This example streams PLY Gaussians and uploads each batch to a GPU buffer.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example stream-read-ply -- "path/to/input.ply" [batch_size]
//! ```
//! `batch_size` is the non-zero number of Gaussians read and uploaded per step
//! (defaults to 4096). CPU memory use stays bounded by the batch size rather
//! than the size of the whole model.

use std::{io::BufReader, num::NonZeroUsize};

use glam::*;
use wgpu_3dgs_core::{self as gs, BufferWrapper, GaussianStream};

type GaussianPod = gs::GaussianPodWithShHalfCov3dHalfConfigs;

#[pollster::main]
async fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/model.ply".to_string());
    let batch_size = std::env::args()
        .nth(2)
        .map(|size| {
            size.parse::<NonZeroUsize>()
                .expect("batch_size must be a non-zero integer")
        })
        .unwrap_or(NonZeroUsize::new(4096).unwrap());

    println!("Reading gaussians from {}", model_path);
    let file = std::fs::File::open(&model_path).expect("open PLY file");
    let mut stream = gs::PlyGaussianStream::new(BufReader::new(file)).expect("PLY stream");

    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("device");

    let gaussians_buffer =
        gs::GaussiansBuffer::<GaussianPod>::new_empty(&device, stream.total_gaussians());
    let mut batch = Vec::new();
    let mut decoded = Vec::new();

    while !stream.progress().done {
        batch.clear();
        let start = stream.progress().completed_units;
        stream
            .read_gaussians(batch_size, &mut batch)
            .expect("read PLY batch");

        decoded.clear();
        decoded.extend(batch.iter().map(gs::Gaussian::from_ply));
        gaussians_buffer
            .update_range(&queue, start, &decoded)
            .expect("upload PLY batch");

        let progress = stream.progress();
        println!(
            "Reading {}: {}/{} ({}/{})",
            progress.phase,
            progress.completed_in_phase,
            progress.total_in_phase,
            progress.completed_units,
            progress.total_units,
        );
    }

    println!(
        "Loaded {} gaussians ({:.3} KB) into GPU buffer.",
        gaussians_buffer.len(),
        gaussians_buffer.buffer().size() as f32 / 1024.0,
    );
}

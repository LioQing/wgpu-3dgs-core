//! This example reads a PLY file containing Gaussians and uploads them to a GPU buffer.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example read-ply -- "path/to/input.ply" [batch_size]
//! ```
//! Omit `batch_size` to read normally, or provide a non-zero number of items per
//! step to print reading progress.

use std::{io::BufReader, num::NonZeroUsize};

use glam::*;
use wgpu_3dgs_core::{self as gs, BatchRead, BufferWrapper, ReadIterGaussian};

type GaussianPod = gs::PackedGaussian<gs::ShHalf, gs::CovHalf>;

#[pollster::main]
async fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/model.ply".to_string());
    let batch_size = std::env::args().nth(2).map(|size| {
        size.parse::<NonZeroUsize>()
            .expect("batch_size must be a non-zero integer")
    });

    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

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

    let gaussians = if let Some(batch_size) = batch_size {
        let file = std::fs::File::open(&model_path).expect("open PLY file");
        let mut reader = gs::PlyBatchReader::new(BufReader::new(file)).expect("PLY reader");
        while !reader.progress().done {
            let progress = reader.step(batch_size).expect("read PLY batch");
            println!(
                "Reading {}: {}/{} ({}/{})",
                progress.phase,
                progress.completed_in_phase,
                progress.total_in_phase,
                progress.completed_units,
                progress.total_units,
            );
        }
        reader.finish().expect("finish PLY reading")
    } else {
        gs::PlyGaussians::read_from_file(&model_path).expect("gaussians")
    };

    let gaussians_buffer = gs::GaussiansBuffer::<GaussianPod>::new(&device, &gaussians);

    println!(
        "Loaded {} gaussians ({:.3} KB) into GPU buffer.",
        gaussians_buffer.len(),
        gaussians_buffer.buffer().size() as f32 / 1024.0,
    );
}

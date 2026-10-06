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

use wgpu_3dgs_core::{self as gs, BatchRead, IterGaussian, ReadIterGaussian};

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

    let gaussians_buffer = upload(&device, &gaussians);

    println!(
        "Loaded {} gaussians ({:.3} KB) into GPU buffer.",
        gaussians.len(),
        gaussians_buffer.size() as f32 / 1024.0,
    );
}

/// Dispatch a runtime model once, then upload its matching compact layout.
fn upload(device: &wgpu::Device, gaussians: &impl IterGaussian) -> wgpu::Buffer {
    macro_rules! upload_sh_degree {
        ($degree:ty) => {
            gs::GaussiansBuffer::<gs::PackedGaussian<gs::ShHalf<$degree>, gs::CovHalf>>::try_new(
                device, gaussians,
            )
            .expect("matching storage degree")
            .into()
        };
    }

    match gaussians.sh_degree().get() {
        0 => upload_sh_degree!(gs::ShDegree0),
        1 => upload_sh_degree!(gs::ShDegree1),
        2 => upload_sh_degree!(gs::ShDegree2),
        3 => upload_sh_degree!(gs::ShDegree3),
        4 => upload_sh_degree!(gs::ShDegree4),
        _ => unreachable!(),
    }
}

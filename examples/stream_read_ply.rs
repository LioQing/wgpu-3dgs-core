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

use wgpu_3dgs_core::{self as gs, BufferWrapper, GaussianStream};

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
    let stream = gs::PlyGaussianStream::new(BufReader::new(file)).expect("PLY stream");

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

    macro_rules! upload_sh_degree {
        ($degree:ty) => {
            upload::<gs::PackedGaussian<gs::ShHalf<$degree>, gs::CovHalf>>(
                &device, &queue, stream, batch_size,
            )
        };
    }

    match stream.header().sh_degree {
        0 => upload_sh_degree!(gs::ShDegree0),
        1 => upload_sh_degree!(gs::ShDegree1),
        2 => upload_sh_degree!(gs::ShDegree2),
        3 => upload_sh_degree!(gs::ShDegree3),
        4 => upload_sh_degree!(gs::ShDegree4),
        _ => unreachable!(),
    }
}

fn upload<G: gs::GaussianPod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut stream: gs::PlyGaussianStream<impl std::io::BufRead>,
    batch_size: NonZeroUsize,
) {
    let gaussians_buffer = gs::GaussiansBuffer::<G>::new_empty(device, stream.total_gaussians());

    let mut batch = Vec::new();
    let mut decoded = Vec::new();

    while !stream.progress().done {
        batch.clear();

        let start = stream.progress().completed_units;

        stream
            .next_batch(batch_size, &mut batch)
            .expect("read PLY batch");

        decoded.clear();
        decoded.extend(
            batch.iter().map(|ply| {
                gs::Gaussian::<G::ShDegree>::from_ply(ply).expect("matching PLY degree")
            }),
        );

        gaussians_buffer
            .update_range(queue, start, &decoded)
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

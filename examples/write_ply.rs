//! This example generates a PLY file containing 3 hardcoded Gaussians.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example write-ply -- "path/to/output.ply" [batch_size]
//! ```
//! Omit `batch_size` to write normally, or provide a non-zero number of items per
//! step to print writing progress.

use std::{
    io::{BufWriter, Write},
    num::NonZeroUsize,
};

use glam::*;
use wgpu_3dgs_core::{self as gs, BatchWrite, WriteIterGaussian};

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/output.ply".to_string());
    let batch_size = std::env::args().nth(2).map(|size| {
        size.parse::<NonZeroUsize>()
            .expect("batch_size must be a non-zero integer")
    });

    let gaussians = [
        gs::Gaussian {
            rot: Quat::from_axis_angle((Vec3::X + Vec3::Y / 2.0 + Vec3::Z).normalize(), 0.5),
            pos: Vec3::ZERO,
            scale: Vec3::new(0.5, 1.0, 0.75),
            color: U8Vec4::new(255, 0, 0, 255),
            sh: [Vec3::ZERO; 15],
        },
        gs::Gaussian {
            rot: Quat::from_axis_angle((Vec3::X + Vec3::Z / 3.0).normalize(), 0.3),
            pos: Vec3::new(0.0, 8.0, 4.0),
            scale: Vec3::new(1.0, 1.9, 0.75),
            color: U8Vec4::new(0, 255, 0, 255),
            sh: [Vec3::ZERO; 15],
        },
        gs::Gaussian {
            rot: Quat::from_axis_angle((Vec3::X - Vec3::Z).normalize(), 0.2),
            pos: Vec3::new(4.0, 0.0, 6.0),
            scale: Vec3::new(1.0, 1.1, 0.8),
            color: U8Vec4::new(0, 0, 255, 255),
            sh: [Vec3::ZERO; 15],
        },
    ];

    let gaussians = gs::PlyGaussians::from(
        gaussians
            .iter()
            .map(gs::Gaussian::to_ply)
            .collect::<Vec<_>>(),
    );

    println!("Writing {} gaussians to {}", gaussians.0.len(), model_path);

    if let Some(batch_size) = batch_size {
        let file = std::fs::File::create(&model_path).expect("create PLY file");
        let mut writer =
            gs::PlyBatchWriter::new(BufWriter::new(file), &gaussians).expect("PLY writer");
        while !writer.progress().done {
            let progress = writer.step(batch_size).expect("write PLY batch");
            println!(
                "Writing {}: {}/{} ({}/{})",
                progress.phase,
                progress.completed_in_phase,
                progress.total_in_phase,
                progress.completed_units,
                progress.total_units,
            );
        }
        writer
            .finish()
            .expect("finish PLY writing")
            .flush()
            .expect("flush PLY file");
    } else {
        gaussians
            .write_to_file(&model_path)
            .expect("write PLY file");
    }
}

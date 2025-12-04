//! This example generates a PLY file containing 3 hardcoded Gaussians.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example write_ply -- "path/to/output.ply"
//! ```

use glam::*;
use wgpu_3dgs_core::{self as gs, WriteIterGaussian};

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/output.ply".to_string());

    let gaussians = vec![
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

    gaussians
        .write_to_file(&model_path)
        .expect("write PLY file");
}

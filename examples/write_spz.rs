//! This example generates a SPZ file containing 3 hardcoded Gaussians.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example write_spz -- "path/to/output.spz"
//! ```

use glam::*;
use wgpu_3dgs_core::{self as gs, WriteIterGaussian};

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/output.spz".to_string());

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

    // Alternatively, use `SpzGaussians::from_gaussians` for default options.
    let gaussians = gs::SpzGaussians::from_gaussians_with_options(
        gaussians.as_slice(),
        &gs::SpzGaussiansFromGaussianSliceOptions {
            version: 2,
            ..Default::default()
        },
    )
    .expect("valid options");

    println!("Header: {:?}", gaussians.header);

    println!(
        "Writing {} gaussians to {}",
        gaussians.header.num_points(),
        model_path
    );

    gaussians
        .write_to_file(&model_path)
        .expect("write SPZ file");
}

use assert_matches::assert_matches;
use wgpu_3dgs_core::{BufferWrapper, Gaussian, GaussianPod, Gaussians, GaussiansBuffer};

use crate::{
    common::{TestContext, given},
    for_each_gaussian_pod,
};

#[test]
fn test_gaussians_buffer_new_should_return_correct_buffer() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let size = (std::mem::size_of::<G>() * gaussians.len()) as u64;
        let gaussians_buffer = GaussiansBuffer::<G>::new(&ctx.device, &gaussians);

        assert_eq!(gaussians_buffer.buffer().size(), size);
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_new_with_usage_should_return_correct_buffer() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let gaussian_pods = gaussians.iter().map(|g| G::from(&g)).collect::<Vec<_>>();
        let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(&ctx.device, &gaussians, usage);

        let gaussian_pods_downloaded =
            pollster::block_on(gaussians_buffer.download::<G>(&ctx.device, &ctx.queue));

        assert_matches!(gaussian_pods_downloaded, Ok(pods) if pods == gaussian_pods);
        assert_eq!(gaussians_buffer.buffer().usage(), usage);
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_new_with_pods_should_return_correct_buffer() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3).map(given::gaussian_with_seed).collect::<Vec<_>>();
        let gaussian_pods = gaussians.iter().map(|g| G::from(g)).collect::<Vec<_>>();
        let size = (std::mem::size_of::<G>() * gaussian_pods.len()) as u64;
        let gaussians_buffer = GaussiansBuffer::<G>::new_with_pods(&ctx.device, &gaussian_pods);

        assert_eq!(gaussians_buffer.buffer().size(), size);
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_new_with_pods_and_usage_should_return_correct_buffer() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3).map(given::gaussian_with_seed).collect::<Vec<_>>();
        let gaussian_pods = gaussians.iter().map(|g| G::from(g)).collect::<Vec<_>>();
        let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let gaussians_buffer =
            GaussiansBuffer::<G>::new_with_pods_and_usage(&ctx.device, &gaussian_pods, usage);

        let gaussian_pods_downloaded =
            pollster::block_on(gaussians_buffer.download::<G>(&ctx.device, &ctx.queue));

        assert_matches!(gaussian_pods_downloaded, Ok(pods) if pods == gaussian_pods);
        assert_eq!(gaussians_buffer.buffer().usage(), usage);
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_new_empty_should_return_correct_buffer() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let count = 10;
        let size = (std::mem::size_of::<G>() * count) as u64;
        let gaussians_buffer = GaussiansBuffer::<G>::new_empty(&ctx.device, count);

        assert_eq!(gaussians_buffer.buffer().size(), size);
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_new_empty_with_usage_should_return_correct_buffer() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let count = 10;
        let size = (std::mem::size_of::<G>() * count) as u64;
        let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let gaussians_buffer =
            GaussiansBuffer::<G>::new_empty_with_usage(&ctx.device, count, usage);

        assert_eq!(gaussians_buffer.buffer().size(), size);
        assert_eq!(gaussians_buffer.buffer().usage(), usage);
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_len_should_return_correct_length() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let gaussians_buffer = GaussiansBuffer::<G>::new(&ctx.device, &gaussians);

        assert_eq!(gaussians_buffer.len(), gaussians.len());
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_is_empty_should_return_correct_value() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let gaussians_buffer = GaussiansBuffer::<G>::new(&ctx.device, &gaussians);
        let empty_gaussians_buffer = GaussiansBuffer::<G>::new_empty(&ctx.device, 0);

        assert!(!gaussians_buffer.is_empty());
        assert!(empty_gaussians_buffer.is_empty());
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_update_should_update_buffer_correctly() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let new_gaussians = (3..6)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let gaussian_pods = new_gaussians
            .iter()
            .map(|g| G::from(&g))
            .collect::<Vec<_>>();
        let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(
            &ctx.device,
            &gaussians,
            GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        );

        gaussians_buffer
            .update(&ctx.queue, &new_gaussians)
            .expect("update");

        let gaussian_pods_downloaded =
            pollster::block_on(gaussians_buffer.download::<G>(&ctx.device, &ctx.queue));

        assert_matches!(gaussian_pods_downloaded, Ok(pods) if pods == gaussian_pods);
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_update_when_new_data_length_is_different_should_return_error() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let new_gaussians = (3..5)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(
            &ctx.device,
            &gaussians,
            GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        );

        let result = gaussians_buffer.update(&ctx.queue, &new_gaussians);

        assert_matches!(
            result,
            Err(wgpu_3dgs_core::GaussiansBufferUpdateError::CountMismatch {
                expected_count: 3,
                count: 2,
            })
        );
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_update_range_should_update_buffer_correctly() {
    fn body<G: GaussianPod>() {
        const START_INDEX: usize = 2;

        let ctx = TestContext::new();
        let gaussians = (0..10)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let new_partial_gaussians = (10..15)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let gaussian_pods = gaussians
            .iter()
            .take(START_INDEX)
            .chain(new_partial_gaussians.iter())
            .chain(
                gaussians
                    .iter()
                    .skip(START_INDEX + new_partial_gaussians.len()),
            )
            .map(|g| G::from(&g))
            .collect::<Vec<_>>();
        let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(
            &ctx.device,
            &gaussians,
            GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        );

        gaussians_buffer
            .update_range(
                &ctx.queue,
                START_INDEX,
                new_partial_gaussians.iter().collect::<Vec<_>>().as_slice(),
            )
            .expect("update_range");

        let gaussian_pods_downloaded =
            pollster::block_on(gaussians_buffer.download::<G>(&ctx.device, &ctx.queue));

        assert_matches!(gaussian_pods_downloaded, Ok(pods) if pods == gaussian_pods);
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_download_gaussians_should_download_buffer_successfully() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(
            &ctx.device,
            &gaussians,
            GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        );

        let gaussian_pods_downloaded =
            pollster::block_on(gaussians_buffer.download::<G>(&ctx.device, &ctx.queue))
                .expect("download");
        let gaussian_pods_gaussians = gaussian_pods_downloaded
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Gaussian>>();
        let gaussians_downloaded =
            pollster::block_on(gaussians_buffer.download_gaussians(&ctx.device, &ctx.queue))
                .expect("download_gaussians");

        assert_eq!(gaussians_downloaded, gaussian_pods_gaussians);
    }

    body::<wgpu_3dgs_core::GaussianPodWithShSingleCov3dRotScaleConfigs>();
    body::<wgpu_3dgs_core::GaussianPodWithShHalfCov3dRotScaleConfigs>();
    body::<wgpu_3dgs_core::GaussianPodWithShNorm8Cov3dRotScaleConfigs>();
}

mod test_gaussians_buffer_download_gaussians_when_configs_unsupported_should_panic {
    use super::*;

    pub(crate) fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(
            &ctx.device,
            &gaussians,
            GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        );

        let _ = pollster::block_on(gaussians_buffer.download_gaussians(&ctx.device, &ctx.queue));
    }
}

#[test]
#[should_panic]
fn test_gaussians_buffer_download_gaussians_when_sh_single_cov3d_single_should_panic() {
    test_gaussians_buffer_download_gaussians_when_configs_unsupported_should_panic::body::<
        wgpu_3dgs_core::GaussianPodWithShSingleCov3dSingleConfigs,
    >();
}

#[test]
#[should_panic]
fn test_gaussians_buffer_download_gaussians_when_sh_single_cov3d_half_should_panic() {
    test_gaussians_buffer_download_gaussians_when_configs_unsupported_should_panic::body::<
        wgpu_3dgs_core::GaussianPodWithShSingleCov3dHalfConfigs,
    >();
}

#[test]
#[should_panic]
fn test_gaussians_buffer_download_gaussians_when_sh_half_cov3d_single_should_panic() {
    test_gaussians_buffer_download_gaussians_when_configs_unsupported_should_panic::body::<
        wgpu_3dgs_core::GaussianPodWithShHalfCov3dSingleConfigs,
    >();
}

#[test]
#[should_panic]
fn test_gaussians_buffer_download_gaussians_when_sh_half_cov3d_half_should_panic() {
    test_gaussians_buffer_download_gaussians_when_configs_unsupported_should_panic::body::<
        wgpu_3dgs_core::GaussianPodWithShHalfCov3dHalfConfigs,
    >();
}

#[test]
#[should_panic]
fn test_gaussians_buffer_download_gaussians_when_sh_norm8_cov3d_single_should_panic() {
    test_gaussians_buffer_download_gaussians_when_configs_unsupported_should_panic::body::<
        wgpu_3dgs_core::GaussianPodWithShNorm8Cov3dSingleConfigs,
    >();
}

#[test]
#[should_panic]
fn test_gaussians_buffer_download_gaussians_when_sh_norm8_cov3d_half_should_panic() {
    test_gaussians_buffer_download_gaussians_when_configs_unsupported_should_panic::body::<
        wgpu_3dgs_core::GaussianPodWithShNorm8Cov3dHalfConfigs,
    >();
}

#[test]
#[should_panic]
fn test_gaussians_buffer_download_gaussians_when_sh_none_cov3d_rot_scale_should_panic() {
    test_gaussians_buffer_download_gaussians_when_configs_unsupported_should_panic::body::<
        wgpu_3dgs_core::GaussianPodWithShNoneCov3dRotScaleConfigs,
    >();
}

#[test]
#[should_panic]
fn test_gaussians_buffer_download_gaussians_when_sh_none_cov3d_single_should_panic() {
    test_gaussians_buffer_download_gaussians_when_configs_unsupported_should_panic::body::<
        wgpu_3dgs_core::GaussianPodWithShNoneCov3dSingleConfigs,
    >();
}

#[test]
#[should_panic]
fn test_gaussians_buffer_download_gaussians_when_sh_none_cov3d_half_should_panic() {
    test_gaussians_buffer_download_gaussians_when_configs_unsupported_should_panic::body::<
        wgpu_3dgs_core::GaussianPodWithShNoneCov3dHalfConfigs,
    >();
}

#[test]
fn test_gaussians_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let gaussians = (0..3)
            .map(given::gaussian_with_seed)
            .collect::<Gaussians<_>>();
        let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(
            &ctx.device,
            &gaussians,
            GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        );
        let wgpu_buffer = gaussians_buffer.buffer().clone();

        let converted_gaussians_buffer =
            GaussiansBuffer::<G>::try_from(wgpu_buffer).expect("try_from");
        let wgpu_converted_buffer = wgpu::Buffer::from(converted_gaussians_buffer.clone());

        let wgpu_downloaded = pollster::block_on(
            gaussians_buffer
                .buffer()
                .download::<G>(&ctx.device, &ctx.queue),
        )
        .expect("download");
        let converted_downloaded = pollster::block_on(
            converted_gaussians_buffer
                .buffer()
                .download::<G>(&ctx.device, &ctx.queue),
        )
        .expect("download");
        let wgpu_converted_downloaded =
            pollster::block_on(wgpu_converted_buffer.download::<G>(&ctx.device, &ctx.queue))
                .expect("download");

        assert_eq!(wgpu_downloaded, converted_downloaded);
        assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
    }

    for_each_gaussian_pod!(G => body::<G>());
}

#[test]
fn test_gaussians_buffer_try_from_wgpu_buffer_when_size_is_not_multiple_should_return_error() {
    fn body<G: GaussianPod>() {
        let ctx = TestContext::new();
        let size = (std::mem::size_of::<G>() * 3 + 1) as u64; // +1 to make it not multiple
        let wgpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let result = GaussiansBuffer::<G>::try_from(wgpu_buffer);

        assert_matches!(
            result,
            Err(
                wgpu_3dgs_core::GaussiansBufferTryFromBufferError::BufferSizeNotMultiple {
                    buffer_size,
                    expected_multiple_size,
                }
            ) if buffer_size == size && expected_multiple_size == std::mem::size_of::<G>() as u64
        );
    }

    for_each_gaussian_pod!(G => body::<G>());
}

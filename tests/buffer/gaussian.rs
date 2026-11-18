use assert_matches::assert_matches;
use pollster::FutureExt;
use wgpu_3dgs_core::{
    BufferWrapper, CovHalf, CovRotScale, CovSingle, Gaussian, GaussianPod, GaussiansBuffer,
    IterGaussian, PackedGaussian, ShHalf, ShNone, ShNorm8, ShSingle,
};

use crate::{
    common::{TestContext, given},
    for_each_gaussian_pod,
};

fn assert_new_buffer<G: GaussianPod>() {
    let ctx = TestContext::new();
    let gaussians = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let size = (std::mem::size_of::<G>() * gaussians.len()) as u64;
    let gaussians_buffer = GaussiansBuffer::<G>::new(&ctx.device, &gaussians);

    assert_eq!(gaussians_buffer.buffer().size(), size);
}

fn assert_new_buffer_with_usage<G: GaussianPod>() {
    let ctx = TestContext::new();
    let gaussians = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let gaussian_pods = gaussians
        .iter_gaussian()
        .map(|g| G::from(&g))
        .collect::<Vec<_>>();
    let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(&ctx.device, &gaussians, usage);

    let gaussian_pods_downloaded = gaussians_buffer
        .download::<G>(&ctx.device, &ctx.queue)
        .block_on();

    assert_matches!(gaussian_pods_downloaded, Ok(pods) if pods == gaussian_pods);
    assert_eq!(gaussians_buffer.buffer().usage(), usage);
}

fn assert_new_buffer_with_pods<G: GaussianPod>() {
    let ctx = TestContext::new();
    let gaussians = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let gaussian_pods = gaussians.iter().map(|g| G::from(g)).collect::<Vec<_>>();
    let size = (std::mem::size_of::<G>() * gaussian_pods.len()) as u64;
    let gaussians_buffer = GaussiansBuffer::<G>::new_with_pods(&ctx.device, &gaussian_pods);

    assert_eq!(gaussians_buffer.buffer().size(), size);
}

fn assert_new_buffer_with_pods_and_usage<G: GaussianPod>() {
    let ctx = TestContext::new();
    let gaussians = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let gaussian_pods = gaussians.iter().map(|g| G::from(g)).collect::<Vec<_>>();
    let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let gaussians_buffer =
        GaussiansBuffer::<G>::new_with_pods_and_usage(&ctx.device, &gaussian_pods, usage);

    let gaussian_pods_downloaded = gaussians_buffer
        .download::<G>(&ctx.device, &ctx.queue)
        .block_on();

    assert_matches!(gaussian_pods_downloaded, Ok(pods) if pods == gaussian_pods);
    assert_eq!(gaussians_buffer.buffer().usage(), usage);
}

fn assert_new_empty_buffer<G: GaussianPod>() {
    let ctx = TestContext::new();
    let count = 10;
    let size = (std::mem::size_of::<G>() * count) as u64;
    let gaussians_buffer = GaussiansBuffer::<G>::new_empty(&ctx.device, count);

    assert_eq!(gaussians_buffer.buffer().size(), size);
}

fn assert_new_empty_buffer_with_usage<G: GaussianPod>() {
    let ctx = TestContext::new();
    let count = 10;
    let size = (std::mem::size_of::<G>() * count) as u64;
    let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let gaussians_buffer = GaussiansBuffer::<G>::new_empty_with_usage(&ctx.device, count, usage);

    assert_eq!(gaussians_buffer.buffer().size(), size);
    assert_eq!(gaussians_buffer.buffer().usage(), usage);
}

fn assert_buffer_len<G: GaussianPod>() {
    let ctx = TestContext::new();
    let gaussians = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let gaussians_buffer = GaussiansBuffer::<G>::new(&ctx.device, &gaussians);

    assert_eq!(gaussians_buffer.len(), gaussians.len());
}

fn assert_buffer_is_empty<G: GaussianPod>() {
    let ctx = TestContext::new();
    let gaussians = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let gaussians_buffer = GaussiansBuffer::<G>::new(&ctx.device, &gaussians);
    let empty_gaussians_buffer = GaussiansBuffer::<G>::new_empty(&ctx.device, 0);

    assert!(!gaussians_buffer.is_empty());
    assert!(empty_gaussians_buffer.is_empty());
}

fn assert_buffer_update<G: GaussianPod>() {
    let ctx = TestContext::new();
    let gaussians = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let new_gaussians = (3..6)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let gaussian_pods = new_gaussians
        .iter_gaussian()
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

    let gaussian_pods_downloaded = gaussians_buffer
        .download::<G>(&ctx.device, &ctx.queue)
        .block_on();

    assert_matches!(gaussian_pods_downloaded, Ok(pods) if pods == gaussian_pods);
}

fn assert_buffer_update_count_mismatch<G: GaussianPod>() {
    let ctx = TestContext::new();
    let gaussians = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let new_gaussians = (3..5)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
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

fn assert_buffer_update_range<G: GaussianPod>() {
    const START_INDEX: usize = 2;

    let ctx = TestContext::new();
    let gaussians = (0..10)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let new_partial_gaussians = (10..15)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let gaussian_pods = gaussians
        .iter_gaussian()
        .take(START_INDEX)
        .chain(new_partial_gaussians.iter_gaussian())
        .chain(
            gaussians
                .iter_gaussian()
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
            new_partial_gaussians
                .iter_gaussian()
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .expect("update_range");

    let gaussian_pods_downloaded = gaussians_buffer
        .download::<G>(&ctx.device, &ctx.queue)
        .block_on();

    assert_matches!(gaussian_pods_downloaded, Ok(pods) if pods == gaussian_pods);
}

fn assert_buffer_update_range_count_mismatch<G: GaussianPod>() {
    const START_INDEX: usize = 8;

    let ctx = TestContext::new();
    let gaussians = (0..10)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let new_partial_gaussians = (10..15)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let result = gaussians_buffer.update_range(
        &ctx.queue,
        START_INDEX,
        new_partial_gaussians
            .iter_gaussian()
            .collect::<Vec<_>>()
            .as_slice(),
    );

    assert_matches!(
        result,
        Err(wgpu_3dgs_core::GaussiansBufferUpdateRangeError::CountMismatch {
            count,
            start,
            expected_count,
        }) if count == new_partial_gaussians.len()
            && start == START_INDEX
            && expected_count == gaussians.len()
    );
}

fn assert_buffer_download_gaussians<G: GaussianPod>() {
    let ctx = TestContext::new();
    let gaussians = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    let gaussian_pods_downloaded = gaussians_buffer
        .download::<G>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");
    let gaussian_pods_gaussians = gaussian_pods_downloaded
        .into_iter()
        .map(Into::into)
        .collect::<Vec<Gaussian<G::ShDegree>>>();
    let gaussians_downloaded = gaussians_buffer
        .download_gaussians(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download_gaussians");

    assert_eq!(gaussians_downloaded, gaussian_pods_gaussians);
}

fn download_buffer_gaussians<G: GaussianPod>() {
    let ctx = TestContext::new();
    let gaussians = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    gaussians_buffer
        .download_gaussians(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download_gaussians");
}

fn assert_wgpu_buffer_round_trip<G: GaussianPod>() {
    let ctx = TestContext::new();
    let gaussians = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );
    let wgpu_buffer = gaussians_buffer.buffer().clone();

    let converted_gaussians_buffer = GaussiansBuffer::<G>::try_from(wgpu_buffer).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_gaussians_buffer.clone());

    let wgpu_downloaded = gaussians_buffer
        .buffer()
        .download::<G>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");
    let converted_downloaded = converted_gaussians_buffer
        .buffer()
        .download::<G>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");
    let wgpu_converted_downloaded = wgpu_converted_buffer
        .download::<G>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

fn assert_wgpu_buffer_size_mismatch<G: GaussianPod>() {
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

fn assert_typed_and_runtime_buffer_updates<G: GaussianPod>() {
    let ctx = TestContext::new();
    let original = (0..3)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let replacement = (3..6)
        .map(given::gaussian_for_sh_degree::<G::ShDegree>)
        .collect::<Vec<_>>();
    let ply = wgpu_3dgs_core::PlyGaussians::from_gaussians(&original).unwrap();
    let usage = GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC;
    let buffer = GaussiansBuffer::<G>::try_new_with_usage(&ctx.device, &ply, usage).unwrap();

    assert_eq!(
        GaussiansBuffer::<G>::try_new(&ctx.device, &ply)
            .unwrap()
            .len(),
        original.len()
    );

    // PLY conversion normalizes rotations and applies floating-point transforms.
    let expected = ply
        .iter()
        .map(|g| G::from_gaussian(&Gaussian::<G::ShDegree>::from_ply(g).unwrap()))
        .collect::<Vec<_>>();

    assert_eq!(
        buffer
            .download::<G>(&ctx.device, &ctx.queue)
            .block_on()
            .unwrap(),
        expected
    );

    buffer.update(&ctx.queue, &replacement).unwrap();
    buffer.update_range(&ctx.queue, 1, &original[1..2]).unwrap();

    let mut expected = replacement.iter().map(G::from_gaussian).collect::<Vec<_>>();
    expected[1] = G::from_gaussian(&original[1]);

    assert_eq!(
        buffer
            .download::<G>(&ctx.device, &ctx.queue)
            .block_on()
            .unwrap(),
        expected
    );

    buffer.try_update(&ctx.queue, &ply).unwrap();

    let expected = ply
        .iter()
        .map(|g| G::from_gaussian(&Gaussian::<G::ShDegree>::from_ply(g).unwrap()))
        .collect::<Vec<_>>();

    assert_eq!(
        buffer
            .download::<G>(&ctx.device, &ctx.queue)
            .block_on()
            .unwrap(),
        expected
    );

    buffer
        .try_update_range(&ctx.queue, 1, &replacement[1..2].to_vec())
        .unwrap();

    let mut expected = expected;
    expected[1] = G::from_gaussian(&replacement[1]);

    assert_eq!(
        buffer
            .download::<G>(&ctx.device, &ctx.queue)
            .block_on()
            .unwrap(),
        expected
    );
    if <G::Cov3dConfig as wgpu_3dgs_core::GaussianCov3dConfig>::FEATURE == "cov3d_rot_scale" {
        let restored = buffer
            .download_gaussians(&ctx.device, &ctx.queue)
            .block_on()
            .unwrap();

        assert_eq!(
            restored.iter().map(G::from_gaussian).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn test_gaussians_buffer_new_should_return_correct_buffer() {
    for_each_gaussian_pod!(G => assert_new_buffer::<G>());
}

#[test]
fn test_gaussians_buffer_new_with_usage_should_return_correct_buffer() {
    for_each_gaussian_pod!(G => assert_new_buffer_with_usage::<G>());
}

#[test]
fn test_gaussians_buffer_new_with_pods_should_return_correct_buffer() {
    for_each_gaussian_pod!(G => assert_new_buffer_with_pods::<G>());
}

#[test]
fn test_gaussians_buffer_new_with_pods_and_usage_should_return_correct_buffer() {
    for_each_gaussian_pod!(G => assert_new_buffer_with_pods_and_usage::<G>());
}

#[test]
fn test_gaussians_buffer_new_empty_should_return_correct_buffer() {
    for_each_gaussian_pod!(G => assert_new_empty_buffer::<G>());
}

#[test]
fn test_gaussians_buffer_new_empty_with_usage_should_return_correct_buffer() {
    for_each_gaussian_pod!(G => assert_new_empty_buffer_with_usage::<G>());
}

#[test]
fn test_gaussians_buffer_len_should_return_correct_length() {
    for_each_gaussian_pod!(G => assert_buffer_len::<G>());
}

#[test]
fn test_gaussians_buffer_is_empty_should_return_correct_value() {
    for_each_gaussian_pod!(G => assert_buffer_is_empty::<G>());
}

#[test]
fn test_gaussians_buffer_update_should_update_buffer_correctly() {
    for_each_gaussian_pod!(G => assert_buffer_update::<G>());
}

#[test]
fn test_gaussians_buffer_update_when_new_data_length_is_different_should_return_error() {
    for_each_gaussian_pod!(G => assert_buffer_update_count_mismatch::<G>());
}

#[test]
fn test_gaussians_buffer_update_range_should_update_buffer_correctly() {
    for_each_gaussian_pod!(G => assert_buffer_update_range::<G>());
}

#[test]
fn test_gaussians_buffer_update_range_when_new_data_overflows_buffer_should_return_error() {
    for_each_gaussian_pod!(G => assert_buffer_update_range_count_mismatch::<G>());
}

#[test]
fn test_gaussians_buffer_download_gaussians_should_download_buffer_successfully() {
    assert_buffer_download_gaussians::<PackedGaussian<ShSingle, CovRotScale>>();
    assert_buffer_download_gaussians::<PackedGaussian<ShHalf, CovRotScale>>();
    assert_buffer_download_gaussians::<PackedGaussian<ShNorm8, CovRotScale>>();
}

#[test]
#[should_panic(expected = "Cannot convert from Cov3d Single configuration")]
fn test_gaussians_buffer_download_gaussians_when_sh_single_cov3d_single_should_panic() {
    download_buffer_gaussians::<PackedGaussian<ShSingle, CovSingle>>();
}

#[test]
#[should_panic(expected = "Cannot convert from Cov3d Half configuration")]
fn test_gaussians_buffer_download_gaussians_when_sh_single_cov3d_half_should_panic() {
    download_buffer_gaussians::<PackedGaussian<ShSingle, CovHalf>>();
}

#[test]
#[should_panic(expected = "Cannot convert from Cov3d Single configuration")]
fn test_gaussians_buffer_download_gaussians_when_sh_half_cov3d_single_should_panic() {
    download_buffer_gaussians::<PackedGaussian<ShHalf, CovSingle>>();
}

#[test]
#[should_panic(expected = "Cannot convert from Cov3d Half configuration")]
fn test_gaussians_buffer_download_gaussians_when_sh_half_cov3d_half_should_panic() {
    download_buffer_gaussians::<PackedGaussian<ShHalf, CovHalf>>();
}

#[test]
#[should_panic(expected = "Cannot convert from Cov3d Single configuration")]
fn test_gaussians_buffer_download_gaussians_when_sh_norm8_cov3d_single_should_panic() {
    download_buffer_gaussians::<PackedGaussian<ShNorm8, CovSingle>>();
}

#[test]
#[should_panic(expected = "Cannot convert from Cov3d Half configuration")]
fn test_gaussians_buffer_download_gaussians_when_sh_norm8_cov3d_half_should_panic() {
    download_buffer_gaussians::<PackedGaussian<ShNorm8, CovHalf>>();
}

#[test]
fn test_gaussians_buffer_download_gaussians_when_sh_none_cov3d_rot_scale_should_return_gaussians() {
    assert_buffer_download_gaussians::<PackedGaussian<ShNone, CovRotScale>>();
}

#[test]
#[should_panic(expected = "Cannot convert from Cov3d Single configuration")]
fn test_gaussians_buffer_download_gaussians_when_sh_none_cov3d_single_should_panic() {
    download_buffer_gaussians::<PackedGaussian<ShNone, CovSingle>>();
}

#[test]
#[should_panic(expected = "Cannot convert from Cov3d Half configuration")]
fn test_gaussians_buffer_download_gaussians_when_sh_none_cov3d_half_should_panic() {
    download_buffer_gaussians::<PackedGaussian<ShNone, CovHalf>>();
}

#[test]
fn test_gaussians_buffer_wgpu_buffer_round_trip_should_be_equal() {
    for_each_gaussian_pod!(G => assert_wgpu_buffer_round_trip::<G>());
}

#[test]
fn test_gaussians_buffer_try_from_wgpu_buffer_when_size_is_not_multiple_should_return_error() {
    for_each_gaussian_pod!(G => assert_wgpu_buffer_size_mismatch::<G>());
}

#[test]
fn test_gaussians_buffer_should_support_typed_and_runtime_uploads_and_updates_for_all_degrees() {
    crate::for_each_sh_degree_gaussian_pod!(G => assert_typed_and_runtime_buffer_updates::<G>());
}

#[test]
fn test_gaussians_buffer_when_runtime_degrees_mismatch_should_leave_buffer_unchanged() {
    use wgpu_3dgs_core::{
        GaussiansBufferUpdateError, GaussiansBufferUpdateRangeError, PlyGaussians, ShDegree1,
        ShDegree4, ShDegreeMismatchError,
    };

    let ctx = TestContext::new();
    type G = PackedGaussian<ShHalf<ShDegree4>, CovRotScale>;

    let high = vec![given::gaussian_for_sh_degree::<ShDegree4>(0)];
    let low = [high[0].convert_sh_degree::<ShDegree1>()];
    let low_ply = PlyGaussians::from_gaussians(low).unwrap();
    let empty = PlyGaussians::new(Vec::new(), 1).unwrap();

    for source in [&low_ply, &empty] {
        assert_matches!(
            GaussiansBuffer::<G>::try_new(&ctx.device, source),
            Err(ShDegreeMismatchError {
                actual_degree: 1,
                expected_degree: 4
            })
        );
    }

    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &high,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );

    assert_matches!(
        buffer.try_update(&ctx.queue, &low_ply),
        Err(GaussiansBufferUpdateError::InvalidDegree(
            ShDegreeMismatchError {
                actual_degree: 1,
                expected_degree: 4,
            }
        ))
    );
    for source in [&low_ply, &empty] {
        assert_matches!(
            buffer.try_update_range(&ctx.queue, 0, source),
            Err(GaussiansBufferUpdateRangeError::InvalidDegree(
                ShDegreeMismatchError {
                    actual_degree: 1,
                    expected_degree: 4,
                }
            ))
        );
    }

    assert_eq!(
        buffer
            .download::<G>(&ctx.device, &ctx.queue)
            .block_on()
            .unwrap(),
        vec![G::from_gaussian(&high[0])]
    );
}

#[test]
fn test_gaussians_buffer_try_update_when_count_mismatches_should_leave_buffer_unchanged() {
    let ctx = TestContext::new();
    type G = PackedGaussian<ShHalf, CovRotScale>;

    let original = given::gaussians();
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &original,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );
    let short = wgpu_3dgs_core::PlyGaussians::from_gaussians(&original[..1]).unwrap();

    assert_matches!(
        buffer.try_update(&ctx.queue, &short),
        Err(wgpu_3dgs_core::GaussiansBufferUpdateError::CountMismatch {
            count: 1,
            expected_count: 2,
        })
    );

    assert_eq!(
        buffer
            .download::<G>(&ctx.device, &ctx.queue)
            .block_on()
            .unwrap(),
        original.iter().map(G::from_gaussian).collect::<Vec<_>>()
    );
}

#[test]
fn test_gaussians_buffer_update_range_when_start_overflows_should_leave_buffer_unchanged() {
    let ctx = TestContext::new();
    type G = PackedGaussian<ShHalf, CovRotScale>;

    let original = given::gaussians();
    let buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &original,
        GaussiansBuffer::<G>::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
    );
    let replacement = wgpu_3dgs_core::PlyGaussians::from_gaussians(&original[..1]).unwrap();

    for start in [original.len() + 1, usize::MAX] {
        for pods in [Vec::new(), vec![G::from_gaussian(&original[0])]] {
            assert_matches!(
                buffer.update_range_with_pod(&ctx.queue, start, &pods),
                Err(wgpu_3dgs_core::GaussiansBufferUpdateRangeError::CountMismatch {
                    start: actual_start,
                    count,
                    expected_count: 2,
                }) if actual_start == start && count == pods.len()
            );
        }
        assert_matches!(
            buffer.try_update_range(&ctx.queue, start, &replacement),
            Err(wgpu_3dgs_core::GaussiansBufferUpdateRangeError::CountMismatch {
                start: actual_start,
                count: 1,
                expected_count: 2,
            }) if actual_start == start
        );
    }

    assert_eq!(
        buffer
            .download::<G>(&ctx.device, &ctx.queue)
            .block_on()
            .unwrap(),
        original.iter().map(G::from_gaussian).collect::<Vec<_>>()
    );
}

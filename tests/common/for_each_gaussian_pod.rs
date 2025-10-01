#[macro_export]
macro_rules! for_each_gaussian_pod {
    ($pod:ident => $body:expr) => {
        fn _body<$pod: wgpu_3dgs_core::GaussianPod>() {
            $body
        }
        _body::<wgpu_3dgs_core::GaussianPodWithShSingleCov3dRotScaleConfigs>();
        _body::<wgpu_3dgs_core::GaussianPodWithShSingleCov3dSingleConfigs>();
        _body::<wgpu_3dgs_core::GaussianPodWithShSingleCov3dHalfConfigs>();
        _body::<wgpu_3dgs_core::GaussianPodWithShHalfCov3dRotScaleConfigs>();
        _body::<wgpu_3dgs_core::GaussianPodWithShHalfCov3dSingleConfigs>();
        _body::<wgpu_3dgs_core::GaussianPodWithShHalfCov3dHalfConfigs>();
        _body::<wgpu_3dgs_core::GaussianPodWithShNorm8Cov3dRotScaleConfigs>();
        _body::<wgpu_3dgs_core::GaussianPodWithShNorm8Cov3dSingleConfigs>();
        _body::<wgpu_3dgs_core::GaussianPodWithShNorm8Cov3dHalfConfigs>();
        _body::<wgpu_3dgs_core::GaussianPodWithShNoneCov3dRotScaleConfigs>();
        _body::<wgpu_3dgs_core::GaussianPodWithShNoneCov3dSingleConfigs>();
        _body::<wgpu_3dgs_core::GaussianPodWithShNoneCov3dHalfConfigs>();
    };
}

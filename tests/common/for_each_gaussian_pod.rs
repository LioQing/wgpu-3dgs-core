#[macro_export]
macro_rules! for_each_gaussian_pod {
    ($pod:ident => $body:expr) => {
        fn _body<$pod: wgpu_3dgs_core::GaussianPod>() {
            $body
        }
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShSingle, wgpu_3dgs_core::CovRotScale>>();
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShSingle, wgpu_3dgs_core::CovSingle>>();
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShSingle, wgpu_3dgs_core::CovHalf>>();
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShHalf, wgpu_3dgs_core::CovRotScale>>();
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShHalf, wgpu_3dgs_core::CovSingle>>();
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShHalf, wgpu_3dgs_core::CovHalf>>();
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShNorm8, wgpu_3dgs_core::CovRotScale>>();
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShNorm8, wgpu_3dgs_core::CovSingle>>();
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShNorm8, wgpu_3dgs_core::CovHalf>>();
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShNone, wgpu_3dgs_core::CovRotScale>>();
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShNone, wgpu_3dgs_core::CovSingle>>();
        _body::<wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShNone, wgpu_3dgs_core::CovHalf>>();
    };
}

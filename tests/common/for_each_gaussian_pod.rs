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

/// All 45 degree/encoding/covariance layouts plus the three ShNone aliases.
#[macro_export]
macro_rules! for_each_sh_degree_gaussian_pod {
    ($pod:ident => $body:expr) => {
        fn _sh_degree_body<$pod: wgpu_3dgs_core::GaussianPod>() {
            $body
        }

        macro_rules! sh_degree {
            ($degree:ty) => {
                macro_rules! encoding {
                    ($sh:ty) => {
                        _sh_degree_body::<
                            wgpu_3dgs_core::PackedGaussian<$sh, wgpu_3dgs_core::CovRotScale>,
                        >();
                        _sh_degree_body::<
                            wgpu_3dgs_core::PackedGaussian<$sh, wgpu_3dgs_core::CovSingle>,
                        >();
                        _sh_degree_body::<
                            wgpu_3dgs_core::PackedGaussian<$sh, wgpu_3dgs_core::CovHalf>,
                        >();
                    };
                }

                encoding!(wgpu_3dgs_core::ShSingle<$degree>);
                encoding!(wgpu_3dgs_core::ShHalf<$degree>);
                encoding!(wgpu_3dgs_core::ShNorm8<$degree>);
            };
        }

        sh_degree!(wgpu_3dgs_core::ShDegree0);
        sh_degree!(wgpu_3dgs_core::ShDegree1);
        sh_degree!(wgpu_3dgs_core::ShDegree2);
        sh_degree!(wgpu_3dgs_core::ShDegree3);
        sh_degree!(wgpu_3dgs_core::ShDegree4);

        _sh_degree_body::<
            wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShNone, wgpu_3dgs_core::CovRotScale>,
        >();
        _sh_degree_body::<
            wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShNone, wgpu_3dgs_core::CovSingle>,
        >();
        _sh_degree_body::<
            wgpu_3dgs_core::PackedGaussian<wgpu_3dgs_core::ShNone, wgpu_3dgs_core::CovHalf>,
        >();
    };
}

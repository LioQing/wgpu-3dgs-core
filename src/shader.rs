use wesl::{Pkg, PkgModule};

pub const PACKAGE: Pkg = Pkg {
    crate_name: "wgpu-3dgs-core",
    root: &MODULE,
    dependencies: &[],
};

pub const MODULE: PkgModule = PkgModule {
    name: "wgpu_3dgs_core",
    source: "",
    submodules: &[
        &gaussian::MODULE,
        &gaussian_transform::MODULE,
        &model_transform::MODULE,
        &compute_bundle::MODULE,
    ],
};

pub mod gaussian {
    use super::PkgModule;

    pub const MODULE: PkgModule = PkgModule {
        name: "gaussian",
        source: include_str!("shader/gaussian.wesl"),
        submodules: &[],
    };
}

pub mod gaussian_transform {
    use super::PkgModule;

    pub const MODULE: PkgModule = PkgModule {
        name: "gaussian_transform",
        source: include_str!("shader/gaussian_transform.wesl"),
        submodules: &[],
    };
}

pub mod model_transform {
    use super::PkgModule;

    pub const MODULE: PkgModule = PkgModule {
        name: "model_transform",
        source: include_str!("shader/model_transform.wesl"),
        submodules: &[],
    };
}

pub mod compute_bundle {
    use super::PkgModule;

    pub const MODULE: PkgModule = PkgModule {
        name: "compute_bundle",
        source: include_str!("shader/compute_bundle.wesl"),
        submodules: &[],
    };
}

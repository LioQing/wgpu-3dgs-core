//! Shader modules for the [`wesl::Pkg`] `wgpu-3dgs-core`.
//!
//! See the documentation of each module for details.

use wesl::{CodegenModule, CodegenPkg};

/// The `wgpu-3dgs-core` [`wesl::Pkg`].
pub const PACKAGE: CodegenPkg = CodegenPkg {
    crate_name: "wgpu-3dgs-core",
    root: &MODULE,
    dependencies: &[],
};

/// The root module of the `wgpu-3dgs-core` package.
pub const MODULE: CodegenModule = CodegenModule {
    name: "wgpu_3dgs_core",
    source: "",
    submodules: &[
        &gaussian::MODULE,
        &gaussian_transform::MODULE,
        &model_transform::MODULE,
    ],
};

#[doc = concat!("```wgsl\n", include_str!("shader/gaussian.wesl"), "\n```")]
pub mod gaussian {
    use super::CodegenModule;

    pub const MODULE: CodegenModule = CodegenModule {
        name: "gaussian",
        source: include_str!("shader/gaussian.wesl"),
        submodules: &[],
    };
}

#[doc = concat!("```wgsl\n", include_str!("shader/gaussian_transform.wesl"), "\n```")]
pub mod gaussian_transform {
    use super::CodegenModule;

    pub const MODULE: CodegenModule = CodegenModule {
        name: "gaussian_transform",
        source: include_str!("shader/gaussian_transform.wesl"),
        submodules: &[],
    };
}

#[doc = concat!("```wgsl\n", include_str!("shader/model_transform.wesl"), "\n```")]
pub mod model_transform {
    use super::CodegenModule;

    pub const MODULE: CodegenModule = CodegenModule {
        name: "model_transform",
        source: include_str!("shader/model_transform.wesl"),
        submodules: &[],
    };
}

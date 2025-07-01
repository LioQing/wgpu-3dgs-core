use wesl::PkgModule;

use crate::wesl::VarDirective;

pub struct Mod;

impl PkgModule for Mod {
    fn name(&self) -> &'static str {
        "wgpu_3dgs_core"
    }

    fn source(&self) -> &'static str {
        ""
    }

    fn submodules(&self) -> &[&dyn PkgModule] {
        static SUBMODULES: &[&dyn PkgModule] = &[
            &gaussian::Mod,
            &gaussian_transform::Mod,
            &model_transform::Mod,
        ];
        SUBMODULES
    }

    fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
        match name {
            "gaussian" => Some(&gaussian::Mod),
            "gaussian_transform" => Some(&gaussian_transform::Mod),
            "model_transform" => Some(&model_transform::Mod),
            _ => None,
        }
    }
}

impl VarDirective for Mod {
    fn vars(&self) -> &'static [&'static str] {
        &["gaussians", "gaussian_transform", "model_transform"]
    }
}

macro_rules! submodule {
    ($name:ident $(, $($vars:ident),* $(,)?)?) => {
        paste::paste! {
            pub mod $name {
                pub struct Mod;

                impl wesl::PkgModule for Mod {
                    fn name(&self) -> &'static str {
                        stringify!($name)
                    }

                    fn source(&self) -> &'static str {
                        include_str!(concat!("shader/", stringify!($name), ".wesl"))
                    }

                    fn submodules(&self) -> &[&dyn wesl::PkgModule] {
                        &[]
                    }

                    fn submodule(&self, _name: &str) -> Option<&dyn wesl::PkgModule> {
                        None
                    }
                }

                $(
                    impl super::VarDirective for Mod {
                        fn vars(&self) -> &'static [&'static str] {
                            &[$(stringify!($vars)),*]
                        }
                    }
                )?
            }
        }
    };
}

submodule!(gaussian, gaussians);
submodule!(gaussian_transform, gaussian_transform);
submodule!(model_transform, model_transform);

use std::{borrow::Cow, collections::HashMap, path::Path};

use crate::Error;

/// A resolver to provide dynamic entry shader.
pub struct DynEntryResolver<R: wesl::Resolver> {
    pub resolver: R,
    pub shader_source: String,
}

impl<R: wesl::Resolver> DynEntryResolver<R> {
    /// The entry shader module path.
    pub const ENTRY_SHADER_PATH: &'static str = "__entry__";

    /// Create a new compute bundle resolver.
    pub fn new(resolver: R, shader_source: String) -> Self {
        Self {
            resolver,
            shader_source,
        }
    }
}

impl<R: wesl::Resolver> wesl::Resolver for DynEntryResolver<R> {
    fn resolve_source<'a>(
        &'a self,
        path: &wesl::ModulePath,
    ) -> Result<Cow<'a, str>, wesl::ResolveError> {
        if path == &Self::ENTRY_SHADER_PATH.into() {
            return Ok(Cow::Borrowed(&self.shader_source));
        }

        self.resolver.resolve_source(path)
    }
}

/// A trait to enable the use of `@var(name)` directive in WESL packages.
pub trait VarDirective: wesl::PkgModule {
    /// Returns all the variable names that can be resolved.
    fn vars(&self) -> &'static [&'static str] {
        &[]
    }
}

/// A resolver extended from [`wesl::StandardResolver`] to support `@var(name)` directive.
pub struct VarStandardResolver {
    std: wesl::StandardResolver,
    vars: HashMap<&'static str, String>,
}

impl VarStandardResolver {
    pub fn new(base: impl AsRef<Path>) -> Self {
        Self {
            std: wesl::StandardResolver::new(base),
            vars: HashMap::new(),
        }
    }

    pub fn add_package(&mut self, pkg: &'static dyn VarDirective) {
        self.std.add_package(pkg);
        for &var in pkg.vars() {
            self.vars.insert(var, var.to_string());
        }
    }

    pub fn with_package(mut self, pkg: &'static dyn VarDirective) -> Self {
        self.add_package(pkg);
        self
    }

    pub fn add_var(&mut self, from: impl AsRef<str>, to: impl Into<String>) -> Result<(), Error> {
        if let Some(var) = self.vars.get_mut(from.as_ref()) {
            *var = to.into();
            Ok(())
        } else {
            Err(Error::VarNotFound(from.as_ref().to_string()))
        }
    }

    pub fn with_var(mut self, from: impl AsRef<str>, to: impl Into<String>) -> Result<Self, Error> {
        self.add_var(from, to)?;
        Ok(self)
    }
}

impl wesl::Resolver for VarStandardResolver {
    fn resolve_source<'a>(
        &'a self,
        path: &wesl::ModulePath,
    ) -> Result<Cow<'a, str>, wesl::ResolveError> {
        let source = self.std.resolve_source(path)?;

        Ok(self.vars.iter().fold(source, |source, (from, to)| {
            source.replace(format!("@var({from})").as_str(), &to).into()
        }))
    }

    fn resolve_module(
        &self,
        path: &wesl::ModulePath,
    ) -> Result<wesl::syntax::TranslationUnit, wesl::ResolveError> {
        self.std.resolve_module(path)
    }

    fn display_name(&self, path: &wesl::ModulePath) -> Option<String> {
        self.std.display_name(path)
    }
}

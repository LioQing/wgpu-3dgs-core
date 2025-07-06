use std::{borrow::Cow, collections::HashMap};

/// A resolver to provide dynamic shaders.
pub struct DynResolver<R: wesl::Resolver> {
    pub resolver: R,
    pub dyn_shaders: HashMap<wesl::ModulePath, String>,
}

impl<R: wesl::Resolver> DynResolver<R> {
    /// Create a new dynamic resolver.
    pub fn new(resolver: R) -> Self {
        Self {
            resolver,
            dyn_shaders: HashMap::new(),
        }
    }

    /// Add a dynamic shader.
    pub fn add_shader(&mut self, path: wesl::ModulePath, source: String) {
        self.dyn_shaders.insert(path, source);
    }

    /// Add a dynamic shader.
    pub fn with_shader(mut self, path: wesl::ModulePath, source: String) -> Self {
        self.add_shader(path, source);
        self
    }

    /// Get a dynamic shader by path.
    pub fn get_shader(&self, path: &wesl::ModulePath) -> Option<&String> {
        self.dyn_shaders.get(path)
    }
}

impl<R: wesl::Resolver> wesl::Resolver for DynResolver<R> {
    fn resolve_source<'a>(
        &'a self,
        path: &wesl::ModulePath,
    ) -> Result<Cow<'a, str>, wesl::ResolveError> {
        if let Some(source) = self.dyn_shaders.get(path) {
            return Ok(Cow::Borrowed(source));
        }

        self.resolver.resolve_source(path)
    }
}

use std::borrow::Cow;

/// A resolver to provide dynamic entry shader.
pub struct DynEntryResolver<R: wesl::Resolver> {
    pub resolver: R,
    pub shader_source: String,
}

impl<R: wesl::Resolver> DynEntryResolver<R> {
    /// The entry shader module path.
    pub const ENTRY_SHADER_PATH: &'static str = "_entry_";

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
        if path.last() == Some(&Self::ENTRY_SHADER_PATH) {
            return Ok(Cow::Borrowed(&self.shader_source));
        }

        self.resolver.resolve_source(path)
    }
}

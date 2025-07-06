use crate::{BufferWrapper, Error, wesl::DynEntryResolver};

macro_rules! label_for_components {
    ($label:expr, $component:expr) => {
        format!(
            "{} {}",
            $label.as_deref().unwrap_or("Compute Bundle"),
            $component,
        )
    };
}

/// A bundle of [`wgpu::ComputePipeline`], its [`wgpu::BindGroupLayout`]
/// and optionally [`wgpu::BindGroup`].
#[derive(Debug)]
pub struct ComputeBundle<B = wgpu::BindGroup> {
    /// The label of the compute bundle.
    label: Option<String>,
    /// The workgroup size.
    workgroup_size: u32,
    /// The bind group layouts.
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    /// The bind groups.
    bind_groups: Vec<B>,
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
}

impl<B> ComputeBundle<B> {
    /// Create the bind group.
    pub fn create_bind_group<'a>(
        &self,
        device: &wgpu::Device,
        index: usize,
        buffers: impl IntoIterator<Item = &'a dyn BufferWrapper>,
    ) -> Option<wgpu::BindGroup> {
        Some(ComputeBundle::create_bind_group_static(
            self.label.as_deref(),
            device,
            index,
            self.bind_group_layouts().get(index)?,
            buffers,
        ))
    }

    /// Get the number of invocations in one workgroup.
    pub fn workgroup_size(&self) -> u32 {
        self.workgroup_size
    }

    /// Get the label.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Get the bind group layouts.
    pub fn bind_group_layouts(&self) -> &[wgpu::BindGroupLayout] {
        &self.bind_group_layouts
    }

    /// Get the compute pipeline.
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

impl ComputeBundle {
    /// Create a new compute bundle.
    pub fn new<'a>(
        label: Option<&str>,
        device: &wgpu::Device,
        bind_group_layout_descriptors: impl IntoIterator<Item = &'a wgpu::BindGroupLayoutDescriptor<'a>>,
        buffers: impl IntoIterator<Item = impl IntoIterator<Item = &'a dyn BufferWrapper>>,
        shader_source: wgpu::ShaderSource,
    ) -> Result<Self, Error> {
        let this = ComputeBundle::new_without_bind_groups(
            label,
            device,
            bind_group_layout_descriptors,
            shader_source,
        )?;

        let buffers = buffers.into_iter().collect::<Vec<_>>();

        if buffers.len() != this.bind_group_layouts.len() {
            return Err(Error::BindGroupLayoutCountMismatch {
                buffer_count: buffers.len(),
                bind_group_layout_count: this.bind_group_layouts.len(),
            });
        }

        log::debug!(
            "Creating {} bind groups",
            label.as_deref().unwrap_or("compute bundle")
        );
        let bind_groups = this
            .bind_group_layouts
            .iter()
            .zip(buffers.into_iter())
            .enumerate()
            .map(|(i, (layout, buffers))| {
                ComputeBundle::create_bind_group_static(this.label(), device, i, layout, buffers)
            })
            .collect::<Vec<_>>();

        Ok(Self {
            label: label.map(String::from),
            workgroup_size: this.workgroup_size,
            bind_group_layouts: this.bind_group_layouts,
            bind_groups,
            pipeline: this.pipeline,
        })
    }

    /// Get the bind groups.
    pub fn bind_groups(&self) -> &[wgpu::BindGroup] {
        &self.bind_groups
    }

    /// Dispatch the compute bundle for `count` instances.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label_for_components!(self.label, "Compute Pass").as_str()),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);

        for (i, group) in self.bind_groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }

        pass.dispatch_workgroups(count.div_ceil(self.workgroup_size()), 1, 1);
    }

    /// Create a bind group statically.
    ///
    /// `index` is only for labeling.
    fn create_bind_group_static<'a>(
        label: Option<&str>,
        device: &wgpu::Device,
        index: usize,
        bind_group_layout: &wgpu::BindGroupLayout,
        buffers: impl IntoIterator<Item = &'a dyn BufferWrapper>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label_for_components!(label, format!("Bind Group {index}")).as_str()),
            layout: bind_group_layout,
            entries: &buffers
                .into_iter()
                .enumerate()
                .map(|(i, buffer)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buffer.buffer().as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        })
    }
}

impl ComputeBundle<()> {
    /// Create a new compute bundle without internally managed bind group.
    ///
    /// To create a bind group with layout matched to one of the layout in this compute bundle,
    /// use the [`ComputeBundle::create_bind_group`] method.
    pub fn new_without_bind_groups<'a>(
        label: Option<&str>,
        device: &wgpu::Device,
        bind_group_layout_descriptors: impl IntoIterator<Item = &'a wgpu::BindGroupLayoutDescriptor<'a>>,
        shader_source: wgpu::ShaderSource,
    ) -> Result<Self, Error> {
        let workgroup_size = device
            .limits()
            .max_compute_workgroup_size_x
            .min(device.limits().max_compute_invocations_per_workgroup);

        log::debug!(
            "Creating {} bind group layouts",
            label.as_deref().unwrap_or("compute bundle")
        );
        let bind_group_layouts = bind_group_layout_descriptors
            .into_iter()
            .map(|desc| device.create_bind_group_layout(desc))
            .collect::<Vec<_>>();

        log::debug!(
            "Creating {} pipeline layout",
            label.as_deref().unwrap_or("compute bundle"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label_for_components!(label, "Pipeline Layout").as_str()),
            bind_group_layouts: &bind_group_layouts.iter().collect::<Vec<_>>(),
            push_constant_ranges: &[],
        });

        log::debug!(
            "Creating {} shader module",
            label.as_deref().unwrap_or("compute bundle"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label_for_components!(label, "Shader").as_str()),
            source: shader_source,
        });

        let compilation_options = wgpu::PipelineCompilationOptions {
            constants: &[("workgroup_size", workgroup_size as f64)],
            ..Default::default()
        };

        log::debug!(
            "Creating {} pipeline",
            label.as_deref().unwrap_or("compute bundle"),
        );
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label_for_components!(label, "Pipeline").as_str()),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("_entry_"),
            compilation_options: compilation_options.clone(),
            cache: None,
        });

        log::info!("{} created", label.as_deref().unwrap_or("Compute Bundle"));

        Ok(Self {
            label: label.map(String::from),
            workgroup_size,
            bind_group_layouts,
            bind_groups: Vec::new(),
            pipeline,
        })
    }

    /// Dispatch the compute bundle for `count` instances.
    pub fn dispatch<'a>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_groups: impl IntoIterator<Item = &'a wgpu::BindGroup>,
        count: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label_for_components!(self.label, "Compute Pass").as_str()),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);

        for (i, group) in bind_groups.into_iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }

        pass.dispatch_workgroups(count.div_ceil(self.workgroup_size()), 1, 1);
    }
}

/// A builder for [`ComputeBundle`].
///
/// The shader is built with WESL, and the entry point is defined by the template
/// in [`ComputeBundleBuilder::WESL_TEMPLATE`].
pub struct ComputeBundleBuilder<'a, R: wesl::Resolver> {
    pub label: Option<&'a str>,
    pub headers: Vec<&'a str>,
    pub bind_group_layouts: Vec<&'a wgpu::BindGroupLayoutDescriptor<'a>>,
    pub main: Option<&'a str>,
    pub compile_options: wesl::CompileOptions,
    pub resolver: Option<R>,
}

impl<'a, R: wesl::Resolver> ComputeBundleBuilder<'a, R> {
    /// The WESL template.
    pub const WESL_TEMPLATE: &'static str = include_str!("shader/compute_bundle_template.wesl");

    /// Create a new compute bundle builder.
    pub fn new() -> Self {
        Self {
            label: None,
            headers: Vec::new(),
            bind_group_layouts: Vec::new(),
            main: None,
            compile_options: wesl::CompileOptions::default(),
            resolver: None,
        }
    }

    /// Set the label of the compute bundle.
    pub fn label(mut self, label: impl Into<&'a str>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add a header to the main entry shader.
    ///
    /// These can be constants, functions, imporst, etc.
    /// The headers are only joined with newlines,
    /// so anything that is valid WGSL code can be used.
    pub fn header(mut self, header: &'a str) -> Self {
        self.headers.push(header);
        self
    }

    /// Add the headers to the main entry shader.
    ///
    /// These can be constants, functions, imporst, etc.
    /// The headers are only joined with newlines,
    /// so anything that is valid WGSL code can be used.
    pub fn headers(mut self, header: Vec<&'a str>) -> Self {
        self.headers.extend(header);
        self
    }

    /// Add a bind group descriptor.
    ///
    /// The `bind_group_decl` is in the format of `var<{access space}, {mode}> name: {type}`.
    pub fn bind_group(
        mut self,
        bind_group_layout: &'a wgpu::BindGroupLayoutDescriptor<'a>,
    ) -> Self {
        self.bind_group_layouts.push(bind_group_layout);
        self
    }

    /// Add the bind group descriptors.
    ///
    /// The `bind_group_decls` are in the format of `var<{access space}, {mode}> name: {type}`.
    pub fn bind_groups(
        mut self,
        bind_groups: impl IntoIterator<Item = &'a wgpu::BindGroupLayoutDescriptor<'a>>,
    ) -> Self {
        self.bind_group_layouts.extend(bind_groups);
        self
    }

    /// Set the main function of the compute shader.
    ///
    /// The format is any statements, most likely of `"my_main(arg1, arg2, ...);"`.
    /// Since my_main is likely from another module, you may want to use
    /// [`ComputeBundleBuilder::header`] to import it by adding a header like
    /// `"import my_module::my_main;"`.
    ///
    /// `arg1, arg2, ...` can be one of the followings:
    /// - `index`: The index of the current invocation.
    /// - `workgroup_size`: The size of the workgroup.
    /// - Any of the compute builtins:
    ///     - `local_invocation_id`
    ///     - `local_invocation_index`
    ///     - `global_invocation_id`
    ///     - `workgroup_id`
    ///     - `num_workgroups`
    pub fn main(mut self, main: &'a str) -> Self {
        self.main = Some(main);
        self
    }

    /// Set the compile options for the WESL compiler.
    pub fn compile_options(mut self, options: wesl::CompileOptions) -> Self {
        self.compile_options = options;
        self
    }

    /// Set the WESL resolver.
    pub fn resolver(mut self, resolver: R) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Build the compute bundle with bind groups.
    pub fn build(
        self,
        device: &wgpu::Device,
        buffers: impl IntoIterator<Item = impl IntoIterator<Item = &'a dyn BufferWrapper>>,
    ) -> Result<ComputeBundle<wgpu::BindGroup>, Error> {
        if self.bind_group_layouts.is_empty() {
            return Err(Error::MissingBindGroupLayout);
        }

        let Some(resolver) = self.resolver else {
            return Err(Error::MissingResolver);
        };

        let Some(main) = self.main else {
            return Err(Error::MissingMainFunction);
        };

        let resolver = Self::build_dyn_entry_resolver(resolver, self.headers, main);

        let shader_source = wgpu::ShaderSource::Wgsl(
            wesl::Wesl::new("placeholder") // Base will be replaced by DynEntryResolver
                .set_custom_resolver(resolver)
                .set_options(self.compile_options)
                .compile(DynEntryResolver::<R>::ENTRY_SHADER_PATH)?
                .to_string()
                .into(),
        );

        ComputeBundle::new(
            self.label,
            device,
            self.bind_group_layouts.into_iter().collect::<Vec<_>>(),
            buffers,
            shader_source,
        )
    }

    /// Build the compute bundle without bind groups.
    pub fn build_without_bind_groups(
        self,
        device: &wgpu::Device,
    ) -> Result<ComputeBundle<()>, Error> {
        if self.bind_group_layouts.is_empty() {
            return Err(Error::MissingBindGroupLayout);
        }

        let Some(resolver) = self.resolver else {
            return Err(Error::MissingResolver);
        };

        let Some(main) = self.main else {
            return Err(Error::MissingMainFunction);
        };

        let resolver = Self::build_dyn_entry_resolver(resolver, self.headers, main);

        let shader_source = wgpu::ShaderSource::Wgsl(
            wesl::Wesl::new("placeholder") // Base will be replaced by DynEntryResolver
                .set_custom_resolver(resolver)
                .set_options(self.compile_options)
                .compile(DynEntryResolver::<R>::ENTRY_SHADER_PATH)?
                .to_string()
                .into(),
        );

        ComputeBundle::new_without_bind_groups(
            self.label,
            device,
            self.bind_group_layouts.into_iter().collect::<Vec<_>>(),
            shader_source,
        )
    }

    /// Build the dynamic entry resolver.
    fn build_dyn_entry_resolver(
        resolver: R,
        headers: Vec<&'a str>,
        main: &'a str,
    ) -> DynEntryResolver<R> {
        let source = format!(
            "{}\n{}",
            headers.join("\n"),
            Self::WESL_TEMPLATE.replace("{{main}}", main),
        );

        DynEntryResolver::new(resolver, source)
    }
}

impl<R: wesl::Resolver> Default for ComputeBundleBuilder<'_, R> {
    fn default() -> Self {
        Self::new()
    }
}

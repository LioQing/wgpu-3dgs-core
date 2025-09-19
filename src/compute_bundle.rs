use crate::{ComputeBundleBuildError, ComputeBundleCreateError};

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
///
/// ## Overview
///
/// This is an abstraction of a compute pipeline with its associated resources, so that any
/// compute operations can be easily setup and dispatched.
///
/// It is recommended to use [`ComputeBundleBuilder`] to create a compute bundle
///
/// ## Shader Format
///
/// The compute shader is suggested to be in the following form:
///
/// ```wgsl
/// override workgroup_size: u32;
///
/// @compute @workgroup_size(workgroup_size)
/// fn main(@builtin(global_invocation_id) id: vec3<u32>) {
///     let index = id.x;
///
///     if index >= arrayLength(&data) {
///         return;
///     }
///
///     // Do something with `data[index]`
/// }
/// ```
///
/// - `workgroup_size` is an overridable variable of type `u32`.
/// - The entry point function (here `main`) must have the `@compute` attribute and a
///   `@workgroup_size(workgroup_size)` attribute.
/// - The entry point function is suggested to have a parameter with
///   [`@builtin(global_invocation_id)`](https://www.w3.org/TR/WGSL/#global-invocation-id-builtin-value)
///   attribute to get the global invocation ID for indexing into the data.
#[derive(Debug, Clone)]
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
    /// Create the bind group at the given index.
    ///
    /// `index` refers to the index in [`ComputeBundle::bind_group_layouts`].
    ///
    /// Returns [`None`] if the `index` is out of bounds.
    ///
    /// As a good practice, if you are designing API for others to use, do not let the user
    /// create bind groups manually as they will have to make sure the binding resources match
    /// the layout.
    pub fn create_bind_group<'a>(
        &self,
        device: &wgpu::Device,
        index: usize,
        resources: impl IntoIterator<Item = wgpu::BindingResource<'a>>,
    ) -> Option<wgpu::BindGroup> {
        Some(ComputeBundle::create_bind_group_static(
            self.label.as_deref(),
            device,
            index,
            self.bind_group_layouts().get(index)?,
            resources,
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
    ///
    /// This corresponds to the `bind_group_layout_descriptors` provided
    /// when creating the compute bundle.
    pub fn bind_group_layouts(&self) -> &[wgpu::BindGroupLayout] {
        &self.bind_group_layouts
    }

    /// Get the compute pipeline.
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Dispatch the compute bundle for `count` instances with provided bind group.
    ///
    /// Each bind group in `bind_groups` corresponds to the bind group layout
    /// at the same index in [`ComputeBundle::bind_group_layouts`].
    pub fn dispatch_with_bind_groups<'a>(
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

impl ComputeBundle {
    /// Create a new compute bundle.
    ///
    /// `shader_source` requires an overridable variable `workgroup_size` of `u32`, see docs of
    /// [`ComputeBundle`].
    pub fn new<'a, 'b>(
        label: Option<&str>,
        device: &wgpu::Device,
        bind_group_layout_descriptors: impl IntoIterator<Item = &'a wgpu::BindGroupLayoutDescriptor<'a>>,
        resources: impl IntoIterator<Item = impl IntoIterator<Item = wgpu::BindingResource<'a>>>,
        compilation_options: wgpu::PipelineCompilationOptions,
        shader_source: wgpu::ShaderSource,
        entry_point: &str,
    ) -> Result<Self, ComputeBundleCreateError> {
        let this = ComputeBundle::new_without_bind_groups(
            label,
            device,
            bind_group_layout_descriptors,
            compilation_options,
            shader_source,
            entry_point,
        );

        let resources = resources.into_iter().collect::<Vec<_>>();

        if resources.len() != this.bind_group_layouts.len() {
            return Err(ComputeBundleCreateError::ResourceCountMismatch {
                resource_count: resources.len(),
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
            .zip(resources.into_iter())
            .enumerate()
            .map(|(i, (layout, resources))| {
                ComputeBundle::create_bind_group_static(this.label(), device, i, layout, resources)
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
        self.dispatch_with_bind_groups(encoder, self.bind_groups(), count);
    }

    /// Update the bind group at `index`.
    ///
    /// Returns [`Some`] of the previous bind group if it was updated,
    /// or [`None`] if the index is out of bounds.
    pub fn update_bind_group(
        &mut self,
        index: usize,
        bind_group: wgpu::BindGroup,
    ) -> Option<wgpu::BindGroup> {
        if index >= self.bind_groups.len() {
            return None;
        }

        Some(std::mem::replace(&mut self.bind_groups[index], bind_group))
    }

    /// Update the bind group at `index` with the provided binding resources.
    ///
    /// Returns [`Some`] of the previous bind group if it was updated,
    /// or [`None`] if the index is out of bounds.
    pub fn update_bind_group_with_binding_resources<'a>(
        &mut self,
        device: &wgpu::Device,
        index: usize,
        resources: impl IntoIterator<Item = wgpu::BindingResource<'a>>,
    ) -> Option<wgpu::BindGroup> {
        let bind_group = self.create_bind_group(device, index, resources)?;
        self.update_bind_group(index, bind_group)
    }

    /// Create a bind group statically.
    ///
    /// `index` is only for labeling.
    fn create_bind_group_static<'a>(
        label: Option<&str>,
        device: &wgpu::Device,
        index: usize,
        bind_group_layout: &wgpu::BindGroupLayout,
        resources: impl IntoIterator<Item = wgpu::BindingResource<'a>>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label_for_components!(label, format!("Bind Group {index}")).as_str()),
            layout: bind_group_layout,
            entries: &resources
                .into_iter()
                .enumerate()
                .map(|(i, resource)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource,
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
        compilation_options: wgpu::PipelineCompilationOptions,
        shader_source: wgpu::ShaderSource,
        entry_point: &str,
    ) -> Self {
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

        let constants = [
            &[("workgroup_size", workgroup_size as f64)],
            compilation_options.constants,
        ]
        .concat();

        let compilation_options = wgpu::PipelineCompilationOptions {
            constants: &constants,
            zero_initialize_workgroup_memory: compilation_options.zero_initialize_workgroup_memory,
        };

        log::debug!(
            "Creating {} pipeline",
            label.as_deref().unwrap_or("compute bundle"),
        );
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label_for_components!(label, "Pipeline").as_str()),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: compilation_options.clone(),
            cache: None,
        });

        log::info!("{} created", label.as_deref().unwrap_or("Compute Bundle"));

        Self {
            label: label.map(String::from),
            workgroup_size,
            bind_group_layouts,
            bind_groups: Vec::new(),
            pipeline,
        }
    }

    /// Dispatch the compute bundle for `count` instances.
    pub fn dispatch<'a>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        count: u32,
        bind_groups: impl IntoIterator<Item = &'a wgpu::BindGroup>,
    ) {
        self.dispatch_with_bind_groups(encoder, bind_groups, count);
    }
}

/// A builder for [`ComputeBundle`].
///
/// The shader is compiled using the WESL compiler,
///
/// The following fields should be set before calling [`ComputeBundleBuilder::build`] or
/// [`ComputeBundleBuilder::build_without_bind_groups`]:
/// - [`ComputeBundleBuilder::bind_group_layouts`]
/// - [`ComputeBundleBuilder::resolver`]
/// - [`ComputeBundleBuilder::entry_point`]
/// - [`ComputeBundleBuilder::main_shader`]
pub struct ComputeBundleBuilder<'a, R: wesl::Resolver = wesl::StandardResolver> {
    pub label: Option<&'a str>,
    pub bind_group_layouts: Vec<&'a wgpu::BindGroupLayoutDescriptor<'a>>,
    pub pipeline_compile_options: wgpu::PipelineCompilationOptions<'a>,
    pub entry_point: Option<&'a str>,
    pub main_shader: Option<wesl::ModulePath>,
    pub wesl_compile_options: wesl::CompileOptions,
    pub resolver: Option<R>,
    pub mangler: Box<dyn wesl::Mangler + Send + Sync + 'static>,
}

impl ComputeBundleBuilder<'_> {
    /// Create a new compute bundle builder.
    pub fn new() -> Self {
        Self {
            label: None,
            bind_group_layouts: Vec::new(),
            pipeline_compile_options: wgpu::PipelineCompilationOptions::default(),
            entry_point: None,
            main_shader: None,
            wesl_compile_options: wesl::CompileOptions::default(),
            resolver: None,
            mangler: Box::new(wesl::NoMangler),
        }
    }
}

impl<'a, R: wesl::Resolver> ComputeBundleBuilder<'a, R> {
    /// Set the label of the compute bundle.
    pub fn label(mut self, label: impl Into<&'a str>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add a [`wgpu::BindGroupLayoutDescriptor`].
    pub fn bind_group_layout(
        mut self,
        bind_group_layout: &'a wgpu::BindGroupLayoutDescriptor<'a>,
    ) -> Self {
        self.bind_group_layouts.push(bind_group_layout);
        self
    }

    /// Add [`wgpu::BindGroupLayoutDescriptor`]s.
    pub fn bind_group_layouts(
        mut self,
        bind_group_layouts: impl IntoIterator<Item = &'a wgpu::BindGroupLayoutDescriptor<'a>>,
    ) -> Self {
        self.bind_group_layouts.extend(bind_group_layouts);
        self
    }

    /// Set the [`wgpu::PipelineCompilationOptions`].
    pub fn pipeline_compile_options(
        mut self,
        compilation_options: wgpu::PipelineCompilationOptions<'a>,
    ) -> Self {
        self.pipeline_compile_options = compilation_options;
        self
    }

    /// Set the entry point of the compute shader.
    ///
    /// This should be the function name of the entry point in the compute shader, which is
    /// passed into [`wgpu::ComputePipelineDescriptor::entry_point`].
    pub fn entry_point(mut self, main: &'a str) -> Self {
        self.entry_point = Some(main);
        self
    }

    /// Set the main shader of the compute bundle.
    ///
    /// The shader is required to have an overridable variable `workgroup_size` of `u32`, which is
    /// set to the workgroup size of at the entry point of the compute pipeline.
    pub fn main_shader(self, main: wesl::ModulePath) -> ComputeBundleBuilder<'a, R> {
        ComputeBundleBuilder {
            label: self.label,
            bind_group_layouts: self.bind_group_layouts,
            pipeline_compile_options: self.pipeline_compile_options,
            entry_point: self.entry_point,
            main_shader: Some(main),
            wesl_compile_options: self.wesl_compile_options,
            resolver: self.resolver,
            mangler: self.mangler,
        }
    }

    /// Set the [`wesl::CompileOptions`].
    pub fn wesl_compile_options(mut self, options: wesl::CompileOptions) -> Self {
        self.wesl_compile_options = options;
        self
    }

    /// Set the [`wesl::Resolver`].
    pub fn resolver<S: wesl::Resolver>(self, resolver: S) -> ComputeBundleBuilder<'a, S> {
        ComputeBundleBuilder {
            label: self.label,
            bind_group_layouts: self.bind_group_layouts,
            pipeline_compile_options: self.pipeline_compile_options,
            entry_point: self.entry_point,
            main_shader: self.main_shader,
            wesl_compile_options: self.wesl_compile_options,
            resolver: Some(resolver),
            mangler: self.mangler,
        }
    }

    /// Set the [`wesl::Mangler`].
    pub fn mangler(
        self,
        mangler: impl wesl::Mangler + Send + Sync + 'static,
    ) -> ComputeBundleBuilder<'a, R> {
        ComputeBundleBuilder {
            label: self.label,
            bind_group_layouts: self.bind_group_layouts,
            pipeline_compile_options: self.pipeline_compile_options,
            entry_point: self.entry_point,
            main_shader: self.main_shader,
            wesl_compile_options: self.wesl_compile_options,
            resolver: self.resolver,
            mangler: Box::new(mangler),
        }
    }

    /// Build the compute bundle with bindings.
    pub fn build<'b>(
        self,
        device: &wgpu::Device,
        resources: impl IntoIterator<Item = impl IntoIterator<Item = wgpu::BindingResource<'a>>>,
    ) -> Result<ComputeBundle<wgpu::BindGroup>, ComputeBundleBuildError> {
        if self.bind_group_layouts.is_empty() {
            return Err(ComputeBundleBuildError::MissingBindGroupLayout);
        }

        let Some(resolver) = self.resolver else {
            return Err(ComputeBundleBuildError::MissingResolver);
        };

        let Some(entry_point) = self.entry_point else {
            return Err(ComputeBundleBuildError::MissingEntryPoint);
        };

        let Some(main_shader) = self.main_shader else {
            return Err(ComputeBundleBuildError::MissingMainShader);
        };

        let shader_source = wgpu::ShaderSource::Wgsl(
            wesl::compile_sourcemap(
                &main_shader.into(),
                &resolver,
                &self.mangler,
                &self.wesl_compile_options,
            )?
            .to_string()
            .into(),
        );

        ComputeBundle::new(
            self.label,
            device,
            self.bind_group_layouts.into_iter().collect::<Vec<_>>(),
            resources,
            self.pipeline_compile_options,
            shader_source,
            entry_point,
        )
        .map_err(Into::into)
    }

    /// Build the compute bundle without bind groups.
    pub fn build_without_bind_groups(
        self,
        device: &wgpu::Device,
    ) -> Result<ComputeBundle<()>, ComputeBundleBuildError> {
        if self.bind_group_layouts.is_empty() {
            return Err(ComputeBundleBuildError::MissingBindGroupLayout);
        }

        let Some(resolver) = self.resolver else {
            return Err(ComputeBundleBuildError::MissingResolver);
        };

        let Some(entry_point) = self.entry_point else {
            return Err(ComputeBundleBuildError::MissingEntryPoint);
        };

        let Some(main_shader) = self.main_shader else {
            return Err(ComputeBundleBuildError::MissingMainShader);
        };

        let shader_source = wgpu::ShaderSource::Wgsl(
            wesl::compile_sourcemap(
                &main_shader.into(),
                &resolver,
                &self.mangler,
                &self.wesl_compile_options,
            )?
            .to_string()
            .into(),
        );

        Ok(ComputeBundle::new_without_bind_groups(
            self.label,
            device,
            self.bind_group_layouts.into_iter().collect::<Vec<_>>(),
            self.pipeline_compile_options,
            shader_source,
            entry_point,
        ))
    }
}

impl Default for ComputeBundleBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

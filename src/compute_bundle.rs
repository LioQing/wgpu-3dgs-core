use crate::{BufferWrapper, Error, GaussianPod};

macro_rules! label_for_components {
    ($label:expr, $component:expr) => {
        Some(
            format!(
                "{} {}",
                $label.as_deref().unwrap_or("Compute Bundle"),
                $component,
            )
            .as_str(),
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
    pub fn create_bind_group<'a, G: GaussianPod>(
        &self,
        device: &wgpu::Device,
        index: usize,
        buffers: impl IntoIterator<Item = &'a (impl BufferWrapper + 'a)>,
    ) -> wgpu::BindGroup {
        ComputeBundle::create_bind_group_static(
            self.label(),
            device,
            index,
            &self.bind_group_layouts[index],
            buffers,
        )
    }

    /// Get the number of invocations in one workgroup.
    pub fn workgroup_size(&self) -> u32 {
        self.workgroup_size
    }

    /// Get the label.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

impl ComputeBundle {
    /// Create a new compute bundle.
    pub fn new<'a>(
        label: Option<&str>,
        device: &wgpu::Device,
        bind_group_layout_descriptors: impl IntoIterator<Item = &'a wgpu::BindGroupLayoutDescriptor<'a>>,
        resolver: &wesl::Wesl<impl wesl::Resolver>,
        buffers: impl IntoIterator<Item = impl IntoIterator<Item = &'a (impl BufferWrapper + 'a)>>,
    ) -> Result<Self, Error> {
        let this = ComputeBundle::new_without_bind_groups(
            label,
            device,
            bind_group_layout_descriptors,
            resolver,
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

    /// Dispatch the compute bundle for `count` instances.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: label_for_components!(self.label, "Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);

        for (i, group) in self.bind_groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }

        pass.dispatch_workgroups(count.div_ceil(self.workgroup_size()), 1, 1);
    }

    /// Create a bind group statically.
    fn create_bind_group_static<'a>(
        label: Option<&str>,
        device: &wgpu::Device,
        index: usize,
        bind_group_layout: &wgpu::BindGroupLayout,
        buffers: impl IntoIterator<Item = &'a (impl BufferWrapper + 'a)>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: label_for_components!(label, format!("Bind Group {index}")),
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
        resolver: &wesl::Wesl<impl wesl::Resolver>,
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
            label: label_for_components!(label, "Pipeline Layout"),
            bind_group_layouts: &bind_group_layouts.iter().collect::<Vec<_>>(),
            push_constant_ranges: &[],
        });

        log::debug!(
            "Creating {} shader module",
            label.as_deref().unwrap_or("compute bundle"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: label_for_components!(label, "Shader"),
            source: wgpu::ShaderSource::Wgsl(resolver.compile("main")?.to_string().into()),
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
            label: label_for_components!(label, "Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: compilation_options.clone(),
            cache: None,
        });

        log::info!("Compute Bundle created");

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
            label: label_for_components!(self.label, "Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);

        for (i, group) in bind_groups.into_iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }

        pass.dispatch_workgroups(count.div_ceil(self.workgroup_size()), 1, 1);
    }
}

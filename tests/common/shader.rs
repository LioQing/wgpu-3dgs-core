use wgpu::util::DeviceExt;

pub const SHADER_DIR: &str = "tests/common/shader";
pub const SHADER_ENTRY_POINT: &str = "main";

pub const ARRAY_MAP_ADD_MODULE_PATH: &str = "package::array_map_add";
pub const ARRAY_MAP_ADD_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<'static> =
    wgpu::BindGroupLayoutDescriptor {
        label: Some("Array Map Add Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    };
pub const ARRAY_MAP_ADD_SECOND_BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor<
    'static,
> = wgpu::BindGroupLayoutDescriptor {
    label: Some("Array Map Add Second Bind Group Layout"),
    entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }],
};
pub const ARRAY_MAP_ADD_ADDITIONAL_CONSTANT: u32 = 20;
pub const ARRAY_MAP_ADD_ADDITIONAL_CONSTANTS: &[(&str, f64)] = &[(
    "additional_constant",
    ARRAY_MAP_ADD_ADDITIONAL_CONSTANT as f64,
)];
pub const ARRAY_MAP_ADD_WESL_FEATURE_FLAGS: &[(&str, wesl::Feature)] = &[
    ("second_group", wesl::Feature::Enable),
    ("additional_constant", wesl::Feature::Enable),
];

pub const ARRAY_MAP_ADD_DEFAULT_DATA: [u32; 5] = [1, 2, 3, 4, 5];

pub const ARRAY_MAP_ADD_DEFAULT_ADDITIONAL_UNIFORM: u32 = 10;

pub mod given {
    use super::*;

    pub fn array_map_add_data(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Array Map Add Data Buffer"),
            contents: bytemuck::cast_slice(&ARRAY_MAP_ADD_DEFAULT_DATA),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn array_map_add_additional_uniform(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Array Map Add Additional Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ARRAY_MAP_ADD_DEFAULT_ADDITIONAL_UNIFORM]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }
}

@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@if(second_group) @group(1) @binding(0)
var<uniform> additional_value: u32;

@if(additional_constant)
override additional_constant: u32;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    if index >= arrayLength(&data) {
        return;
    }

    data[index] = data[index] + 1;

    @if(second_group) {
        data[index] = data[index] + additional_value;
    }

    @if(additional_constant) {
        data[index] = data[index] + additional_constant;
    }
}
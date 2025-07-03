override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
) {
    let index = workgroup_id.x * workgroup_size + local_invocation_id.x;

    if index >= arrayLength(&gaussians) {
        return;
    }

    {{main}}
}
override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn _entry_(
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let index = workgroup_id.x * workgroup_size + local_invocation_id.x;

    {{main}}
}
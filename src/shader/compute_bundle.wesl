// Compute the index for this invocation.
fn index(
    workgroup_id: vec3<u32>,
    workgroup_size: u32,
    local_invocation_id: vec3<u32>,
) -> u32 {
    return workgroup_id.x * workgroup_size + local_invocation_id.x;
}
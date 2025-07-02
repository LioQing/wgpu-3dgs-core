override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let index = wid.x * workgroup_size + lid.x;

    if index >= arrayLength(&gaussians) {
        return;
    }

    {{main}}
}
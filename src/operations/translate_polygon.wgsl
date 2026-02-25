@group(0) @binding(0)
var<storage, read_write> coordinates: array<i32>;

@compute @workgroup_size(64)
fn translate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let num_coords = arrayLength(&coordinates);
    if(index >= num_coords) {
        return;
    }

    if index % 2 == 0 { //Translate X coordinate.
        coordinates[index] += 1;
    } else { //Translate Y coordinate.
        coordinates[index] += 2;
    }
}
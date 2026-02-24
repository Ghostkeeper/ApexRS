@compute @workgroup_size(1)
fn main(@group(0) @binding(0) dataA: array<i32>, @group(0) @binding(2) result: array<i32>)
/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

struct ScaleFactors {
    x: f64,
    y: f64,
}

@group(0) @binding(0) var<uniform> scale_factors: ScaleFactors;

@group(0) @binding(1)
var<storage, read_write> coordinates: array<i32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let num_coords = arrayLength(&coordinates);
    if(index >= num_coords) {
        return;
    }

    if index % 2 == 0 { //Translate X coordinate.
        coordinates[index] = (coordinates[index] as f64 * scale_factors.x).round() as i32;
    } else { //Translate Y coordinate.
        coordinates[index] = (coordinates[index] as f64 * scale_factors.y).round() as i32;
    }
}
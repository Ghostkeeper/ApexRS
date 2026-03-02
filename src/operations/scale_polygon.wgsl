/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

/// The structure of the uniform buffer is a combination of two floats: The X and Y scale factors.
struct ScaleFactors {
    /// The scale factor in the X direction.
    x: f32,

    /// The scale factor in the Y direction.
    y: f32,
}
@group(0) @binding(0) var<uniform> scale_factors: ScaleFactors;

/// The structure of the first binding is an array of coordinates.
///
/// There should always be an even number of coordinates: one X, Y pair for each vertex of the
/// polygon to scale.
@group(0) @binding(1)
var<storage, read_write> coordinates: array<i32>;

/// Round fractional coordinate points to the nearest coordinate.
///
/// Rounding coordinates is done slightly non-standard in order to maintain better accuracy: It is
/// always rounded half-up. If we were to round half-away-from-zero, or round half-to-even, moving a
/// shape may cause its size to change.
///
/// # Arguments
/// * `coordinate` - The coordinate to round, representing as a floating-point value.
fn round(coordinate: f32) -> i32 {
    if fract(coordinate) >= 0.5 {
        return i32(ceil(coordinate));
    } else {
        return i32(floor(coordinate));
    }
}

/// Perform the scale operation on the polygon in-place.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let num_coords = arrayLength(&coordinates);
    if(index >= num_coords) {
        return;
    }

    if index % 2 == 0 { //Scale X coordinate.
        coordinates[index] = round(f32(coordinates[index]) * scale_factors.x);
    } else { //Scale Y coordinate.
        coordinates[index] = round(f32(coordinates[index]) * scale_factors.y);
    }
}
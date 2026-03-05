/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

struct EmulatedF64 {
    high: f32,
    low: f32,
    _pad_a: f32,
    _pad_b: f32,
}

fn split(a: f32) -> EmulatedF64 {
    let splitter = 4097.0;
    let t = a * splitter;
    let high = t - (t - a);
    let low = a - high;
    return EmulatedF64(high, low, 0.0, 0.0);
}

fn split_i32(a: i32) -> EmulatedF64 {
    let high = f32(a);
    let low = f32(a - i32(high));
    return EmulatedF64(high, low, 0.0, 0.0);
}

fn twoprod(a: f32, b: f32) -> EmulatedF64 {
    let p = a * b;
    let a_split = split(a);
    let b_split = split(b);
    let err = (a_split.high * b_split.high - p) + a_split.high * b_split.low + a_split.low * b_split.high + a_split.low * b_split.low;
    return EmulatedF64(p, err, 0.0, 0.0);
}

fn quicktwosum(a: f32, b: f32) -> EmulatedF64 {
    let s = a + b;
    let e = b - (s - a);
    return EmulatedF64(s, e, 0.0, 0.0);
}

fn mul(lhs: EmulatedF64, rhs: EmulatedF64) -> EmulatedF64 {
    var p = twoprod(lhs.high, rhs.high);
    p.low += lhs.high * rhs.low;
    p.low += lhs.low * rhs.high;
    return quicktwosum(p.high, p.low);
}

fn round(value: EmulatedF64) -> i32 {
	let high_part = i32(select(floor(value.high), ceil(value.high), value.high < 0.0));
	let low_part = i32(select(floor(value.low), ceil(value.low), value.low < 0.0));
	let remainders = i32((f32(high_part) - value.high) + (f32(low_part) - value.low) + 0.5);
	return high_part + low_part + remainders;
}

/// The structure of the uniform buffer is a combination of two floats: The X and Y scale factors.
struct ScaleFactors {
    /// The scale factor in the X direction.
    x: EmulatedF64,

    /// The scale factor in the Y direction.
    y: EmulatedF64,
}
@group(0) @binding(0) var<uniform> scale_factors: ScaleFactors;

/// The structure of the first binding is an array of coordinates.
///
/// There should always be an even number of coordinates: one X, Y pair for each vertex of the
/// polygon to scale.
@group(0) @binding(1)
var<storage, read_write> coordinates: array<i32>;

/// Perform the scale operation on the polygon in-place.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let num_coords = arrayLength(&coordinates);
    if(index >= num_coords) {
        return;
    }

    let to_f64 = split_i32(coordinates[index]);
    if index % 2 == 0 { //Scale X coordinate.
        coordinates[index] = round(mul(to_f64, scale_factors.x));
    } else { //Scale Y coordinate.
        coordinates[index] = round(mul(to_f64, scale_factors.y));
    }
}
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
}

fn split(a: f32) -> EmulatedF64 {
	let splitter = 4097.0;
	let t = a * splitter;
	let high = t - (t - a);
	let low = a - high;
	return EmulatedF64(high, low);
}

fn split_i32(a: i32) -> EmulatedF64 {
	let high = f32(a);
	let low = f32(a - i32(high));
	return EmulatedF64(high, low);
}

fn twoprod(a: f32, b: f32) -> EmulatedF64 {
	let p = a * b;
	let a_split = split(a);
	let b_split = split(b);
	let err = (a_split.high * b_split.high - p) + a_split.high * b_split.low + a_split.low * b_split.high + a_split.low * b_split.low;
	return EmulatedF64(p, err);
}

fn twosum(a: f32, b: f32) -> EmulatedF64 {
	let s = a + b;
	let v = s - a;
	let e = (a - (s - v)) + (b - v);
	return EmulatedF64(s, e);
}

fn quicktwosum(a: f32, b: f32) -> EmulatedF64 {
	let s = a + b;
	let e = b - (s - a);
	return EmulatedF64(s, e);
}

fn mul(lhs: EmulatedF64, rhs: EmulatedF64) -> EmulatedF64 {
	var p = twoprod(lhs.high, rhs.high);
	p.low += lhs.high * rhs.low;
	p.low += lhs.low * rhs.high;
	return quicktwosum(p.high, p.low);
}

fn add(lhs: EmulatedF64, rhs: EmulatedF64) -> EmulatedF64 {
	var s = twosum(lhs.high, rhs.high);
	let t = twosum(lhs.low, rhs.low);
	s.low += t.high;
	s = quicktwosum(s.high, s.low);
	s.low += t.low;
	return quicktwosum(s.high, s.low);
}

fn sub(lhs: EmulatedF64, rhs: EmulatedF64) -> EmulatedF64 {
	let negative = EmulatedF64(-rhs.high, -rhs.low);
	return add(lhs, negative);
}

fn round(value: EmulatedF64) -> i32 {
	let half = EmulatedF64(0.5, 0.0);
	let halfup = add(value, half);
	let high_part = i32(halfup.high);
	let high_frac = halfup.high % 1.0;
	let low_part = i32(halfup.low);
	let low_frac = halfup.low % 1.0;
	let remainders = i32(floor(high_frac + low_frac));
	return high_part + low_part + remainders;
}

@group(0) @binding(0) var<uniform> rotation_trigonometry: vec4<f32>;

struct Vertex {
	x: i32,
	y: i32,
}

/// The structure of the first binding is an array of coordinates.
///
/// There should always be an even number of coordinates: one X, Y pair for each vertex of the
/// polygon to scale.
@group(0) @binding(1)
var<storage, read_write> vertices: array<Vertex>;

/// Perform the scale operation on the polygon in-place.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let index = global_id.x;
	let num_verts = arrayLength(&vertices);
	if(index >= num_verts) {
		return;
	}

	let vertex = vertices[index];
	let x = split_i32(vertices[index].x);
	let y = split_i32(vertices[index].y);
	let sine = EmulatedF64(rotation_trigonometry.x, rotation_trigonometry.y);
	let cosine = EmulatedF64(rotation_trigonometry.z, rotation_trigonometry.w);
	vertices[index].x = round(sub(mul(x, cosine), mul(y, sine)));
	vertices[index].y = round(add(mul(x, sine), mul(y, cosine)));
}
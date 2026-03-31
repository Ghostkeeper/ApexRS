/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

/// The structure of the uniform buffer is a combination of two integers: The delta-X and delta-Y.
struct TranslationVector {
	/// The delta-X to move the polygon in the X direction.
	x: i32,

	/// The delta-Y to move the polygon in the Y direction.
	y: i32,
}
@group(0) @binding(0) var<uniform> translation_vector: TranslationVector;

/// One corner of the polygon.
struct Vertex {
	/// The X coordinate of the vertex.
	x: i32,

	/// The Y coordinate of the vertex.
	y: i32,
}

/// The structure of the first binding is an array of coordinates.
///
/// There should always be an even number of coordinates: one X, Y pair for each vertex of the
/// polygon to scale.
@group(0) @binding(1)
var<storage, read_write> vertices: array<Vertex>;

/// Perform the translate operation on the polygon in-place.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let index = global_id.x;
	let num_verts = arrayLength(&vertices);
	if(index >= num_verts) {
		return;
	}

	vertices[index].x += translation_vector.x;
	vertices[index].y += translation_vector.y;
}
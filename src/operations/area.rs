/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! This module contains the implementations of operations to calculate the area of geometric
//! objects.

use embed_doc_image::embed_doc_image; //Documenting with diagrams.
use rayon::current_num_threads; //For multi-threaded implementations.
use rayon::iter::ParallelIterator; //For multi-threaded implementations.
use rayon::prelude::ParallelSlice; //For multi-threaded implementations.
use std::cmp;
use std::sync::LazyLock;
use wgpu::{include_wgsl, ShaderModule}; //For loading the area GPU kernel.

use crate::Area; //Outputting the area gives this Area object.
use crate::Polygon; //Get the area of polygons.
use crate::detail::emulated_i64::EmulatedI64;
use crate::detail::gpu::GPU; //To perform calculations on the GPU.

/// Calculate the area of a polygon.
///
/// # Arguments
/// - `polygon` The polygon to calculate the area of.
///
/// # Examples
/// ```
/// use apex::{Point2D, Polygon};
/// //Create a triangular polygon.
/// let poly = Polygon::from_iter([
/// 	Point2D { x: 0, y: 0 },
/// 	Point2D { x: 100, y: 0 },
/// 	Point2D { x: 67, y: 100 },
/// ]);
/// //Get the area of it.
/// let area = apex::operations::area::area_polygon_st(&poly);
/// assert_eq!(area, 5000);
/// ```
///
/// # Implementation
/// This uses the shoelace formula to compute the area. The shoelace formula sums the areas of the
/// individual triangles formed by two adjacent vertices and the coordinate origin.
///
/// To calculate the area of a triangle with one vertex on the origin, we'll calculate the area of a
/// parallelogram formed by the original triangle and that triangle mirrored around the line segment
/// we're calculating the area for. Visualise this:
///
/// ![A parallelogram with the original line segment as diagonal, one vertex on the 0,0 coordinate, and the last vertex mirrored past that diagonal.][shoelace_algorithm_parallelogram]
///
/// The area of the parallelogram can be visualised by starting with a rectangle that encloses the
/// original triangle like this. The green areas fall outside of the parallelogram and shouldn't be
/// counted towards its area.
///
/// ![A bounding box surrounds the original triangle, and the area inside the box but outside the parallelogram is marked in green, forming two right trianges with dimensions x1, y1 and x2, y2.][shoelace_algorithm_rectangle_overlay]
///
/// The green areas are not part of the parallelogram here, but they can be shifted towards the
/// missing part that falls outside of the rectangle like this.
///
/// ![The green triangles are shifted to fill the areas of the parallelogram that were outside of the box, creating a rectangular area in the top right, part of which being overlap and the rest being outside of the parallelogram.][shoelace_algorithm_multiple_rectangles]
///
/// This forms a second rectangle, in this case a smaller one in the upper right hand corner. The
/// two green triangles partially overlap and go partially outside of the parallelogram we're trying
/// to get the area of. The part that is overlap plus the part that goes outside of the
/// parallelogram together forms an area of x₁ ⋅ y₂.
///
/// The total area of the parallelogram then becomes the area of the rectangle formed by x₂ ⋅ y₁
/// minus the area formed by the other rectangle formed by x₁ ⋅ y₂. In other words, the area of the
/// parallelogram is x₂ ⋅ y₁ - x₁ ⋅ y₂. This needs to be divided by two to arrive at the area of the
/// triangle. The surface area of a polygon is the sum of all these triangles. This is the shoelace
/// formula.
///
/// In this implementation, the areas of these parallelograms are calculated in sequence and summed
/// together to get the area of the polygon.
#[embed_doc_image("shoelace_algorithm_parallelogram", "doc/images/shoelace_algorithm_parallelogram.svg")]
#[embed_doc_image("shoelace_algorithm_rectangle_overlay", "doc/images/shoelace_algorithm_rectangle_overlay.svg")]
#[embed_doc_image("shoelace_algorithm_multiple_rectangles", "doc/images/shoelace_algorithm_multiple_rectangles.svg")]
pub fn area_polygon_st(polygon: &Polygon) -> Area {
	let vertices = polygon.host_vertices();
	if vertices.len() < 3 {
		return 0;
	}
	let mut area: Area = 0;
	let mut previous = vertices.len() - 1;
	for vertex in 0..vertices.len() {
		area += vertices[previous].x as Area * vertices[vertex].y as Area - vertices[previous].y as Area * vertices[vertex].x as Area;
		previous = vertex;
	}
	area / 2
}

/// Calculate the area of a polygon.
///
/// # Arguments
/// - `polygon` The polygon to calculate the area of.
///
/// # Examples
/// ```
/// use apex::{Point2D, Polygon};
/// //Create a triangular polygon.
/// let poly = Polygon::from_iter([
/// 	Point2D { x: 0, y: 0 },
/// 	Point2D { x: 100, y: 0 },
/// 	Point2D { x: 67, y: 100 },
/// ]);
/// //Get the area of it.
/// let area = apex::operations::area::area_polygon_mt(&poly);
/// assert_eq!(area, 5000);
/// ```
///
/// # Implementation
/// This uses the shoelace formula to compute the area. The shoelace formula sums the areas of the
/// individual triangles formed by two adjacent vertices and the coordinate origin.
///
/// To calculate the area of a triangle with one vertex on the origin, we'll calculate the area of a
/// parallelogram formed by the original triangle and that triangle mirrored around the line segment
/// we're calculating the area for. Visualise this:
///
/// ![A parallelogram with the original line segment as diagonal, one vertex on the 0,0 coordinate, and the last vertex mirrored past that diagonal.][shoelace_algorithm_parallelogram]
///
/// The area of the parallelogram can be visualised by starting with a rectangle that encloses the
/// original triangle like this. The green areas fall outside of the parallelogram and shouldn't be
/// counted towards its area.
///
/// ![A bounding box surrounds the original triangle, and the area inside the box but outside the parallelogram is marked in green, forming two right trianges with dimensions x1, y1 and x2, y2.][shoelace_algorithm_rectangle_overlay]
///
/// The green areas are not part of the parallelogram here, but they can be shifted towards the
/// missing part that falls outside of the rectangle like this.
///
/// ![The green triangles are shifted to fill the areas of the parallelogram that were outside of the box, creating a rectangular area in the top right, part of which being overlap and the rest being outside of the parallelogram.][shoelace_algorithm_multiple_rectangles]
///
/// This forms a second rectangle, in this case a smaller one in the upper right hand corner. The
/// two green triangles partially overlap and go partially outside of the parallelogram we're trying
/// to get the area of. The part that is overlap plus the part that goes outside of the
/// parallelogram together forms an area of x₁ ⋅ y₂.
///
/// The total area of the parallelogram then becomes the area of the rectangle formed by x₂ ⋅ y₁
/// minus the area formed by the other rectangle formed by x₁ ⋅ y₂. In other words, the area of the
/// parallelogram is x₂ ⋅ y₁ - x₁ ⋅ y₂. This needs to be divided by two to arrive at the area of the
/// triangle. The surface area of a polygon is the sum of all these triangles. This is the shoelace
/// formula.
///
/// In this implementation, the vertices are broken up in non-overlapping chunks. Each chunk is
/// processed in parallel, with the line segments in the chunk being summed up in sequence in a
/// thread. The line segments that connect these chunks together are calculated afterwards and added
/// to the sum of the chunks.
#[embed_doc_image("shoelace_algorithm_parallelogram", "doc/images/shoelace_algorithm_parallelogram.svg")]
#[embed_doc_image("shoelace_algorithm_rectangle_overlay", "doc/images/shoelace_algorithm_rectangle_overlay.svg")]
#[embed_doc_image("shoelace_algorithm_multiple_rectangles", "doc/images/shoelace_algorithm_multiple_rectangles.svg")]
pub fn area_polygon_mt(polygon: &Polygon) -> Area {
	let vertices = polygon.host_vertices();
	if vertices.len() < 3 {
		return 0;
	}
	//Create chunks of contiguous vertices that are fast to access in sequence by one thread.
	//We'll compute the area sum of each chunk in parallel.
	let chunk_size = cmp::max(1000, vertices.len() / current_num_threads());
	let mut result: Area = vertices.par_chunks(chunk_size).map(
		|slice| {
			let mut sum = 0;
			for index in 1..slice.len() {
				sum += slice[index - 1].x as Area * slice[index].y as Area - slice[index - 1].y as Area * slice[index].x as Area;
			}
			sum
		}
	).sum();
	//The seams between chunks were not yet calculated. Calculate those afterwards in single-threaded mode.
	for index in (chunk_size..vertices.len()).step_by(chunk_size) {
		result += vertices[index - 1].x as Area * vertices[index].y as Area - vertices[index - 1].y as Area * vertices[index].x as Area;
	}
	result += vertices[vertices.len() - 1].x as Area * vertices[0].y as Area - vertices[vertices.len() - 1].y as Area * vertices[0].x as Area;
	result / 2
}

/// The shader for calculating the area of polygons on the GPU.
static AREA_POLYGON_SHADER: LazyLock<ShaderModule> = LazyLock::new(|| {
	GPU.device.create_shader_module(include_wgsl!("area_polygon.wgsl"))
});

/// Calculate the area of a polygon.
///
/// # Arguments
/// - `polygon` The polygon to calculate the area of.
///
/// # Examples
/// ```
/// use apex::{Point2D, Polygon};
/// //Create a triangular polygon.
/// let poly = Polygon::from_iter([
/// 	Point2D { x: 0, y: 0 },
/// 	Point2D { x: 100, y: 0 },
/// 	Point2D { x: 67, y: 100 },
/// ]);
/// //Get the area of it.
/// let area = apex::operations::area::area_polygon_gpu(&poly);
/// assert_eq!(area, 5000);
/// ```
///
/// # Implementation
/// This uses the shoelace formula to compute the area. The shoelace formula sums the areas of the
/// individual triangles formed by two adjacent vertices and the coordinate origin.
///
/// To calculate the area of a triangle with one vertex on the origin, we'll calculate the area of a
/// parallelogram formed by the original triangle and that triangle mirrored around the line segment
/// we're calculating the area for. Visualise this:
///
/// ![A parallelogram with the original line segment as diagonal, one vertex on the 0,0 coordinate, and the last vertex mirrored past that diagonal.][shoelace_algorithm_parallelogram]
///
/// The area of the parallelogram can be visualised by starting with a rectangle that encloses the
/// original triangle like this. The green areas fall outside of the parallelogram and shouldn't be
/// counted towards its area.
///
/// ![A bounding box surrounds the original triangle, and the area inside the box but outside the parallelogram is marked in green, forming two right trianges with dimensions x1, y1 and x2, y2.][shoelace_algorithm_rectangle_overlay]
///
/// The green areas are not part of the parallelogram here, but they can be shifted towards the
/// missing part that falls outside of the rectangle like this.
///
/// ![The green triangles are shifted to fill the areas of the parallelogram that were outside of the box, creating a rectangular area in the top right, part of which being overlap and the rest being outside of the parallelogram.][shoelace_algorithm_multiple_rectangles]
///
/// This forms a second rectangle, in this case a smaller one in the upper right hand corner. The
/// two green triangles partially overlap and go partially outside of the parallelogram we're trying
/// to get the area of. The part that is overlap plus the part that goes outside of the
/// parallelogram together forms an area of x₁ ⋅ y₂.
///
/// The total area of the parallelogram then becomes the area of the rectangle formed by x₂ ⋅ y₁
/// minus the area formed by the other rectangle formed by x₁ ⋅ y₂. In other words, the area of the
/// parallelogram is x₂ ⋅ y₁ - x₁ ⋅ y₂. This needs to be divided by two to arrive at the area of the
/// triangle. The surface area of a polygon is the sum of all these triangles. This is the shoelace
/// formula.
///
/// This implementation calculates the areas of these parallelograms for each line segment in
/// parallel using the massive concurrency of the GPU. The areas are then summed using a
/// tree-reduction in the GPU to arrive at a single summed area.
pub fn area_polygon_gpu(polygon: &Polygon) -> Area {
	let parameters = [0 as Area];
	let uniform_buffer = bytemuck::cast_slice(&parameters);
	let result_bytes = polygon.execute_gpu_kernel(&AREA_POLYGON_SHADER, uniform_buffer, uniform_buffer.len());
	let binding = result_bytes.unwrap();
	let output = bytemuck::cast_slice::<u8, EmulatedI64>(&binding.as_slice());
	<EmulatedI64 as Into<i64>>::into(output[0]) / 2
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::test::data::polygon;
	use std::f64::consts::{PI, TAU}; //To calculate the area of regular polygons.
	use test_case::test_case;

	/// Test getting the area of an empty polygon.
	#[test]
	fn area_polygon_empty() {
		let poly = Polygon::new();
		assert_eq!(area_polygon_st(&poly), 0, "An empty polygon has no area.");
		assert_eq!(area_polygon_mt(&poly), 0, "An empty polygon has no area.");
	}

	/// Test getting the area of a 1000x1000 square.
	#[test_case(polygon::square_1000(); "Basic")]
	#[test_case(polygon::square_1000_negative_x(); "Negative X")]
	#[test_case(polygon::square_1000_negative_y(); "Negative Y")]
	#[test_case(polygon::square_1000_negative_xy(); "Negative X and Y")]
	#[test_case(polygon::square_1000_centred(); "Centred")]
	fn area_polygon_square(poly: Polygon) {
		assert_eq!(area_polygon_st(&poly), 1000 * 1000, "A 1000 by 1000 square");
		assert_eq!(area_polygon_mt(&poly), 1000 * 1000, "A 1000 by 1000 square");
	}

	/// Test getting the area of a triangle.
	#[test]
	fn area_polygon_triangle() {
		let poly = polygon::triangle_1000();
		assert_eq!(area_polygon_st(&poly), 1000 * 1000 / 2, "A triangle with base 1000, height 1000");
		assert_eq!(area_polygon_mt(&poly), 1000 * 1000 / 2, "A triangle with base 1000, height 1000");
	}

	/// Test getting the area of a concave shape.
	#[test]
	fn area_polygon_concave() {
		let poly = polygon::arrowhead(); //The arrowhead is a concave shape.
		assert_eq!(area_polygon_st(&poly), 1000 * 1000 / 2 - 1000 * 500 / 2, "The 1000x1000 triangle with a 1000x500 triangle cut out");
		assert_eq!(area_polygon_mt(&poly), 1000 * 1000 / 2 - 1000 * 500 / 2, "The 1000x1000 triangle with a 1000x500 triangle cut out");
	}

	/// Test getting the area of various degenerate shapes.
	///
	/// This tests how it deals with these degeneracies, and also whether it won't crash on such
	/// inputs.
	#[test_case(polygon::zero_width(); "Zero width")]
	#[test_case(polygon::degenerate_line(); "Line")]
	#[test_case(polygon::degenerate_point(); "Point")]
	fn area_polygon_degenerate(poly: Polygon) {
		assert_eq!(area_polygon_st(&poly), 0, "Degenerate shapes have no surface area.");
		assert_eq!(area_polygon_mt(&poly), 0, "Degenerate shapes have no surface area.");
	}

	/// Test getting the area of a self-intersecting shape with both positive and negative regions.
	#[test]
	fn area_polygon_self_intersecting() {
		let poly = polygon::hourglass(); //The hourglass is a self-intersecting shape.
		assert_eq!(area_polygon_st(&poly), 0, "The positive areas cancel out the negative areas.");
		assert_eq!(area_polygon_mt(&poly), 0, "The positive areas cancel out the negative areas.");
	}

	/// Test getting the area of a regular polygon that approximates a circle.
	///
	/// The test involves many vertices, which tests cases where we break up the polygon into
	/// multiple chunks and have to connect those together.
	///
	/// The testing polygon approximates a circle. The ground truth for the area is calculated with
	/// the formula for a regular polygon. For a regular polygon with N sides and outer radius r,
	/// the area is exactly: ½N ⋅ r² ⋅ sin(2π/N).
	#[test]
	fn area_polygon_circle() {
		const NUM_VERTICES: usize = 1000_000;
		let poly = polygon::regular(NUM_VERTICES);
		let radius = poly.vertex(0).x as Area;
		let ground_truth = ((NUM_VERTICES as Area * radius * radius) as f64 * ((TAU / NUM_VERTICES as f64).sin() / 2.0)).round() as Area;
		let error_margin = ((NUM_VERTICES as f64).sqrt() / NUM_VERTICES as f64 / 6.0 * (PI * (radius * radius) as f64 - PI * ((radius - 1) * (radius - 1)) as f64)) as Area;

		assert!((area_polygon_st(&poly) - ground_truth).abs() <= error_margin, "The area is equal to the ground truth (with an allowed error margin).");
		assert!((area_polygon_mt(&poly) - ground_truth).abs() <= error_margin, "The area is equal to the ground truth (with an allowed error margin).");
	}
}
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
use std::cmp;
use rayon::current_num_threads; //For multi-threaded implementations.
use rayon::iter::ParallelIterator; //For multi-threaded implementations.
use rayon::prelude::ParallelSlice; //For multi-threaded implementations.

use crate::Area; //Outputting the area gives this Area object.
use crate::Polygon; //Get the area of polygons.

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
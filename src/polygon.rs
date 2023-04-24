/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines the Polygon struct.

use crate::Area; //To return the polygon's surface area.
use crate::Convexity; //To return the polygon's convexity.
use crate::Coordinate;
use crate::Point2D; //The vertices of the Polygon are Point2D.
use crate::Shape2D; //This is a 2D shape.
use crate::TwoDimensional; //This is a two-dimensional object.

/// A plane figure consisting of a single contour of straight line segments.
///
/// This is a closed shape, represented by a list of vertices in 2D. Between every two adjacent
/// vertices, as well as between the first and last vertices, is an edge. These edges together form
/// a closed shape that is the contents of the polygon.
///
/// Since the shape only has a single closed polygonal chain, it cannot have multiple boundaries,
/// like a doughnut-shape with a hole inside. This would be a multi-polygon. However the polygon may
/// be self-intersecting. It does not need to be a simple polygon. Operations on the polygon are
/// expected to deal with all such kinds of polygons correctly.
///
/// The polygon does not publicly have a start or end point. However its data model must start
/// somewhere and iterating over the vertices must choose a vertex to start and end at. Operations
/// on the polygons should behave the same regardless of where the polygon starts its iteration.
///
/// If the vertices of the polygon are winding counter-clockwise, the polygon is positive. Otherwise
/// it is negative.
pub struct Polygon {
	/// The vertices that form the closed polygonal chain around this polygon.
	///
	/// These vertices are not publicly accessible, since access to the most up-to-date version may
	/// require a sync from the GPU to the CPU.
	vertices: Vec<Point2D>
}

impl TwoDimensional for Polygon {
	fn translate(&mut self, dx: Coordinate, dy: Coordinate) {
		for vertex in &mut self.vertices {
			vertex.translate(dx, dy);
		}
	}
}

impl Shape2D for Polygon {
	fn area(&self) -> Area {
		return 0; //TODO: Implement.
	}

	fn convexity(&self) -> Convexity {
		return Convexity::UNKNOWN; //TODO: Implement.
	}
}
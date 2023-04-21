/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines a struct that represents single points in a 2-dimensional space.

use crate::Area; //To implement Shape2D.
use crate::Coordinate; //The position of the point is stored with coordinates.
use crate::TwoDimensional; //This point is in two-dimensional space.
use crate::Shape2D; //A point is a shape, with a bounded (zero) area.

/// Specifies a point in 2D space.
///
/// The two dimensions are called X and Y, by convention.
///
/// The point can be considered a degenerate shape. It has no surface area or width or height. It
/// collides with other geometry only when the borders of the objects are considered.
pub struct Point2D {
	/// The projection of this point on the X dimension.
	pub x: Coordinate,

	/// The projection of this point on the Y dimension.
	pub y: Coordinate,
}

impl Shape2D for Point2D {
	fn area(&self) -> Area {
		return 0; //A point has no area.
	}
}

impl TwoDimensional for Point2D {
	fn translate(&mut self, dx: Coordinate, dy: Coordinate) {
		self.x += dx;
		self.y += dy;
	}
}
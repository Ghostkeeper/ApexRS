/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines an enum for different types of convexity.

/// These are the possible states of convexity that a shape can have.
#[derive(Debug, Eq, PartialEq)]
pub enum Convexity {
	/// The convexity has not yet been computed. This should never be returned as a result of a
	/// public function.
	UNKNOWN,

	/// The shape is convex, meaning that any straight line segment through the inside of the shape
	/// will remain inside of the shape.
	CONVEX,

	/// The shape is concave, meaning that there are straight line segments with two endpoints
	/// inside the shape that pass out of the bounds of the shape.
	CONCAVE,

	/// The shape is degenerate, meaning there are parts of the shape with zero width, negative
	/// parts, or the shape has no area.
	DEGENERATE,
}
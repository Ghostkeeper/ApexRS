/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

use crate::Area; //To return the area of the shape.
use crate::Convexity; //To return the convexity of the shape.
use crate::TwoDimensional; //All Shape2Ds are two-dimensional.

/// A trait for finitely-bounded shapes in a 2D space.
pub trait Shape2D : TwoDimensional {
	/// Get the surface area of this shape.
	///
	/// It is possible that this area is negative. Shapes can be negative shapes, representing holes
	/// in other shapes.
	///
	/// # Result
	/// The surface area of this shape.
	fn area(&self) -> Area;

	/// Get the convexity of this shape.
	///
	/// A shape is convex if and only if all line segments starting and ending inside of the shape
	/// are wholly inside of the shape. That is, there are no line segments that start and end
	/// inside the shape which pass partially outside of the shape. If there are, the shape is
	/// concave.
	///
	/// If the shape is degenerate, degenerate convexity will be returned.
	///
	/// # Result
	/// The convexity of the shape.
	fn convexity(&self) -> Convexity;
}
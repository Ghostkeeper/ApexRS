/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

use crate::{Area, TwoDimensional};

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
}
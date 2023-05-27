/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

use crate::Coordinate; //To transform objects across the two-dimensional space.

/// This trait is for geometrical objects that are in a two-dimensional space.
///
/// The traits provides operations that are available for all two-dimensional geometries.
pub trait TwoDimensional {
	/// Move the object across the two-dimensional space.
	///
	/// This causes the position of the object to change, but doesn't otherwise transform it. The
	/// object is not rotated, scaled or deformed in any way.
	///
	/// # Arguments
	/// * `dx` - How far to move the object in the X direction. Use a positive number to increase
	/// the X position, or a negative number to reduce the X position.
	/// * `dy` - How far to move the object in the Y direction. Use a positive number to increase
	/// the Y position, or a negative number to reduce the Y position.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, TwoDimensional};
	/// let mut point = Point2D{ x: 100, y: 500 }; //Create a two-dimensional object, such as Point2D.
	/// point.translate(50, -130); //The point's position is now [150, 370].
	/// ```
	fn translate(&mut self, dx: Coordinate, dy: Coordinate);
}
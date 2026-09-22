/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines a structure that represents a ray that lies on a 2-dimensional plane.

use crate::Point2D;

/// A geometric object that represents a ray, a one-dimensional line in a 2-dimensional plane.
///
/// A ray, otherwise called a half-line, is a part of a line which starts in a point, and extends to
/// infinity on the other side. The starting point where the line does not extend to infinity is
/// called the "initial" point. Other than that it also has a two-dimensional direction vector.
pub struct Ray {
	/// The endpoint of the ray.
	///
	/// This defines the point where the ray ends. It is infinite on the other side of the ray.
	pub initial: Point2D,

	/// The direction vector of the ray.
	///
	/// The direction vector is implemented as a point. It is not (necessarily) a unit vector,
	/// because if it did the integer coordinate system would only allow the line to have one of the
	/// four cardinal directions.
	///
	/// The direction vector can be taken relative to the initial point. Adding the direction vector
	/// to the initial point points to a second point that is also on the ray.
	pub direction: Point2D,
}

impl Ray {
	/// Find the opposite ray of this ray.
	///
	/// The opposite ray is the ray that starts in the same place, but has an opposite direction.
	/// Combining a ray with its opposite ray forms a line, infinite on both sides. The opposite of
	/// the opposite ray is the original ray again.
	///
	/// # Returns
	/// The opposite ray of this ray.
	pub fn opposite(self) -> Ray {
		Ray {
			initial: self.initial,
			direction: -self.direction, //Opposite direction.
		}
	}
}
/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines an enum for ways in which geometric objects may intersect with each other.

/// Ways in which geometric objects can intersect with other objects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Intersecting {
	/// The boundaries of the objects cross each other fully.
	INTERSECTING,

	/// The geometric objects are not intersecting and fully separated.
	///
	/// There is space between the objects.
	SEPARATE,

	/// The geometric objects touch, but the boundaries do not fully intersect.
	EDGE,
}
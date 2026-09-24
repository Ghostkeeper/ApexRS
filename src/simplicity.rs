/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines an enum for different ways in which geometric objects can be simple or not.

/// These are the possible states of simplicity that a geometric object can have.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Simplicity {
	/// The object is simple, meaning that the bounds of the object do not intersect or hit each
	/// other.
	SIMPLE,

	/// The object is complex, meaning that the bounds of the object properly intersect each other.
	///
	/// If the bounds properly intersect and also (in other places) touch each other without
	/// intersecting, the simplicity is considered `COMPLEX` rather than `EDGE`.
	COMPLEX,

	/// The simplicity of the object is an edge case, meaning that the bounds of the object touch
	/// each other, but do not intersect.
	EDGE,
}
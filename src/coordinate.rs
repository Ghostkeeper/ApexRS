/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! These are the types used to represent coordinate and related data in Apex.

/// The type used to store coordinates in space.
///
/// This type is an integer type rather than a floating point type, so no partial unit coordinates
/// are possible. This is intended to make rounding errors more predictable so that they can be
/// accounted for in geometric algorithms, to minimise their effect.
///
/// The type is 32-bits to allow for single-width entries in graphics card integer processors.
/// Anything else kills performance.
pub type Coordinate = i32;
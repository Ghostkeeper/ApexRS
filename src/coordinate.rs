/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
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

/// Round fractional coordinate points to the nearest coordinate.
///
/// Rounding coordinates is done slightly non-standard in order to maintain better accuracy: It is
/// always rounded half-up. If we were to round half-away-from-zero, or round half-to-even, moving a
/// shape may cause its size to change.
///
/// # Arguments
/// * `coordinate` - The coordinate to round, representing as a floating-point value.
///
/// # Examples
/// ```
/// use apex::coordinate::round;
/// let close_down = round(3.1);
/// assert_eq!(close_down, 3);
/// let close_up = round(2.9);
/// assert_eq!(close_up, 3);
/// let half_up_even = round(3.5);
/// assert_eq!(half_up_even, 4);
/// let half_up_odd = round(4.5);
/// assert_eq!(half_up_odd, 5);
/// let half_up_negative = round(-3.5);
/// assert_eq!(half_up_negative, -3);
/// ```
pub fn round(coordinate: f64) -> Coordinate {
	if coordinate >= 0.0 {
		if coordinate.fract().abs() >= 0.5 {
			return coordinate.ceil() as Coordinate;
		} else {
			return coordinate.floor() as Coordinate;
		}
	} else {
		if coordinate.fract().abs() > 0.5 {
			return coordinate.floor() as Coordinate;
		} else {
			return coordinate.ceil() as Coordinate;
		}
	}
}

/// The type used to store the size of areas in 2D space.
///
/// Areas can be negative. This is used to indicate the surface areas of parts of complex shapes and
/// self-intersecting shapes.
///
/// This type is meant to guarantee that every shape that can be represented by the coordinate
/// system of `Coordinate` can have its area properly calculated. However, the maximum area that can
/// be represented by `Coordinate`-specified shapes is 2^32 by 2^32 (from the negative minimum
/// coordinate to the positive maximum, in both dimensions). This would be an area of 2^64 units,
/// but the area can also be negative. The maximum signed area would need 65 bits to be represented.
/// The available integer types would require this to use `i128`. However this size is so rarely
/// used and would incur such a performance penalty that it's better to use `i64`. This type can at
/// most represent only half of the total coordinate space in 2D, but it's much more performant.
pub type Area = i64;

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	/// Test whether the possible range of `Coordinate` is as expected.
	fn coordinate_range() {
		let x: Coordinate = 0x7FFFFFFF;
		assert_eq!(x, 0x7FFFFFFF, "We need to be able to store at least this coordinate.");
		let result = x.checked_add(1);
		assert_eq!(result, None, "It needs to overflow then.");
	}

	#[test]
	/// Test whether the possible range of `Area` is as expected.
	fn area_range() {
		let area: Area = 0x7FFFFFFFFFFFFFFF;
		assert_eq!(area, 0x7FFFFFFFFFFFFFFF, "We need to be able to store at least this area.");
		let result = area.checked_add(1);
		assert_eq!(result, None, "It needs to overflow then.");
	}
}
/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines a struct that represents single points in a 2-dimensional space.

use bytemuck::{Pod, Zeroable}; //Point2D is plain-old-data.

use crate::Angle; //To implement TwoDimensional.
use crate::Area; //To implement Shape2D.
use crate::Convexity; //To implement Shape2D.
use crate::Coordinate; //The position of the point is stored with coordinates.
use crate::TwoDimensional; //This point is in two-dimensional space.
use crate::Shape2D; //A point is a shape, with a bounded (zero) area.
use crate::coordinate::round; //To properly round after transformations.

/// Specifies a point in 2D space.
///
/// The two dimensions are called X and Y, by convention.
///
/// The point can be considered a degenerate shape. It has no surface area or width or height. It
/// collides with other geometry only when the borders of the objects are considered.
///
/// Points can be compared lexicographically. While this has no real geometric meaning, this can be
/// useful for certain geometric algorithms. When compared, points with lower X coordinates will be
/// considered lower. If points have the same X coordinate, points with lower Y coordinates will be
/// considered lower. Thus the points are compared lexicographically with X before Y.
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Pod, Zeroable)]
pub struct Point2D {
	/// The projection of this point on the X dimension.
	pub x: Coordinate,

	/// The projection of this point on the Y dimension.
	pub y: Coordinate,
}

impl Point2D {
	/// Construct a new point in 2D space.
	///
	/// # Arguments
	/// * `x` - The coordinate along the first dimension where the point will be located.
	/// * `y` - The coordinate along the second dimension where the point will be located.
	pub fn new(x: Coordinate, y: Coordinate) -> Point2D {
		Point2D { x, y }
	}
}

impl Shape2D for Point2D {
	/// Get the surface area of the point.
	///
	/// Points have no surface area, so this will always return 0.
	fn area(&self) -> Area {
		return 0; //A point has no area.
	}

	/// Get the convexity of the point.
	///
	/// Points don't have any dimensions or surface area, so they are always degenerate.
	fn convexity(&self) -> Convexity {
		return Convexity::DEGENERATE; //Points are degenerate shapes.
	}
}

impl TwoDimensional for Point2D {
	/// Move the point across the two-dimensional space.
	///
	/// This causes the position of the point to change. The point is modified in-place.
	///
	/// # Arguments
	/// * `dx` - How far to move the point in the X direction. Use a positive number to increase the
	/// X position, or a negative number to reduce the X position.
	/// * `dy` - How far to move the point in the Y direction. Use a positive number to increase the
	/// Y position, or a negative number to reduce the Y position.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, TwoDimensional};
	/// let mut point = Point2D{ x: 100, y: 500 };
	/// point.translate(50, -130);
	/// assert_eq!(point, Point2D { x: 150, y: 370 });
	/// ```
	fn translate(&mut self, dx: Coordinate, dy: Coordinate) {
		self.x += dx;
		self.y += dy;
	}

	/// Scale the point away from the coordinate origin.
	///
	/// This causes the point to move away from or closer to the coordinate origin.
	///
	/// The point is modified in-place.
	///
	/// # Arguments
	/// * `x` - The scaling factor for the X axis. Use a number greater than 1 to move the point
	/// farther away from the coordinate origin, or smaller than 1 to move it closer. Use a negative
	/// number to mirror the position horizontally.
	/// * `y` - The scaling factor for the Y axis. Use a number greater than 1 to move the point
	/// farther away from the coordinate origin, or smaller than 1 to move it closer. Use a negative
	/// number to mirror the position horizontally.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, TwoDimensional};
	/// let mut point = Point2D { x: 100, y: 500 };
	/// point.scale(2.0, -0.5);
	/// assert_eq!(point, Point2D { x: 200, y: -250 });
	/// ```
	fn scale(&mut self, x: f64, y: f64) {
		self.x = round(self.x as f64 * x);
		self.y = round(self.y as f64 * y);
	}

	/// Rotate the point around the coordinate origin.
	///
	/// The rotation is mathematically around the 0,0 origin rather than around its own centre. The
	/// rotation is counter-clockwise, so a rotation of 0.1 rad will cause the object to rotate
	/// slightly counter-clockwise, while a rotation of 6.1 rad (almost 2 pi) will cause the object
	/// to rotate slightly clockwise.
	///
	/// # Arguments
	/// * `angle` - The amount of counter-clockwise rotation to apply.
	///
	/// # Examples
	/// ```
	/// use apex::{Angle, Point2D, TwoDimensional};
	/// let mut point = Point2D { x: 100, y: 0 }; //Create a point with initially only an X-offset.
	/// point.rotate(Angle::EIGHTH_TURN);
	/// assert_eq!(point, Point2D { x: 71, y: 71 }); //Now rotated counter-clockwisely to 100/sqrt(2).
	/// ```
	fn rotate(&mut self, angle: Angle) {
		let cosine = angle.cos();
		let sine = angle.sin();
		//Calculate X first without adjusting the real X, so that we can use the old value for the Y too.
		let new_x = round(self.x as f64 * cosine - self.y as f64 * sine);
		self.y = round(self.x as f64 * sine + self.y as f64 * cosine);
		self.x = new_x;
	}
}

impl_op_ex!(+ |a: &Point2D, b: &Point2D| -> Point2D { Point2D::new(a.x + b.x, a.y + b.y) });
impl_op_ex!(- |a: &Point2D, b: &Point2D| -> Point2D { Point2D::new(a.x - b.x, a.y - b.y) });

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	/// Test the area of a point.
	fn point2d_area() {
		let point = Point2D { x: 10, y: 10 };
		assert_eq!(point.area(), 0, "Points have no surface area, so it should be 0.");
	}

	#[test]
	/// Test moving a point by 0,0. It should not be modified.
	fn point2d_translate_zero() {
		let mut point = Point2D { x: 10, y: 20 };
		point.translate(0, 0);
		assert_eq!(point.x, 10, "Moving the point by 0,0 should not change it.");
		assert_eq!(point.y, 20, "Moving the point by 0,0 should not change it.");
	}

	#[test]
	/// Test moving a point in a positive direction.
	fn point2d_translate_positive() {
		let mut point = Point2D { x: 100, y: 200 };
		point.translate(20, 10);
		assert_eq!(point.x, 100 + 20, "We moved the X coordinate into the positive direction by 20.");
		assert_eq!(point.y, 200 + 10, "We moved the Y coordinate into the positive direction by 10.");
	}

	#[test]
	/// Test moving a point in a negative direction.
	fn point2d_translate_negative() {
		let mut point = Point2D { x: 1000, y: -2000 };
		point.translate(-400, -500);
		assert_eq!(point.x, 1000 - 400, "We moved the X coordinate into the negative direction by 400.");
		assert_eq!(point.y, -2000 - 500, "We moved the Y coordinate into the negative direction by 500.");
	}

	#[test]
	/// Test moving a point in a mixed direction.
	fn point2d_translate_mixed() {
		let mut point = Point2D { x: 20000, y: -10000 };
		point.translate(100, -200);
		assert_eq!(point.x, 20000 + 100, "We moved the X coordinate into the positive direction by 100.");
		assert_eq!(point.y, -10000 - 200, "We moved the Y coordinate into the negative direction by 200.");
		point.translate(-500, 1000);
		assert_eq!(point.x, 20000 + 100 - 500, "We further moved the X coordinate into the negative direction by 500.");
		assert_eq!(point.y, -10000 - 200 + 1000, "We further moved the Y coordinate into the positive direction by 1000.");
	}

	#[test]
	/// Test scaling a point to be larger.
	fn point2d_scale_larger() {
		let mut point = Point2D { x: 1000, y: -200 };
		point.scale(2.0, 3.5);
		assert_eq!(point.x, 2000, "We scaled the X coordinate by 2, so 1000 * 2 = 2000.");
		assert_eq!(point.y, -700, "We scaled the Y coordinate by 3.5, so -200 * 3.5 = -700.");
	}

	#[test]
	/// Test scaling a point to be smaller.
	fn point2d_scale_smaller() {
		let mut point = Point2D { x: -1000, y: 200 };
		point.scale(0.5, 0.7);
		assert_eq!(point.x, -500, "We scaled the X coordinate by 0.5, so -1000 * 0.5 = -500.");
		assert_eq!(point.y, 140, "We scaled the Y coordinate by 0.7, so 200 * 0.7 = 140.");
	}

	#[test]
	/// Test scaling a point to be mirrored around the origin.
	fn point2d_scale_negative() {
		let mut point = Point2D { x: 40000, y: -90 };
		point.scale(-0.2, -25.0);
		assert_eq!(point.x, -8000, "We scaled the X coordinate by -0.2, so 40000 * -0.2 = -8000.");
		assert_eq!(point.y, 2250, "We scaled the Y coordinate by -25, so -90 * -25 = 2250.");
	}

	#[test]
	/// Test proper rounding and rounding errors when scaling.
	fn point2d_scale_rounding() {
		let mut point = Point2D { x: 1, y: 25 };
		point.scale(4.5, -0.5);
		assert_eq!(point.x, 5, "1 * 4.5 would be 4.5, which gets rounded up to 5.");
		assert_eq!(point.y, -12, "25 * -0.5 would be -12.5, which gets rounded up to -12.");
	}

	#[test]
	/// Test the equality operator on Point2D.
	fn point2d_equality() {
		let point1 = Point2D { x: 400, y: 600 };
		let point2 = Point2D { x: 400, y: 600 };
		let different = Point2D { x: -400, y: 600 }; //Different from the other two.
		assert_eq!(point1, point1, "Reflexive: The point must be equal to itself.");
		assert_eq!(point1, point2, "If the coordinates of the points are the same, the points are the same.");
		assert_eq!(point2, point1, "Commutative: It doesn't matter in what order points are equated.");
		assert_ne!(point1, different, "If the coordinates of the points are different, the points are different.");
		assert_ne!(different, point1, "Commutative: It doesn't matter in what order points are equated.");
	}

	#[test]
	/// Test comparing the order of Point2Ds if they are the same.
	fn point2d_compare_equal() {
		let point1 = Point2D { x: 100, y: 150 };
		let point2 = Point2D { x: 100, y: 150 };
		assert!(point1 <= point2, "The points are equal, so they must also be less-than-or-equal.");
		assert!(point2 <= point1, "Commutative: It doesn't matter in what order the points are compared.");
		assert!(point1 >= point2, "The points are equal, so they must also be greater-than-or-equal.");
		assert!(point2 >= point1, "Commutative: It doesn't matter in what order the points are compared.");
		assert!(!(point1 < point2), "The points are equal, so one is not less than the other.");
		assert!(!(point2 < point1), "The points are equal, so one is not less than the other.");
		assert!(!(point1 > point2), "The points are equal, so one is not greater than the other.");
		assert!(!(point2 > point1), "The points are equal, so one is not greater than the other.");
	}

	#[test]
	/// Test comparing the order of Point2Ds if they have different coordinates.
	fn point2d_compare_different() {
		let point1 = Point2D { x: 100, y: 150 };
		let point2 = Point2D { x: 101, y: 100 }; //X is greater, which is more significant, so point2 > point1.
		assert!(point1 < point2, "The X coordinate is more significant, so point1 is less than point2.");
		assert!(point1 <= point2, "If point1 < point2, then also point1 <= point2.");
		assert!(!(point1 > point2), "The X coordinate is more significant, so point1 is not greater than point2.");
		assert!(!(point1 >= point2), "If not point1 > point2 and not equal, then not point1 >= point2.");
		assert!(!(point2 < point1), "Commutative: It doesn't matter in what order the points are compared.");
		assert!(!(point2 <= point1), "Commutative: It doesn't matter in what order the points are compared.");
		assert!(point2 > point1, "Commutative: It doesn't matter in what order the points are compared.");
		assert!(point2 >= point1, "Commutative: It doesn't matter in what order the points are compared.");
	}

	#[test]
	/// Test comparing the order of Point2Ds if they have the same X coordinate, but different Y
	/// coordinates.
	///
	/// Since X is the same, Y is the less significant comparison, but determines the outcome.
	fn point2d_compare_same_x() {
		let point1 = Point2D { x: 100, y: 100 };
		let point2 = Point2D { x: 100, y: 150 }; //X is the same, but Y is greater.
		assert!(point1 < point2, "The X coordinate is the same, but point1.y < point2.y.");
		assert!(point1 <= point2, "If point1 < point2, then also point1 <= point2.");
		assert!(!(point1 > point2), "The X coordinate is the same, but point1.y < point2.y.");
		assert!(!(point1 >= point2), "If not point1 > point2 and not equal, then not point1 >= point2.");
		assert!(!(point2 < point1), "Commutative: It doesn't matter in what order the points are compared.");
		assert!(!(point2 <= point1), "Commutative: It doesn't matter in what order the points are compared.");
		assert!(point2 > point1, "Commutative: It doesn't matter in what order the points are compared.");
		assert!(point2 >= point1, "Commutative: It doesn't matter in what order the points are compared.");
	}

	#[test]
	/// Test summing Point2Ds coordinate-wise.
	fn point2d_sum() {
		let point1 = Point2D { x: 100, y: 200 };
		let point2 = Point2D { x: 4000, y: 5000 };
		assert_eq!(&point1 + &point2, Point2D { x: 100 + 4000, y: 200 + 5000 }, "We simply sum the coordinates separately.");
		assert_eq!(point2 + point1, Point2D { x: 100 + 4000, y: 200 + 5000 }, "Commutative: It doesn't matter in what order the points are summed.");
	}

	#[test]
	/// Test subtracting Point2Ds coordinate-wise.
	fn point2d_subtract() {
		let point1 = Point2D { x: 100, y: 200 };
		let point2 = Point2D { x: 10, y: -20 };
		assert_eq!(&point1 - &point2, Point2D { x: 100 - 10, y: 200 + 20 }, "We simply subtract the coordinates separately.");
	}

	#[test]
	/// Test the convexity of Point2D.
	fn point2d_convexity() {
		let point = Point2D { x: 100, y: 200 };
		assert_eq!(point.convexity(), Convexity::DEGENERATE, "Points are always degenerate convexity.");
	}
}
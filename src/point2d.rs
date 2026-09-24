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
use crate::Area; //For the area of a vector represented by this point.
use crate::Coordinate; //The position of the point is stored with coordinates.
use crate::TwoDimensional; //This point is in two-dimensional space.
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
///
/// A point can also be viewed as an Euclidean vector. In this case, the vector has no defined
/// starting point, so it can be considered a free vector. The struct has methods that help with
/// using it in this way.
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
	///
	/// # Returns
	/// A point that has the given coordinates.
	pub fn new(x: Coordinate, y: Coordinate) -> Point2D {
		Point2D { x, y }
	}

	/// Get the squared distance of this point to the coordinate origin.
	///
	/// This is equal to the length of the vector from the coordinate origin to the point, squared.
	///
	/// The result is returned as an `Area`, because it needs the increased range given by the
	/// `Area` type and because the square of the vector length is always integer, without rounding.
	/// It can also be seen as the area formed by a square with the vector as one of its sides.
	///
	/// Computing the squared length of the vector is much faster than computing the actual length,
	/// since no square root call is involved. The result though is perhaps less useful if multiple
	/// lengths have to be summed together or for more complex algorithms, but it is still useful
	/// for comparing distances. Perhaps even more so than the actual length, because the squared
	/// length does not get rounded.
	///
	/// # Returns
	/// The length of the vector squared.
	///
	/// # Examples
	/// ```
	/// use apex::Point2D;
	/// let point1 = Point2D { x: 10, y: 6 };
	/// assert_eq!(point1.vector_length_squared(), 136); //Actual length: ~11.662.
	/// let point2 = Point2D { x: -4, y: 3 };
	/// assert_eq!(point2.vector_length_squared(), 25); //Actual length: 5.
	/// ```
	pub fn vector_length_squared(self) -> Area {
		self.x as Area * self.x as Area + self.y as Area * self.y as Area
	}

	/// Compute the Z-component of the cross product of this 2D vector with another 2D vector.
	///
	/// The cross product is normally only defined for 3D vectors. This function takes the cross
	/// product between two 3D vectors, where the third coordinate is 0 and the first two are the
	/// two coordinates of this `Point2D`. The cross product would then become the 3D vector that is
	/// perpendicular to both of these vectors. Since both of these vectors are on the plane of the
	/// first two dimensions, the cross product will always have X and Y coordinates 0, pointing
	/// along the Z axis. The dimension along the Z axis is returned here.
	///
	/// The direction of the cross product is characterised with the right-hand rule, with the two
	/// vectors of the input as the index and middle fingers, and the resulting cross product in the
	/// direction of the thumb. It is anticommutative, meaning that taking the cross product of this
	/// `Point2D` with another will yield an inverted result (negative) as when the two points are
	/// swapped.
	///
	/// The magnitude of the cross product is the area of the parallelogram containing the two
	/// vectors as two of its sides. This property can be useful to work with the angle between the
	/// two vectors. The result of this function is an area, as a result. The result could possibly
	/// be so big that a normal `Coordinate` couldn't represent it.
	///
	/// # Arguments
	/// * `other` - The X and Y components of the vector to take the cross product with. The Z
	/// component will be set to 0.
	///
	/// # Returns
	/// The Z component of the resulting cross product of this vector with the given vector. Since
	/// the X and Y components will be 0, this is also the magnitude of the cross product.
	///
	/// # Examples
	/// ```
	/// use apex::Point2D;
	/// let point1 = Point2D { x: 10, y: 0 };
	/// let point2 = Point2D { x: 10, y: 10 };
	/// assert_eq!(point1.cross_product_z(&point2), 100); //Forms a parallelogram with area 100.
	/// ```
	pub fn cross_product_z(self, other: &Point2D) -> Area {
		self.x as Area * other.y as Area - self.y as Area * other.x as Area
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
	/// let mut point = Point2D { x: 100, y: 500 };
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
impl_op_ex!(- |a: &Point2D| -> Point2D { Point2D::new(-a.x, -a.y) });

#[cfg(test)]
mod tests {
	use super::*;
	use test_case::test_case;

	/// Test getting the squared vector length of a point.
	#[test_case(4, 3, 25; "Four-three-five triangle")]
	#[test_case(-10, 20, 500; "X negative")]
	#[test_case(10, -20, 500; "Y negative")]
	#[test_case(-20, -10, 500; "Both negative")]
	#[test_case(0, 40, 1600; "X zero")]
	#[test_case(-40, 0, 1600; "Y zero")]
	#[test_case(0, 0, 0; "Both zero")]
	#[test_case(Coordinate::MAX, Coordinate::MAX, 9223372028264841218; "Maximum vector")] //Minimum vector results in an overflow, but that is accepted as a limitation.
	fn vector_length_squared(x: Coordinate, y: Coordinate, result: Area) {
		let point = Point2D { x, y };
		assert_eq!(point.vector_length_squared(), result);
	}

	/// Test calculating the Z component of the cross product between two vectors.
	#[test_case(0, 0, 0, 0, 0; "Zeroes")]
	#[test_case(4, 3, 0, 0, 0; "RHS zero")]
	#[test_case(0, 0, 5, 6, 0; "LHS zero")]
	#[test_case(10, 0, 0, 10, 100; "Perpendicular vectors")]
	#[test_case(10, 0, 5, 0, 0; "Parallel vectors")]
	#[test_case(10, 0, -5, 0, 0; "Opposite vectors")]
	#[test_case(3, 4, 4, 3, -7; "Right winding")]
	#[test_case(Coordinate::MAX, 0, 0, Coordinate::MAX, 4611686014132420609; "Maximum vector")]
	#[test_case(Coordinate::MAX, 0, 0, Coordinate::MAX, 4611686014132420609; "Minimum vector")]
	fn cross_product_z(x1: Coordinate, y1: Coordinate, x2: Coordinate, y2: Coordinate, result: Area) {
		let vector1 = Point2D { x: x1, y: y1 };
		let vector2 = Point2D { x: x2, y: y2 };
		assert_eq!(vector1.cross_product_z(&vector2), result);
	}

	#[test_case(0, 0, 0, 0; "Zeroes")]
	#[test_case(10, 0, 0, 10; "Perpendicular vectors")]
	#[test_case(3, 4, 4, 3; "Right winding")]
	fn cross_product_z_anticommutative(x1: Coordinate, y1: Coordinate, x2: Coordinate, y2: Coordinate) {
		let vector1 = Point2D { x: x1, y: y1 };
		let vector2 = Point2D { x: x2, y: y2 };
		let result = vector1.cross_product_z(&vector2);
		assert_eq!(vector2.cross_product_z(&vector1), -result);
	}

	/// Test moving a point by 0,0. It should not be modified.
	#[test]
	fn translate_zero() {
		let mut point = Point2D { x: 10, y: 20 };
		point.translate(0, 0);
		assert_eq!(point.x, 10, "Moving the point by 0,0 should not change it.");
		assert_eq!(point.y, 20, "Moving the point by 0,0 should not change it.");
	}

	/// Test moving a point in a positive direction.
	#[test]
	fn translate_positive() {
		let mut point = Point2D { x: 100, y: 200 };
		point.translate(20, 10);
		assert_eq!(point.x, 100 + 20, "We moved the X coordinate into the positive direction by 20.");
		assert_eq!(point.y, 200 + 10, "We moved the Y coordinate into the positive direction by 10.");
	}

	/// Test moving a point in a negative direction.
	#[test]
	fn translate_negative() {
		let mut point = Point2D { x: 1000, y: -2000 };
		point.translate(-400, -500);
		assert_eq!(point.x, 1000 - 400, "We moved the X coordinate into the negative direction by 400.");
		assert_eq!(point.y, -2000 - 500, "We moved the Y coordinate into the negative direction by 500.");
	}

	/// Test moving a point in a mixed direction.
	#[test]
	fn translate_mixed() {
		let mut point = Point2D { x: 20000, y: -10000 };
		point.translate(100, -200);
		assert_eq!(point.x, 20000 + 100, "We moved the X coordinate into the positive direction by 100.");
		assert_eq!(point.y, -10000 - 200, "We moved the Y coordinate into the negative direction by 200.");
		point.translate(-500, 1000);
		assert_eq!(point.x, 20000 + 100 - 500, "We further moved the X coordinate into the negative direction by 500.");
		assert_eq!(point.y, -10000 - 200 + 1000, "We further moved the Y coordinate into the positive direction by 1000.");
	}

	/// Test scaling a point to be larger.
	#[test]
	fn scale_larger() {
		let mut point = Point2D { x: 1000, y: -200 };
		point.scale(2.0, 3.5);
		assert_eq!(point.x, 2000, "We scaled the X coordinate by 2, so 1000 * 2 = 2000.");
		assert_eq!(point.y, -700, "We scaled the Y coordinate by 3.5, so -200 * 3.5 = -700.");
	}

	/// Test scaling a point to be smaller.
	#[test]
	fn scale_smaller() {
		let mut point = Point2D { x: -1000, y: 200 };
		point.scale(0.5, 0.7);
		assert_eq!(point.x, -500, "We scaled the X coordinate by 0.5, so -1000 * 0.5 = -500.");
		assert_eq!(point.y, 140, "We scaled the Y coordinate by 0.7, so 200 * 0.7 = 140.");
	}

	/// Test scaling a point to be mirrored around the origin.
	#[test]
	fn scale_negative() {
		let mut point = Point2D { x: 40000, y: -90 };
		point.scale(-0.2, -25.0);
		assert_eq!(point.x, -8000, "We scaled the X coordinate by -0.2, so 40000 * -0.2 = -8000.");
		assert_eq!(point.y, 2250, "We scaled the Y coordinate by -25, so -90 * -25 = 2250.");
	}

	/// Test proper rounding and rounding errors when scaling.
	#[test]
	fn scale_rounding() {
		let mut point = Point2D { x: 1, y: 25 };
		point.scale(4.5, -0.5);
		assert_eq!(point.x, 5, "1 * 4.5 would be 4.5, which gets rounded up to 5.");
		assert_eq!(point.y, -12, "25 * -0.5 would be -12.5, which gets rounded up to -12.");
	}

	/// Test the equality operator on Point2D.
	#[test]
	fn equality() {
		let point1 = Point2D { x: 400, y: 600 };
		let point2 = Point2D { x: 400, y: 600 };
		let different = Point2D { x: -400, y: 600 }; //Different from the other two.
		assert_eq!(point1, point1, "Reflexive: The point must be equal to itself.");
		assert_eq!(point1, point2, "If the coordinates of the points are the same, the points are the same.");
		assert_eq!(point2, point1, "Commutative: It doesn't matter in what order points are equated.");
		assert_ne!(point1, different, "If the coordinates of the points are different, the points are different.");
		assert_ne!(different, point1, "Commutative: It doesn't matter in what order points are equated.");
	}

	/// Test comparing the order of Point2Ds if they are the same.
	#[test]
	fn compare_equal() {
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

	/// Test comparing the order of Point2Ds if they have different coordinates.
	#[test]
	fn compare_different() {
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

	/// Test comparing the order of Point2Ds if they have the same X coordinate, but different Y
	/// coordinates.
	///
	/// Since X is the same, Y is the less significant comparison, but determines the outcome.
	#[test]
	fn compare_same_x() {
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

	/// Test summing Point2Ds coordinate-wise.
	#[test]
	fn sum() {
		let point1 = Point2D { x: 100, y: 200 };
		let point2 = Point2D { x: 4000, y: 5000 };
		assert_eq!(&point1 + &point2, Point2D { x: 100 + 4000, y: 200 + 5000 }, "We simply sum the coordinates separately.");
		assert_eq!(point2 + point1, Point2D { x: 100 + 4000, y: 200 + 5000 }, "Commutative: It doesn't matter in what order the points are summed.");
	}

	/// Test subtracting Point2Ds coordinate-wise.
	#[test]
	fn subtract() {
		let point1 = Point2D { x: 100, y: 200 };
		let point2 = Point2D { x: 10, y: -20 };
		assert_eq!(&point1 - &point2, Point2D { x: 100 - 10, y: 200 + 20 }, "We simply subtract the coordinates separately.");
	}
}
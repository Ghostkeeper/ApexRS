/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

use apex;
use apex::Convexity;
use apex::Shape2D;
use apex::TwoDimensional;

#[test]
/// Test the area of a point.
fn point2d_area() {
	let point = apex::Point2D { x: 10, y: 10 };
	assert_eq!(point.area(), 0, "Points have no surface area, so it should be 0.");
}

#[test]
/// Test moving a point by 0,0. It should not be modified.
fn point2d_translate_zero() {
	let mut point = apex::Point2D { x: 10, y: 20 };
	point.translate(0, 0);
	assert_eq!(point.x, 10, "Moving the point by 0,0 should not change it.");
	assert_eq!(point.y, 20, "Moving the point by 0,0 should not change it.");
}

#[test]
/// Test moving a point in a positive direction.
fn point2d_translate_positive() {
	let mut point = apex::Point2D { x: 100, y: 200 };
	point.translate(20, 10);
	assert_eq!(point.x, 100 + 20, "We moved the X coordinate into the positive direction by 20.");
	assert_eq!(point.y, 200 + 10, "We moved the Y coordinate into the positive direction by 10.");
}

#[test]
/// Test moving a point in a negative direction.
fn point2d_translate_negative() {
	let mut point = apex::Point2D { x: 1000, y: -2000 };
	point.translate(-400, -500);
	assert_eq!(point.x, 1000 - 400, "We moved the X coordinate into the negative direction by 400.");
	assert_eq!(point.y, -2000 - 500, "We moved the Y coordinate into the negative direction by 500.");
}

#[test]
/// Test moving a point in a mixed direction.
fn point2d_translate_mixed() {
	let mut point = apex::Point2D { x: 20000, y: -10000 };
	point.translate(100, -200);
	assert_eq!(point.x, 20000 + 100, "We moved the X coordinate into the positive direction by 100.");
	assert_eq!(point.y, -10000 - 200, "We moved the Y coordinate into the negative direction by 200.");
	point.translate(-500, 1000);
	assert_eq!(point.x, 20000 + 100 - 500, "We further moved the X coordinate into the negative direction by 500.");
	assert_eq!(point.y, -10000 - 200 + 1000, "We further moved the Y coordinate into the positive direction by 1000.");
}

#[test]
/// Test the equality operator on Point2D.
fn point2d_equality() {
	let point1 = apex::Point2D { x: 400, y: 600 };
	let point2 = apex::Point2D { x: 400, y: 600 };
	let different = apex::Point2D { x: -400, y: 600 }; //Different from the other two.
	assert_eq!(point1, point1, "Reflexive: The point must be equal to itself.");
	assert_eq!(point1, point2, "If the coordinates of the points are the same, the points are the same.");
	assert_eq!(point2, point1, "Commutative: It doesn't matter in what order points are equated.");
	assert_ne!(point1, different, "If the coordinates of the points are different, the points are different.");
	assert_ne!(different, point1, "Commutative: It doesn't matter in what order points are equated.");
}

#[test]
/// Test comparing the order of Point2Ds if they are the same.
fn point2d_compare_equal() {
	let point1 = apex::Point2D { x: 100, y: 150 };
	let point2 = apex::Point2D { x: 100, y: 150 };
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
	let point1 = apex::Point2D { x: 100, y: 150 };
	let point2 = apex::Point2D { x: 101, y: 100 }; //X is greater, which is more significant, so point2 > point1.
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
	let point1 = apex::Point2D { x: 100, y: 100 };
	let point2 = apex::Point2D { x: 100, y: 150 }; //X is the same, but Y is greater.
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
	let point1 = apex::Point2D { x: 100, y: 200 };
	let point2 = apex::Point2D { x: 4000, y: 5000 };
	assert_eq!(&point1 + &point2, apex::Point2D { x: 100 + 4000, y: 200 + 5000 }, "We simply sum the coordinates separately.");
	assert_eq!(point2 + point1, apex::Point2D { x: 100 + 4000, y: 200 + 5000 }, "Commutative: It doesn't matter in what order the points are summed.");
}

#[test]
/// Test subtracting Point2Ds coordinate-wise.
fn point2d_subtract() {
	let point1 = apex::Point2D { x: 100, y: 200 };
	let point2 = apex::Point2D { x: 10, y: -20 };
	assert_eq!(&point1 - &point2, apex::Point2D { x: 100 - 10, y: 200 + 20 }, "We simply subtract the coordinates separately.");
}

#[test]
/// Test the convexity of Point2D.
fn point2d_convexity() {
	let point = apex::Point2D { x: 100, y: 200 };
	assert_eq!(point.convexity(), Convexity::DEGENERATE, "Points are always degenerate convexity.");
}
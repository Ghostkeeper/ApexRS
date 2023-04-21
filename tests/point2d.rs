/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

use apex;
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
fn point2d_move_zero() {
	let mut point = apex::Point2D { x: 10, y: 20 };
	point.translate(0, 0);
	assert_eq!(point.x, 10, "Moving the point by 0,0 should not change it.");
	assert_eq!(point.y, 20, "Moving the point by 0,0 should not change it.");
}
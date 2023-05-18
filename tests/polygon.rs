/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

use std::mem;
use apex;

/// Test creating a new, empty polygon.
///
/// This asserts that the new polygon is empty.
#[test]
fn new() {
	let poly = apex::Polygon::new();
	assert_eq!(poly.len(), 0, "The new polygon has no vertices.");
}

/// Test creating a polygon with a given capacity.
///
/// This asserts that the new polygon is empty, and that the polygon has the given capacity.
#[test]
fn with_capacity() {
	let poly = apex::Polygon::with_capacity(10);
	assert_eq!(poly.capacity(), 10, "We require the capacity to be exactly 10 then.");
	assert_eq!(poly.len(), 0, "The new polygon has no vertices.");
}

/// Test getting the capacity of a polygon.
#[test]
fn capacity() {
	let mut poly = apex::Polygon::with_capacity(3);
	assert_eq!(poly.capacity(), 3, "The polygon was initially created with capacity 3.");
	//The memory is guaranteed to not be reallocated as long as the capacity is not reached.
	//We can sort of see that by testing that the capacity was not increased.
	poly.push(apex::Point2D { x: 0, y: 0 });
	assert_eq!(poly.capacity(), 3, "The capacity was not expanded since there is only 1 vertex in the polygon.");
	poly.push(apex::Point2D { x: 100, y: 0 });
	assert_eq!(poly.capacity(), 3, "The capacity was not expanded since there are only 2 vertices in the polygon.");
	poly.push(apex::Point2D { x: 100, y: 100 });
	assert_eq!(poly.capacity(), 3, "The capacity was not expanded since there are exactly 3 vertices in the polygon.");
	poly.push(apex::Point2D { x: 0, y: 100 });
	assert!(poly.capacity() > 3, "The capacity is now expanded since the number of vertices was over capacity.");
}
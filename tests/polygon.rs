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
/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

use apex; //The unit under test.

mod data; //Polygon test cases to test with.
use data::polygon;

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

/// Test getting the number of vertices/sides of a polygon.
#[test]
fn len() {
	let mut poly = apex::Polygon::new();
	assert_eq!(poly.len(), 0, "The polygon was created without any vertices.");
	poly.push(apex::Point2D { x: 0, y: 0 });
	assert_eq!(poly.len(), 1, "After adding a vertex, the length is now 1.");
	for i in 0..10 { //Add 10 more vertices.
		poly.push(apex::Point2D { x: i + 100, y: i + 100 });
	}
	assert_eq!(poly.len(), 11, "After adding 10 more vertices, the length is now 11.");
}

/// Test adding new vertices to a polygon.
#[test]
fn push() {
	let mut poly = polygon::square_1000();
	assert_eq!(poly.len(), 4, "The square starts with 4 vertices.");
	poly.push(apex::Point2D { x: 0, y: 100 });
	assert_eq!(poly.len(), 5, "After adding 1 more vertex, there are now 5 vertices.");
	assert_eq!(poly[4], apex::Point2D { x: 0, y: 100 }, "The newly added vertex is at the seam.");
}

/// Test creating a polygon from an iterable object, this time an array.
#[test]
fn from_iter_array() {
	let poly = apex::Polygon::from_iter([
		apex::Point2D { x: 0, y: 0 },
		apex::Point2D { x: 500, y: 0 },
		apex::Point2D { x: 250, y: 1000 }
	]);
	assert_eq!(poly[0], apex::Point2D { x: 0, y: 0 }, "The first vertex in the newly created polygon.");
	assert_eq!(poly[1], apex::Point2D { x: 500, y: 0 }, "The second vertex in the newly created polygon.");
	assert_eq!(poly[2], apex::Point2D { x: 250, y: 1000 }, "The third vertex in the newly created polygon.");
}

/// Test creating a polygon from an iterable object, this time a vector.
#[test]
fn from_iter_vec() {
	let vertices = vec![
		apex::Point2D { x: 0, y: 0 },
		apex::Point2D { x: 500, y: 0 },
		apex::Point2D { x: 250, y: 1000 }
	];
	let poly = apex::Polygon::from_iter(vertices);
	assert_eq!(poly[0], apex::Point2D { x: 0, y: 0 }, "The first vertex in the newly created polygon.");
	assert_eq!(poly[1], apex::Point2D { x: 500, y: 0 }, "The second vertex in the newly created polygon.");
	assert_eq!(poly[2], apex::Point2D { x: 250, y: 1000 }, "The third vertex in the newly created polygon.");
}

/// Test creating a polygon from an iterable object, this time a different polygon.
#[test]
fn from_iter_polygon() {
	let original = apex::Polygon::from_iter([
		apex::Point2D { x: 0, y: 0 },
		apex::Point2D { x: 500, y: 0 },
		apex::Point2D { x: 250, y: 1000 }
	]);
	let poly = apex::Polygon::from_iter(original);
	assert_eq!(poly[0], apex::Point2D { x: 0, y: 0 }, "The first vertex in the newly created polygon.");
	assert_eq!(poly[1], apex::Point2D { x: 500, y: 0 }, "The second vertex in the newly created polygon.");
	assert_eq!(poly[2], apex::Point2D { x: 250, y: 1000 }, "The third vertex in the newly created polygon.");
}

/// Test iterating over vertices in a polygon.
#[test]
fn into_iter() {
	let vertices = [
		apex::Point2D { x: 0, y: 0 },
		apex::Point2D { x: 100, y: 0 },
		apex::Point2D { x: 50, y: 100 }
	];
	let poly = apex::Polygon::from_iter(vertices);
	let mut i = 0;
	for vertex in poly {
		assert_eq!(vertex, vertices[i], "The iterator must iterate over the vertices in order.");
		i += 1;
	}
}

/// Test accessing vertices of the polygon.
///
/// All access in this test is within range.
#[test]
fn index_in_range() {
	let poly = apex::Polygon::from_iter([
		apex::Point2D { x: 0, y: 0 },
		apex::Point2D { x: 50, y: 10 },
		apex::Point2D { x: 10, y: 100 }
	]);
	assert_eq!(poly[0], apex::Point2D { x: 0, y: 0 }, "Getting the first vertex at index 0.");
	assert_eq!(poly[1], apex::Point2D { x: 50, y: 10 }, "Getting the second vertex at index 1.");
	assert_eq!(poly[2], apex::Point2D { x: 10, y: 100 }, "Getting the third vertex at index 2.");
}

/// Test accessing a vertex beyond the size of the polygon.
///
/// This test should cause a panic.
#[test]
#[should_panic(expected = "the len is 3 but the index is 3")]
fn index_out_of_range() {
	let poly = polygon::triangle_1000();
	std::panic::set_hook(Box::new(|_| {})); //Disable stack trace from this panic.
	poly[3]; //Panic here. This is out of range.
}

/// Test modifying a vertex of the polygon.
#[test]
fn index_mut() {
	let mut poly = apex::Polygon::from_iter([
		apex::Point2D { x: 0, y: 0 },
		apex::Point2D { x: 50, y: 10 },
		apex::Point2D { x: 10, y: 100 }
	]);
	poly[1] = apex::Point2D { x: 200, y: 400 };
	assert_eq!(poly[0], apex::Point2D { x: 0, y: 0 }, "The first vertex was not modified.");
	assert_eq!(poly[1], apex::Point2D { x: 200, y: 400 }, "The second vertex was modified.");
	assert_eq!(poly[2], apex::Point2D { x: 10, y: 100 }, "The third vertex was not modified.");
}
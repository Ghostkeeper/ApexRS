/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines the Polygon struct.

use std::iter::FromIterator; //Constructing polygons from iterable lists of vertices.
use std::ops::Index; //Indexing polygons accesses its vertices.
use std::ops::IndexMut; //Indexing polygons accesses its vertices.
use crate::Area; //To return the polygon's surface area.
use crate::Convexity; //To return the polygon's convexity.
use crate::Coordinate;
use crate::Point2D; //The vertices of the Polygon are Point2D.
use crate::Shape2D; //This is a 2D shape.
use crate::TwoDimensional; //This is a two-dimensional object.

use crate::operations::translate; //To translate the polygons.

/// A plane figure consisting of a single contour of straight line segments.
///
/// This is a closed shape, represented by a list of vertices in 2D. Between every two adjacent
/// vertices, as well as between the first and last vertices, is an edge. These edges together form
/// a closed shape that is the contents of the polygon.
///
/// Since the shape only has a single closed polygonal chain, it cannot have multiple boundaries,
/// like a doughnut-shape with a hole inside. This would be a multi-polygon. However the polygon may
/// be self-intersecting. It does not need to be a simple polygon. Operations on the polygon are
/// expected to deal with all such kinds of polygons correctly.
///
/// The polygon does not publicly have a start or end point. However its data model must start
/// somewhere and iterating over the vertices must choose a vertex to start and end at. Operations
/// on the polygons should behave the same regardless of where the polygon starts its iteration.
///
/// If the vertices of the polygon are winding counter-clockwise, the polygon is positive. Otherwise
/// it is negative.
pub struct Polygon {
	/// The vertices that form the closed polygonal chain around this polygon.
	///
	/// These vertices are not publicly accessible, since access to the most up-to-date version may
	/// require a sync from the GPU to the CPU.
	pub(crate) vertices: Vec<Point2D>
}

impl Polygon {
	/// Create a new, empty polygon, without any vertices.
	///
	/// The polygon will be degenerate, since it has no vertices.
	pub const fn new() -> Self {
		Polygon { vertices: Vec::new() }
	}

	/// Create a new, empty polygon, without any vertices.
	///
	/// The polygon will be degenerate, since it has no vertices.
	///
	/// The polygon will reserve memory for a given number of vertices. This guarantees that as long
	/// as the polygon doesn't contain more vertices than that, it will not need to move its
	/// contents to a bigger sized piece of memory. If any more vertices are added, new memory will
	/// need to be allocated and the contents will need to be moved.
	///
	/// # Arguments
	/// * `capacity` - The amount of vertices that this polygon needs to be able to contain without
	/// needing to allocate more memory.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let mut poly = Polygon::with_capacity(4);
	/// //Now add some vertices to it.
	/// //The first 4 vertices are guaranteed to not need additional memory.
	/// poly.push(Point2D { x: 0, y: 0 });
	/// poly.push(Point2D { x: 100, y: 0 });
	/// poly.push(Point2D { x: 200, y: 50 });
	/// poly.push(Point2D { x: 300, y: 150 });
	/// poly.push(Point2D { x: 400, y: 300 }); //But the 5th vertex might cause reallocation!
	/// ```
	pub fn with_capacity(capacity: usize) -> Self {
		Polygon { vertices: Vec::with_capacity(capacity) }
	}

	/// Get the capacity of the polygon's memory allocation to hold vertices.
	///
	/// This is the number of vertices that the polygon could hold without needing to allocate more
	/// memory. Allocating more memory would require the geometric data to be copied, which takes
	/// additional computational resources.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let mut poly = Polygon::with_capacity(4); //Create a polygon with capacity 4.
	/// poly.push(Point2D { x: 0, y: 0 });
	/// poly.push(Point2D { x: 100, y: 0 });
	/// poly.push(Point2D { x: 100, y: 100 });
	/// //We filled 3 of the 4 vertices that the polygon has capacity for.
	/// assert_eq!(poly.capacity(), 4);
	/// //Try adding one more.
	/// poly.push(Point2D { x: 0, y: 100 });
	/// assert_eq!(poly.capacity(), 4); //We're now up to capacity.
	/// //Try adding another.
	/// poly.push(Point2D { x: 50, y: 50 }); //This will be over capacity, causing new memory allocation and copying the data.
	/// assert!(poly.capacity() > 4); //We've had to increase the capacity.
	/// ```
	pub fn capacity(&self) -> usize {
		self.vertices.capacity()
	}

	/// Reserve memory for at least the given amount of vertices to be added to this polygon.
	///
	/// This guarantees that as long as the polygon doesn't receive more additional vertices than
	/// that, it will not need to move its contents to a bigger sized piece of memory. If any more
	/// vertices are added, new memory may need to be allocated and the contents may need to be
	/// moved.
	///
	/// If the capacity is already sufficient, this will not do anything.
	///
	/// # Arguments
	/// * `additional` - How many additional vertices this polygon will need to contain.
	pub fn reserve(&mut self, additional: usize) {
		self.vertices.reserve(additional);
	}

	/// Get the number of vertices (or the number of sides) of a polygon.
	///
	/// This struct represents simple polygons, so the number of sides is equal to the number of
	/// vertices.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// //Construct two polygons with different amounts of vertices.
	/// let triangle = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 100, y: 0 },
	/// 	Point2D { x: 50, y: 87 }
	/// ]);
	/// let pentagon = Polygon::from_iter([
	/// 	Point2D { x: 31, y: 0 },
	/// 	Point2D { x: 131, y: 0 },
	/// 	Point2D { x: 162, y: 95 },
	/// 	Point2D { x: 81, y: 154 },
	/// 	Point2D { x: 0, y: 95 }
	/// ]);
	/// //Now test how many sides each of these polygons has.
	/// assert_eq!(triangle.len(), 3, "A triangle has 3 sides.");
	/// assert_eq!(pentagon.len(), 5, "A pentagon has 5 sides.");
	/// ```
	pub fn len(&self) -> usize {
		self.vertices.len()
	}

	/// Add an extra vertex to this polygon.
	///
	/// The vertex will be connected in the seam of the polygon, after the last vertex and connected
	/// to the first vertex. Adding a vertex can change the properties of the polygon significantly.
	/// Not only does it change the shape of the polygon or its surface area, but it can also
	/// make it degenerate, introduce self-intersections, and so on.
	///
	/// If there is not enough space in the memory reserved for this polygon, a bigger area of
	/// memory will be allocated for it. As such, this operation will often be quick, but can
	/// sometimes take a long time to execute.
	///
	/// # Arguments
	/// * `vertex` - The vertex to add to the polygon.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let mut poly = Polygon::new();
	/// //Create an isosceles triangle by adding these three vertices.
	/// poly.push(Point2D { x: 0, y: 0 });
	/// poly.push(Point2D { x: 100, y: 0 });
	/// poly.push(Point2D { x: 50, y: 100 });
	/// ```
	pub fn push(&mut self, vertex: Point2D) {
		self.vertices.push(vertex);
	}
}

impl TwoDimensional for Polygon {
	fn translate(&mut self, dx: Coordinate, dy: Coordinate) {
		translate::translate_polygon_st(self, dx, dy);
	}
}

impl Shape2D for Polygon {
	fn area(&self) -> Area {
		return 0; //TODO: Implement.
	}

	fn convexity(&self) -> Convexity {
		return Convexity::UNKNOWN; //TODO: Implement.
	}
}

impl FromIterator<Point2D> for Polygon {
	/// Construct a new polygon from a collection of vertices.
	///
	/// The vertices will be copied into the new polygon.
	///
	/// # Arguments
	/// * `iter` - An object that can be converted into an iterator. In other words, an iterable
	/// object. The elements of the objects must be `Point2D` instances which will become the
	/// vertices of the new polygon.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// //Here we feed a literal array of Point2D objects as argument.
	/// let poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 100, y: 0 },
	/// 	Point2D { x: 50, y: 100 }
	/// ]);
	/// assert_eq!(poly[0], Point2D { x: 0, y: 0 });
	/// assert_eq!(poly[1], Point2D { x: 100, y: 0 });
	/// assert_eq!(poly[2], Point2D { x: 50, y: 100 });
	/// ```
	fn from_iter<T>(iter: T) -> Self
			where T: IntoIterator<Item = Point2D> {
		Polygon { vertices: Vec::from_iter(iter) }
	}
}

impl IntoIterator for Polygon {
	type Item = Point2D;
	type IntoIter = std::vec::IntoIter<Point2D>;

	/// Allows iterating over the vertices of the polygon.
	///
	/// This will return an iterator over the vertices, as `Point2D` instances. This will start
	/// iterating at the seam of the polygon, and will enumerate all vertices in counter-clockwise
	/// order (for a positive polygon) or clockwise order (for a negative polygon) until reaching
	/// the seam again.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let vertices = [
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 100, y: 100 },
	/// 	Point2D { x: 0, y: 100 }
	/// ];
	/// let poly = Polygon::from_iter(vertices);
	/// //Now iterate over the vertices.
	/// let mut i = 0;
	/// for vertex in poly { //This for-each loop is possible because Polygon implements IntoIterator.
	/// 	assert_eq!(vertex, vertices[i]);
	/// 	i += 1;
	/// }
	/// ```
	fn into_iter(self) -> Self::IntoIter {
		self.vertices.into_iter()
	}
}

impl Index<usize> for Polygon {
	type Output = Point2D;

	/// Indexes the vertices of this polygon.
	///
	/// This will obtain a single vertex of the polygon. This is typically used to process one
	/// vertex at a time by a custom algorithm, or to extract the resulting computed geometry from
	/// the library into the rest of your application.
	///
	/// This indexing only supports single indices. The polygon can't produce slices.
	///
	/// # Arguments
	/// * `index` - The index of the vertex to obtain.
	///
	/// # Panics
	/// Will panic if the index is equal to or greater than the number of vertices in the polygon.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// //Create a square, with 4 vertices.
	/// let mut poly = Polygon::new();
	/// poly.push(Point2D { x: 0, y: 0 });
	/// poly.push(Point2D { x: 100, y: 0 });
	/// poly.push(Point2D { x: 100, y: 100 });
	/// poly.push(Point2D { x: 0, y: 100 });
	/// //Access one of the vertices.
	/// let third_vertex = poly[2];
	/// assert_eq!(third_vertex, Point2D { x: 100, y: 100 });
	/// ```
	fn index(&self, index: usize) -> &Point2D {
		self.vertices.index(index)
	}
}

impl IndexMut<usize> for Polygon {
	/// Indexes the vertices of this polygon.
	///
	/// This will allow mutating a single vertex of the polygon. This is typically used to process
	/// one vertex at a time by a custom algorithm. Beware though that this doesn't allow processing
	/// the polygon on a graphics card.
	///
	/// This indexing only supports single indices. The polygon can't produce slices.
	///
	/// # Arguments
	/// * `index` - The index of the vertex to mutate.
	///
	/// # Panics
	/// Will panic if the index is equal to or greater than the number of vertices in the polygon.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// //Create a square, with 4 vertices.
	/// let mut poly = Polygon::new();
	/// poly.push(Point2D { x: 0, y: 0 });
	/// poly.push(Point2D { x: 100, y: 0 });
	/// poly.push(Point2D { x: 100, y: 100 });
	/// poly.push(Point2D { x: 0, y: 100 });
	/// //Change one of the vertices.
	/// poly[1].x = 50;
	/// assert_eq!(poly[1], Point2D { x: 50, y: 0 });
	/// ```
	fn index_mut(&mut self, index: usize) -> &mut Point2D {
		self.vertices.index_mut(index)
	}
}
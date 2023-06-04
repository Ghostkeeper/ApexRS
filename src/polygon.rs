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
use crate::detail::sync_status; //To track whether the GPU or CPU copies are up-to-date.
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
#[derive(Debug)]
pub struct Polygon {
	/// The vertices that form the closed polygonal chain around this polygon.
	///
	/// These vertices are not publicly accessible, since access to the most up-to-date version may
	/// require a sync from the GPU to the CPU.
	vertices: Vec<Point2D>,

	/// The up-to-date-ness of the vertex data on the CPU (host) or the GPU.
	///
	/// This tracks whether the CPU version is the most up-to-date version of the vertex data, or
	/// the GPU is, or whether both are in sync (so both are the most up-to-date version).
	///
	/// If the CPU version is the most up-to-date,
	sync_status: sync_status::SyncStatus
}

impl Polygon {
	/// Create a new, empty polygon, without any vertices.
	///
	/// The polygon will be degenerate, since it has no vertices.
	pub const fn new() -> Self {
		Polygon {
			vertices: Vec::new(),
			sync_status: sync_status::SyncStatus::HOST
		}
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
		Polygon {
			vertices: Vec::with_capacity(capacity),
			sync_status: sync_status::SyncStatus::HOST
		}
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
		self.host_vertices().capacity()
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
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let mut poly = Polygon::with_capacity(10); //Create a polygon with capacity 10.
	/// poly.push(Point2D { x: 0, y: 0 });
	/// poly.push(Point2D { x: 100, y: 100 });
	/// poly.push(Point2D { x: 100, y: 0 }); //At this point, there is 7 capacity left.
	/// poly.reserve(5); //Does nothing, since we already had more than 5 capacity left.
	/// assert_eq!(poly.capacity(), 10); //So this is still 10.
	/// poly.reserve(8); //We have too little capacity for 8 additional vertices, so this should reserve more memory.
	/// assert!(poly.capacity() >= 11); //We must have capacity now for at least 3 + 8 vertices (current length + 8 additional).
	/// ```
	pub fn reserve(&mut self, additional: usize) {
		self.host_vertices_mut().reserve(additional);
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
		self.host_vertices().len()
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
		self.host_vertices_mut().push(vertex);
	}

	/// Remove the last vertex before the seam of the polygon and return it.
	///
	/// The vertex before the last vertex will be connected to the first vertex after the seam to
	/// form a new edge.
	///
	/// If the polygon is already empty, return `None`.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let mut poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 1000, y: 0 },
	/// 	Point2D { x: 500, y: 1000 }
	/// ]);
	/// let mut removed = poly.pop();
	/// assert_eq!(removed.unwrap(), Point2D { x: 500, y: 1000 }); //The last vertex was removed.
	/// assert_eq!(poly.len(), 2); //Only 2 vertices left now.
	/// removed = poly.pop();
	/// assert_eq!(removed.unwrap(), Point2D { x: 1000, y: 0 }); //Remove the one that is now last.
	/// assert_eq!(poly.len(), 1); //Only 1 left.
	/// removed = poly.pop();
	/// assert_eq!(removed.unwrap(), Point2D { x: 0, y: 0 }); //Remove the one that is left, which was the first vertex.
	/// assert_eq!(poly.len(), 0); //Nothing left.
	/// removed = poly.pop();
	/// assert_eq!(removed, None); //Since there is nothing to remove, returns None.
	/// ```
	pub fn pop(&mut self) -> Option<Point2D> {
		self.host_vertices_mut().pop()
	}

	/// Inserts a vertex at the given position in the polygonal chain.
	///
	/// The given index is the number of vertices between the new vertex and the seam going
	/// clockwise around the polygon. The vertex with the given index and everything after it will
	/// adjust its index.
	///
	/// # Arguments
	/// * `index` - The position along the polygonal chain where to insert the new vertex.
	/// * `vertex` - The new vertex to insert.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let mut poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 1000, y: 0 },
	/// 	Point2D { x: 1000, y: 1000 },
	/// 	Point2D { x: 0, y: 1000 }
	/// ]);
	/// //Insert a new vertex halfway.
	/// poly.insert(3, Point2D { x: 500, y: 500 });
	/// //The first 3 vertices are not moved.
	/// assert_eq!(poly[0], Point2D { x: 0, y: 0 });
	/// assert_eq!(poly[1], Point2D { x: 1000, y: 0 });
	/// assert_eq!(poly[2], Point2D { x: 1000, y: 1000 });
	/// //Here is where the new vertex was inserted.
	/// assert_eq!(poly[3], Point2D { x: 500, y: 500 });
	/// //The remaining vertices were shifted.
	/// assert_eq!(poly[4], Point2D { x: 0, y: 1000 });
	/// ```
	pub fn insert(&mut self, index: usize, vertex: Point2D) {
		self.host_vertices_mut().insert(index, vertex);
	}

	/// Removes a vertex from the polygonal chain around this polygon and returns the removed
	/// vertex.
	///
	/// The vertices around the removed vertex will be linked together to form a new edge.
	///
	/// # Arguments
	/// * `index` - The index of the vertex to remove.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let mut poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 1000, y: 0 },
	/// 	Point2D { x: 1000, y: 1000 },
	/// 	Point2D { x: 0, y: 1000 }
	/// ]);
	/// //Remove one of the vertices.
	/// let removed_vertex = poly.remove(2);
	/// assert_eq!(removed_vertex, Point2D { x: 1000, y: 1000 }); //This is the removed vertex.
	/// assert_eq!(poly[2], Point2D { x: 0, y: 1000 }); //The last vertex has shifted in its place.
	/// ```
	pub fn remove(&mut self, index: usize) -> Point2D {
		self.host_vertices_mut().remove(index)
	}

	/// Removes all vertices from this polygon, leaving it empty.
	///
	/// The resulting polygon will be degenerate, since it no longer has any vertices.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let mut poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 1000, y: 0 },
	/// 	Point2D { x: 500, y: 1000 }
	/// ]); //This polygon has 3 vertices.
	/// poly.clear(); //But this will remove all of them.
	/// assert_eq!(poly.len(), 0); //No more vertices.
	/// ```
	pub fn clear(&mut self) {
		self.host_vertices_mut().clear();
	}

	/// Obtain a reference to a particular vertex in the polygon.
	///
	/// The index counts the number of vertices from the seam of the polygon. The result is a
	/// reference to the vertex at that position.
	///
	/// If the index is out of bounds of the polygon, this will return `None`.
	///
	/// # Arguments
	/// * `index` - The index of the vertex to get a reference to.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 1000, y: 0 },
	/// 	Point2D { x: 500, y: 1000 }
	/// ]);
	/// let vertex = poly.get(1); //Get the second vertex.
	/// assert_eq!(*vertex.unwrap(), Point2D { x: 1000, y: 0 });
	/// let other_vertex = poly.get(3); //But this is out of range.
	/// assert_eq!(other_vertex, None);
	/// ```
	pub fn get(&self, index: usize) -> Option<&Point2D> {
		self.host_vertices().get(index)
	}

	/// Obtain a mutable reference to a particular vertex in the polygon.
	///
	/// The index counts the number of vertices from the seam of the polygon. The result is a
	/// reference to the vertex at that position.
	///
	/// If the index is out of bounds of the polygon, this will return `None`.
	///
	/// This reference can be used to change a vertex of the polygon in-place.
	///
	/// # Arguments
	/// * `index` - The index of the vertex to get a mutable reference to.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let mut poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 1000, y: 0 },
	/// 	Point2D { x: 500, y: 1000 }
	/// ]);
	/// let vertex = poly.get_mut(1).unwrap(); //Get the second vertex.
	/// vertex.x = 500; //Changes the polygon in-place.
	/// assert_eq!(poly[1], Point2D { x: 500, y: 0 });
	/// ```
	pub fn get_mut(&mut self, index: usize) -> Option<&mut Point2D> {
		self.host_vertices_mut().get_mut(index)
	}

	/// Create an iterator over the vertices of this polygon.
	///
	/// The iterator will enumerate all of the vertices of this polygon in order. The order will be
	/// counter-clockwise if the polygon is a positive shape, starting from the seam.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 667, y: 0 },
	/// 	Point2D { x: 333, y: 1000 }
	/// ]);
	/// let mut iter = poly.iter();
	/// assert_eq!(iter.next(), Some(&Point2D { x: 0, y: 0 }));
	/// assert_eq!(iter.next(), Some(&Point2D { x: 667, y: 0 })); //Counter-clockwise along the polygon's boundary.
	/// assert_eq!(iter.next(), Some(&Point2D { x: 333, y: 1000 }));
	/// assert_eq!(iter.next(), None); //It ran out of vertices, so it stops iterating here.
	/// ```
	pub fn iter(&self) -> core::slice::Iter<Point2D> {
		self.host_vertices().iter()
	}

	/// Create an iterator that allows modifying the vertices of this polygon.
	///
	/// The iterator will enumerate all of the vertices of this polygon in order, and allow the user
	/// to modify them in-place. The order will be counter-clockwise if the polygon is a positive
	/// shape, starting from the seam.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let mut poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 1000, y: 700 },
	/// 	Point2D { x: 0, y: 1000 }
	/// ]);
	/// for vertex in poly.iter_mut() { //Now iterate over the vertices, multiplying all Y coordinates by 2.
	/// 	vertex.y *= 2;
	/// }
	/// assert_eq!(poly[0], Point2D { x: 0, y: 0 });
	/// assert_eq!(poly[1], Point2D { x: 1000, y: 1400 });
	/// assert_eq!(poly[2], Point2D { x: 0, y: 2000 });
	/// ```
	pub fn iter_mut(&mut self) -> core::slice::IterMut<Point2D> {
		self.host_vertices_mut().iter_mut()
	}

	/// Obtain the vertices of this polygon on the host.
	///
	/// If the latest version of the vertices is in the GPU rather than the host, it will be copied
	/// to the host's RAM. If the latest version of the vertices is on the CPU (or they are in
	/// sync), it will simply give a reference to those.
	pub(crate) fn host_vertices<'a>(&'a self) -> &'a Vec<Point2D> {
		&self.vertices //TODO: Sync from GPU if necessary.
	}

	/// Obtain the vertices of this polygon on the host, allowing their modification.
	///
	/// If the latest version of the vertices is in the GPU rather than the host, it will be copied
	/// to the host's RAM. If the latest version of the vertices is on the CPU (or they are in
	/// sync), it will simply give a reference to those.
	pub(crate) fn host_vertices_mut<'a>(&'a mut self) -> &'a mut Vec<Point2D> {
		&mut self.vertices //TODO: Sync from GPU if necessary.
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
		Polygon {
			vertices: Vec::from_iter(iter),
			sync_status: sync_status::SyncStatus::HOST
		}
	}
}

impl<'a> IntoIterator for &'a Polygon {
	type Item = &'a Point2D;
	type IntoIter = std::slice::Iter<'a, Point2D>;

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
		self.host_vertices().into_iter()
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
		self.host_vertices().index(index)
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
		self.host_vertices_mut().index_mut(index)
	}
}

impl AsRef<Polygon> for Polygon {
	/// Convert a polygon into a reference to the same polygon.
	fn as_ref(&self) -> &Polygon {
		self
	}
}

impl AsMut<Polygon> for Polygon {
	/// Convert a polygon into a mutable reference to the same polygon.
	fn as_mut(&mut self) -> &mut Polygon {
		self
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::test::data::polygon;

	/// Test creating a new, empty polygon.
	///
	/// This asserts that the new polygon is empty.
	#[test]
	fn new() {
		let poly = Polygon::new();
		assert_eq!(poly.len(), 0, "The new polygon has no vertices.");
	}

	/// Test creating a polygon with a given capacity.
	///
	/// This asserts that the new polygon is empty, and that the polygon has the given capacity.
	#[test]
	fn with_capacity() {
		let poly = Polygon::with_capacity(10);
		assert_eq!(poly.capacity(), 10, "We require the capacity to be exactly 10 then.");
		assert_eq!(poly.len(), 0, "The new polygon has no vertices.");
	}

	/// Test getting the capacity of a polygon.
	#[test]
	fn capacity() {
		let mut poly = Polygon::with_capacity(3);
		assert_eq!(poly.capacity(), 3, "The polygon was initially created with capacity 3.");
		//The memory is guaranteed to not be reallocated as long as the capacity is not reached.
		//We can sort of see that by testing that the capacity was not increased.
		poly.push(Point2D { x: 0, y: 0 });
		assert_eq!(poly.capacity(), 3, "The capacity was not expanded since there is only 1 vertex in the polygon.");
		poly.push(Point2D { x: 100, y: 0 });
		assert_eq!(poly.capacity(), 3, "The capacity was not expanded since there are only 2 vertices in the polygon.");
		poly.push(Point2D { x: 100, y: 100 });
		assert_eq!(poly.capacity(), 3, "The capacity was not expanded since there are exactly 3 vertices in the polygon.");
		poly.push(Point2D { x: 0, y: 100 });
		assert!(poly.capacity() > 3, "The capacity is now expanded since the number of vertices was over capacity.");
	}

	/// Test reserving memory for more vertices.
	///
	/// This will test whether it will reserve more memory when it doesn't have enough capacity yet, and
	/// that it will do nothing if it does have enough capacity.
	#[test]
	fn reserve() {
		let mut poly = Polygon::with_capacity(10);
		for _ in 0..3 {
			poly.push(Point2D { x: 0, y: 0 });
		}

		//We already have capacity for 7 additional vertices, so this shouldn't do anything.
		poly.reserve(7);
		assert_eq!(poly.capacity(), 10, "The capacity is still 10, since we already had enough space for 7 additional vertices.");

		//We don't have capacity for 8 additional vertices, so this should increase the capacity.
		poly.reserve(8);
		assert!(poly.capacity() >= 11, "We need capacity for at least 8 additional vertices above the current 3.");
	}

	/// Test getting the number of vertices/sides of a polygon.
	#[test]
	fn len() {
		let mut poly = Polygon::new();
		assert_eq!(poly.len(), 0, "The polygon was created without any vertices.");
		poly.push(Point2D { x: 0, y: 0 });
		assert_eq!(poly.len(), 1, "After adding a vertex, the length is now 1.");
		for i in 0..10 { //Add 10 more vertices.
			poly.push(Point2D { x: i + 100, y: i + 100 });
		}
		assert_eq!(poly.len(), 11, "After adding 10 more vertices, the length is now 11.");
	}

	/// Test adding new vertices to a polygon.
	#[test]
	fn push() {
		let mut poly = polygon::square_1000();
		assert_eq!(poly.len(), 4, "The square starts with 4 vertices.");
		poly.push(Point2D { x: 0, y: 100 });
		assert_eq!(poly.len(), 5, "After adding 1 more vertex, there are now 5 vertices.");
		assert_eq!(poly[4], Point2D { x: 0, y: 100 }, "The newly added vertex is at the seam.");
	}

	/// Test removing the last element from the polygon.
	#[test]
	fn pop() {
		let mut poly = polygon::triangle_1000();
		let mut removed = poly.pop();
		assert_eq!(removed.unwrap(), Point2D { x: 524, y: 1024 }, "The last vertex was removed.");
		assert_eq!(poly.len(), 2, "The triangle had 3 vertices, but now only 2.");
		removed = poly.pop();
		assert_eq!(removed.unwrap(), Point2D { x: 1024, y: 24 }, "The second vertex was removed, which was now last.");
		assert_eq!(poly.len(), 1, "The polygon had 2 vertices left, but now only 1.");
		removed = poly.pop();
		assert_eq!(removed.unwrap(), Point2D { x: 24, y: 24 }, "The first vertex was removed, which was the only one remaining.");
		assert_eq!(poly.len(), 0, "This was the last remaining vertex. Nothing is left.");
		removed = poly.pop();
		assert_eq!(removed, None, "There was nothing to remove any more.");
	}

	/// Test inserting a new vertex at the start of a polygon.
	#[test]
	fn insert_start() {
		let mut poly = polygon::triangle_1000();
		poly.insert(0, Point2D { x: 500, y: 500 }); //Insert at the start.
		assert_eq!(poly.len(), 4, "With one additional vertex inserted, there are now 4 vertices.");
		assert_eq!(poly[0], Point2D { x: 500, y: 500 }, "This is the newly inserted vertex.");
		assert_eq!(poly[1], Point2D { x: 24, y: 24 }, "This is the vertex that used to be the first one.");
	}

	/// Test inserting a new vertex in the middle of a polygon.
	#[test]
	fn insert_middle() {
		let mut poly = polygon::triangle_1000();
		poly.insert(2, Point2D { x: 500, y: 500 }); //Insert with 2 vertices before it, and 1 vertex after it.
		assert_eq!(poly.len(), 4, "With one additional vertex inserted, there are now 4 vertices.");
		assert_eq!(poly[0], Point2D { x: 24, y: 24 }, "The first vertex is not moved.");
		assert_eq!(poly[1], Point2D { x: 1024, y: 24 }, "The second vertex is not moved.");
		assert_eq!(poly[2], Point2D { x: 500, y: 500 }, "This is the newly inserted vertex.");
		assert_eq!(poly[3], Point2D { x: 524, y: 1024 }, "This is the vertex that used to be the third vertex.");
	}

	/// Test inserting a new vertex at the end of a polygon.
	#[test]
	fn insert_end() {
		let mut poly = polygon::triangle_1000();
		poly.insert(3, Point2D { x: 500, y: 500 }); //Insert at the end.
		assert_eq!(poly.len(), 4, "With one additional vertex inserted, there are now 4 vertices.");
		assert_eq!(poly[0], Point2D { x: 24, y: 24 }, "The first vertex is not moved.");
		assert_eq!(poly[1], Point2D { x: 1024, y: 24 }, "The second vertex is not moved.");
		assert_eq!(poly[2], Point2D { x: 524, y: 1024 }, "The third vertex is not moved.");
		assert_eq!(poly[3], Point2D { x: 500, y: 500 }, "This is the newly inserted vertex.");
	}

	/// Test removing a vertex from the start of a polygon.
	#[test]
	fn remove_start() {
		let mut poly = polygon::square_1000();
		let removed = poly.remove(0);
		assert_eq!(removed, Point2D { x: 0, y: 0 }, "The first vertex was removed.");
		assert_eq!(poly[0], Point2D { x: 1000, y: 0 }, "The second vertex shifted into the first position.");
		assert_eq!(poly[1], Point2D { x: 1000, y: 1000 }, "The third vertex shifted into the second position.");
		assert_eq!(poly[2], Point2D { x: 0, y: 1000 }, "The fourth vertex shifted into the third position.");
	}

	/// Test removing a vertex from the middle of a polygon.
	#[test]
	fn remove_middle() {
		let mut poly = polygon::square_1000();
		let removed = poly.remove(2);
		assert_eq!(removed, Point2D { x: 1000, y: 1000 }, "The third vertex was removed.");
		assert_eq!(poly[0], Point2D { x: 0, y: 0 }, "The first vertex is still in place.");
		assert_eq!(poly[1], Point2D { x: 1000, y: 0 }, "The second vertex is still in place.");
		assert_eq!(poly[2], Point2D { x: 0, y: 1000 }, "The fourth vertex shifted into the third position.");
	}

	/// Test removing a vertex from the end of a polygon.
	#[test]
	fn remove_end() {
		let mut poly = polygon::square_1000();
		let removed = poly.remove(3);
		assert_eq!(removed, Point2D { x: 0, y: 1000 }, "The fourth vertex was removed.");
		assert_eq!(poly[0], Point2D { x: 0, y: 0 }, "The first vertex is still in place.");
		assert_eq!(poly[1], Point2D { x: 1000, y: 0 }, "The second vertex is still in place.");
		assert_eq!(poly[2], Point2D { x: 1000, y: 1000 }, "The third vertex is still in place.");
	}

	/// Test clearing a polygon.
	#[test]
	fn clear() {
		let mut poly = polygon::square_1000();
		poly.clear();
		assert_eq!(poly.len(), 0, "After clearing, there should no longer be any vertices.");
	}

	/// Test getting a vertex from the polygon.
	///
	/// The vertex we obtain is within the range of the polygon.
	#[test]
	fn get_in_range() {
		let poly = polygon::square_1000();
		let vertex = poly.get(2);
		assert_eq!(*vertex.unwrap(), Point2D { x: 1000, y: 1000 }, "The second vertex was obtained.");
	}

	/// Test getting a vertex from the polygon.
	///
	/// However in this test, the vertex we obtain is out of range, so we should get `None`.
	#[test]
	fn get_out_of_range() {
		let poly = polygon::square_1000();
		let vertex = poly.get(4); //There are 4 vertices, so the 5th element is out of range.
		assert_eq!(vertex, None, "The 5th vertex is out of range, so it should return None.");
	}

	/// Test getting a vertex from the polygon.
	///
	/// The vertex we obtain is within the range of the polygon.
	#[test]
	fn get_mut_in_range() {
		let mut poly = polygon::square_1000();
		let vertex = poly.get_mut(2).unwrap();
		assert_eq!(*vertex, Point2D { x: 1000, y: 1000 }, "The second vertex was obtained.");
		vertex.x = 2000;
		assert_eq!(*vertex, Point2D { x: 2000, y: 1000 }, "The second vertex is now modified.");
		assert_eq!(poly[2], Point2D { x: 2000, y: 1000 }, "And this is also reflected in the polygon itself.");
	}

	/// Test getting a vertex from the polygon.
	///
	/// However in this test, the vertex we obtain is out of range, so we should get `None`.
	#[test]
	fn get_mut_out_of_range() {
		let mut poly = polygon::square_1000();
		let vertex = poly.get_mut(4); //There are 4 vertices, so the 5th element is out of range.
		assert_eq!(vertex, None, "The 5th vertex is out of range, so it should return None.");
	}

	/// Test iterating over the polygon with `iter()`.
	#[test]
	fn iter() {
		let poly = polygon::square_1000();
		let mut iterator = poly.iter();
		assert_eq!(iterator.next(), Some(&Point2D { x: 0, y: 0 }), "First it should encounter the vertex at the seam.");
		assert_eq!(iterator.next(), Some(&Point2D { x: 1000, y: 0 }), "Next the second vertex counter-clockwisely.");
		assert_eq!(iterator.next(), Some(&Point2D { x: 1000, y: 1000 }), "Next the third vertex.");
		assert_eq!(iterator.next(), Some(&Point2D { x: 0, y: 1000 }), "And finally the last vertex.");
		assert_eq!(iterator.next(), None, "After all vertices are iterated over, it should return None.");
	}

	/// Test iterating over the polygon while modifying it with `iter_mut()`.
	#[test]
	fn iter_mut() {
		let mut poly = polygon::square_1000();
		let copy = polygon::square_1000();
		let mut i = 0;
		for vertex in poly.iter_mut() {
			assert_eq!(*vertex, copy[i], "We must iterate over the polygon in index order.");
			i += 1;
			vertex.x += 33;
			vertex.y += 10;
		}
		assert_eq!(poly[0], Point2D { x: 33, y: 10 }, "The first vertex is now shifted by 33,10.");
		assert_eq!(poly[1], Point2D { x: 1033, y: 10 }, "The second vertex is now shifted by 33,10.");
		assert_eq!(poly[2], Point2D { x: 1033, y: 1010 }, "The third vertex is now shifted by 33,10.");
		assert_eq!(poly[3], Point2D { x: 33, y: 1010 }, "The fourth vertex is now shifted by 33,10.");
	}

	/// Test creating a polygon from an iterable object, this time an array.
	#[test]
	fn from_iter_array() {
		let poly = Polygon::from_iter([
			Point2D { x: 0, y: 0 },
			Point2D { x: 500, y: 0 },
			Point2D { x: 250, y: 1000 }
		]);
		assert_eq!(poly[0], Point2D { x: 0, y: 0 }, "The first vertex in the newly created polygon.");
		assert_eq!(poly[1], Point2D { x: 500, y: 0 }, "The second vertex in the newly created polygon.");
		assert_eq!(poly[2], Point2D { x: 250, y: 1000 }, "The third vertex in the newly created polygon.");
	}

	/// Test creating a polygon from an iterable object, this time a vector.
	#[test]
	fn from_iter_vec() {
		let vertices = vec![
			Point2D { x: 0, y: 0 },
			Point2D { x: 500, y: 0 },
			Point2D { x: 250, y: 1000 }
		];
		let poly = Polygon::from_iter(vertices);
		assert_eq!(poly[0], Point2D { x: 0, y: 0 }, "The first vertex in the newly created polygon.");
		assert_eq!(poly[1], Point2D { x: 500, y: 0 }, "The second vertex in the newly created polygon.");
		assert_eq!(poly[2], Point2D { x: 250, y: 1000 }, "The third vertex in the newly created polygon.");
	}

	/// Test creating a polygon from an iterable object, this time a different polygon.
	#[test]
	fn from_iter_polygon() {
		let original = Polygon::from_iter([
			Point2D { x: 0, y: 0 },
			Point2D { x: 500, y: 0 },
			Point2D { x: 250, y: 1000 }
		]);
		let poly = Polygon::from_iter(original.iter());
		assert_eq!(poly[0], Point2D { x: 0, y: 0 }, "The first vertex in the newly created polygon.");
		assert_eq!(poly[1], Point2D { x: 500, y: 0 }, "The second vertex in the newly created polygon.");
		assert_eq!(poly[2], Point2D { x: 250, y: 1000 }, "The third vertex in the newly created polygon.");
	}

	/// Test iterating over vertices in a polygon.
	#[test]
	fn into_iter() {
		let vertices = [
			Point2D { x: 0, y: 0 },
			Point2D { x: 100, y: 0 },
			Point2D { x: 50, y: 100 }
		];
		let poly = Polygon::from_iter(vertices);
		let mut i = 0;
		for vertex in &poly {
			assert_eq!(*vertex, vertices[i], "The iterator must iterate over the vertices in order.");
			i += 1;
		}
	}

	/// Test accessing vertices of the polygon.
	///
	/// All access in this test is within range.
	#[test]
	fn index_in_range() {
		let poly = Polygon::from_iter([
			Point2D { x: 0, y: 0 },
			Point2D { x: 50, y: 10 },
			Point2D { x: 10, y: 100 }
		]);
		assert_eq!(poly[0], Point2D { x: 0, y: 0 }, "Getting the first vertex at index 0.");
		assert_eq!(poly[1], Point2D { x: 50, y: 10 }, "Getting the second vertex at index 1.");
		assert_eq!(poly[2], Point2D { x: 10, y: 100 }, "Getting the third vertex at index 2.");
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
		let mut poly = Polygon::from_iter([
			Point2D { x: 0, y: 0 },
			Point2D { x: 50, y: 10 },
			Point2D { x: 10, y: 100 }
		]);
		poly[1] = Point2D { x: 200, y: 400 };
		assert_eq!(poly[0], Point2D { x: 0, y: 0 }, "The first vertex was not modified.");
		assert_eq!(poly[1], Point2D { x: 200, y: 400 }, "The second vertex was modified.");
		assert_eq!(poly[2], Point2D { x: 10, y: 100 }, "The third vertex was not modified.");
	}
}
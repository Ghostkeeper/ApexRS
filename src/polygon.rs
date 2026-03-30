/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines the Polygon struct.

use std::cell::{Ref, RefCell, RefMut}; //For interior mutability to keep CPU and GPU in sync.
use std::fmt; //You can print polygons as text.
use std::iter::FromIterator; //Constructing polygons from iterable lists of vertices.
use std::num::NonZeroU64; //For communicating buffer sizes to the GPU.
use std::rc::Rc; //For interior mutability to keep CPU and GPU in sync.
use wgpu::{ //For computing on the GPU.
	BindGroupDescriptor,
	BindGroupEntry,
	BindGroupLayoutDescriptor,
	BindGroupLayoutEntry,
	BindingType,
	Buffer,
	BufferBindingType,
	BufferDescriptor,
	BufferUsages,
	CommandEncoderDescriptor,
	ComputePassDescriptor,
	ComputePipelineDescriptor,
	MapMode,
	PipelineCompilationOptions,
	PipelineLayoutDescriptor,
	PollType,
	ShaderModule,
	ShaderStages
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::Angle; //To implement TwoDimensional.
use crate::Area; //To return the polygon's surface area.
use crate::Convexity; //To return the polygon's convexity.
use crate::Coordinate;
use crate::Point2D; //The vertices of the Polygon are Point2D.
use crate::Shape2D; //This is a 2D shape.
use crate::TwoDimensional; //This is a two-dimensional object.
use crate::detail::gpu::GPU; //To perform calculations on the GPU.
use crate::detail::sync_status::SyncStatus; //To track whether the GPU or CPU copies are up-to-date.
use crate::operations::{area, rotate, scale, translate}; //To translate the polygons.

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
///
/// # Basic usage
/// A polygon can be constructed from an iterable data source, like so:
/// ```
/// use apex::{Point2D, Polygon};
/// //Some vertex data from your application's input data.
/// let verts = vec!(
/// 	Point2D { x: 0, y: 0 }, //In this case, a square.
/// 	Point2D { x: 1000, y: 0 },
/// 	Point2D { x: 1000, y: 1000 },
/// 	Point2D { x: 0, y: 1000 },
/// );
/// let square = Polygon::from_iter(verts); //Put the vertices in a polygon.
///
/// //You can also construct a polygon by creating an empty one and then pushing vertices onto it.
/// let mut triangle = Polygon::new();
/// triangle.push(Point2D { x: 0, y: 0 });
/// triangle.push(Point2D { x: 1000, y: 0 });
/// triangle.push(Point2D { x: 500, y: 1000 });
/// ```
///
/// # Host vs. GPU
/// Apex decides for itself whether to use the host or the GPU for each operation, based on where
/// the data is at the moment and how long an operation would take on each device. If the data
/// currently resides on the host (RAM), but the operation would be much faster on a GPU, the data
/// will be copied to the GPU and kept there. If the data is currently in the GPU (VRAM) but is
/// needed on the CPU for outputting or because the operation is more efficient there, then it will
/// be copied to the CPU.
///
/// Transfer time between the host and the CPU is significant, but this is taken into account in
/// order to decide where an operation should be calculated.
pub struct Polygon {
	/// The vertices that form the closed polygonal chain around this polygon.
	///
	/// This is the copy of the vertices that is in the host CPU's RAM. These vertices are not
	/// publicly accessible, since access to the most up-to-date version may require a sync from the
	/// GPU to the CPU.
	vertices: Rc<RefCell<Vec<Point2D>>>,

	/// The vertices that form the closed polygonal chain around this polygon.
	///
	/// This is the copy of the vertices that is on the GPU slave. These vertices are not publicly
	/// accessible, since access to the most up-to-date version may require a sync from the CPU to
	/// the GPU.
	///
	/// Before the first time that the polygon gets synced to the GPU, this will be `None`.
	gpu_buffer: Rc<RefCell<Option<Buffer>>>,

	/// A WGPU buffer used to transfer the vertex data between the CPU (host) and the GPU.
	///
	/// Sometimes this buffer will contain the vertex data, but it cannot be relied on to be
	/// up-to-date.
	transfer_buffer: Rc<RefCell<Option<Buffer>>>,

	/// The up-to-date-ness of the vertex data on the CPU (host) or the GPU.
	///
	/// This tracks whether the CPU version is the most up-to-date version of the vertex data, or
	/// the GPU is, or whether both are in sync (so both are the most up-to-date version).
	///
	/// If the CPU version is the most up-to-date,
	sync_status: Rc<RefCell<SyncStatus>>,
}

impl Polygon {
	/// Create a new, empty polygon, without any vertices.
	///
	/// The polygon will be degenerate, since it has no vertices.
	pub fn new() -> Self {
		Polygon {
			vertices: Rc::new(RefCell::new(vec!())),
			gpu_buffer: Rc::new(RefCell::new(None)),
			transfer_buffer: Rc::new(RefCell::new(None)),
			sync_status: Rc::new(RefCell::new(SyncStatus::HOST)),
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
			vertices: Rc::new(RefCell::new(Vec::with_capacity(capacity))),
			gpu_buffer: Rc::new(RefCell::new(None)),
			transfer_buffer: Rc::new(RefCell::new(None)),
			sync_status: Rc::new(RefCell::new(SyncStatus::HOST)),
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
	/// 	Point2D { x: 50, y: 87 },
	/// ]);
	/// let pentagon = Polygon::from_iter([
	/// 	Point2D { x: 31, y: 0 },
	/// 	Point2D { x: 131, y: 0 },
	/// 	Point2D { x: 162, y: 95 },
	/// 	Point2D { x: 81, y: 154 },
	/// 	Point2D { x: 0, y: 95 },
	/// ]);
	/// //Now test how many sides each of these polygons has.
	/// assert_eq!(triangle.len(), 3, "A triangle has 3 sides.");
	/// assert_eq!(pentagon.len(), 5, "A pentagon has 5 sides.");
	/// ```
	pub fn len(&self) -> usize {
		self.host_vertices().len()
	}

	/// Get a reference to a vertex in the polygon.
	///
	/// The vertex is a point where two of the edges meet. The polygon consists of a chain of
	/// vertices connected by edges. The vertices are addressed by an index, starting from the seam
	/// of the polygon, numbering from 0.
	///
	/// Using the vertex, you can get the coordinates of this corner of the polygon.
	///
	/// # Arguments
	/// * `index` - The index of the vertex to address.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// //When we access the vertices, it will be indexed in the same order as it is constructed.
	/// let poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 100, y: 0 },
	/// 	Point2D { x: 50, y: 87 },
	/// ]);
	/// //Now let's get the vertices.
	/// assert_eq!(*poly.vertex(0), Point2D{ x: 0, y: 0 }, "Accessing the 0th index gets the vertex at the seam of the polygon.");
	/// assert_eq!(*poly.vertex(1), Point2D{ x: 100, y: 0 }, "Accessing the 1st index gets the next down the list.");
	/// assert_eq!(*poly.vertex(2), Point2D{ x: 50, y: 87 }, "Accessing the 2nd vertex gets the next vertex (in this case the last).");
	/// //Accessing poly.vertex(3) would panic, since it's out of range.
	/// ```
	pub fn vertex<'a>(&'a self, index: usize) -> Ref<'a, Point2D> {
		Ref::map(self.host_vertices(), |verts| &verts[index])
	}

	/// Get a mutable reference to a vertex in the polygon.
	///
	/// The vertex is a point where two of the edges meet. The polygon consists of a chain of
	/// vertices connected by edges. The vertices are addressed by an index, starting from the seam
	/// of the polygon, numbering from 0.
	///
	/// The reference to the vertex is mutable, and changing the contents of the reference will
	/// cause that vertex of the polygon to change.
	///
	/// # Arguments
	/// * `index` - The index of the vertex to address.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// //When we access the vertices, it will be indexed in the same order as it is constructed.
	/// let mut poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 100, y: 0 },
	/// 	Point2D { x: 50, y: 87 },
	/// ]);
	/// //Now let's get the vertices.
	/// assert_eq!(*poly.vertex_mut(0), Point2D{ x: 0, y: 0 }, "This gets the vertex at the seam. Although the reference is mutable, we're not mutating it here.");
	/// *poly.vertex_mut(1) = Point2D { x: 200, y: 200 }; //We can change the vertices this way.
	/// assert_eq!(*poly.vertex(0), Point2D{ x: 0, y: 0 }, "The 0th vertex didn't change.");
	/// assert_eq!(*poly.vertex(1), Point2D{ x: 200, y: 200 }, "The 1st vertex was mutated.");
	/// assert_eq!(*poly.vertex(2), Point2D{ x: 50, y: 87 }, "The 2nd vertex didn't change.");
	/// ```
	pub fn vertex_mut<'a>(&'a mut self, index: usize) -> RefMut<'a, Point2D> {
		RefMut::map(self.host_vertices_mut(), |verts| &mut verts[index])
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
	/// 	Point2D { x: 500, y: 1000 },
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
	/// 	Point2D { x: 0, y: 1000 },
	/// ]);
	/// //Insert a new vertex halfway.
	/// poly.insert(3, Point2D { x: 500, y: 500 });
	/// //The first 3 vertices are not moved.
	/// assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 });
	/// assert_eq!(*poly.vertex(1), Point2D { x: 1000, y: 0 });
	/// assert_eq!(*poly.vertex(2), Point2D { x: 1000, y: 1000 });
	/// //Here is where the new vertex was inserted.
	/// assert_eq!(*poly.vertex(3), Point2D { x: 500, y: 500 });
	/// //The remaining vertices were shifted.
	/// assert_eq!(*poly.vertex(4), Point2D { x: 0, y: 1000 });
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
	/// 	Point2D { x: 0, y: 1000 },
	/// ]);
	/// //Remove one of the vertices.
	/// let removed_vertex = poly.remove(2);
	/// assert_eq!(removed_vertex, Point2D { x: 1000, y: 1000 }); //This is the removed vertex.
	/// assert_eq!(*poly.vertex(2), Point2D { x: 0, y: 1000 }); //The last vertex has shifted in its place.
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
	/// 	Point2D { x: 500, y: 1000 },
	/// ]); //This polygon has 3 vertices.
	/// poly.clear(); //But this will remove all of them.
	/// assert_eq!(poly.len(), 0); //No more vertices.
	/// ```
	pub fn clear(&mut self) {
		self.host_vertices_mut().clear();
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
	/// 	Point2D { x: 333, y: 1000 },
	/// ]);
	/// let mut iter = poly.iter();
	/// assert_eq!(*iter.next().expect("There should be 3 vertices."), Point2D { x: 0, y: 0 });
	/// assert_eq!(*iter.next().expect("There should be 3 vertices."), Point2D { x: 667, y: 0 }); //Counter-clockwise along the polygon's boundary.
	/// assert_eq!(*iter.next().expect("There should be 3 vertices."), Point2D { x: 333, y: 1000 });
	/// assert!(iter.next().is_none()); //It ran out of vertices, so it stops iterating here.
	/// ```
	pub fn iter<'a>(&'a self) -> PolygonIterator<'a> {
		PolygonIterator {
			vertices_ref: Some(Ref::map(self.vertices.borrow(), |v| &v[..])),
		}
	}

	/// Create an iterator over the vertices of this polygon that allows modification.
	///
	/// The iterator will enumerate all of the vertices of this polygon in order. The order will be
	/// counter-clockwise if the polygon is a positive shape, starting from the seam.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon};
	/// let mut poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 667, y: 0 },
	/// 	Point2D { x: 333, y: 1000 },
	/// ]);
	/// for mut vertex in poly.iter_mut() {
	/// 	vertex.x *= 2;  // vertex is a RefMut<Point2D> so it can be edited by reference.
	/// }
	/// //The X coordinates are now all doubled.
	/// assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0});
	/// assert_eq!(*poly.vertex(1), Point2D { x: 1334, y: 0});
	/// assert_eq!(*poly.vertex(2), Point2D { x: 666, y: 1000});
	/// ```
	pub fn iter_mut<'a>(&'a mut self) -> PolygonIteratorMut<'a> {
		PolygonIteratorMut {
			vertices_ref: Some(RefMut::map(self.vertices.borrow_mut(), |v| &mut v[..])),
		}
	}

	/// Obtain the vertices of this polygon on the host.
	///
	/// If the latest version of the vertices is in the GPU rather than the host, it will be copied
	/// to the host's RAM. If the latest version of the vertices is on the CPU (or they are in
	/// sync), it will simply give a reference to those.
	pub(crate) fn host_vertices<'a>(&'a self) -> Ref<'a, Vec<Point2D>> {
		if self.sync_status.borrow().eq(&SyncStatus::GPU) { //Host is outdated.
			self.sync_gpu_to_host();
		}
		self.vertices.borrow()
	}

	/// Obtain the vertices of this polygon on the host, allowing their modification.
	///
	/// If the latest version of the vertices is in the GPU rather than the host, it will be copied
	/// to the host's RAM. If the latest version of the vertices is on the CPU (or they are in
	/// sync), it will simply give a reference to those.
	pub(crate) fn host_vertices_mut<'a>(&'a mut self) -> RefMut<'a, Vec<Point2D>> {
		if self.sync_status.borrow().eq(&SyncStatus::GPU) { //Host is outdated.
			self.sync_gpu_to_host();
		}
		self.vertices.borrow_mut()
	}

	/// Obtain the vertices of this polygon on the GPU.
	///
	/// If the latest version of the vertices is in the host rather than the GPU, it will be copied
	/// to the GPU first. If the latest version of the vertices is in the GPU (or they are in sync),
	/// it will simply give a reference to those.
	///
	/// While this returns an ``Option`` due to the internal data structure in this polygon, the
	/// resulting ``Option`` is guaranteed to be ``Some``.
	///
	/// There is no mutable version of this function because the GPU buffers are detached from this
	/// polygon in CPU memory. Mutability has to be enforced through the operations that may modify
	/// the polygonal data.
	pub(crate) fn gpu_vertices<'a>(&'a self) -> Ref<'a, Option<Buffer>> {
		if self.sync_status.borrow().eq(&SyncStatus::HOST) { //GPU is outdated.
			self.sync_host_to_gpu();
		}
		self.gpu_buffer.borrow()
	}

	/// Synchronise the vertex data of this polygon from the host's memory to the GPU.
	///
	/// This is necessary for calculations on the GPU. This should be performed if the
	/// ``sync_status`` is set to ``HOST``. If the ``sync_status`` is set to ``GPU`` or ``SYNCED``,
	/// it will have no effect.
	fn sync_host_to_gpu(&self) {
		if self.host_vertices().is_empty() {
			*self.gpu_buffer.borrow_mut() = None;
		} else {
			self.gpu_buffer.borrow_mut().replace(GPU.device.create_buffer_init(&BufferInitDescriptor {
				label: None,
				contents: bytemuck::cast_slice(self.host_vertices().as_slice()),
				usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, //Both reading and writing.
			}));
		}
		*self.sync_status.borrow_mut() = SyncStatus::SYNCED;
	}

	/// Synchronise the vertex data of this polygon from the GPU's memory to the host.
	///
	/// If the GPU has changed the vertex data through some transformation, and the information is
	/// accessed by the CPU, it will need to be synchronised first. This should be performed if the
	/// ``sync_status`` is set to ``GPU``. If the ``sync_status`` is set to ``HOST`` or ``SYNCED``,
	/// it will have no effect.
	fn sync_gpu_to_host(&self) {
		let mut encoder = GPU.device.create_command_encoder(&CommandEncoderDescriptor {
			label: None,
		});
		if self.gpu_buffer.borrow().is_none() {
			self.vertices.borrow_mut().clear();
		} else {
			let buffer_size = self.gpu_vertices().as_ref().expect("The GPU needs to have data before we can synchronise it to the host.").size();
			self.transfer_buffer.borrow_mut().replace(GPU.device.create_buffer(&BufferDescriptor {
				label: None,
				size: buffer_size,
				usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
				mapped_at_creation: false,
			}));
			encoder.copy_buffer_to_buffer(
				self.gpu_vertices().as_ref().unwrap(), 0,
				self.transfer_buffer.borrow().as_ref().unwrap(), 0,
				buffer_size,
			);
			let command_buffer = encoder.finish(); //Finish the compilation.
			GPU.queue.submit([command_buffer]); //Execute the commands.

			self.transfer_buffer.borrow().as_ref().unwrap().slice(..).map_async(MapMode::Read, |_| {});
			let _ = GPU.device.poll(PollType::wait_indefinitely());
			let slice: &[u8] = &self.transfer_buffer.borrow().as_ref().unwrap().slice(..).get_mapped_range();
			*self.vertices.borrow_mut() = bytemuck::cast_slice(slice).to_vec();
		}
		*self.sync_status.borrow_mut() = SyncStatus::SYNCED;
	}

	/// Execute a compute kernel on the GPU that would mutate the polygon.
	///
	/// This creates a uniform buffer for the parameters, a binding group layout, a binding group,
	/// a pipeline layout, a pipeline, an encoder and a compute pass. It then uses that encoder to
	/// gather up all of these instructions for the GPU and submits that to the device.
	///
	/// The kernel is a bit restricted, for the purpose of calling this function easier:
	/// * The entrypoint of the kernel must be called `main`. That's the function we'll call from
	///   the compute pipeline.
	/// * The kernel will have two buffers bound to it: binding 0 will be the uniform buffer, and
	///   binding 1 will be the vertex data of this polygon.
	///
	/// # Arguments
	/// * `shader_module` - The kernel to execute on the GPU.
	/// * `uniform_data` - Uniform data to pass to the GPU in order to configure parameters to the
	///   kernel.
	pub(crate) fn execute_gpu_kernel_mut(&mut self, shader_module: &ShaderModule, uniform_data: &[u8]) {
		//The parameters of the kernel are communicated via Uniforms, in this uniform buffer.
		let uniform_buffer = GPU.device.create_buffer_init(&BufferInitDescriptor {
			label: Some("Uniform"),
			contents: uniform_data,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
		});

		//All data communicated to execute the kernel is put in buffers.
		//We need to tell the GPU what these buffers are, where to find them and how to call them in the shader.
		let bind_group_layout = GPU.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			label: Some("Bind Group Layout"),
			entries: &[
				//In binding position 0: The uniform buffer.
				BindGroupLayoutEntry {
					binding: 0,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Buffer {
						ty: BufferBindingType::Uniform { },
						min_binding_size: Some(NonZeroU64::new(uniform_data.len() as u64).unwrap()),
						has_dynamic_offset: false,
					},
					count: None,
				},
				//In binding position 1: The vertex data.
				BindGroupLayoutEntry {
					binding: 1,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Buffer {
						ty: BufferBindingType::Storage { read_only: false },
						min_binding_size: None,
						has_dynamic_offset: false,
					},
					count: None,
				},
			],
		});
		//Then bind the actual buffers according to the layout above.
		//If there is no data, create a dummy array, because WGPU doesn't allow zero-length buffers.
		//Hopefully the compiler can manage to safely move this into the if-statement.
		let dummy_buffer = GPU.device.create_buffer_init(&BufferInitDescriptor {
			label: None,
			contents: &[0u8; 8],
			usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, //Both reading and writing.
		});
		let gpu_vertices = self.gpu_vertices();
		let bind_group = GPU.device.create_bind_group(&BindGroupDescriptor {
			label: Some("Bind Group"),
			layout: &bind_group_layout,
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: uniform_buffer.as_entire_binding(),
				},
				BindGroupEntry {
					binding: 1,
					resource: if gpu_vertices.is_none() {
						&dummy_buffer
					} else {
						gpu_vertices.as_ref().expect("Upload the polygon to the GPU first.")
					}.as_entire_binding(),
				},
			],
		});

		//Also communicated to the GPU is the pipeline: Compiled code telling it what to do.
		let pipeline_layout = GPU.device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("Pipeline Layout"),
			bind_group_layouts: &[&bind_group_layout],
			immediate_size: 0,
		});
		let pipeline = GPU.device.create_compute_pipeline(&ComputePipelineDescriptor {
			label: Some("Pipeline"),
			layout: Some(&pipeline_layout),
			module: &shader_module,
			entry_point: Some("main"),
			compilation_options: PipelineCompilationOptions::default(),
			cache: None,
		});
		let mut encoder = GPU.device.create_command_encoder(&CommandEncoderDescriptor {
			label: Some("Encoder"),
		});
		let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: Some("Compute Pass"),
			timestamp_writes: None,
		});
		compute_pass.set_pipeline(&pipeline);
		compute_pass.set_bind_group(0, &bind_group, &[]);
		compute_pass.dispatch_workgroups(64, 1, 1);
		drop(compute_pass); //Now that we've dispatched the workgroups, we can drop the compute pass so that we can access the encoder again.
		let command_buffer = encoder.finish(); //Finish the compilation.

		*self.sync_status.borrow_mut() = SyncStatus::GPU; //From here on out, the CPU data may be out of date.
		GPU.queue.submit([command_buffer]); //Execute the commands.
	}
}

impl TwoDimensional for Polygon {
	/// Move the polygon across the two-dimensional space.
	///
	/// This causes the position of the polygon to change, but doesn't otherwise transform it. The
	/// polygon is not rotated, scaled or deformed in any way.
	///
	/// The polygon is translated in-place, causing the polygon to be modified.
	///
	/// # Arguments
	/// * `dx` - How far to move the polygon in the X direction. Use a positive number to increase
	/// the X position, or a negative number to reduce the X position.
	/// * `dy` - How far to move the polygon in the Y direction. Use a positive number to increase
	/// the Y position, or a negative number to reduce the Y position.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon, TwoDimensional};
	/// //Create a triangular polygon.
	/// let mut poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 100, y: 0 },
	/// 	Point2D { x: 67, y: 100 }
	/// ]);
	/// //Move the polygon.
	/// poly.translate(100, -150);
	/// //Now, all of the vertices will have moved.
	/// assert_eq!(*poly.vertex(0), Point2D { x: 100, y: -150 });
	/// assert_eq!(*poly.vertex(1), Point2D { x: 200, y: -150 });
	/// assert_eq!(*poly.vertex(2), Point2D { x: 167, y: -50 });
	/// ```
	fn translate(&mut self, dx: Coordinate, dy: Coordinate) {
		translate::translate_polygon_st(self, dx, dy);
	}

	/// Scale the polygon away from the coordinate origin.
	///
	/// This causes the polygon to become bigger or smaller, and simultaneously to move away from or
	/// closer to the coordinate origin. It can also cause the shape to appear squished or
	/// stretched, because the scale factors can be different between the X and the Y axis.
	///
	/// The polygon is scaled in-place, causing the polygon to be modified.
	///
	/// # Arguments
	/// * `x` - The scaling factor for the X axis. Use a number greater than 1 to make the polygon
	/// wider, or smaller than 1 to make the polygon smaller. Use a negative number to mirror the
	/// polygon horizontally.
	/// * `y` - The scaling factor for the Y axis. Use a number greater than 1 to make the polygon
	/// taller, or smaller than 1 to make the polygon shorter. Use a negative number to mirror the
	/// polygon vertically.
	///
	/// # Examples
	/// ```
	/// use apex::{Point2D, Polygon, TwoDimensional};
	/// //Create a triangular polygon.
	/// let mut poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 100, y: 0 },
	/// 	Point2D { x: 67, y: 100},
	/// ]);
	/// //Scale the polygon.
	/// poly.scale(2.0, 1.5);
	/// //Now, the polygon will be scaled to be bigger.
	/// assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 });
	/// assert_eq!(*poly.vertex(1), Point2D { x: 200, y: 0 });
	/// assert_eq!(*poly.vertex(2), Point2D { x: 134, y: 150 });
	/// ```
	fn scale(&mut self, x: f64, y: f64) {
		scale::scale_polygon_st(self, x, y);
	}

	/// Rotate the polygon around the coordinate origin.
	///
	/// This causes the polygon to turn. It doesn't necessarily turn around its own centre, just
	/// around the 0,0 coordinate. The polygon is not scaled or deformed in any way.
	///
	/// The polygon is scaled in-place, causing the polygon to be modified.
	///
	/// # Arguments
	/// * `angle` - How much to rotate the polygon.
	///
	/// # Examples
	/// ```
	/// use apex::{Angle, Point2D, Polygon, TwoDimensional};
	/// //Create a triangular polygon.
	/// let mut poly = Polygon::from_iter([
	/// 	Point2D { x: 0, y: 0 },
	/// 	Point2D { x: 100, y: 0 },
	/// 	Point2D { x: 67, y: 100},
	/// ]);
	/// //Rotate the polygon.
	/// apex::operations::rotate::rotate_polygon_gpu(&mut poly, Angle::EIGHTH_TURN);
	/// //Now, the polygon will be rotated 45 degrees counter-clockwise.
	/// assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 });
	/// assert_eq!(*poly.vertex(1), Point2D { x: 71, y: 71 });
	/// assert_eq!(*poly.vertex(2), Point2D { x: -23, y: 118 });
	/// ```
	fn rotate(&mut self, angle: Angle) {
		rotate::rotate_polygon_st(self, angle.clone());
	}
}

impl Shape2D for Polygon {
	fn area(&self) -> Area {
		area::area_polygon_st(self)
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
	/// 	Point2D { x: 50, y: 100 },
	/// ]);
	/// assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 });
	/// assert_eq!(*poly.vertex(1), Point2D { x: 100, y: 0 });
	/// assert_eq!(*poly.vertex(2), Point2D { x: 50, y: 100 });
	/// ```
	fn from_iter<T>(iter: T) -> Self
			where T: IntoIterator<Item = Point2D> {
		Polygon {
			vertices: Rc::new(RefCell::new(Vec::from_iter(iter))),
			gpu_buffer: Rc::new(RefCell::new(None)),
			transfer_buffer: Rc::new(RefCell::new(None)),
			sync_status: Rc::new(RefCell::new(SyncStatus::HOST)),
		}
	}
}

impl fmt::Debug for Polygon {
	/// A reference string representing this polygon, for debugging the polygon in a log or CLI
	/// output.
	///
	/// The resulting formatting looks something like this:
	/// `Polygon { vertices: [Point2D { x: 0, y: 0 }, Point2D { x: 100, y: 0 }, Point2D { x: 50, y: 100 }] }`
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("Polygon")
			.field("vertices", &self.host_vertices().as_slice())
			.finish()
	}
}

/// An iterator over the vertices of a polygon.
///
/// This iterator holds a reference to the vertex data in the polygon. The reference is a guard to
/// borrow the polygon's data. While the iterator is in use, the reference will be kept alive so
/// that iteration can continue safely.
pub struct PolygonIterator<'a> {
	/// A reference to a slice of polygon data.
	///
	/// This uses a slice of the vertex data in order to use the slice's built-in ability to get a
	/// reference to all of its elements.
	vertices_ref: Option<Ref<'a, [Point2D]>>,
}

impl<'a> Iterator for PolygonIterator<'a> {
	/// The type of element we're iterating over.
	type Item = Ref<'a, Point2D>;

	/// Get the next item of the iteration.
	fn next(&mut self) -> Option<Self::Item> {
		if self.vertices_ref.is_none() {
			return None;
		}
		if let Some(borrow) = self.vertices_ref.take() {
			if borrow.is_empty() {
				return None;
			}
			let (head, tail) = Ref::map_split(borrow, |slice| {
				slice.split_at(1)
			});
			self.vertices_ref.replace(tail);
			return Some(Ref::map(head, |slice| &slice[0]));
		}
		None
	}
}

/// A mutable iterator over the vertices of a polygon.
///
/// This iterator holds a reference to the vertex data in the polygon. The reference is a guard to
/// borrow the polygon's data. While the iterator is in use, the reference will be kept alive so
/// that iteration can continue safely.
///
/// This iterator requires a mutable polygon, and then returns mutable references.
pub struct PolygonIteratorMut<'a> {
	/// A reference to a slice of polygon data.
	///
	/// This uses a slice of the vertex data in order to use the slice's built-in ability to get a
	/// reference to all of its elements.
	vertices_ref: Option<RefMut<'a, [Point2D]>>,
}

impl<'a> Iterator for PolygonIteratorMut<'a> {
	/// The type of element we're iterating over.
	type Item = RefMut<'a, Point2D>;

	/// Get the next item of the iteration.
	fn next(&mut self) -> Option<Self::Item> {
		if self.vertices_ref.is_none() {
			return None;
		}
		if let Some(borrow) = self.vertices_ref.take() {
			if borrow.is_empty() {
				return None;
			}
			let (head, tail) = RefMut::map_split(borrow, |slice| {
				slice.split_at_mut(1)
			});
			self.vertices_ref.replace(tail);
			return Some(RefMut::map(head, |slice| &mut slice[0]));
		}
		None
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
		assert_eq!(*poly.vertex(4), Point2D { x: 0, y: 100 }, "The newly added vertex is at the seam.");
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
		assert_eq!(*poly.vertex(0), Point2D { x: 500, y: 500 }, "This is the newly inserted vertex.");
		assert_eq!(*poly.vertex(1), Point2D { x: 24, y: 24 }, "This is the vertex that used to be the first one.");
	}

	/// Test inserting a new vertex in the middle of a polygon.
	#[test]
	fn insert_middle() {
		let mut poly = polygon::triangle_1000();
		poly.insert(2, Point2D { x: 500, y: 500 }); //Insert with 2 vertices before it, and 1 vertex after it.
		assert_eq!(poly.len(), 4, "With one additional vertex inserted, there are now 4 vertices.");
		assert_eq!(*poly.vertex(0), Point2D { x: 24, y: 24 }, "The first vertex is not moved.");
		assert_eq!(*poly.vertex(1), Point2D { x: 1024, y: 24 }, "The second vertex is not moved.");
		assert_eq!(*poly.vertex(2), Point2D { x: 500, y: 500 }, "This is the newly inserted vertex.");
		assert_eq!(*poly.vertex(3), Point2D { x: 524, y: 1024 }, "This is the vertex that used to be the third vertex.");
	}

	/// Test inserting a new vertex at the end of a polygon.
	#[test]
	fn insert_end() {
		let mut poly = polygon::triangle_1000();
		poly.insert(3, Point2D { x: 500, y: 500 }); //Insert at the end.
		assert_eq!(poly.len(), 4, "With one additional vertex inserted, there are now 4 vertices.");
		assert_eq!(*poly.vertex(0), Point2D { x: 24, y: 24 }, "The first vertex is not moved.");
		assert_eq!(*poly.vertex(1), Point2D { x: 1024, y: 24 }, "The second vertex is not moved.");
		assert_eq!(*poly.vertex(2), Point2D { x: 524, y: 1024 }, "The third vertex is not moved.");
		assert_eq!(*poly.vertex(3), Point2D { x: 500, y: 500 }, "This is the newly inserted vertex.");
	}

	/// Test removing a vertex from the start of a polygon.
	#[test]
	fn remove_start() {
		let mut poly = polygon::square_1000();
		let removed = poly.remove(0);
		assert_eq!(removed, Point2D { x: 0, y: 0 }, "The first vertex was removed.");
		assert_eq!(*poly.vertex(0), Point2D { x: 1000, y: 0 }, "The second vertex shifted into the first position.");
		assert_eq!(*poly.vertex(1), Point2D { x: 1000, y: 1000 }, "The third vertex shifted into the second position.");
		assert_eq!(*poly.vertex(2), Point2D { x: 0, y: 1000 }, "The fourth vertex shifted into the third position.");
	}

	/// Test removing a vertex from the middle of a polygon.
	#[test]
	fn remove_middle() {
		let mut poly = polygon::square_1000();
		let removed = poly.remove(2);
		assert_eq!(removed, Point2D { x: 1000, y: 1000 }, "The third vertex was removed.");
		assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 }, "The first vertex is still in place.");
		assert_eq!(*poly.vertex(1), Point2D { x: 1000, y: 0 }, "The second vertex is still in place.");
		assert_eq!(*poly.vertex(2), Point2D { x: 0, y: 1000 }, "The fourth vertex shifted into the third position.");
	}

	/// Test removing a vertex from the end of a polygon.
	#[test]
	fn remove_end() {
		let mut poly = polygon::square_1000();
		let removed = poly.remove(3);
		assert_eq!(removed, Point2D { x: 0, y: 1000 }, "The fourth vertex was removed.");
		assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 }, "The first vertex is still in place.");
		assert_eq!(*poly.vertex(1), Point2D { x: 1000, y: 0 }, "The second vertex is still in place.");
		assert_eq!(*poly.vertex(2), Point2D { x: 1000, y: 1000 }, "The third vertex is still in place.");
	}

	/// Test clearing a polygon.
	#[test]
	fn clear() {
		let mut poly = polygon::square_1000();
		poly.clear();
		assert_eq!(poly.len(), 0, "After clearing, there should no longer be any vertices.");
	}

	/// Test iterating over the polygon with `iter()`.
	#[test]
	fn iter() {
		let poly = polygon::square_1000();
		let mut iterator = poly.iter();
		assert_eq!(*iterator.next().expect("There should be 4 vertices."), Point2D { x: 0, y: 0 }, "First it should encounter the vertex at the seam.");
		assert_eq!(*iterator.next().expect("There should be 4 vertices."), Point2D { x: 1000, y: 0 }, "Next the second vertex counter-clockwisely.");
		assert_eq!(*iterator.next().expect("There should be 4 vertices."), Point2D { x: 1000, y: 1000 }, "Next the third vertex.");
		assert_eq!(*iterator.next().expect("There should be 4 vertices."), Point2D { x: 0, y: 1000 }, "And finally the last vertex.");
		assert!(iterator.next().is_none(), "After all vertices are iterated over, it should return None.");
	}

	/// Test iterating over the polygon while modifying it with `iter_mut()`.
	#[test]
	fn iter_mut() {
		let mut poly = polygon::square_1000();
		let copy = polygon::square_1000();
		let mut i = 0;
		for mut vertex in poly.iter_mut() {
			assert_eq!(*vertex, *copy.vertex(i), "We must iterate over the polygon in index order.");
			i += 1;
			vertex.x += 33;
			vertex.y += 10;
		}
		assert_eq!(*poly.vertex(0), Point2D { x: 33, y: 10 }, "The first vertex is now shifted by 33,10.");
		assert_eq!(*poly.vertex(1), Point2D { x: 1033, y: 10 }, "The second vertex is now shifted by 33,10.");
		assert_eq!(*poly.vertex(2), Point2D { x: 1033, y: 1010 }, "The third vertex is now shifted by 33,10.");
		assert_eq!(*poly.vertex(3), Point2D { x: 33, y: 1010 }, "The fourth vertex is now shifted by 33,10.");
	}

	/// Test creating a polygon from an iterable object, this time an array.
	#[test]
	fn from_iter_array() {
		let poly = Polygon::from_iter([
			Point2D { x: 0, y: 0 },
			Point2D { x: 500, y: 0 },
			Point2D { x: 250, y: 1000 }
		]);
		assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 }, "The first vertex in the newly created polygon.");
		assert_eq!(*poly.vertex(1), Point2D { x: 500, y: 0 }, "The second vertex in the newly created polygon.");
		assert_eq!(*poly.vertex(2), Point2D { x: 250, y: 1000 }, "The third vertex in the newly created polygon.");
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
		assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 }, "The first vertex in the newly created polygon.");
		assert_eq!(*poly.vertex(1), Point2D { x: 500, y: 0 }, "The second vertex in the newly created polygon.");
		assert_eq!(*poly.vertex(2), Point2D { x: 250, y: 1000 }, "The third vertex in the newly created polygon.");
	}

	/// Test iterating over vertices in a polygon.
	#[test]
	fn into_iter() {
		let vertices = [
			Point2D { x: 0, y: 0 },
			Point2D { x: 100, y: 0 },
			Point2D { x: 50, y: 100 },
		];
		let poly = Polygon::from_iter(vertices);
		let mut i = 0;
		for vertex in poly.iter() {
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
			Point2D { x: 10, y: 100 },
		]);
		assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 }, "Getting the first vertex at index 0.");
		assert_eq!(*poly.vertex(1), Point2D { x: 50, y: 10 }, "Getting the second vertex at index 1.");
		assert_eq!(*poly.vertex(2), Point2D { x: 10, y: 100 }, "Getting the third vertex at index 2.");
	}

	/// Test accessing a vertex beyond the size of the polygon.
	///
	/// This test should cause a panic.
	#[test]
	#[should_panic(expected = "the len is 3 but the index is 3")]
	fn index_out_of_range() {
		let poly = polygon::triangle_1000();
		std::panic::set_hook(Box::new(|_| {})); //Disable stack trace from this panic.
		poly.vertex(3); //Panic here. This is out of range.
	}

	/// Test modifying a vertex of the polygon.
	#[test]
	fn index_mut() {
		let mut poly = Polygon::from_iter([
			Point2D { x: 0, y: 0 },
			Point2D { x: 50, y: 10 },
			Point2D { x: 10, y: 100 },
		]);
		*poly.vertex_mut(1) = Point2D { x: 200, y: 400 };
		assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 }, "The first vertex was not modified.");
		assert_eq!(*poly.vertex(1), Point2D { x: 200, y: 400 }, "The second vertex was modified.");
		assert_eq!(*poly.vertex(2), Point2D { x: 10, y: 100 }, "The third vertex was not modified.");
	}
}
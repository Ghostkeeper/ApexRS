/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! This module contains the implementations of operations to scale geometric objects.

use std::cmp;
use std::sync::LazyLock;
use rayon::current_num_threads; //For multi-threaded implementations.
use rayon::iter::ParallelIterator; //For multi-threaded implementations.
use rayon::prelude::ParallelSliceMut; //For multi-threaded implementations.
use wgpu::{include_wgsl, ShaderModule}; //For loading the translate GPU kernel.

use crate::Polygon; //Translate polygons.
use crate::TwoDimensional; //The scale operation is part of TwoDimensional.
use crate::detail::emulated_f64::EmulatedF64; //To get high accuracy on the GPU.
use crate::detail::gpu::GPU; //To perform calculations on the GPU.

/// Scale a polygon by a certain scale factor.
///
/// This implementation is single-threaded and simply scales each vertex one by one.
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
/// apex::operations::scale::scale_polygon_st(&mut poly, 2.0, 1.5);
/// //Now, the polygon will be scaled to be bigger.
/// assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 });
/// assert_eq!(*poly.vertex(1), Point2D { x: 200, y: 0 });
/// assert_eq!(*poly.vertex(2), Point2D { x: 134, y: 150 });
/// ```
pub fn scale_polygon_st(polygon: &mut Polygon, x: f64, y: f64) {
	for vertex in polygon.host_vertices_mut().iter_mut() {
		vertex.scale(x, y);
	}
}

/// Scale a polygon by a certain scale factor.
///
/// This implementation is multi-threaded and will apply multiple threads to scale the polygon
/// quickly.
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
/// apex::operations::scale::scale_polygon_mt(&mut poly, 2.0, 1.5);
/// //Now, the polygon will be scaled to be bigger.
/// assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 });
/// assert_eq!(*poly.vertex(1), Point2D { x: 200, y: 0 });
/// assert_eq!(*poly.vertex(2), Point2D { x: 134, y: 150 });
/// ```
pub fn scale_polygon_mt(polygon: &mut Polygon, x: f64, y: f64) {
	let chunk_size = cmp::max(10000, polygon.host_vertices().len() / current_num_threads());
	polygon.host_vertices_mut().par_chunks_mut(chunk_size).for_each(
		|slice| slice.iter_mut().for_each(
			|vertex| vertex.scale(x, y)
		)
	);
}

/// The shader for scaling polygons on the GPU.
static SCALE_POLYGON_SHADER: LazyLock<ShaderModule> = LazyLock::new(|| {
	GPU.device.create_shader_module(include_wgsl!("scale_polygon.wgsl"))
});

/// Scale a polygon by a certain scale factor.
///
/// This implementation runs on the GPU to use its massively parallel processing ability to scale
/// the polygon quickly.
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
/// apex::operations::scale::scale_polygon_gpu(&mut poly, 2.0, 1.5);
/// //Now, the polygon will be scaled to be bigger.
/// assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 });
/// assert_eq!(*poly.vertex(1), Point2D { x: 200, y: 0 });
/// assert_eq!(*poly.vertex(2), Point2D { x: 134, y: 150 });
/// ```
pub fn scale_polygon_gpu(polygon: &mut Polygon, x: f64, y: f64) {
	let parameters = [EmulatedF64::from(x), EmulatedF64::from(y)];
	let uniform_buffer = bytemuck::cast_slice(&parameters);
	polygon.execute_gpu_kernel_mut(&SCALE_POLYGON_SHADER, uniform_buffer);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::test::data::polygon;
	use test_case::test_case;

	/// Test scaling an empty polygon.
	///
	/// This mainly just tests that it won't panic on that.
	#[test]
	fn scale_polygon_empty() {
		let mut poly = Polygon::new();
		scale_polygon_st(&mut poly, 2.0, 2.0);
		assert_eq!(poly.len(), 0, "The polygon must still be unchanged.");
		scale_polygon_mt(&mut poly, 2.0, 2.0);
		assert_eq!(poly.len(), 0, "The polygon must still be unchanged.");
		scale_polygon_gpu(&mut poly, 2.0, 2.0);
		assert_eq!(poly.len(), 0, "The polygon must still be unchanged.");
	}

	/// Test whether scaling a polygon with factor 1,1 yields the original polygon.
	#[test]
	fn scale_polygon_one() {
		let original = polygon::square_1000(); //An original to compare to.
		let mut poly = polygon::square_1000(); //A copy that we can scale.
		scale_polygon_st(&mut poly, 1.0, 1.0); //Scale by factor 1,1.
		assert_eq!(*poly.host_vertices(), *original.host_vertices(), "The polygon's vertices may not have changed by scaling with factor 1,1.");
		scale_polygon_mt(&mut poly, 1.0, 1.0);
		assert_eq!(*poly.host_vertices(), *original.host_vertices(), "The polygon's vertices may not have changed by scaling with factor 1,1.");
		scale_polygon_gpu(&mut poly, 1.0, 1.0);
		assert_eq!(*poly.host_vertices(), *original.host_vertices(), "The polygon's vertices may not have changed by scaling with factor 1,1.");
	}

	/// Test scaling a polygon by a certain factor.
	#[test_case(2.0, 2.0; "Both positive")]
	#[test_case(-2.0, 2.0; "X negative")]
	#[test_case(2.0, -2.0; "Y negative")]
	#[test_case(-2.0, -2.0; "Both negative")]
	#[test_case(0.0, 0.0; "Zero")]
	#[test_case(10.0, 0.0; "Y zero")]
	fn scale_polygon_vector(x: f64, y: f64) {
		let original = polygon::square_1000(); //An original to compare to.
		let mut poly = polygon::square_1000(); //A copy that we can scale.

		scale_polygon_st(&mut poly, x, y);
		for i in 0..poly.len() {
			let mut scaled_vertex = original.vertex(i).clone();
			scaled_vertex.scale(x, y);
			assert_eq!(*(&poly).vertex(i), scaled_vertex);
		}

		poly = polygon::square_1000(); //Reset to original.
		scale_polygon_mt(&mut poly, x, y);
		for i in 0..poly.len() {
			let mut scaled_vertex = original.vertex(i).clone();
			scaled_vertex.scale(x, y);
			assert_eq!(*(&poly).vertex(i), scaled_vertex);
		}

		poly = polygon::square_1000(); //Reset to original.
		scale_polygon_gpu(&mut poly, x, y);
		for i in 0..poly.len() {
			let mut scaled_vertex = original.vertex(i).clone();
			scaled_vertex.scale(x, y);
			assert_eq!(*(&poly).vertex(i), scaled_vertex);
		}
	}
}
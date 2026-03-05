/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! This module contains the implementations of operations to rotate geometric objects.

use std::cmp;
use std::sync::LazyLock;
use rayon::current_num_threads; //For multi-threaded implementations.
use rayon::iter::ParallelIterator; //For multi-threaded implementations.
use rayon::prelude::ParallelSliceMut; //For multi-threaded implementations.
use wgpu::{include_wgsl, ShaderModule}; //For loading the translate GPU kernel.

use crate::Angle; //To measure how much to rotate objects.
use crate::coordinate::round; //To accurately round coordinates after rotating them.
use crate::Polygon; //Translate polygons.
use crate::TwoDimensional; //The rotate operation is part of TwoDimensional.
use crate::detail::emulated_f64::EmulatedF64; //To get greater accuracy on the GPU.
use crate::detail::gpu::GPU; //To perform calculations on the GPU.

/// Rotate a polygon around the coordinate origin by a certain angle.
///
/// This implementation is single-threaded and simply rotates each vertex one by one.
///
/// # Arguments
/// * `angle` - The amount of counter-clockwise rotation to apply, in radians.
///
/// # Examples
/// ```
/// use apex::{Angle, Point2D, Polygon, TwoDimensional};
/// //Create a triangular polygon.
/// let mut poly = Polygon::from_iter([
///     Point2D { x: 0, y: 0 },
///     Point2D { x: 100, y: 0 },
///     Point2D { x: 67, y: 100},
/// ]);
/// //Rotate the polygon.
/// apex::operations::rotate::rotate_polygon_st(&mut poly, Angle::degrees(45.0));
/// //Now, the polygon will be rotated 45 degrees counter-clockwise.
/// assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 });
/// assert_eq!(*poly.vertex(1), Point2D { x: 71, y: 71 });
/// assert_eq!(*poly.vertex(2), Point2D { x: -24, y: 118 });
/// ```
pub fn rotate_polygon_st(polygon: &mut Polygon, angle: Angle) {
	let cosine = angle.cos();
	let sine = angle.sin();
    for vertex in polygon.host_vertices_mut().iter_mut() {
		let new_x = round(vertex.x as f64 * cosine - vertex.y as f64 * sine);
		vertex.y = round(vertex.x as f64 * sine + vertex.y as f64 * cosine);
		vertex.x = new_x;
    }
}

/// Rotate a polygon around the coordinate origin by a certain angle.
///
/// This implementation is multi-threaded and will apply multiple threads to rotate the polygon
/// quickly.
///
/// # Arguments
/// * `angle` - The amount of counter-clockwise rotation to apply, in radians.
///
/// # Examples
/// ```
/// use apex::{Angle, Point2D, Polygon, TwoDimensional};
/// //Create a triangular polygon.
/// let mut poly = Polygon::from_iter([
///     Point2D { x: 0, y: 0 },
///     Point2D { x: 100, y: 0 },
///     Point2D { x: 67, y: 100},
/// ]);
/// //Rotate the polygon.
/// apex::operations::rotate::rotate_polygon_mt(&mut poly, Angle::degrees(45.0));
/// //Now, the polygon will be rotated 45 degrees counter-clockwise.
/// assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 });
/// assert_eq!(*poly.vertex(1), Point2D { x: 71, y: 71 });
/// assert_eq!(*poly.vertex(2), Point2D { x: -24, y: 118 });
/// ```
pub fn rotate_polygon_mt(polygon: &mut Polygon, angle: Angle) {
	let cosine = angle.cos();
	let sine = angle.sin();
    let chunk_size = cmp::max(10000, polygon.host_vertices().len() / current_num_threads());
    polygon.host_vertices_mut().par_chunks_mut(chunk_size).for_each(
        |slice| slice.iter_mut().for_each(
            |vertex| {
				let new_x = round(vertex.x as f64 * cosine - vertex.y as f64 * sine);
				vertex.y = round(vertex.x as f64 * sine + vertex.y as f64 * cosine);
				vertex.x = new_x;
			}
        )
    );
}

/// The shader for rotating polygons on the GPU.
static ROTATE_POLYGON_SHADER: LazyLock<ShaderModule> = LazyLock::new(|| {
    GPU.device.create_shader_module(include_wgsl!("rotate_polygon.wgsl"))
});

/// Rotate a polygon around the coordinate origin by a certain angle.
///
/// This implementation runs on the GPU to use its massively parallel processing ability to rotate
/// the polygon quickly.
///
/// # Arguments
/// * `angle` - The amount of counter-clockwise rotation to apply, in radians.
///
/// # Examples
/// ```
/// use apex::{Angle, Point2D, Polygon, TwoDimensional};
/// //Create a triangular polygon.
/// let mut poly = Polygon::from_iter([
///     Point2D { x: 0, y: 0 },
///     Point2D { x: 100, y: 0 },
///     Point2D { x: 67, y: 100},
/// ]);
/// //Rotate the polygon.
/// apex::operations::rotate::rotate_polygon_gpu(&mut poly, Angle::degrees(45.0));
/// //Now, the polygon will be rotated 45 degrees counter-clockwise.
/// assert_eq!(*poly.vertex(0), Point2D { x: 0, y: 0 });
/// assert_eq!(*poly.vertex(1), Point2D { x: 71, y: 71 });
/// assert_eq!(*poly.vertex(2), Point2D { x: -24, y: 118 });
/// ```
pub fn rotate_polygon_gpu(polygon: &mut Polygon, angle: Angle) {
	let cosine = EmulatedF64::new(angle.cos());
	let sine = EmulatedF64::new(angle.sin());
    let parameters = [sine, cosine];
    let uniform_buffer = bytemuck::cast_slice(&parameters);
    polygon.execute_gpu_kernel_mut(&ROTATE_POLYGON_SHADER, uniform_buffer);
}
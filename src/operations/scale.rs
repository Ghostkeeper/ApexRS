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
use rayon::prelude; //For multi-threaded implementations.
use wgpu::{include_wgsl, ShaderModule}; //For loading the translate GPU kernel.

use crate::Coordinate; //As parameter for how much to scale.
use crate::Polygon; //Translate polygons.
use crate::TwoDimensional; //The scale operation is part of TwoDimensional.
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
///     Point2D { x: 0, y: 0 },
///     Point2D { x: 100, y: 0 },
///     Point2D { x: 67, y: 100},
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
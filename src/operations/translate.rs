/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! This module contains the implementations of operations to translate (move) geometric objects.

use std::cmp;
use rayon::prelude::*; //For multi-threaded implementations.

use crate::Coordinate; //As parameter for how far to translate.
use crate::Polygon; //Translate polygons.
use crate::TwoDimensional; //The translate function is part of TwoDimensional.

/// Move a polygon by a certain delta coordinate.
///
/// This implementation is single-threaded and simply translates every vertex one by one.
///
/// # Arguments
/// * `dx` - How far to move the object in the X direction. Use a positive number to increase the X
/// position, or a negative number to reduce the X position.
/// * `dy` - How far to move the object in the Y direction. Use a positive number to increase the Y
/// position, or a negative number to reduce the Y position.
///
/// # Examples
/// ```ignore //Since it uses crate-private functions.
/// use apex::{Point2D, Polygon, TwoDimensional};
/// //Create a triangular polygon.
/// let mut poly = Polygon::from_iter([
/// 	Point2D { x: 0, y: 0 },
/// 	Point2D { x: 100, y: 0 },
/// 	Point2D { x: 67, y: 100 }
/// ]);
/// //Move the polygon.
/// translate_polygon_st(poly, 100, -150);
/// //Now, all of the vertices will have moved.
/// assert_eq!(poly[0], Point2D { x: 100, y: -150 });
/// assert_eq!(poly[1], Point2D { x: 200, y: -150 });
/// assert_eq!(poly[2], Point2D { x: 167, y: -50 });
/// ```
pub(crate) fn translate_polygon_st(polygon: &mut Polygon, dx: Coordinate, dy: Coordinate) {
	for vertex in &mut polygon.vertices {
		vertex.translate(dx, dy);
	}
}

/// Move a polygon by a certain delta coordinate.
///
/// This implementation is multi-threaded and will apply multiple threads to move the polygon
/// quickly.
///
/// # Aruments
/// * `dx` - How far to move the object in the X direction. Use a positive number to increase the X
/// position, or a negative number to reduce the X position.
/// * `dy` - How far to move the object in the Y direction. Use a positive number to increase the Y
/// position, or a negative number to reduce the Y position.
///
/// # Examples
/// ```ignore //Since it uses crate-private functions.
/// use apex::{Point2D, Polygon, TwoDimensional};
/// //Create a triangular polygon.
/// let mut poly = Polygon::from_iter([
/// 	Point2D { x: 0, y: 0 },
/// 	Point2D { x: 100, y: 0 },
/// 	Point2D { x: 67, y: 100 }
/// ]);
/// //Move the polygon.
/// translate_polygon_mt(poly, 100, -150);
/// //Now, all of the vertices will have moved.
/// assert_eq!(poly[0], Point2D { x: 100, y: -150 });
/// assert_eq!(poly[1], Point2D { x: 200, y: -150 });
/// assert_eq!(poly[2], Point2D { x: 167, y: -50 });
/// ```
pub(crate) fn translate_polygon_mt(polygon: &mut Polygon, dx: Coordinate, dy: Coordinate) {
	let chunk_size = cmp::max(10000, polygon.vertices.len() / rayon::current_num_threads());
	polygon.vertices.par_chunks_mut(chunk_size).for_each(
		|slice| slice.iter_mut().for_each(
			|vertex| vertex.translate(dx, dy)
		)
	);
}
/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! This module contains the implementations of operations to translate (move) geometric objects.

use crate::Coordinate; //As parameter for how far to translate.
use crate::Polygon; //Translate polygons.
use crate::TwoDimensional; //The translate function is part of TwoDimensional.

pub(crate) fn translate_polygon_st(polygon: &mut Polygon, dx: Coordinate, dy: Coordinate) {
	for vertex in &mut polygon.vertices {
		vertex.translate(dx, dy);
	}
}
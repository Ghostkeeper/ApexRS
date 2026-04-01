/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! A set of polygons to test with.
//!
//! These polygons are loaded from SVG files. This is compact, and also allows for easy development
//! because SVG files can easily be observed to understand what the polygon's shape really is, and
//! external tools handle SVG files well.

use crate::Coordinate; //To parse coordinates from SVG files.
use crate::Point2D; //To parse coordinates from SVG files.
use crate::Polygon; //We're loading polygons here.

/// A 1000x1000 square.
///
/// The square starts at the coordinate origin with the seam. It is centred at 500,500.
pub fn square_1000() -> Polygon {
	load_polygon(include_str!("polygon/square_1000.svg"))
}

/// A 1000x1000 square that is completely in the negative X space.
///
/// The square has two corners in X coordinate -1000 and two at X coordinate 0. It is centred at
/// -500,500.
pub fn square_1000_negative_x() -> Polygon {
	load_polygon(include_str!("polygon/square_1000_negative_x.svg"))
}

/// A 1000x1000 square that is completely in the negative Y space.
///
/// The square has two corners in Y coordinate -1000 and two at Y coordinate 0. It is centred at
/// 500,-500.
pub fn square_1000_negative_y() -> Polygon {
	load_polygon(include_str!("polygon/square_1000_negative_y.svg"))
}

/// A 1000x1000 square that is completely in the negative X and Y space.
///
/// The square starts at coordinate -1000,-1000 and reaches up to the coordinate origin with the far
/// corner. It is centred at -500,-500.
pub fn square_1000_negative_xy() -> Polygon {
	load_polygon(include_str!("polygon/square_1000_negative_xy.svg"))
}

/// A 1000x1000 square that is centred on the coordinate origin.
///
/// The square fills a part of all four quadrants of coordinates. It is centred at 0,0.
pub fn square_1000_centred() -> Polygon {
	load_polygon(include_str!("polygon/square_1000_centred.svg"))
}

/// A triangle with base 1000.
///
/// The triangle starts at 24,24 with the seam. The 1000-length base extends from there parallel to
/// the X-axis.
pub fn triangle_1000() -> Polygon {
	load_polygon(include_str!("polygon/triangle_1000.svg"))
}

/// A 1000x1000 square with an inverted winding, causing it to have a negative area.
pub fn negative_square() -> Polygon {
	load_polygon(include_str!("polygon/negative_square.svg"))
}

/// The shape of a barbed arrowhead or chevron.
///
/// This is effectively a triangle with another triangle (half the height) cut out of the base. The
/// arrow is pointing towards positive Y.
pub fn arrowhead() -> Polygon {
	load_polygon(include_str!("polygon/arrowhead.svg"))
}

/// An hourglass shape where half of the shape is negative.
///
/// It consists of only four sides, and two of the sides are intersecting. This makes it the
/// simplest possible self-intersecting shape.
pub fn hourglass() -> Polygon {
	load_polygon(include_str!("polygon/hourglass.svg"))
}

/// A 1000x1 rectangle, which is wide but very thin.
pub fn thin_rectangle() -> Polygon {
	load_polygon(include_str!("polygon/thin_rectangle.svg"))
}

/// A triangle where all three vertices are on a single line, giving the polygon zero surface area.
pub fn zero_width() -> Polygon {
	load_polygon(include_str!("polygon/zero_width.svg"))
}

/// A polygon consisting of only two vertices, making it just a line.
pub fn degenerate_line() -> Polygon {
	load_polygon(include_str!("polygon/degenerate_line.svg"))
}

/// A polygon consisting of only a single vertex, making it just a point.
pub fn degenerate_point() -> Polygon {
	load_polygon(include_str!("polygon/degenerate_point.svg"))
}

/// Parse an SVG file to load a polygon from it.
///
/// This will find the first `<polygon>` tag in the file, and take the `points` attribute from it to
/// create a polygon. If there are multiple `<polygon>` tags, the rest are ignored. Groups and
/// transforms are ignored completely, so the polygon may end up differently from how the file is
/// actually rendered by an SVG renderer.
///
/// This implementation is meant to be simple. It will not deal with generic SVGs. It will just deal
/// with the hand-crafted SVGs that are the data inputs for the tests. This way, the data inputs can
/// easily be visualised to see what the test is dealing with.
///
/// # Arguments
/// * `svg` - An SVG document, the contents of an SVG file, as text mark-up.
///
/// # Examples
/// ```
/// let poly = load_polygon(include_str!("polygon/square_1000.svg")); //Statically load this polygon.
/// assert_eq!(poly.area(), 1000000);
/// ```
fn load_polygon(svg: &str) -> Polygon {
	let tag_start = svg.find("<polygon ").expect("The <polygon> tag is missing.") + 9;
	let points_start = tag_start + svg[tag_start..].find("points=\"").expect("The points attribute is missing.") + 8;
	let points_end = points_start + svg[points_start..].find("\"").expect("The points attribute never closes.");

	let coordinates = svg[points_start..points_end] //Take the points attribute's contents.
		.split([' ', ',']) //Split at spaces or commas.
		.map(|coordinate_str| coordinate_str.parse::<Coordinate>().expect(["One of the coordinates is not integer:", coordinate_str].join(" ").as_str()))
		.collect::<Vec<Coordinate>>();
	let vertices = coordinates.chunks(2) //Pair them up into coordinate-pairs.
		.map(|chunk| Point2D { x: chunk[0], y: chunk[1] }); //Group them up into points. If this panics, there's not an even number of coordinates.
	return Polygon::from_iter(vertices);
}
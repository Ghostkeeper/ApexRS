/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

/// A 1000x1000 square.
///
/// The square starts at the coordinate origin with the seam. It is centred at 500,500.
pub fn square_1000() -> crate::Polygon {
	load_polygon(include_str!("polygon/square_1000.svg"))
}

/// A triangle with base 1000.
///
/// The triangle starts at 24,24 with the seam. The 1000-length base extends from there parallel to
/// the X-axis.
pub fn triangle_1000() -> crate::Polygon {
	load_polygon(include_str!("polygon/triangle_1000.svg"))
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
fn load_polygon(svg: &str) -> crate::Polygon {
	let tag_start = svg.find("<polygon ").expect("The <polygon> tag is missing.") + 9;
	let points_start = tag_start + svg[tag_start..].find("points=\"").expect("The points attribute is missing.") + 8;
	let points_end = points_start + svg[points_start..].find("\"").expect("The points attribute never closes.");

	let coordinates = svg[points_start..points_end] //Take the points attribute's contents.
		.split([' ', ',']) //Split at spaces or commas.
		.map(|coordinate_str| coordinate_str.parse::<crate::Coordinate>().expect(["One of the coordinates is not integer:", coordinate_str].join(" ").as_str()))
		.collect::<Vec<crate::Coordinate>>();
	let vertices = coordinates.chunks(2) //Pair them up into coordinate-pairs.
		.map(|chunk| crate::Point2D { x: chunk[0], y: chunk[1] }); //Group them up into points. If this panics, there's not an even number of coordinates.
	return crate::Polygon::from_iter(vertices);
}
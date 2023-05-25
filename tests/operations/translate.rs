/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

use apex;

/// Test moving an empty polygon.
///
/// This mainly just tests that it won't panic on that.
#[test]
fn translate_polygon_st_empty() {
    let mut poly = apex::Polygon::new();
    apex::operations::translate::translate_polygon_st(&mut poly, 100, 100);
    assert_eq!(poly.len(), 0, "The polygon must still be unchanged.");
}
/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Test the coordinate types of Apex.

use apex;

#[test]
/// Test whether the possible range of `Coordinate` is as expected.
fn coordinate_range() {
    let mut x: apex::Coordinate = 2147483647;
    assert_eq!(x, 2147483647, "We need to be able to store at least this coordinate.");
    x = x.wrapping_add(1);
    assert_eq!(x, -2147483648, "It needs to overflow to the lowest coordinate.");
}
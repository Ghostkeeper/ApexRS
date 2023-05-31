/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

use apex;

#[test]
/// Test whether the possible range of `Coordinate` is as expected.
fn coordinate_range() {
    let x: apex::Coordinate = 0x7FFFFFFF;
    assert_eq!(x, 0x7FFFFFFF, "We need to be able to store at least this coordinate.");
    let result = x.checked_add(1);
    assert_eq!(result, None, "It needs to overflow then.");
}

#[test]
/// Test whether the possible range of `Area` is as expected.
fn area_range() {
    let area: apex::Area = 0x7FFFFFFFFFFFFFFF;
    assert_eq!(area, 0x7FFFFFFFFFFFFFFF, "We need to be able to store at least this area.");
    let result = area.checked_add(1);
    assert_eq!(result, None, "It needs to overflow then.");
}
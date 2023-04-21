/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Importing this module imports all of Apex, making it available for use in your project.

pub mod coordinate;
pub use coordinate::*;

pub mod point2d;
pub use point2d::*;

pub mod shape2d;
pub use shape2d::*;

pub mod two_dimensional;
pub use two_dimensional::*;
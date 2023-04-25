/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! This module contains the algorithms that perform actual operations on the geometric objects in
//! this library.
//!
//! These algorithms are implemented as free functions. They can be called separately, but the
//! intended way to use the library is to use the methods of the geometric objects themselves. The
//! functions are organised by the type of operation performed. This causes the algorithms that
//! perform the same type of operation to be located in the same module, which juxtaposes the
//! algorithms together since they are likely going to be very similar. This makes the code easier
//! to read. The methods in the geometric objects will simply call these free functions to implement
//! them.

pub(crate) mod translate;
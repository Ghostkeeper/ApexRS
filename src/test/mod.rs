/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2023 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! This module contains integration tests and helper utilities for all tests.
//!
//! This module does not contain the unit tests. Those are organised inside of the files that
//! implement the units under test, as per the recommended Rust code organisation guidelines. This
//! makes access control and scope easier for those tests.
//!
//! To run these tests, use cargo's `test` target. If Cargo is integrated in your shell, you can run
//! these tests in the usual way for Rust, like so:
//! ```bash
//! cargo test
//! ```
//!
//! This should run all of this library's tests and provide you with an overview of which tests
//! failed (if any) and how they failed.

pub mod data;
pub use data::*;
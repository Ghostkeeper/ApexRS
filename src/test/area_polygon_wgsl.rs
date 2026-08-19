/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Tests for the area_polygon.wgsl file.
//!
//! These tests load up the area_polygon.wgsl kernel and execute functions from it to test them.

#[cfg(test)]
mod tests {
	use test_case::test_case;
	use wgpu::include_wgsl;
	use crate::detail::gpu::GPU;
	use crate::test::kernel_call::kernel_call;

	/// Test taking the absolute value of an i32 number.
	///
	/// The conversion is not allowed to lose any range.
	#[test_case(0; "Zero")]
	#[test_case(1; "One")]
	#[test_case(-1; "Negative one")]
	#[test_case(1000; "Thousand")]
	#[test_case(-1000; "Negative thousand")]
	#[test_case(2147483647; "Max i32")]
	#[test_case(-2147483648; "Min i32")]
	fn abs_i32(value: i32) {
		let ground_truth = (value as i64).abs() as u32;
		let module = GPU.device.create_shader_module(include_wgsl!("../../src/operations/area_polygon.wgsl"));
		let result = kernel_call(&module, "test_abs_i32", value, 3, 4);
		assert_eq!(result, ground_truth);
	}
}
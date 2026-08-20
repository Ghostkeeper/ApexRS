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
	use crate::detail::emulated_i64::EmulatedI64;
	use crate::detail::gpu::GPU;
	use crate::test::kernel_call::kernel_call;

	#[test_case(0, 0; "Zeroes")]
	#[test_case(1, 0; "One and zero")]
	#[test_case(0, 1; "Zero and one")]
	#[test_case(2_000_000_000, 2_000_000_000; "Two billions")]
	#[test_case(1_000_000_000, 10; "Billion and ten")]
	#[test_case(2, -4; "Positive and negative")]
	#[test_case(-2, 4; "Negative and positive")]
	#[test_case(1_000_000_000, -10; "Billion times minus ten")]
	#[test_case(-1_000_000_000, 10; "Minus billion times ten")]
	#[test_case(-1_000_000_000, -10; "Minus billion times minus ten")]
	#[test_case(1_000_000_000, -2_000_000_000; "Billion minus two billion")]
	#[test_case(-2_000_000_000, 1_000_000_000; "Minus two billion times billion")]
	#[test_case(-1_000_000_000, -1_000_000_000; "Minus billion times minus billion")]
	#[test_case(2147483647, 2147483647; "Maximums")]
	#[test_case(-2147483648, -2147483648; "Minimums")]
	fn multiply_i32(lhs: i32, rhs: i32) {
		let using_i64 = lhs as i64 * rhs as i64;
		let module = GPU.device.create_shader_module(include_wgsl!("../../src/operations/area_polygon.wgsl"));
		let result = kernel_call(&module, "test_multiply_i32", &[lhs.to_le_bytes(), rhs.to_le_bytes()].concat(), 3, 4, 8);
		let emulated_result = EmulatedI64::from(result);
		assert_eq!(using_i64, emulated_result.into());
	}

	/// Test taking the absolute value of an i32 number.
	///
	/// The conversion is not allowed to lose any range.
	#[test_case(0; "Zero")]
	#[test_case(1; "One")]
	#[test_case(4; "Four")]
	#[test_case(-1; "Negative one")]
	#[test_case(1000; "Thousand")]
	#[test_case(-1000; "Negative thousand")]
	#[test_case(2147483647; "Max i32")]
	#[test_case(-2147483648; "Min i32")]
	fn abs_i32(value: i32) {
		let using_i64 = (value as i64).abs() as u32;
		let module = GPU.device.create_shader_module(include_wgsl!("../../src/operations/area_polygon.wgsl"));
		let result_bytes = kernel_call(&module, "test_abs_i32", &value.to_le_bytes(), 5, 6, 4);
		let result = u32::from_le_bytes(result_bytes.try_into().expect("Should be 4 bytes of output."));
		assert_eq!(using_i64, result);
	}

	/// Test the addition operator.
	///
	/// The addition operator should give the same result as with a real `i64`.
	#[test_case(0, 0; "Zeroes")]
	#[test_case(1, 0; "One and zero")]
	#[test_case(0, 1; "Zero and one")]
	#[test_case(2_000_000_000, 2_000_000_000; "i32 overflows")]
	#[test_case(1_000_000_000_000, 1; "Trillion and one")]
	#[test_case(1, 1_000_000_000_000; "One and trillion")]
	#[test_case(1_000_000_000_000, 1_000_000_000_000; "Trillions")]
	#[test_case(2, -4; "Positive and negative")]
	#[test_case(-2, 4; "Negative and positive")]
	#[test_case(1_000_000_000_000, -1; "Trillion minus one")]
	#[test_case(-1_000_000_000_000, 1; "Minus trillion plus one")]
	#[test_case(-1_000_000_000_000, -1; "Minus trillion minus one")]
	#[test_case(1_000_000_000_000, -3_000_000_000_000; "Trillion minus three trillion")]
	#[test_case(-3_000_000_000_000, 1_000_000_000_000; "Minus three trillion plus trillion")]
	#[test_case(-1_000_000_000_000, -1_000_000_000_000; "Minus trillion minus trillion")]
	fn add(lhs: i64, rhs: i64) {
		let using_i64 = lhs + rhs;
		let emulated_lhs = EmulatedI64::from(lhs);
		let emulated_rhs = EmulatedI64::from(rhs);
		let mut combined: Vec<u8> = emulated_lhs.into();
		combined.append(&mut emulated_rhs.into());
		let module = GPU.device.create_shader_module(include_wgsl!("../../src/operations/area_polygon.wgsl"));
		let result = kernel_call(&module, "test_add", &combined, 7, 4, 8);
		let emulated_result = EmulatedI64::from(result);
		assert_eq!(using_i64, emulated_result.into());
	}
}
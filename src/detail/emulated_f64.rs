/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! This provides a structure for using 64-bit floating point numbers in the GPU.
//!
//! GPUs don't normally support 64-bit floats. Their processing cores consist of many parallel
//! compute modules for 32-bit floats, but many of them don't have any 64-bit support and the ones
//! that do only have very few 64-bit compute modules. We don't want to require the user to have a
//! GPU that supports 64-bit floats, and we don't want to incur the performance hit of using those
//! compute modules anyway. Instead, we emulate the accuracy 64-bit floats by using a combination of
//! two 32-bit floats. The performance is somewhat worse, and the accuracy slightly less, but both
//! of those metrics get somewhat close to 64-bit floating points.
//!
//! This implementation is based on Extended-Precision Floating-Point Numbers for GPU Computation
//! (2007, A. Thall).

use bytemuck::{Pod, Zeroable}; //To be able to send the EmulatedF64 struct to the GPU.
use std::ops::{Add, Mul, Sub}; //Implement arithmetic summation and multiplication for EmulatedF64.

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct EmulatedF64 {
	high: f32,
	low: f32,
	_pad_a: f32,
	_pad_b: f32,
}

impl EmulatedF64 {
	pub fn new(value: f64) -> EmulatedF64 {
		const SPLITTER: f64 = ((1 << 29) + 1) as f64;
		let split = value * SPLITTER;
		let high = split - (split - value);
		let low = value - high;
		println!("EmulatedF64 original: {}, high: {}, low: {}, combined: {}", value, high, low, high + low);
		EmulatedF64 { high: high as f32, low: low as f32, _pad_a: 0.0, _pad_b: 0.0 }
	}

	pub fn round(self) -> i32 {
		let high_part = if self.high < 0.0 {self.high.ceil()} else {self.high.floor()} as i32;
		let low_part = if self.low < 0.0 {self.low.ceil()} else {self.low.floor()} as i32;
		let remainders = ((high_part as f32 - self.high) + (low_part as f32 - self.low) + 0.5) as i32;
		println!("high part: {}, low_part: {}, remainders: {}", high_part, low_part, remainders);
		high_part + low_part + remainders
	}

    fn split(a: f32) -> EmulatedF64 {
        let splitter = 4097.0;
        let t = a * splitter;
        let high = t - (t - a);
        let low = a - high;
        EmulatedF64 { high: high, low: low, _pad_a: 0.0, _pad_b: 0.0 }
    }

    fn twoprod(a: f32, b: f32) -> EmulatedF64 {
        let p = a * b;
        let a_split = Self::split(a);
        let b_split = Self::split(b);
        let err = (a_split.high * b_split.high - p) + a_split.high * b_split.low + a_split.low * b_split.high + a_split.low * b_split.low;
        EmulatedF64 { high: p, low: err, _pad_a: 0.0, _pad_b: 0.0 }
    }

	fn twosum(a: f32, b: f32) -> EmulatedF64 {
		let s = a + b;
		let v = s - a;
		let e = (a - (s - v)) + (b - v);
		EmulatedF64 { high: s, low: e, _pad_a: 0.0, _pad_b: 0.0 }
	}

    fn quicktwosum(a: f32, b: f32) -> EmulatedF64 {
        let s = a + b;
        let e = b - (s - a);
        EmulatedF64 { high: s, low: e, _pad_a: 0.0, _pad_b: 0.0 }
    }
}

impl From<f64> for EmulatedF64 {
	fn from(value: f64) -> EmulatedF64 {
		EmulatedF64::new(value)
	}
}

impl From<i32> for EmulatedF64 {
	fn from(value: i32) -> EmulatedF64 {
		let high = value as f32;
		let low = (value - high as i32) as f32;
		EmulatedF64 { high: high, low: low, _pad_a: 0.0, _pad_b: 0.0 }
	}
}

impl Into<f64> for EmulatedF64 {
	fn into(self) -> f64 {
		self.high as f64 + self.low as f64
	}
}

impl Mul for EmulatedF64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut p = Self::twoprod(self.high, rhs.high);
        p.low += self.high * rhs.low;
        p.low += self.low * rhs.high;
        Self::quicktwosum(p.high, p.low)
    }
}

impl Add for EmulatedF64 {
	type Output = Self;
	fn add(self, rhs: Self) -> Self::Output {
		let mut s = Self::twosum(self.high, rhs.high);
		let t = Self::twosum(self.low, rhs.low);
		s.low += t.high;
		s = Self::quicktwosum(s.high, s.low);
		s.low += t.low;
		Self::quicktwosum(s.high, s.low)
	}
}

impl Sub for EmulatedF64 {
	type Output = Self;
	fn sub(self, rhs: Self) -> Self::Output {
		let negative = EmulatedF64 { high: -rhs.high, low: -rhs.low, _pad_a: 0.0, _pad_b: 0.0 };
		self + negative
	}
}

#[cfg(test)]
mod tests {
	use assert_float_eq::assert_float_absolute_eq;
	use test_case::test_case;
	use super::*;

	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(10_000_000_000.0; "Ten billion")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.7999999999; "Almost 0.8")]
	#[test_case(0.8000000001; "Just over 0.8")]
	#[test_case(1_000_000_000.01; "Just over a billion")]
	#[test_case(123456789.0; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(3.141592653589793; "Pi")]
	#[test_case(-123456789.0; "Big negative")]
	fn convert_loop(value: f64) {
		let emulated = EmulatedF64::from(value);
		let converted: f64 = emulated.into();
		assert_float_absolute_eq!(value, converted);
	}

	#[test_case(0.0, 0.0; "Zeroes")]
	#[test_case(1.0, 0.0; "One and zero")]
	#[test_case(10_000_000_000.0, 0.0000000001; "High and low")]
	#[test_case(0.0000000001, 10_000_000_000.0; "Low and high")]
	#[test_case(0.7999999999, 10_000_000_000.0; "Just below 0.8")]
	#[test_case(123456789.0, 0.71; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(-123456789.0, 123456789.0; "Negative and positive")]
	#[test_case(123456789.0, -123456789.0; "Positive and negative")]
	#[test_case(-123456789.0, -123456789.0; "Negative and negative")]
	fn multiply(lhs: f64, rhs: f64) {
		let emulated_lhs = EmulatedF64::from(lhs);
		let emulated_rhs = EmulatedF64::from(rhs);
		let using_f64 = lhs * rhs;
		let result = (emulated_lhs * emulated_rhs).into();
		assert_float_absolute_eq!(using_f64, result);
	}

	#[test_case(0.0, 0.0; "Zeroes")]
	#[test_case(1.0, 0.0; "One and zero")]
	#[test_case(10_000_000_000.0, 0.0000000001; "High and low")]
	#[test_case(0.0000000001, 10_000_000_000.0; "Low and high")]
	#[test_case(0.7999999999, 10_000_000_000.0; "Just below 0.8")]
	#[test_case(123456789.0, 0.71; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(-123456789.0, 123456789.0; "Negative and positive")]
	#[test_case(123456789.0, -123456789.0; "Positive and negative")]
	#[test_case(-123456789.0, -123456789.0; "Negative and negative")]
	fn add(lhs: f64, rhs: f64) {
		let emulated_lhs = EmulatedF64::from(lhs);
		let emulated_rhs = EmulatedF64::from(rhs);
		let using_f64 = lhs + rhs;
		let result = (emulated_lhs + emulated_rhs).into();
		assert_float_absolute_eq!(using_f64, result);
	}

	#[test_case(0.0, 0.0; "Zeroes")]
	#[test_case(1.0, 0.0; "One and zero")]
	#[test_case(10_000_000_000.0, 0.0000000001; "High and low")]
	#[test_case(0.0000000001, 10_000_000_000.0; "Low and high")]
	#[test_case(0.7999999999, 10_000_000_000.0; "Just below 0.8")]
	#[test_case(123456789.0, 0.71; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(-123456789.0, 123456789.0; "Negative and positive")]
	#[test_case(123456789.0, -123456789.0; "Positive and negative")]
	#[test_case(-123456789.0, -123456789.0; "Negative and negative")]
	fn subtract(lhs: f64, rhs: f64) {
		let emulated_lhs = EmulatedF64::from(lhs);
		let emulated_rhs = EmulatedF64::from(rhs);
		let using_f64 = lhs - rhs;
		let result = (emulated_lhs - emulated_rhs).into();
		assert_float_absolute_eq!(using_f64, result);
	}

	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(10_000_000_000.0; "Ten billion")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.4999999999; "Almost 0.5")]
	#[test_case(0.5000000001; "Just over 0.5")]
	#[test_case(0.5; "Exactly 0.5")]
	#[test_case(-0.4999999999; "Almost negative 0.5")]
	#[test_case(-0.5000000001; "Just under negative 0.5")]
	#[test_case(-0.5; "Exactly negative 0.5")]
	#[test_case(1_000_000_000.01; "Just over a billion")]
	#[test_case(123456789.0; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(3.141592653589793; "Pi")]
	#[test_case(-123456789.0; "Big negative")]
	fn round(value: f64) {
		let emulated = EmulatedF64::from(value);
		let rounded = emulated.round();
		assert_eq!(rounded, value.round() as i32);
	}
}
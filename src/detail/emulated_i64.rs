/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! This provides a structure for using 64-bit integers in the GPU.

use bytemuck::{Pod, Zeroable}; //To be able to send the EmulatedI64 struct to the GPU.
use std::fmt; //To print in debugging.
use std::ops::{Add, AddAssign, Sub, SubAssign, Neg}; //Implement arithmetic operators for EmulatedI64.

/// A structure that mimics the behaviour of a 64-bit signed integer by using two 32-bit integers.
///
/// Some compute devices, in particular GPUs and FPGA's, don't have 64-bit integer units. Their
/// processing cores consist of many parallel integer units for 32-bit ints, but some of them don't
/// have any 64-bit units. Many modern GPUs do have them, but then don't support atomic operations.
/// We do need the range of 64-bit integers for certain results, like surface area. Instead, we
/// emulate the range of 64-bit integers by using a combination of a signed 32-bit integer and an
/// unsigned 32-bit integer. GPUs generally have many 32-bit integer units so the performance is
/// much better.
///
/// The 64-bits of the integer are simply split up into two pieces. The first 32 bits are stored in
/// the `high` field, and the other 32 bits are stored in the `low` field. As such, a `i64` can in
/// theory simply be cast into the `EmulatedI64` struct, although Rust does not really allow this.
/// Because integers are represented using
/// [two's complement](https://en.wikipedia.org/wiki/Two%27s_complement), addition and subtraction
/// can be executed using 32-bit unsigned integers and simply be reinterpreted as signed integers.
///
/// All of the operations (except conversions) on this number are implemented without using 64-bit
/// integers. While many of them could be implemented more efficiently on a CPU using 64-bit
/// operations, by implementing them without, they can be copied into a kernel that runs on GPUs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct EmulatedI64 {
	/// The high-significance part of the number.
	///
	/// This stores the most-significant 32 bits of the 64-bit integer. To obtain the number
	/// represented by this struct, we shift this high-significance part left by 32 bits, and then
	/// add (or union) the low-significance part to that.
	high: u32,

	/// The low-significance part of the number.
	///
	/// This stores the least-significant 32 bits of the 64-bit integer. To obtain the number
	/// represented by this struct, we shift the high-significance part left by 32 bits, and then
	/// add (or union) this low-significance part to that.
	low: u32,
}

#[allow(dead_code)] //This struct also serves as example from where we copy code to GPU kernels, so not all methods are actually used.
impl EmulatedI64 {
	/// Multiply two `i32` numbers together without overflow, producing an emulated 64-bit number.
	///
	/// # Arguments
	/// * `lhs` - The number to multiply with the `rhs`.
	/// * `rhs` - The number to multiply with the `lhs`.
	///
	/// # Implementation
	/// To multiply the numbers R = A ⋅ B without using 64-bit operators, we will use some bit-wise
	/// tricks. So first we will get rid of the sign of the operands by converting both of them to
	/// unsigned integers (`u32`). We will apply the correct sign of the result at the end.
	///
	/// We can't safely multiply 31-bit integers without overflowing, so instead we'll split the two
	/// operands into 16-bit halves. To define some names for those:
	/// A = Aₗ + 2¹⁶Aₕ
	/// B = Bₗ + 2¹⁶Bₕ
	/// The resulting multiplication of A ⋅ B would become (Aₗ + 2¹⁶Aₕ) ⋅ (Bₗ + 2¹⁶Bₕ). This can be
	/// factored out to multiply all four pairs of the halves and sum those together:
	/// A ⋅ B = AₗBₗ + 2¹⁶AₗBₕ + 2¹⁶AₕBₗ + 2³²AₕBₕ.
	///
	/// We want to calculate the result as two components: R = Rₗ + 2³²Rₕ. For the lower component
	/// Rₗ, we need the first three factors (AₗBₗ, AₗBₕ and AₕBₗ). For the higher component Rₕ, we
	/// need the last three (AₗBₕ, AₕBₗ and AₕBₕ). The middle two components are repeated, so we
	/// calculate them once. Since they are multiplied by 2¹⁶, we add the lowest 16 bits of it to Rₗ
	/// and the highest 16 bits of it to Rₕ.
	///
	/// Summing the first three factors for Rₗ may overflow. We see if that happened by testing if
	/// Rₗ < AₗBₗ. Since all inputs were positive, Rₗ must be greater or equal to AₗBₗ, so if it
	/// becomes smaller there must have been an overflow. This overflow is dealt with by adding an
	/// extra 1 to Rₕ. Rₕ can never overflow, because the 31 bits from the input operands can never
	/// become greater than 2⁶².
	///
	/// Finally, as described above, we must get the signum of the result back. If none or both of
	/// the operands were positive, the result is positive. If only one of them was positive, the
	/// result is negative. So we take the xor of the signum of both input operands. If it needs to
	/// be negative, we invert the result.
	pub fn multiply_i32(lhs: i32, rhs: i32) -> EmulatedI64 {
		//Compute the absolute values and convert them to unsigned integers.
		let lhs_absolute = EmulatedI64::abs_i32(lhs);
		let rhs_absolute = EmulatedI64::abs_i32(rhs);

		//Split each of these into 16-bit halves.
		let lhs_low = lhs_absolute & 0xFFFF;
		let lhs_high = lhs_absolute >> 16;
		let rhs_low = rhs_absolute & 0xFFFF;
		let rhs_high = rhs_absolute >> 16;

		//Calculate the four pairwise factors of the total multiplication, but leaving out the 2¹⁶ and 2³² factors (or it wouldn't fit in the integer).
		let low_low = lhs_low * rhs_low;
		let low_high = lhs_low * rhs_high; //Leaving out the 2¹⁶ factor.
		let high_low = lhs_high * rhs_low; //Leaving out the 2¹⁶ factor.
		let high_high = lhs_high * rhs_high; //Leaving out the 2³² factor.

		//Calculate the lowest 32 bits of the result, and whether it overflows.
		let middle = low_high.wrapping_add(high_low);
		let (low_result, carry) = low_low.overflowing_add(middle << 16); //Only add the lowest 16 bits.

		//Calculate the highest 32 bits of the result, and carry that overflow if it happened.
		let high_result = high_high + (middle >> 16) + carry as u32;

		let mut result = EmulatedI64 { high: high_result, low: low_result };
		if (lhs < 0) ^ (rhs < 0) { //Result needs to be negative.
			let low = (!low_result).wrapping_add(1); //Invert using two's complement method to prevent overflow of inverting the minimum i32.
			let high = (!high_result).wrapping_add((low == 0) as u32);
			result = EmulatedI64 { low: low, high: high };
		}
		result
	}

	/// Add an `i32` number to this number, in-place.
	///
	/// # Arguments
	/// * `value` - The `i32` value that needs to be added to this number.
	pub fn add_i32(&mut self, value: i32) {
		let (low, carry) = self.low.overflowing_add(value as u32);
		let sign_extension: u32 = if value < 0 {0xFFFFFFFF} else { 0 };
		let (high_with_sign, _) = self.high.overflowing_add(sign_extension);
		let (high, _) = high_with_sign.overflowing_add(carry as u32);
		self.low = low;
		self.high = high;
	}

	/// Helper function to get the absolute value of an `i32` as a `u32`.
	///
	/// Because the minimum negative number is -2147483648, while the maximum positive is
	/// 2147483647, we need a `u32` to safely represent the absolute value. For multiplication, we
	/// also need `u32` values here, so this is doubly convenient.
	///
	/// # Arguments
	/// * `value` - The value to get the absolute number of.
	fn abs_i32(value: i32) -> u32 {
		if value >= 0 { //Already positive
			return value as u32;
		} else { //Is negative, needs to be inverted.
			//We can't just take the negative `(-value) as u32` due to integer underflow at the minimum value.
			//So we'll implement the two's complement method of inverting:
			//Find the inverted binary representation (!value) and add 1.
			return (!(value as u32)).wrapping_add(1);
		}
	}
}

impl fmt::Debug for EmulatedI64 {
	/// Format this number for debugging.
	///
	/// In debugging, this number is formatted as the union of its two components. For instance, the
	/// number `-1_000_000_000_000` gets formatted as `232|3567587328`.
	///
	/// # Arguments
	/// * `formatter` - The formatter used to write the output.
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "{}|{}", self.high, self.low)
	}
}

impl fmt::Display for EmulatedI64 {
	/// Format this number for display.
	///
	/// This shows the number that this emulated `i64` represents. First it calculates the `i64`
	/// itself, and then it simply formats that number in the result.
	///
	/// # Arguments
	/// * `formatter` - The formatter used to write the output.
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		let as_i64: i64 = (*self).into();
		write!(formatter, "{}", as_i64)
	}
}

impl From<i64> for EmulatedI64 {
	/// Transform a real `i64` integer into an `EmulatedI64` that represents the same number.
	///
	/// # Arguments
	/// * `value` - The `i64` value that needs to be transformed to an `EmulatedI64`.
	fn from(value: i64) -> EmulatedI64 {
		let low = (value & 0x00000000FFFFFFFF) as u32;
		let high = ((value as u64 & 0xFFFFFFFF00000000) >> 32) as u32;
		EmulatedI64 { low: low, high: high }
	}
}

impl From<i32> for EmulatedI64 {
	/// Promote an `i32` integer into an `EmulatedI64` that represents the same number.
	///
	/// The resulting `EmulatedI64` represents exactly the same number, but has more range. It will
	/// use more memory though, and operations will be more expensive on it.
	///
	/// # Arguments
	/// * `value` - The `i32` value that needs to be transformed to an `EmulatedI64`.
	fn from(value: i32) -> EmulatedI64 {
		EmulatedI64 { high: if value < 0 { 0xFFFFFFFF } else { 0 }, low: value as u32 }
	}
}

impl Into<i64> for EmulatedI64 {
	/// Calculate the `i64` number that is represented by this emulation.
	fn into(self) -> i64 {
		((self.high as i64) << 32) | (self.low as i64)
	}
}

impl Add for EmulatedI64 {
	/// The output type of the sum.
	///
	/// In this case, the sum results in the same type as its operands.
	type Output = Self;

	/// Add another integer to this integer.
	///
	/// The sum is not done in-place. It will return a new number.
	///
	/// # Arguments
	/// * `rhs` - The number to add to this number.
	fn add(self, rhs: Self) -> Self::Output {
		let (new_low, carry_low) = self.low.overflowing_add(rhs.low);
		let (mut new_high, _) = self.high.overflowing_add(rhs.high);
		new_high = new_high.wrapping_add(carry_low as u32);
		EmulatedI64 { high: new_high, low: new_low }
	}
}

impl AddAssign for EmulatedI64 {
	/// Add another integer to this integer, in-place.
	///
	/// # Arguments
	/// * `rhs` - The number to add to this number.
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl Sub for EmulatedI64 {
	/// The output type of the subtraction.
	///
	/// In this case, the subtraction results in the same type as its operands.
	type Output = Self;

	/// Subtract another integer from this integer.
	///
	/// The subtraction is not done in-place. It will return a new number.
	///
	/// # Arguments
	/// * `rhs` - The number to subtract from this number.
	fn sub(self, rhs: Self) -> Self::Output {
		self + -rhs
	}
}

impl SubAssign for EmulatedI64 {
	/// Subtract another integer from this integer, in-place.
	///
	/// # Arguments
	/// * `rhs` - The number to subtract from this number.
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}

impl Neg for EmulatedI64 {
	/// The output type when negating.
	///
	/// In this case, negating results in the same type as the original.
	type Output = Self;

	/// Get the negation of this number.
	///
	/// The result should equal `0 - x`, where `x` is this number. Negating a negative number
	/// results in a positive number.
	///
	/// # Implementation
	/// The individual high and low components of this number are negated. This results in no loss
	/// of precision, since the sign of the number is stored separately.
	fn neg(self) -> Self::Output {
		let new_low = !self.low;
		let new_high = !self.high;
		let (new_low, carry) = new_low.overflowing_add(1);
		EmulatedI64 { high: new_high.wrapping_add(carry as u32), low: new_low }
	}
}

#[cfg(test)]
mod tests {
	use test_case::test_case;
	use super::*;

	/// Test converting `i64` into `EmulatedI64` and back.
	///
	/// The conversion is not allowed to lose any range.
	#[test_case(0; "Zero")]
	#[test_case(1; "One")]
	#[test_case(-1; "Negative one")]
	#[test_case(2147483647; "Max i32")]
	#[test_case(2147483648; "i32 overflow")]
	#[test_case(4294967295; "Max u32")]
	#[test_case(4294967296; "u32 overflow")]
	#[test_case(1_000_000_000_000; "Trillion")]
	#[test_case(9223372036854775807; "Max i64")]
	#[test_case(-2147483648; "Min i32")]
	#[test_case(-2147483649; "i32 underflow")]
	#[test_case(-4294967295; "Negative min u32")]
	#[test_case(-4294967296; "u32 underflow")]
	#[test_case(-1_000_000_000_000; "Negative trillion")]
	#[test_case(-9223372036854775808; "Min i64")]
	fn convert_loop_i64(value: i64) {
		let emulated = EmulatedI64::from(value);
		let converted: i64 = emulated.into();
		assert_eq!(value, converted);
	}

	/// Test converting `i32` into `EmulatedI64` and then into `i64`.
	///
	/// The number must remain equal throughout these conversions.
	#[test_case(0; "Zero")]
	#[test_case(1; "One")]
	#[test_case(-1; "Negative one")]
	#[test_case(2147483647; "Max i32")]
	#[test_case(1_000_000_000; "Billion")]
	#[test_case(-2147483648; "Min i32")]
	#[test_case(-1_000_000_000; "Negative billion")]
	fn convert_loop_i32(value: i32) {
		let emulated = EmulatedI64::from(value);
		let converted: i64 = emulated.into();
		assert_eq!(value as i64, converted);
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
		let emulated_lhs = EmulatedI64::from(lhs);
		let emulated_rhs = EmulatedI64::from(rhs);
		let using_i64 = lhs + rhs;
		let result = (emulated_lhs + emulated_rhs).into();
		assert_eq!(using_i64, result);
	}

	/// Test the addition assignment operator.
	///
	/// The addition assignment operator should give the same result as with a real `i64`.
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
	fn add_assign(lhs: i64, rhs: i64) {
		let mut emulated_lhs = EmulatedI64::from(lhs);
		let emulated_rhs = EmulatedI64::from(rhs);
		let using_i64 = lhs + rhs;
		emulated_lhs += emulated_rhs;
		assert_eq!(using_i64, emulated_lhs.into());
	}

	#[test_case(0, 0; "Zeroes")]
	#[test_case(1, 0; "One and zero")]
	#[test_case(0, 1; "Zero and one")]
	#[test_case(2_000_000_000, 2_000_000_000; "i32 overflows")]
	#[test_case(1_000_000_000_000, 1; "Trillion and one")]
	#[test_case(1_000_000_000_000, 2_000_000_000; "Trillion vs billion")]
	#[test_case(2, -4; "Positive and negative")]
	#[test_case(-2, 4; "Negative and positive")]
	#[test_case(1_000_000_000_000, -1; "Trillion minus one")]
	#[test_case(-1_000_000_000_000, 1; "Minus trillion plus one")]
	#[test_case(-1_000_000_000_000, -1; "Minus trillion minus one")]
	#[test_case(1_000_000_000_000, -2_000_000_000; "Trillion minus two billion")]
	#[test_case(-3_000_000_000_000, 1_000_000_000; "Minus three trillion plus billion")]
	#[test_case(-1_000_000_000_000, -1_000_000_000; "Minus trillion minus billion")]
	fn add_i32(lhs: i64, rhs: i32) {
		let mut emulated_lhs = EmulatedI64::from(lhs);
		let using_i64 = lhs + rhs as i64;
		emulated_lhs.add_i32(rhs);
		assert_eq!(using_i64, emulated_lhs.into());
	}

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
		let result = EmulatedI64::multiply_i32(lhs, rhs);
		let using_i64 = lhs as i64 * rhs as i64;
		assert_eq!(using_i64, result.into());
	}

	/// Test the subtraction operator.
	///
	/// The subtraction operator should give the same result as with a real `i64`.
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
	fn sub(lhs: i64, rhs: i64) {
		let emulated_lhs = EmulatedI64::from(lhs);
		let emulated_rhs = EmulatedI64::from(rhs);
		let using_i64 = lhs - rhs;
		let result = (emulated_lhs - emulated_rhs).into();
		assert_eq!(using_i64, result);
	}

	/// Test the subtraction assignment operator.
	///
	/// The subtraction assignment operator should give the same result as with a real `i64`.
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
	fn sub_assign(lhs: i64, rhs: i64) {
		let mut emulated_lhs = EmulatedI64::from(lhs);
		let emulated_rhs = EmulatedI64::from(rhs);
		let using_i64 = lhs - rhs;
		emulated_lhs -= emulated_rhs;
		assert_eq!(using_i64, emulated_lhs.into());
	}

	/// Test the negation operator.
	///
	/// The negation operator should give the same result as with a real `i64`.
	#[test_case(0; "Zero")]
	#[test_case(1; "One")]
	#[test_case(-1; "Negative one")]
	#[test_case(2147483647; "Max i32")]
	#[test_case(2147483648; "i32 overflow")]
	#[test_case(4294967295; "Max u32")]
	#[test_case(4294967296; "u32 overflow")]
	#[test_case(1_000_000_000_000; "Trillion")]
	#[test_case(9223372036854775807; "Max i64")]
	#[test_case(-2147483648; "Min i32")]
	#[test_case(-2147483649; "i32 underflow")]
	#[test_case(-4294967295; "Negative min u32")]
	#[test_case(-4294967296; "u32 underflow")]
	#[test_case(-1_000_000_000_000; "Negative trillion")]
	#[test_case(-9223372036854775807; "Min negateable i64")]
	fn neg(value: i64) {
		let emulated = EmulatedI64::from(value);
		let using_i64 = -value;
		let result = (-emulated).into();
		assert_eq!(using_i64, result);
	}
}
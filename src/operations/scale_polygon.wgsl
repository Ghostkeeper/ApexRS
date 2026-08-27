/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

/// A structure that mimics the behaviour of a 64-bit floating point by using two 32-bit floats.
///
/// Many compute devices, in particular GPUs and FPGA's don't have 64-bit floating point units.
/// Their processing cores consist of many parallel floating point units for 32-bit floats, but most
/// of them don't have any 64-bit units and the ones that do have very few of them. We don't want to
/// incur the performance hit of using those anyway. But we do need the accuracy of 64-bit floating
/// point units for certain operations, like rotation or scaling. Instead, we emulate the accuracy
/// of 64-bit floats by using a combination of two 32-bit floats. GPUs generally have many 32-bit
/// float units so the performance is much better.
///
/// The implementation of this number is based on Extended-Precision Floating-Point Numbers for GPU
/// Computation (2007, A. Thall). This paper presents a structure consisting of two 32-bit floating
/// point numbers, each with 23 bits of mantissa information (which is the part of the data that is
/// the limiting factor for precise numbers). The numbers are constructed such that the mantissas do
/// not overlap: The range of numbers that can be represented by the mantissa of the "low"
/// significant number is entirely contained within the inaccuracy of the current "high" significant
/// number. The numbers are added together to reconstruct the accurate, 64-bit float. This results
/// in an effective 46-bit mantissa. The exponent of the "low" significant number is restricted in
/// order to align the mantissas that way, so the effective range of the exponent of this emulation
/// is the same as in a 32-bit float (8 bits).
///
/// The mantissa of a real 64-bit float is 53 bits, and the exponent has 11 bits, so the accuracy of
/// this emulated f64 is still slightly less than a real 64-bit float. The range of the exponent is
/// not really a problem for this library, since 8 bits (up to 2^127) is already more than enough to
/// represent all coordinates that the ordinary coordinate system can represent. The difference in
/// accuracy may be more of a problem, because it could cause rounding to sometimes end up
/// differently, in theory making it possible for the result on a GPU being different from the
/// result on a CPU. Whether that also occurs in practice still has to be proven.
///
/// All of the operations (except conversions) on this number are implemented without using 64-bit
/// floats. While many of them could be implemented more efficiently on a CPU using 64-bit
/// operations, by implementing them without, they can be copied into a kernel that runs on GPUs.
struct EmulatedF64 {
	/// The high-significance part of the number.
	///
	/// The high-significance part never overlaps with the low-significance part. As a result, the
	/// high-significance part has a higher exponent such that the measurement inaccuracy in the
	/// high-significance part will always be greater than the total range of values representable
	/// by the mantissa of the low-significance part.
	///
	/// Adding the high-significance part of the number to the low-significance part results in the
	/// accurate number that is represented by this struct.
	high: f32,

	/// The low-significance part of the number.
	///
	/// The low-significance part never overlaps with the high-significance part. As a result, the
	/// low-significance part has a lower exponent such that the measurement inaccuracy in the high-
	/// significance part will always be greater than the total range of values representable by the
	/// mantissa of the low-significance part.
	///
	/// Adding the low-significance part of the number to the high-significance part results in the
	/// accurate number that is represented by this struct.
	low: f32,
}

/// Split an `f32` number into a high and low component.
///
/// The number is split such that multiplying the components of two split numbers individually will
/// not cause any round-off errors.
///
/// # Arguments
/// * `value` - The number to split.
///
/// # Implementation
/// The number is multiplied by 2^12 + 1, which causes a round-off error of the least significant 12
/// bits in the `f32`'s 23-bit mantissa. This effectively splits the original 23 bits of mantissa
/// into two numbers, one containing the most significant 13 bits, and the other containing the
/// least significant 12 bits of that mantissa. Each of these components can safely be multiplied
/// with one another without round-off error.
fn split(value: f32) -> EmulatedF64 {
	const SPLITTER: f32 = f32((1 << 12) + 1); //2^12 + 1
	let rounded_max = value * SPLITTER; //Maximum round-off error.
	let high = rounded_max - (rounded_max - value); //Mask the mantissa of the original value with this round-off error.
	let low = value - high; //The remainder.
	return EmulatedF64(high, low);
}

/// Convert an `i32` integer to an `EmulatedF64` that represents the same number.
///
/// The resulting `EmulatedF64` represents exactly the same number. The `EmulatedF64` can represent
/// every `i32` value.
///
/// # Arguments:
/// * `value` - The `i32` value that needs to be transformed to an `EmulatedF64`.
fn from_i32(value: i32) -> EmulatedF64 {
	let high = f32(value);
	let low = f32(value - i32(high));
	return EmulatedF64(high, low);
}

/// Compute the multiplication of two `f32` numbers and the exact round-off error.
///
/// The multiplied result together with the round-off error are returned as an `EmulatedF64`. This
/// result represents the same value as the input.
///
/// # Arguments
/// * `a` - One of the numbers to multiply.
/// * `b` - The other number to multiply.
///
/// # Implementation
/// The multiplication is calculated with a simple multiply of the two numbers. The round-off error
/// is more complex though. In order to do this, we need to split each of the operands into a
/// high-order component and a low-order component. Each of these components will fit into an `f32`
/// value without overflow, and sum up to the original operands.
///
/// The product of a and b can then be formulated as follows:
/// a = aₕᵢ + aₗₒ
/// b = bₕᵢ + bₗₒ
/// a ⋅ b = (aₕᵢ + aₗₒ) ⋅ (bₕᵢ + bₗₒ)
///       = aₕᵢbₕᵢ + aₕᵢbₗₒ + aₗₒbₕᵢ + aₗₒbₗₒ
///
/// The error term can be found by subtracting the original "simple" product from the most
/// significant of those terms:
/// error = (aₕᵢbₕᵢ - product) + aₕᵢbₗₒ + aₗₒbₕᵢ + aₗₒbₗₒ
fn two_product(a: f32, b: f32) -> EmulatedF64 {
	let product = a * b;
	let a_split = split(a);
	let b_split = split(b);
	let error = (a_split.high * b_split.high - product) + a_split.high * b_split.low + a_split.low * b_split.high + a_split.low * b_split.low;
	return EmulatedF64(product, error);
}

/// Compute the sum of two `f32` numbers and the exact round-off error.
///
/// This implements the [2Sum](https://en.wikipedia.org/wiki/2Sum) operation, which calculates the
/// sum of two numbers and the round-off error of this sum separately. Assuming that the sum does
/// not overflow, it calculates the sum correctly rounded (returned in the `high` component of the
/// result) and the error correctly rounded (returned in the `low` component) to the nearest
/// available floating point value.
///
/// This algorithm assumes that:
/// * The sum of these numbers do not overflow.
/// * The sum of these numbers may underflow, but it must underflow gradually.
/// * The arithmetic is correctly rounded to the nearest 32-bit floating point value (as in IEEE
///   754).
///
/// Even if these assumptions do not hold, the round-off error is often quite good.
///
/// # Arguments
/// * `a` - One of the numbers to sum.
/// * `b` - The other number to sum.
fn two_sum(a: f32, b: f32) -> EmulatedF64 {
	let rounded_sum = a + b;
	let b_with_error = rounded_sum - a;
	let error = (a - (rounded_sum - b_with_error)) + (b - b_with_error);
	return EmulatedF64(rounded_sum, error);
}

/// Compute the exact round-off error of adding two `f32` numbers where we know that `|a|` >= `|b|`.
///
/// While the original `two_sum` algorithm uses 6 floating point operations, this version uses only
/// 3, but depends on the knowledge that the exponent of `a` is at least as large as the exponent of
/// `b`.
///
/// The `two_sum` algorithm does not use this quick variant, because comparing the two exponents and
/// swapping the values if needed still uses more operations than the original `two_sum` algorithm.
///
/// # Arguments
/// * `a` - The higher of the numbers to sum.
/// * `b` - The lower of the numbers to sum.
fn two_sum_quick(a: f32, b: f32) -> EmulatedF64 {
	let rounded_sum = a + b;
	let error = b - (rounded_sum - a);
	return EmulatedF64(rounded_sum, error);
}

/// Multiply a number with another number.
///
/// # Arguments
/// * `lhs` - One of the numbers to multiply.
/// * `rhs` - The other number to multiply.
fn multiply(lhs: EmulatedF64, rhs: EmulatedF64) -> EmulatedF64 {
	var product_and_error = two_product(lhs.high, rhs.high);
	product_and_error.low += lhs.high * rhs.low;
	product_and_error.low += lhs.low * rhs.high;
	return two_sum_quick(product_and_error.high, product_and_error.low);
}

/// Sum two numbers.
///
/// # Arguments
/// * `lhs` - One of the numbers to sum.
/// * `rhs` - The other number to sum.
fn add(lhs: EmulatedF64, rhs: EmulatedF64) -> EmulatedF64 {
	var sum_highs = two_sum(lhs.high, rhs.high);
	let sum_lows = two_sum(lhs.low, rhs.low);
	sum_highs.low += sum_lows.high;
	sum_highs = two_sum_quick(sum_highs.high, sum_highs.low);
	sum_highs.low += sum_lows.low;
	return two_sum_quick(sum_highs.high, sum_highs.low);
}

/// Round the number to the nearest integer.
///
/// In case of ties, this rounding will always round up, towards positive infinity. This is
/// different from most rounding methods (which are usually rounded away-from-zero or rounded to the
/// nearest even number in case of ties).
///
/// # Arguments
/// * `value` - The floating-point number to round.
///
/// # Implementation
/// The rounding algorithm works as follows:
/// 1. First we calculate the proper precise sum of the given `value` and `0.5`, using the accurate
/// double-float addition algorithm. The resulting sum can be truncated down in order to obtain the
/// rounded result.
/// 2. Then we take the two single-precision components of that sum, and split each of them
/// individually into an integer and fractional part.
/// 3. We then sum together these two fractional parts, and floor the result.
/// 4. Finally, we take the two integer parts, and the summed fractional parts, and add them all
/// together to the final resulting integer.
///
/// In the first step we add 0.5 in order to truncate the result later. This sum is subject to a
/// loss of accuracy, so we must execute it with proper accuracy of double-accuracy floats. At the
/// end of this, we end up with an accurate number that we must truncate rather than a number that
/// we must round, labelled `halfup`.
///
/// The second step is splitting the single-precision components into an integer and fractional
/// part. This step doesn't lose any precision: Casting to integer and computing the modulo are
/// using single-precision floating point operations which are precise according to the IEEE 754
/// specification. The fractional parts of the original components can always exactly be
/// represented, because the new value is always equal or closer to a power of 2 than the original:
/// The integer component is 0, which takes up no part of the mantissa. After the split, the proper
/// accurate number `halfup` is represented by the sum of the four components, `high_int`,
/// `high_frac`, `low_int` and `low_frac`.
///
/// The third step performs the actual truncation. The two fractional parts are added together,
/// which incurs a loss of precision again, this time with single-precision accuracy. However the
/// rounding in this sum can never flow over to the next integer. Since we floor the result
/// afterwards, the result is always the correct integer.
///
/// In the final step, we only add integers together, which incurs no loss of precision.
fn round(value: EmulatedF64) -> i32 {
	let halfup = add(value, EmulatedF64(0.5, 0.0)); //So that we can merely truncate.
	//Split into integer and fractional parts.
	let high_int = i32(halfup.high);
	let high_frac = halfup.high % 1.0;
	let low_int = i32(halfup.low);
	let low_frac = halfup.low % 1.0;
	let remainders = i32(floor(high_frac + low_frac)); //Sum and round the fractional parts separately.
	return high_int + low_int + remainders;
}

/// The structure of the uniform buffer is a combination of two floats: The X and Y scale factors.
struct ScaleFactors {
	/// The scale factor in the X direction.
	x: EmulatedF64,

	/// The scale factor in the Y direction.
	y: EmulatedF64,
}
@group(0) @binding(0) var<uniform> scale_factors: vec4<f32>;

/// One corner of the polygon.
struct Vertex {
	/// The X coordinate of the vertex.
	x: i32,

	/// The Y coordinate of the vertex.
	y: i32,
}

/// The structure of the first binding is an array of vertices, forming the polygon.
@group(0) @binding(1)
var<storage, read_write> vertices: array<Vertex>;

/// Perform the scale operation on the polygon in-place.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let index = global_id.x;
	let num_verts = arrayLength(&vertices);
	if(index >= num_verts) {
		return;
	}

	let x = from_i32(vertices[index].x);
	let y = from_i32(vertices[index].y);
	let scale_x = EmulatedF64(scale_factors.x, scale_factors.y);
	let scale_y = EmulatedF64(scale_factors.z, scale_factors.w);
	vertices[index].x = round(multiply(x, scale_x));
	vertices[index].y = round(multiply(y, scale_y));
}
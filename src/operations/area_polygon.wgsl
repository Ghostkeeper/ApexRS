/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

/// One corner of the polygon.
struct Vertex {
	/// The X coordinate of the vertex.
	x: i32,

	/// The Y coordinate of the vertex.
	y: i32,
}

//The uniform buffer (binding 0) is unused in this one.

/// The structure of the first binding is an array of coordinates, forming the polygon.
@group(0) @binding(0)
var<storage, read_write> vertices: array<Vertex>;

@group(0) @binding(1)
var<storage, read_write> output: array<EmulatedI64>; //One slot per workgroup.

var<workgroup> calculated_areas: array<EmulatedI64, 256>; //One slot per worker in the workgroup.

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
struct EmulatedI64 {
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
fn multiply_i32(lhs: i32, rhs: i32) -> EmulatedI64 {
	//Compute the absolute values and convert them to unsigned integers.
	let lhs_absolute = abs_i32(lhs);
	let rhs_absolute = abs_i32(rhs);

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
	let middle = low_high + high_low;
	let low_result = low_low + (middle << 16); //Only add the lowest 16 bits.
	let carry = select(0u, 1u, low_result < low_low);

	//Calculate the highest 32 bits of the result, and carry that overflow if it happened.
	let high_result = high_high + (middle >> 16) + carry;

	var result = EmulatedI64(high_result, low_result);
	if (lhs < 0 && rhs > 0) || (rhs < 0 && lhs > 0) { //Result needs to be negative.
		let low = (~low_result) + 1; //Invert using two's complement method to prevent overflow of inverting the minimum i32.
		let carry_low = select(0u, 1u, low == 0);
		let high = (~high_result) + carry_low;
		result = EmulatedI64(high, low);
	}
	return result;
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
		return bitcast<u32>(value);
	} else { //Is negative, needs to be inverted.
		//We can't just take the negative `u32(-value)` due to integer underflow at the minimum value.
		//So we'll implement the two's complement method of inverting:
		//Find the inverted binary representation (~value) and add 1.
		return ~bitcast<u32>(value) + 1;
	}
}

/// Add another integer to this integer.
///
/// The sum is not done in-place. It will return a new number.
///
/// # Arguments
/// * `lhs` - The number to add to the `rhs`.
/// * `rhs` - The number to add to the `lhs`.
fn add(lhs: EmulatedI64, rhs: EmulatedI64) -> EmulatedI64 {
	let carry_low = select(0u, 1u, lhs.low > (0xFFFFFFFF - rhs.low)); //Check for overflow.
	let new_low = lhs.low + rhs.low;
	let new_high = lhs.high + rhs.high + carry_low;
	return EmulatedI64(new_high, new_low);
}

/// Subtract another integer from this integer.
///
/// The subtraction is not done in-place. It will return a new number.
///
/// # Arguments
/// * `lhs` - The number to subtract the `rhs` from.
/// * `rhs` - The number to subtract from the `lhs`.
fn sub(lhs: EmulatedI64, rhs: EmulatedI64) -> EmulatedI64 {
	return add(lhs, neg(rhs));
}

/// Get the negation of a number.
///
/// The result should equal `0 - x`, where `x` is this number. Negating a negative number results in
/// a positive number.
///
/// # Implementation
/// The individual high and low components of this number are negated. This results in no loss
/// of precision, since the sign of the number is stored separately.
fn neg(value: EmulatedI64) -> EmulatedI64 {
	var new_low = ~value.low;
	var new_high = ~value.high;
	let carry = select(0u, 1u, new_low == 0xFFFFFFFFu);
	new_high += carry;
	new_low += 1;
	return EmulatedI64(new_high, new_low);
}

/// Calculate the area of the polygon.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
	let index = global_id.x;
	let index_in_workgroup = local_id.x;

	let num_verts = arrayLength(&vertices);
	if(index >= num_verts) {
		return;
	}
	var previous = select(index - 1, num_verts - 1, index == 0);

	//Shoestring formula: vₙ₋₁.x * vₙ.y - vₙ₋₁.y * vₙ.x
	calculated_areas[index_in_workgroup] = sub(multiply_i32(vertices[previous].x, vertices[index].y), multiply_i32(vertices[previous].y, vertices[index].x));

	workgroupBarrier();
	var stride = 1u;
	while stride < 256 {
		let me = index_in_workgroup * stride * 2;
		let them = me + stride;
		if them < 256 {
			calculated_areas[me] = add(calculated_areas[me], calculated_areas[them]);
		}
		stride *= 2;
		workgroupBarrier();
	}
	if index_in_workgroup == 0 {
		output[workgroup_id.x] = EmulatedI64(calculated_areas[0].high, calculated_areas[0].low);
	}
}

//Testing code.

/// Testing input and output for multiply_i32.
@group(0) @binding(2)
var<uniform> test_pair_of_i32_input: vec2<i32>;
@group(0) @binding(3)
var<storage, read_write> test_emulatedi64_output: EmulatedI64;

@compute @workgroup_size(1)
fn test_multiply_i32() {
	test_emulatedi64_output = multiply_i32(test_pair_of_i32_input.x, test_pair_of_i32_input.y);
}

/// Testing input and output for abs_i32.
@group(0) @binding(4)
var<uniform> test_i32_input: i32;
@group(0) @binding(5)
var<storage, read_write> test_u32_output: u32;

@compute @workgroup_size(1)
fn test_abs_i32() {
	test_u32_output = abs_i32(test_i32_input);
}

/// Testing input for add and sub. Output of the correct type already exists at binding 4.
@group(0) @binding(6)
var<uniform> test_pair_of_emulatedi64_input: vec4<u32>;

@compute @workgroup_size(1)
fn test_add() {
	let lhs = EmulatedI64(test_pair_of_emulatedi64_input.x, test_pair_of_emulatedi64_input.y);
	let rhs = EmulatedI64(test_pair_of_emulatedi64_input.z, test_pair_of_emulatedi64_input.w);
	test_emulatedi64_output = add(lhs, rhs);
}

@compute @workgroup_size(1)
fn test_sub() {
	let lhs = EmulatedI64(test_pair_of_emulatedi64_input.x, test_pair_of_emulatedi64_input.y);
	let rhs = EmulatedI64(test_pair_of_emulatedi64_input.z, test_pair_of_emulatedi64_input.w);
	test_emulatedi64_output = sub(lhs, rhs);
}

/// Testing input for neg. Output of the correct type already exists at binding 4.
@group(0) @binding(7)
var<uniform> test_emulatedi64_input: vec2<u32>;

@compute @workgroup_size(1)
fn test_neg() {
	let input = EmulatedI64(test_emulatedi64_input.x, test_emulatedi64_input.y);
	test_emulatedi64_output = neg(input);
}
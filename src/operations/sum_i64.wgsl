/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

/// The first binding is for the uniforms. In this case we only have one uniform, namely how many
/// cells to skip in reading the numbers buffer.
///
/// Summing the numbers on the GPU dispatches a certain number of workgroups (size 256) to calculate
/// the sum in parallel. Between these workgroups there is no way to synchronize them. Since we're
/// using the second binding both as input and as output, we can't write the outputs in a different
/// place than the inputs. That would cause race conditions as the inputs are overwritten before
/// another workgroup has the chance to read them. So we need to write them in the same place in the
/// buffer as where we're reading them. The output will be written in place of the input number of
/// the 0th thread in the workgroup. But this leaves space between the outputs. If we're repeating
/// the sum operation afterwards to sum again, the GPU needs to know how much space is currently
/// between the relevant numbers. Each repetition multiplies that skip space by 256.
@group(0) @binding(0)
var<uniform> skip: u32;

/// The first binding is both the input numbers and where we place the output.
///
/// The output will get written in place of the first thread's number in the inputs. Beware that the
/// rest of the numbers will _not_ get zeroed out though. Only one in 256 of the inputs remains
/// relevant after the kernel has completed. The other numbers can even be used as partial sums and
/// will be changed in the process.
@group(0) @binding(1)
var<storage, read_write> numbers: array<EmulatedI64>;

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

/// Calculate the sum of the numbers.
///
/// This kernel is supposed to be executed in a tree-like fashion where each iteration reduces the
/// number of relevant inputs by the workgroup size.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
	let index = global_id.x;
	let index_in_workgroup = local_id.x;

	let num_numbers = (arrayLength(&numbers) + skip - 1) / skip;
	if(index >= num_numbers) {
		return;
	}

	let offset = index - index_in_workgroup; //Where the data for this workgroup starts.
	var stride = 1u;
	while stride < 256 {
		let me = index_in_workgroup * stride * 2;
		let them = me + stride;
		if them < 256 {
			numbers[(offset + me) * skip] = add(numbers[(offset + me) * skip], numbers[(offset + them) * skip]);
		}
		stride *= 2;
		workgroupBarrier();
	}
}
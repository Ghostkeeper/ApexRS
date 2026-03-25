/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! This provides a structure for using 64-bit floating point numbers in the GPU.

use bytemuck::{Pod, Zeroable}; //To be able to send the EmulatedF64 struct to the GPU.
use std::fmt; //To print in debugging.
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign}; //Implement arithmetic operators for EmulatedF64.

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
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct EmulatedF64 {
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

impl EmulatedF64 {
	/// Shorthand for the value of the mathematical constant τ.
	///
	/// τ is the ratio between the circumference of a circle and its radius. It is equal to 2 * π.
	const TAU: EmulatedF64 = EmulatedF64 { high: 6.2831855, low: -0.00000017484555 };

	/// Shorthand for the value of the mathematical constant π.
	///
	/// π is the ratio between the circumference of a circle and its diameter. It is equal to ½ * τ.
	const PI: EmulatedF64 = EmulatedF64 { high: 3.1415927, low: -0.00000008742278 };

	/// Shorthand for the value of the mathematical constant ¼τ.
	///
	/// ¼τ represents a quarter circle in radians.
	const QUARTER_TAU: EmulatedF64 = EmulatedF64 { high: 1.5707964, low: -0.00000004371139 };

	/// Test if the emulated number is NaN.
	///
	/// The number can end up NaN if it is the result of a calculation that is not defined, such as
	/// division by zero or the square root of a non-positive number.
	pub fn is_nan(self) -> bool {
		self.high.is_nan() || self.low.is_nan()
	}

	/// Return the absolute magnitude of the number.
	///
	/// If the number was negative, it will be made positive. If it was positive or zero, it will be
	/// left as it was.
	pub fn abs(self) -> EmulatedF64 {
		let signum = self.signum();
		EmulatedF64 { high: self.high * signum, low: self.low * signum }
	}

	/// Get the signum of this number.
	///
	/// The signum is +1.0 if the number is positive, -1.0 if the number is negative, and 0.0 if the
	/// number is zero or NaN.
	pub fn signum(self) -> f32 {
		self.high.signum()
	}

	/// Truncate the number towards zero.
	///
	/// This effectively rounds the number towards the nearest integer number. If the number is
	/// positive, it is rounded down (towards zero). If the number is negative, it is rounded up
	/// (towards zero).
	///
	/// # TODO
	/// While the truncation returns an accurate result for most numbers, if the number is just
	/// below an integer (such that a `f32` component is above that integer), the wrong integer is
	/// returned.
	pub fn trunc(self) -> EmulatedF64 {
		let signum = self.signum();
		let absolute = EmulatedF64 { high: self.high * signum, low: self.low * signum };
		let high_int = absolute.high.div_euclid(1.0);
		let high_frac = absolute.high.rem_euclid(1.0);
		let low_int = absolute.low.div_euclid(1.0);
		let low_frac = absolute.low.rem_euclid(1.0);
		let remainders = low_int + (high_frac + low_frac).floor();
		EmulatedF64::two_sum(signum * high_int, signum * remainders)
	}

	/// Round the number to the nearest integer.
	///
	/// In case of ties, this rounding will always round up, towards positive infinity. This is
	/// different from most rounding methods (which are usually rounded away-from-zero or rounded to
	/// the nearest even number in case of ties).
	///
	/// # Implementation
	/// The rounding algorithm works as follows:
	/// 1. First we calculate the proper precise sum of the given `value` and `0.5`, using the
	/// accurate double-float addition algorithm. The resulting sum can be truncated down in order
	/// to obtain the rounded result.
	/// 2. Then we take the two single-precision components of that sum, and split each of them
	/// individually into an integer and fractional part.
	/// 3. We then sum together these two fractional parts, and floor the result.
	/// 4. Finally, we take the two integer parts, and the summed fractional parts, and add them all
	/// together to the final resulting integer.
	///
	/// In the first step we add 0.5 in order to truncate the result later. This sum is subject to a
	/// loss of accuracy, so we must execute it with proper accuracy of double-accuracy floats. At
	/// the end of this, we end up with an accurate number that we must truncate rather than a
	/// number that we must round, labelled `halfup`.
	/// The second step is splitting the single-precision components into an integer and fractional
	/// part. This step doesn't lose any precision: Casting to integer and computing the modulo are
	/// using single-precision floating point operations which are precise according to the IEEE 754
	/// specification. The fractional parts of the original components can always exactly be
	/// represented, because the new value is always equal or closer to a power of 2 than the
	/// original: The integer component is 0, which takes up no part of the mantissa. After the
	/// split, the proper accurate number `halfup` is represented by the sum of the four components,
	/// `high_int`, `high_frac`, `low_int` and `low_frac`.
	/// The third step performs the actual truncation. The two fractional parts are added together,
	/// which incurs a loss of precision again, this time with single-precision accuracy. However
	/// the rounding in this sum can never flow over to the next integer. Since we floor the result
	/// afterwards, the result is always the correct integer.
	/// In the final step, we only add integers together, which incurs no loss of precision.
	pub fn round(self) -> i32 {
		let halfup = self + EmulatedF64::from(0.5_f32); //So that we can merely truncate.
		//Split into integer and fractional parts.
		let high_int = halfup.high as i32;
		let high_frac = halfup.high % 1.0;
		let low_int = halfup.low as i32;
		let low_frac = halfup.low % 1.0;
		let remainders = (high_frac + low_frac).floor() as i32; //Sum and round the fractional parts separately.
		high_int + low_int + remainders
	}

	/// Compute the square root of the number.
	///
	/// The square root is the number that, when multiplied with itself, results in the original
	/// number again.
	///
	/// The square root of a negative number is undefined. The result should then display as NaN.
	///
	/// # Implementation
	/// The square root is estimated with
	/// [Newton's Method](https://en.wikipedia.org/wiki/Newton's_method), which is then enhanced to
	/// need fewer high-precision operations with Karp's Method in their article High Precision
	/// Division and Square Root (1997, Karp & Markstein). Newton's Method tries to approximate the
	/// reciprocal square root of our input α, 1/√α, and then multiply the result by α. The
	/// approximation is done by starting with an arbitrary estimate x, and iteratively approaching
	/// the root with the formula xₙ₊₁ = xₙ - ƒ(xₙ)/ƒ'(xₙ). In the case of the reciprocal square
	/// root, the function ƒ(xₙ) is set to 1/xₙ² - α with its derivative -2/xₙ³. Filling that into
	/// Newton's method gives xₙ₊₁ = xₙ - (1/xₙ² - α)/(-2/xₙ³), simplifying the formula to the
	/// concrete xₙ₊₁ = xₙ + ½xₙ(1 - αxₙ²).
	///
	/// Approximating this needs only multiplication, but requires many high-precision
	/// multiplications. Using multi-component floats we can change this formula to:
	/// yₙ₊₁ = yₙ + ½xₙ(α - yₙ²) with yₙ = αxₙ. Instead of converging to xₙ₊₁ we now converge to
	/// αxₙ₊₁. This brings the multiplication inside of the term that we're converging to, resulting
	/// in fewer high-precision multiplications being necessary. The multiplication in the "αxₙ²"
	/// part of the formula is factored out, and this also immediately applies the multiplication
	/// that we need to get from 1/√α to the final √α. Because Newton's method corrects the initial
	/// estimate sufficiently fast, we don't even need to compute yₙ = αxₙ with high accuracy.
	///
	/// Newton's Method is quadratically convergent, meaning that with every iteration, the accuracy
	/// is doubled. To implement the square root of the `EmulatedF64` then, we simply use the
	/// built-in square root function for `f32` to arrive at our initial estimate. This estimate
	/// should be accurate to the 23 bits of mantissa in `f32`. We then process a single iteration
	/// of Newton's Method, resulting in an accuracy of 46 bits of mantissa, enough for the entire
	/// `EmulatedF64`.
	pub fn sqrt(self) -> EmulatedF64 {
		let initial_estimate = 1.0 / self.high.sqrt(); //xₙ in the above formulas.
		let premultiplied_estimate = self.high * initial_estimate; //yₙ in the above formulas.
		let premultiplied_promoted = Self::from(premultiplied_estimate);
		let difference = (self - premultiplied_promoted.square()).high; //α - yₙ²
		let product = Self::two_product(initial_estimate, difference) * Self::from(0.5_f32); //½xₙ(α - yₙ²)
		Self::from(premultiplied_estimate) + product //yₙ + ½xₙ(α - yₙ²), the complete iteration of Newton's Method.
	}

	/// Compute the natural exponential function of this number.
	///
	/// The natural exponential function is the unique non-zero function which has itself as its
	/// derivative. The function maps ƒ(x) = eˣ where e is Euler's number, a mathematical constant
	/// equal to the limit with n → ∞ of (1 + ¹⁄ₙ)ⁿ, or approximately 2.7182818284590452353602874714.
	/// It is also the inverse of the natural logarithm function `ln`, such that `exp(ln(x)) == x`.
	///
	/// # Implementation
	/// The exponential function is calculated with the power series ∑(xⁿ/n!). However, calculating
	/// xⁿ for sufficiently large values of x and n is problematic since it cannot be represented
	/// with the limited exponent available to this emulated 64-bit float. To increase the accuracy
	/// and to work well with high input values, we first divide by 2 until the input is in the
	/// range [-1, 1]. Division by two merely reduces the exponent of the number, so no accuracy is
	/// lost there. This division is later undone by repeatedly squaring the result the same amount
	/// of time. Some accuracy is lost by repeatedly squaring.
	///
	/// The power series is calculated by maintaining a running partial sum. The numerator and
	/// denominator are separately tracked, the numerator simply being multiplied by x every time
	/// and the denominator being multiplied by a constantly incrementing multiplier. This is
	/// repeated until the term being added to the partial sum is sufficiently small for the
	/// rounding errors of the emulated number (10⁻²⁰ times the original number). If the term ends
	/// up NaN, it means that the numerator or denominator ended up too high. The term added should
	/// be very small then, so we abort the enumeration and return the result.
	pub fn exp(self) -> EmulatedF64 {
		//First divide by 2 until we are in the range [-1, 1].
		let mut shrunk = self;
		let mut power_of_two = 0; //Track how often we did this.
		while shrunk.high.abs() > 1.0 {
			shrunk.high /= 2.0;
			shrunk.low /= 2.0;
			power_of_two += 1;
		}
		//Track the power series.
		let threshold = 1.0e-20 * shrunk.high.exp(); //Iterate until we add sufficiently small terms.
		let mut partial_sum = Self::from(1.0_f32) + shrunk; //First two terms.
		let mut current_power = shrunk.square();
		let mut multiplier = 2.0_f32; //Track as single-precision since this remains integer.
		let mut denominator = 2.0_f32; //Track as single-precision since this remains integer.
		let mut term = current_power / Self::from(denominator);
		while term.high.abs() > threshold {
			partial_sum += term;
			current_power *= shrunk;
			multiplier += 1.0;
			denominator *= multiplier;
			term = current_power / Self::from(denominator);
			if term.is_nan() {
				break;
			}
		}
		if !term.is_nan() {
			partial_sum += term; //Add the last term if we didn't break it off.
		}
		//Undo the shrinking.
		for _ in 0..power_of_two {
			partial_sum = partial_sum.square();
		}
		partial_sum
	}

	/// Compute the natural logarithm of the number.
	///
	/// The natural logarithm is the logarithm to the base of Euler's number: ƒ(x) = logₑ(x).
	/// Euler's number here is a mathematical constant equal to the limit with n → ∞ of (1 + ¹⁄ₙ)ⁿ,
	/// or approximately 2.7182818284590452353602874714. It is also the inverse of the natural
	/// exponential function `ln`, such that `ln(exp(x)) == x`.
	///
	/// # Implementation
	/// The natural logarithm is estimated with
	/// [Newton's Method](https://en.wikipedia.org/wiki/Newton's_method). Newton's Method tries to
	/// approximate a root by starting with an arbitrary estimate x, and iteratively approaching the
	/// root with the formula xₙ₊₁ = xₙ - ƒ(xₙ)/ƒ'(xₙ). In the case of the natural logarithm for our
	/// input α, the function ƒ(xₙ) is set to eˣⁿ - α (which intersects at 0 when x = ln(α) with its
	/// derivative eˣⁿ. Filling that into Newton's method gives xₙ₊₁ = xₙ - (exp(xₙ) - α)/exp(xₙ).
	/// Simplifying the formula to the concrete xₙ₊₁ = xₙ + α ⋅ exp(-xₙ) - 1 removes the costly
	/// division.
	///
	/// Newton's Method is quadratically convergent, meaning that with every iteration, the accuracy
	/// is doubled. To implement the natural logarithm of the `EmulatedF64` then, we simply use the
	/// built-in natural logarithm function for `f32` to arrive at our initial estimate. This
	/// estimate should be accurate to the 23 bits of mantissa in `f32`. We then process a single
	/// iteration of Newton's Method, resulting in an accuracy of 46 bits of mantissa, enough for
	/// the entire `EmulatedF64`.
	pub fn ln(self) -> EmulatedF64 {
		//First check some edge cases.
		if self.high == 1.0 && self.low == 0.0 {
			return Self::from(0.0_f32);
		}
		if self.high <= 0.0 {
			return Self::from(f32::NAN);
		}
		//Create the original estimate by using the built-in 32-bit implementation.
		let mut estimate = Self::from(self.high.ln());
		estimate += (-estimate).exp() * self + Self::from(-1.0_f32);
		estimate
	}

	/// Compute the cosine function applied to this number.
	///
	/// The cosine is defined by a right-angle triangle with the given angle in one of the two non-
	/// right corners, as the ratio between the edge adjacent to the corner and the hypotenuse of
	/// the triangle. Somewhat simpler, it is the X-coordinate of a point around the unit circle at
	/// the given angle, starting from to the right.
	///
	/// ![A right triangle with angle α indicated in the lower left, the "adjacent" on the bottom, "opposite" on the right and "hypotenuse" in the slanted edge.][sine_cosine_triangle]
	/// ![A circle with radius 1, with a line drawn from the centre at angle α, indicating that the line ends on X coordinate cos(α) and Y coordinate sin(α).][sine_cosine_unit_circle]
	///
	/// # Implementation
	/// Here we calculate the cosine by using the sine function. Since cos(α) = sin(π/2 - α), we can
	/// simply calculate π/2 - α and then return the sine of that.
	pub fn cos(self) -> EmulatedF64 {
		let half_pi = EmulatedF64 { high: 1.57079637050628662109375, low: -0.00000004371138828673792886547744274139404296875 };
		let shifted = half_pi - self;
		shifted.sin()
	}

	/// Compute the sine function applied to this number.
	///
	/// The sine is defined by a right-angle triangle with the given angle in one of the two non-
	/// right corners, as the ratio between the edge opposite to the corner and the hypotenuse of
	/// the triangle. Somewhat simpler, it is the Y-coordinate of a point around the unit circle at
	/// the given angle, starting from the right.
	///
	/// ![A right triangle with angle α indicated in the lower left, the "adjacent" on the bottom, "opposite" on the right and "hypotenuse" in the slanted edge.][sine_cosine_triangle]
	/// ![A circle with radius 1, with a line drawn from the centre at angle α, indicating that the line ends on X coordinate cos(α) and Y coordinate sin(α).][sine_cosine_unit_circle]
	///
	/// # Implementation
	/// The sine function is calculated using a its Taylor series. Since the input angle α is given
	/// in radians, the Taylor series approximates the sine using:
	/// sin(α) = α - α³/3! + α⁵/5! - α⁷/7! + ... using sufficient terms to get the accuracy required
	/// for this emulated `f64`.
	///
	/// However, the greater the input α, the more terms we need to achieve that accuracy,
	/// eventually reaching factorials too large for this emulation to represent accurately. For
	/// that reason, the input is modulated to the range [-π,π]. Outside of this range, the output
	/// repeats, so merely clipping the input space without any adjustment to the output will
	/// greatly improve performance.
	///
	/// The Taylor series is calculated by maintaining a running sum of its terms, and continuing to
	/// add more terms until the next term is within the rounding error of the representation, in
	/// this case 10⁻²⁰. At each term, we calculate the next power by multiplying the previous one
	/// with a (pre-calculated) α², and the next denominator by multiplying the previous one by an
	/// incrementing multiplier twice.
	pub fn sin(self) -> EmulatedF64 {
		//TODO: Clip the input!
		if self.high == 0.0 {
			return Self::from(0.0_f32);
		}
		let threshold = 1.0e-20_f32;
		let negative_square = -self.square(); //Pre-compute this multiplier for the numerator of each factor.
		let mut partial_sum = self;
		let mut power = self; //The numerators.
		let mut multiplier = 1.0_f32; //Each iteration, this increases by 2. Since these are integers and stay low, f32 is enough.
		let mut denominator = Self::from(1.0_f32);
		loop {
			power *= negative_square;
			multiplier += 2.0;
			denominator *= Self::from(multiplier * (multiplier - 1.0));
			let term = power / denominator;
			if term.is_nan() {
				//Happens if the power or denominator gets too big to represent.
				break;
			}
			partial_sum += term;
			if term.high.abs() < threshold { //The term is too small to make a difference.
				break;
			}
		}
		partial_sum
	}

	/// Compute the square of the given number, i.e. the number multiplied by itself.
	///
	/// This raises the number to the second power.
	///
	/// The square of the number is a special case of multiplication. This specialised function
	/// performs the multiplication slightly faster.
	pub fn square(self) -> EmulatedF64 {
		let mut p = Self::two_square(self.high); //Specialised two-sum for squaring.
		p.low += self.high * self.low * 2.0; //Multiply by 2 instead of adding the same value twice. Multiplying by 2 incurs no loss of precision.
		Self::two_sum_quick(p.high, p.low)
	}

	/// Split an `f32` number into a high and low component.
	///
	/// The number is split such that multiplying the components of two split numbers individually
	/// will not cause any round-off errors.
	///
	/// # Implementation
	/// The number is multiplied by 2^12 + 1, which causes a round-off error of the least
	/// significant 12 bits in the `f32`'s 23-bit mantissa. This effectively splits the original 23
	/// bits of mantissa into two numbers, one containing the most significant 13 bits, and the
	/// other containing the least significant 12 bits of that mantissa. Each of these components
	/// can safely be multiplied with one another without round-off error.
	fn split(value: f32) -> EmulatedF64 {
		const SPLITTER: f32 = ((1 << 12) + 1) as f32; //2^12 + 1
		let rounded_max = value * SPLITTER; //Maximum round-off error.
		let high = rounded_max - (rounded_max - value); //Mask the mantissa of the original value with this round-off error.
		let low = value - high; //The remainder.
		EmulatedF64 { high: high, low: low }
	}

	/// Compute the multiplication of two `f32` numbers and the exact round-off error.
	///
	/// The multiplied result together with the round-off error are returned as an `EmulatedF64`.
	/// This result represents the same value as the input.
	///
	/// # Implementation
	/// The multiplication is calculated with a simple multiply of the two numbers. The round-off
	/// error is more complex though. In order to do this, we need to split each of the operands
	/// into a high-order component and a low-order component. Each of these components will fit
	/// into an `f32` value without overflow, and sum up to the original operands.
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
		let a_split = Self::split(a);
		let b_split = Self::split(b);
		let error = (a_split.high * b_split.high - product) + a_split.high * b_split.low + a_split.low * b_split.high + a_split.low * b_split.low;
		EmulatedF64 { high: product, low: error }
	}

	/// Specialised version of `two_product` where the two components are the same.
	///
	/// This version can be implemented slightly faster. Instead of splitting the two operands, we
	/// only need to split the one. And two terms in the error calculation become the same, so we
	/// can simply multiply one of them by two and leave out the other.
	fn two_square(value: f32) -> EmulatedF64 {
		let product = value * value;
		let value_split = Self::split(value);
		let error = (value_split.high * value_split.high - product) + value_split.high * value_split.low * 2.0 + value_split.low * value_split.low;
		EmulatedF64 { high: product, low: error }
	}

	/// Compute the sum of two `f32` numbers and the exact round-off error.
	///
	/// This implements the [2Sum](https://en.wikipedia.org/wiki/2Sum) operation, which calculates
	/// the sum of two numbers and the round-off error of this sum separately. Assuming that the sum
	/// does not overflow, it calculates the sum correctly rounded (returned in the `high` component
	/// of the result) and the error correctly rounded (returned in the `low` component) to the
	/// nearest available floating point value.
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
		EmulatedF64 { high: rounded_sum, low: error }
	}

	/// Compute the exact round-off error of adding two `f32` numbers where we know that
	/// `|a|` >= `|b|`.
	///
	/// While the original `two_sum` algorithm uses 6 floating point operations, this version uses
	/// only 3, but depends on the knowledge that the exponent of `a` is at least as large as the
	/// exponent of `b`.
	///
	/// The `two_sum` algorithm does not use this quick variant, because comparing the two exponents
	/// and swapping the values if needed still uses more operations than the original `two_sum`
	/// algorithm.
	fn two_sum_quick(a: f32, b: f32) -> EmulatedF64 {
		let rounded_sum = a + b;
		let error = b - (rounded_sum - a);
		EmulatedF64 { high: rounded_sum, low: error }
	}
}

impl fmt::Debug for EmulatedF64 {
	/// Format this number for debugging.
	///
	/// In debugging, this number is formatted as the sum of its two components. For instance, the
	/// number `0.8000000001` gets formatted as `0.8+0.0000000001`.
	///
	/// # Arguments:
	/// * `formatter` - The formatter used to write the output.
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "{}+{}", self.high, self.low)
	}
}

impl fmt::Display for EmulatedF64 {
	/// Format this number for display.
	///
	/// This shows the number that this emulated `f64` represents. First it calculates the `f64`
	/// itself, and then it simply formats that number in the result.
	///
	/// # Arguments:
	/// * `formatter` - The formatter used to write the output.
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let as_f64: f64 = (*self).into();
		write!(f, "{}", as_f64)
	}
}

impl From<f64> for EmulatedF64 {
	/// Transform a real `f64` number into an `EmulatedF64` that represents approximately the same
	/// number.
	///
	/// Since `EmulatedF64` does not quite have the same accuracy as a real `f64`, the number may
	/// get slightly rounded, but it will not get rounded nearly as much as rounding it to an `f32`.
	///
	/// # Arguments:
	/// * `value` - The `f64` value that needs to be transformed to an `EmulatedF64`.
	///
	/// # Implementation
	/// The number is multiplied by 2^29 + 1, which causes a round-off error of the least
	/// significant 29 bits in the `f64`'s 53-bit mantissa. This effectively splits the original 53
	/// bits of mantissa into two numbers, one containing the most significant 24 bits, and the
	/// other containing the least significant 29 bits of that mantissa. The last of these then gets
	/// rounded off to contain only 24 bits of mantissa.
	fn from(value: f64) -> EmulatedF64 {
		const SPLITTER: f64 = ((1 << 29) + 1) as f64;
		let rounded_max = value * SPLITTER;
		let high = rounded_max - (rounded_max - value);
		let low = value - high;
		EmulatedF64 { high: high as f32, low: low as f32 }
	}
}

impl From<f32> for EmulatedF64 {
	/// Promote an `f32` number to an `EmulatedF64` that represents the same number.
	///
	/// The resulting `EmulatedF64` represents exactly the same number. It will use more memory
	/// though, and operations will be more expensive on it.
	///
	/// # Arguments:
	/// * `value` - The `f32` value that needs to be transformed to an `EmulatedF64`.
	fn from(value: f32) -> EmulatedF64 {
		EmulatedF64 { high: value, low: 0.0 }
	}
}

impl From<i32> for EmulatedF64 {
	/// Convert an `i32` integer to an `EmulatedF64` that represents the same number.
	///
	/// The resulting `EmulatedF64` represents exactly the same number. The `EmulatedF64` can
	/// represent every `i32` value.
	///
	/// # Arguments:
	/// * `value` - The `i32` value that needs to be transformed to an `EmulatedF64`.
	fn from(value: i32) -> EmulatedF64 {
		let high = value as f32;
		let low = (value - high as i32) as f32;
		EmulatedF64 { high: high, low: low }
	}
}

impl Into<f64> for EmulatedF64 {
	/// Calculate the `f64` number that is represented by this emulation.
	///
	/// The emulation holds that the number represented is the sum of the two `f32` components it
	/// holds.
	fn into(self) -> f64 {
		self.high as f64 + self.low as f64
	}
}

impl Into<f32> for EmulatedF64 {
	/// Round the `EmulatedF64` into its closest value that can be represented by an `f32` number.
	///
	/// Due to how the emulation splits up its number into a high-significance and a
	/// low-significance number, where the range of the low-significance number is entierly
	/// contained in the high-significance number, this can simply only return the high-significance
	/// number.
	fn into(self) -> f32 {
		self.high
	}
}

impl Add for EmulatedF64 {
	/// The output type of the sum.
	///
	/// In this case, the sum results in the same type as its operands.
	type Output = Self;

	/// Add another number to this number.
	///
	/// The sum is not done in-place. It will return a new number.
	///
	/// # Arguments
	/// * `rhs` - The number to add to this number.
	fn add(self, rhs: Self) -> Self::Output {
		let mut sum_highs = Self::two_sum(self.high, rhs.high);
		let sum_lows = Self::two_sum(self.low, rhs.low);
		sum_highs.low += sum_lows.high;
		sum_highs = Self::two_sum_quick(sum_highs.high, sum_highs.low);
		sum_highs.low += sum_lows.low;
		Self::two_sum_quick(sum_highs.high, sum_highs.low)
	}
}

impl AddAssign for EmulatedF64 {
	/// Add another number to this number, in-place.
	///
	/// # Arguments
	/// * `rhs` - The number to add to this number.
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl Mul for EmulatedF64 {
	/// The output type of the multiplication.
	///
	/// In this case, the multiplication results in the same type as its operands.
	type Output = Self;

	/// Multiply this number with another number.
	///
	/// The multiplication is not done in-place. It will return a new number.
	///
	/// # Arguments
	/// * `rhs` - The number to multiply this number with.
	fn mul(self, rhs: Self) -> Self::Output {
		//First we calculate the product and error of the multiplication.
		let mut product_and_error = Self::two_product(self.high, rhs.high);
		//We then have to factor in the error of the multiplication into the result.
		product_and_error.low += self.high * rhs.low;
		product_and_error.low += self.low * rhs.high;
		//Re-arrange the result into a properly non-overlapping result by quick-summing the components.
		Self::two_sum_quick(product_and_error.high, product_and_error.low)
	}
}

impl MulAssign for EmulatedF64 {
	/// Multiply this number with another number, in-place.
	///
	/// # Arguments
	/// * `rhs` - The number to multiply this number with.
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}

impl Sub for EmulatedF64 {
	/// The output type of the subtraction.
	///
	/// In this case, the subtraction results in the same type as its operands.
	type Output = Self;

	/// Subtract another number from this number.
	///
	/// The subtraction is not done in-place. It will return a new number.
	///
	/// # Arguments
	/// * `rhs` - The number to subtract from this number.
	fn sub(self, rhs: Self) -> Self::Output {
		//Use the add-operation in combination with a negation for this implementation.
		self + -rhs
	}
}

impl SubAssign for EmulatedF64 {
	/// Subtract another number from this number, in-place.
	///
	/// # Arguments
	/// * `rhs` - The number to subtract from this number.
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}

impl Div for EmulatedF64 {
	/// The output type of the division.
	///
	/// In this case, the division results in the same type as its operands.
	type Output = Self;

	/// Divide this number by another number.
	///
	/// The division is not done in-place. It will return a new number.
	///
	/// # Arguments
	/// * `rhs` - The number to divide this number by.
	///
	/// # Implementation
	/// The division is estimated with
	/// [Newton's Method](https://en.wikipedia.org/wiki/Newton's_method), which is then enhanced to
	/// need fewer high-precision operations with Karp's Method in their article High Precision
	/// Division and Square Root (1997, Karp & Markstein). Newton's Method tries to approximate the
	/// reciprocal of our input β, 1/β, and then multiply this number α by 1/β to effectively divide
	/// it by β. The approximation is done by starting with an arbitrary estimate x, and iteratively
	/// approaching the root with the formula xₙ₊₁ = xₙ - ƒ(xₙ)/ƒ'(xₙ). In the case of the
	/// reciprocal, the function ƒ(xₙ) is set to 1/x - β with its derivative -1/x². Filling that
	/// into Newton's method gives xₙ₊₁ = xₙ - (1/xₙ - β)/(-1/xₙ²), simplifying the formula to the
	/// concrete xₙ₊₁ = xₙ + xₙ(1 - βxₙ).
	///
	/// Approximating this needs only multiplication, but requires many high-precision
	/// multiplications. Using multi-component floats we can change this formula to:
	/// yₙ₊₁ = yₙ + xₙ(α - βyₙ) with yₙ = αxₙ. Instead of converging to xₙ₊₁ we now converge to
	/// αxₙ₊₁. This brings the multiplication inside of the term that we're converging to, resulting
	/// in fewer high-precision multiplications being necessary. The multiplication with α at the
	/// end is factored out. Because Newton's method corrects the initial estimate sufficiently
	/// fast, we don't even need to compute yₙ = αxₙ with high accuracy.
	///
	/// Newton's Method is quadratically convergent, meaning that with every iteration, the accuracy
	/// is doubled. To implement the division of the `EmulatedF64` then, we simply use the built-in
	/// division operator for `f32` to arrive at our initial estimate. This estimate should be
	/// accurate to the 23 bits of mantissa in `f32`. We then process a single iteration of Newton's
	/// Method, resulting in an accuracy of 46 bits of mantissa, enough for the entire
	/// `EmulatedF64`.
	fn div(self, rhs: Self) -> Self::Output {
		let initial_estimate = 1.0 / rhs.high; //xₙ in the above formulas.
		let premultiplied_estimate = self.high * initial_estimate; //yₙ in the above formulas.
		let premultiplied_promoted = Self::from(premultiplied_estimate);
		let difference = (self - rhs * premultiplied_promoted).high; //α - βyₙ
		let product = Self::two_product(initial_estimate, difference); //xₙ(α - βyₙ)
		premultiplied_promoted + product //yₙ + xₙ(α - βyₙ), the complete iteration of Newton's Method.
	}
}

impl DivAssign for EmulatedF64 {
	/// Divide this number by another number, in-place.
	///
	/// # Arguments
	/// * `rhs` - The number to divide this number by.
	fn div_assign(&mut self, rhs: Self) {
		*self = *self / rhs;
	}
}

impl Neg for EmulatedF64 {
	/// The result type when negating.
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
		EmulatedF64 { high: -self.high, low: -self.low }
	}
}

impl Rem for EmulatedF64 {
	/// The result type of the remainder operation.
	///
	/// In this case, calculating the remainder results in the same type as the original.
	type Output = Self;

	/// Performs the remainder operation, calculating the remainder of a division.
	///
	/// The remainder operation pretends to divide the number by the given operand, but instead of
	/// returning the division as a fraction, only returns the remainder if the maximum number of
	/// integer multiples are subtracted from it.
	///
	/// For instance, `10 % 4` would return `2`, because subtracting `4` from `10` twice takes off
	/// `8`, but leaves a remainder of `2`.
	///
	/// Care needs to be taken when dealing with negative numbers. When dealing with numerator 𝒩 and
	/// denominator 𝒟, the output of `𝒩%𝒟`` will be positive or negative depending on which of the
	/// following cases applies:
	/// * Positive 𝒩, positive 𝒟 → positive output
	/// * Positive 𝒩, negative 𝒟 → positive output
	/// * Negative 𝒩, positive 𝒟 → negative output
	/// * Negative 𝒩, negative 𝒟 → negative output
	/// As a result, the output will be negative if the original number is.
	fn rem(self, rhs: Self) -> Self::Output {
		self - (self / rhs).trunc() * rhs
	}
}

#[cfg(test)]
mod tests {
	use assert_float_eq::assert_float_absolute_eq;
	use test_case::test_case;
	use super::*;
	use crate::coordinate;

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
	fn convert_loop_f64(value: f64) {
		let emulated = EmulatedF64::from(value);
		let converted: f64 = emulated.into();
		assert_float_absolute_eq!(value, converted);
	}

	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.7099999; "Almost 0.8")]
	#[test_case(0.8000001; "Just over 0.8")]
	#[test_case(123456792.0; "Big positive")]
	#[test_case(3.141593; "Pi")]
	#[test_case(-123456792.0; "Big negative")]
	fn convert_loop_f32(value: f32) {
		let emulated = EmulatedF64::from(value);
		let converted: f32 = emulated.into();
		assert_float_absolute_eq!(value, converted);
	}

	#[test_case(0; "Zero")]
	#[test_case(1; "One")]
	#[test_case(1_000_000_000; "One billion")]
	#[test_case(123456789; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(-123456789; "Big negative")]
	#[test_case(2_147_483_647; "Maximum i32")]
	#[test_case(-2_147_483_648; "Minimum i32")]
	fn convert_loop_i32_rounding(value: i32) {
		let emulated = EmulatedF64::from(value);
		let rounded = emulated.round();
		assert_eq!(value, rounded);
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
	fn add_assign(lhs: f64, rhs: f64) {
		let mut emulated_lhs = EmulatedF64::from(lhs);
		let emulated_rhs = EmulatedF64::from(rhs);
		let using_f64 = lhs + rhs;
		emulated_lhs += emulated_rhs;
		assert_float_absolute_eq!(using_f64, emulated_lhs.into());
	}

	#[test_case(0.0, 0.0; "Zeroes")]
	#[test_case(1.0, 0.0; "One and zero")]
	#[test_case(10_000_000_000.0, 0.0000000001; "High and low")]
	#[test_case(0.0000000001, 10_000_000_000.0; "Low and high")]
	#[test_case(0.7999999999, 10_000_000_000.0; "Just below 0.8")]
	#[test_case(123456789.0, 0.71; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(-12345678.0, 12345678.0; "Negative and positive")]
	#[test_case(12345678.0, -12345678.0; "Positive and negative")]
	#[test_case(-12345678.0, -12345678.0; "Negative and negative")]
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
	#[test_case(-12345678.0, 12345678.0; "Negative and positive")]
	#[test_case(12345678.0, -12345678.0; "Positive and negative")]
	#[test_case(-12345678.0, -12345678.0; "Negative and negative")]
	fn multiply_assign(lhs: f64, rhs: f64) {
		let mut emulated_lhs = EmulatedF64::from(lhs);
		let emulated_rhs = EmulatedF64::from(rhs);
		let using_f64 = lhs * rhs;
		emulated_lhs *= emulated_rhs;
		assert_float_absolute_eq!(using_f64, emulated_lhs.into());
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

	#[test_case(0.0, 0.0; "Zeroes")]
	#[test_case(1.0, 0.0; "One and zero")]
	#[test_case(10_000_000_000.0, 0.0000000001; "High and low")]
	#[test_case(0.0000000001, 10_000_000_000.0; "Low and high")]
	#[test_case(0.7999999999, 10_000_000_000.0; "Just below 0.8")]
	#[test_case(123456789.0, 0.71; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(-123456789.0, 123456789.0; "Negative and positive")]
	#[test_case(123456789.0, -123456789.0; "Positive and negative")]
	#[test_case(-123456789.0, -123456789.0; "Negative and negative")]
	fn subtract_assign(lhs: f64, rhs: f64) {
		let mut emulated_lhs = EmulatedF64::from(lhs);
		let emulated_rhs = EmulatedF64::from(rhs);
		let using_f64 = lhs - rhs;
		emulated_lhs -= emulated_rhs;
		assert_float_absolute_eq!(using_f64, emulated_lhs.into());
	}

	#[test_case(0.0, 1.0; "Zero and one")]
	#[test_case(10_000_000_000.0, 0.0000000001; "High and low")]
	#[test_case(0.0000000001, 10_000_000_000.0; "Low and high")]
	#[test_case(0.7999999999, 10_000_000_000.0; "Just below 0.8")]
	#[test_case(123456789.0, 0.71; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(-123456789.0, 123456789.0; "Negative and positive")]
	#[test_case(123456789.0, -123456789.0; "Positive and negative")]
	#[test_case(-123456789.0, -123456789.0; "Negative and negative")]
	fn divide(lhs: f64, rhs: f64) {
		let emulated_lhs = EmulatedF64::from(lhs);
		let emulated_rhs = EmulatedF64::from(rhs);
		let using_f64 = lhs / rhs;
		let result = (emulated_lhs / emulated_rhs).into();
		assert_float_absolute_eq!(using_f64, result);
	}

	#[test_case(0.0, 1.0; "Zero and one")]
	#[test_case(10_000_000_000.0, 0.0000000001; "High and low")]
	#[test_case(0.0000000001, 10_000_000_000.0; "Low and high")]
	#[test_case(0.7999999999, 10_000_000_000.0; "Just below 0.8")]
	#[test_case(123456789.0, 0.71; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(-123456789.0, 123456789.0; "Negative and positive")]
	#[test_case(123456789.0, -123456789.0; "Positive and negative")]
	#[test_case(-123456789.0, -123456789.0; "Negative and negative")]
	fn divide_assign(lhs: f64, rhs: f64) {
		let mut emulated_lhs = EmulatedF64::from(lhs);
		let emulated_rhs = EmulatedF64::from(rhs);
		let using_f64 = lhs / rhs;
		emulated_lhs /= emulated_rhs;
		assert_float_absolute_eq!(using_f64, emulated_lhs.into());
	}

	#[test_case(0.0, 1.0; "Zero and one")]
	#[test_case(10.0, 3.0; "Two integers")]
	#[test_case(10.0, 3.5; "Fractional operand, integer result")]
	#[test_case(10.0, 2.0; "Zero result")]
	#[test_case(10.0, 3.141592653589793; "Fractional result")]
	#[test_case(-10.0, 3.0; "Two integers, negative positive")]
	#[test_case(-10.0, 3.5; "Fractional operand, integer result, negative positive")]
	#[test_case(-10.0, 2.0; "Zero result, negative positive")]
	#[test_case(-10.0, 3.141592653589793; "Fractional result, negative positive")]
	#[test_case(10.0, -3.0; "Two integers, positive negative")]
	#[test_case(10.0, -3.5; "Fractional operand, integer result, positive negative")]
	#[test_case(10.0, -2.0; "Zero result, positive negative")]
	#[test_case(10.0, -3.141592653589793; "Fractional result, positive negative")]
	#[test_case(-10.0, -3.0; "Two integers, negative")]
	#[test_case(-10.0, -3.5; "Fractional operand, integer result, negative")]
	#[test_case(-10.0, -2.0; "Zero result, negative")]
	#[test_case(-10.0, -3.141592653589793; "Fractional result, negative")]
	#[test_case(1_000_000_000.0, 0.0000000001; "High and low")]
	#[test_case(0.0000000001, 1_000_000_000.0; "Low and high")]
	#[test_case(0.7999999999, 1_000_000_000.0; "Just below 0.8")]
	#[test_case(-12345678.0, 12345678.0; "Negative and positive")]
	#[test_case(12345678.0, -12345678.0; "Positive and negative")]
	#[test_case(-12345678.0, -12345678.0; "Negative and negative")]
	fn remainder(lhs: f64, rhs: f64) {
		let emulated_lhs = EmulatedF64::from(lhs);
		let emulated_rhs = EmulatedF64::from(rhs);
		let using_f64 = lhs % rhs;
		let result = (emulated_lhs % emulated_rhs).into();
		assert_float_absolute_eq!(using_f64, result);
	}

	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(10_000_000_000.0; "Ten billion")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.9999999999; "Almost 1")]
	#[test_case(0.4999999999; "Almost 0.5")]
	#[test_case(0.5000000001; "Just over 0.5")]
	#[test_case(0.5; "Exactly 0.5")]
	#[test_case(-0.4999999999; "Almost negative 0.5")]
	#[test_case(-0.5000000001; "Just under negative 0.5")]
	#[test_case(-0.5; "Exactly negative 0.5")]
	#[test_case(1_000_000_000.01; "Just over a billion")]
	#[test_case(123456789.0; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(123456793.0; "f32 rounds down to 123456792, f64 doesn't")]
	#[test_case(3.141592653589793; "Pi")]
	#[test_case(-123456789.0; "Big negative")]
	fn negate(value: f64) {
		let emulated = EmulatedF64::from(value);
		let negated: f64 = (-emulated).into();
		assert_float_absolute_eq!(negated, -value);
	}

	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(10_000_000_000.0; "Ten billion")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.9999999999; "Almost 1")]
	#[test_case(0.5; "Exactly 0.5")]
	#[test_case(-0.5; "Exactly negative 0.5")]
	#[test_case(1_000_000_000.01; "Just over a billion")]
	#[test_case(123456789.0; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(123456793.0; "f32 rounds down to 123456792, f64 doesn't")]
	#[test_case(3.141592653589793; "Pi")]
	#[test_case(-123456789.0; "Big negative")]
	fn abs(value: f64) {
		let emulated = EmulatedF64::from(value);
		let absolute = emulated.abs();
		let using_f64 = value.abs();
		assert_float_absolute_eq!(absolute.into(), using_f64);
	}

	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(10_000_000_000.0; "Ten billion")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.9999999999; "Almost 1")]
	#[test_case(0.5; "Exactly 0.5")]
	#[test_case(-0.4999999999; "Almost negative 0.5")]
	#[test_case(1_000_000_000.01; "Just over a billion")]
	#[test_case(123456789.0; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(123456793.0; "f32 rounds down to 123456792, f64 doesn't")]
	#[test_case(3.141592653589793; "Pi")]
	#[test_case(-123456789.0; "Big negative")]
	fn signum(value: f64) {
		let emulated = EmulatedF64::from(value);
		let signum = emulated.signum();
		let using_f64 = value.signum();
		assert_float_absolute_eq!(signum.into(), using_f64);
	}
	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(10_000_000_000.0; "Ten billion")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.5; "Exactly 0.5")]
	#[test_case(-0.5; "Exactly negative 0.5")]
	#[test_case(1_000_000_000.01; "Just over a billion")]
	#[test_case(123456789.0; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(123456793.0; "f32 rounds down to 123456792, f64 doesn't")]
	#[test_case(3.141592653589793; "Pi")]
	#[test_case(-123456789.0; "Big negative")]
	fn trunc(value: f64) {
		let emulated = EmulatedF64::from(value);
		let truncated = emulated.trunc();
		let using_f64 = value.trunc();
		assert_float_absolute_eq!(truncated.into(), using_f64);
	}

	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(10_000_000_000.0; "Ten billion")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.9999999999; "Almost 1")]
	#[test_case(0.4999999999; "Almost 0.5")]
	#[test_case(0.5000000001; "Just over 0.5")]
	#[test_case(0.5; "Exactly 0.5")]
	#[test_case(-0.4999999999; "Almost negative 0.5")]
	#[test_case(-0.5000000001; "Just under negative 0.5")]
	#[test_case(-0.5; "Exactly negative 0.5")]
	#[test_case(1_000_000_000.01; "Just over a billion")]
	#[test_case(123456789.0; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(123456793.0; "f32 rounds down to 123456792, f64 doesn't")]
	#[test_case(3.141592653589793; "Pi")]
	#[test_case(-123456789.0; "Big negative")]
	fn round(value: f64) {
		let emulated = EmulatedF64::from(value);
		let rounded = emulated.round();
		assert_eq!(rounded, coordinate::round(value));
	}

	#[test_case(1.0; "One")]
	#[test_case(10_000_000_000.0; "Ten billion")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.9999999999; "Almost 1")]
	#[test_case(0.4999999999; "Almost 0.5")]
	#[test_case(0.5000000001; "Just over 0.5")]
	#[test_case(0.5; "Exactly 0.5")]
	#[test_case(1_000_000_000.01; "Just over a billion")]
	#[test_case(123456789.0; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(123456793.0; "f32 rounds down to 123456792, f64 doesn't")]
	#[test_case(3.141592653589793; "Pi")]
	fn sqrt(value: f64) {
		let emulated = EmulatedF64::from(value);
		let result = emulated.sqrt().into();
		assert_float_absolute_eq!(value.sqrt(), result);
	}

	#[test_case(-1.0; "One")]
	#[test_case(0.0; "Zero")] //Not negative strictly, but sqrt(0) should also be NaN.
	#[test_case(-10_000_000_000.0; "Ten billion")]
	#[test_case(-3.141592653589793; "Pi")]
	fn sqrt_negative(value: f64) {
		let emulated = EmulatedF64::from(value);
		let result = emulated.sqrt();
		assert!(result.is_nan(), "Square root of a negative number is always NaN.");
	}

	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.7999999999; "Almost 0.8")]
	#[test_case(0.8000000001; "Just over 0.8")]
	#[test_case(-0.4999999999; "Almost negative 0.5")]
	#[test_case(-0.5000000001; "Just under negative 0.5")]
	#[test_case(-0.5; "Exactly negative 0.5")]
	#[test_case(3.141592653589793; "Pi")]
	#[test_case(-3.141592653589793; "Negative pi")]
	#[test_case(-100.0; "Negative 100")]
	fn exp(value: f64) {
		let emulated = EmulatedF64::from(value);
		let result = emulated.exp().into();
		assert_float_absolute_eq!(value.exp(), result);
	}

	#[test_case(1.0; "One")]
	#[test_case(10_000_000_000.0; "Ten billion")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.9999999999; "Almost 1")]
	#[test_case(0.4999999999; "Almost 0.5")]
	#[test_case(0.5000000001; "Just over 0.5")]
	#[test_case(0.5; "Exactly 0.5")]
	#[test_case(1_000_000_000.01; "Just over a billion")]
	#[test_case(123456789.0; "f32 rounds to 123456792, f64 doesn't")]
	#[test_case(123456793.0; "f32 rounds down to 123456792, f64 doesn't")]
	#[test_case(3.141592653589793; "Pi")]
	fn ln(value: f64) {
		let emulated = EmulatedF64::from(value);
		let result = emulated.ln().into();
		assert_float_absolute_eq!(value.ln(), result);
	}

	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(2.0; "Two")]
	#[test_case(3.141592653589793; "Pi")]
	#[test_case(5.0; "Five")]
	#[test_case(6.283185307179586; "Tau")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.9999999999; "Almost 1")]
	#[test_case(0.4999999999; "Almost 0.5")]
	#[test_case(0.5000000001; "Just over 0.5")]
	#[test_case(0.5; "Exactly 0.5")]
	fn cos(value: f64) {
		let half_pi = EmulatedF64::from(1.5707963267948966192313216916397514420985846996875529104874722961_f64);
		println!("Half pi: {:.48},{:.48}", half_pi.high, half_pi.low);
		let emulated = EmulatedF64::from(value);
		let result = emulated.cos().into();
		assert_float_absolute_eq!(value.cos(), result);
	}

	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(2.0; "Two")]
	#[test_case(3.141592653589793; "Pi")]
	#[test_case(5.0; "Five")]
	#[test_case(6.283185307179586; "Tau")]
	#[test_case(0.71; "A fraction")]
	#[test_case(0.9999999999; "Almost 1")]
	#[test_case(0.4999999999; "Almost 0.5")]
	#[test_case(0.5000000001; "Just over 0.5")]
	#[test_case(0.5; "Exactly 0.5")]
	fn sin(value: f64) {
		let emulated = EmulatedF64::from(value);
		let result = emulated.sin().into();
		assert_float_absolute_eq!(value.sin(), result);
	}
}
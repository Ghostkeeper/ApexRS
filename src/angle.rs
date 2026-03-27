/*
 * Library for performing massively parallel computations on polygons.
 * Copyright (C) 2026 Ghostkeeper
 * This library is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
 * You should have received a copy of the GNU Affero General Public License along with this library. If not, see <https://gnu.org/licenses/>.
 */

//! Defines a modular arithmetic number that represents angles.

use bytemuck::{Pod, Zeroable}; //For sending angles to the GPU.
use std::fmt; //To represent angles in text.
use std::f64::consts::TAU; //To convert to degrees, and implement modular arithmetic.

/// Represents the measure of angle in a corner.
///
/// The measure of angle is effectively a measure of how open a corner between two line segments is.
/// It can also function as a measure of direction in two dimensions. When used as a direction, the
/// angle is taken from a vector going to the positive Y direction, with small positive angles going
/// counter-clockwise.
///
/// The angle struct is a modular arithmetic number. It has an internal representation as a floating
/// point number, but that number can never represent more than one full circle of rotation, and
/// never takes on a negative number either. The debugging representation of the angle shows the
/// number of degrees the angle represents. This number should always be between 0 (inclusive) and
/// 360 (exclusive). An angle of 360 or higher will be moved to the range [0, 360).
///
/// The angle does not have a unit, it is not specifically in degrees or in radians or anything
/// else.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Angle {
	/// The angle in radians that this angle represents.
	///
	/// Radians is chosen as storage format because that is the most common format necessary for
	/// trigonometric functions such as cosine and sine. It will be converted to degrees for display
	/// only.
	value: f64,
}

impl Angle {
	/// Short-hand for an angle that represents a half turn.
	///
	/// This is equivalent to π radians or 180°. When used for a direction vector, it represents
	/// inverting the direction of the vector.
	pub const HALF_TURN: Angle = Angle::radians(TAU / 2.0);

	/// Short-hand for an angle that represents a third of a turn.
	///
	/// This is equivalent to ⅔π radians or 120°.
	pub const THIRD_TURN: Angle = Angle::radians(TAU / 3.0);

	/// Short-hand for an angle that represents a quarter of a turn.
	///
	/// This is equivalent to ½π radians or 90°. When used for a direction vector, it represents a
	/// right-angle turn to the left.
	pub const QUARTER_TURN: Angle = Angle::radians(TAU / 4.0);

	/// Short-hand for an angle that represents three quarters of a turn.
	///
	/// This is equivalent to 1½π radians or 270°. When used for a direction vector, it represents a
	/// right-angle turn to the right.
	pub const THREE_QUARTER_TURN: Angle = Angle::radians(TAU * 3.0 / 4.0);

	/// Short-hand for an angle that represents a sixth of a turn.
	///
	/// This is equivalent to ⅓π radians or 60°.
	pub const SIXTH_TURN: Angle = Angle::radians(TAU / 6.0);

	/// Short-hand for an angle that represents an eighth of a turn.
	///
	/// This is equivalent to ¼π radians or 45°.
	pub const EIGHTH_TURN: Angle = Angle::radians(TAU / 8.0);

	/// Short-hand for an angle that represents a twelfth of a turn.
	///
	/// This is equivalent to ⅙π radians or 30°.
	pub const TWELFTH_TURN: Angle = Angle::radians(TAU / 12.0);

	/// Create a new angle from a number of radians.
	///
	/// # Arguments:
	/// * `radians` - The angle that needs to be represented, in radians.
	///
	/// # Examples:
	/// ```
	/// use std::f64::consts::TAU;
	/// use apex::Angle;
	/// let quarter_turn = Angle::radians(TAU / 4.0); //90 degrees, or a "straight angle".
	/// assert_eq!(quarter_turn, TAU / 4.0);
	/// let half_turn = Angle::radians(TAU / 2.0); //180 degrees.
	/// assert_eq!(half_turn, TAU / 2.0);
	/// let zero_angle = Angle::radians(0.0); //0 degrees.
	/// assert_eq!(zero_angle, 0.0);
	/// let full_turn = Angle::radians(TAU); //360 degrees, which gets stored as 0.
	/// assert_eq!(full_turn, 0.0);
	/// let negative_quarter_turn = Angle::radians(-TAU / 4.0); //-90 degrees, which gets stored as 270.
	/// assert_eq!(negative_quarter_turn, TAU * 0.75);
	/// let overturn = Angle::radians(-TAU * 1.75); //-630 degrees, which gets stored as 90.
	/// assert_eq!(overturn, TAU * 0.25);
	/// ```
	pub const fn radians(radians: f64) -> Angle {
		Angle { value: ((radians % TAU) + TAU) % TAU }
	}

	/// Create a new angle from a number of degrees.
	///
	/// # Arguments:
	/// * `degrees` - The angle that needs to be represented, in degrees.
	///
	/// # Examples:
	/// ```
	/// use std::f64::consts::TAU;
	/// use apex::Angle;
	/// let quarter_turn = Angle::degrees(90.0);
	/// assert_eq!(quarter_turn, TAU / 4.0, "Converted to 1/4 TAU.");
	/// let half_turn = Angle::degrees(180.0);
	/// assert_eq!(half_turn, TAU / 2.0, "Converted to 1/2 TAU.");
	/// ```
	pub const fn degrees(degrees: f64) -> Angle {
		Angle::radians(degrees / 360.0 * TAU)
	}

	/// Calculate the cosine function of this angle.
	///
	/// The cosine is defined by a right-angle triangle with the given angle in one of the two non-
	/// right corners, as the ratio between the edge adjacent to the corner and the hypotenuse of
	/// the triangle. Somewhat simpler, it is the X-coordinate of a point around the unit circle at
	/// the given angle, starting from to the right.
	///
	/// ![A right triangle with angle α indicated in the lower left, the "adjacent" on the bottom, "opposite" on the right and "hypotenuse" in the slanted edge.][sine_cosine_triangle]
	/// ![A circle with radius 1, with a line drawn from the centre at angle α, indicating that the line ends on X coordinate cos(α) and Y coordinate sin(α).][sine_cosine_unit_circle]
	///
	/// # Examples:
	/// ```
	/// use std::f64::consts::TAU;
	/// use apex::Angle;
	/// let thirty_degrees_cosine = Angle::radians(TAU / 12.0).cos(); //30 degrees.
	/// use assert_float_eq::assert_float_absolute_eq;
	/// assert_float_absolute_eq!(thirty_degrees_cosine, 0.5 * 3.0_f64.sqrt()); //The cosine of 30 degrees is sqrt(3)/2.
	/// ```
	pub fn cos(&self) -> f64 {
		self.value.cos()
	}

	/// Calculate the sine function of this angle.
	///
	/// The sine is defined by a right-angle triangle with the given angle in one of the two non-
	/// right corners, as the ratio between the edge opposite to the corner and the hypotenuse of
	/// the triangle. Somewhat simpler, it is the Y-coordinate of a point around the unit circle at
	/// the given angle, starting from the right.
	///
	/// ![A right triangle with angle α indicated in the lower left, the "adjacent" on the bottom, "opposite" on the right and "hypotenuse" in the slanted edge.][sine_cosine_triangle]
	/// ![A circle with radius 1, with a line drawn from the centre at angle α, indicating that the line ends on X coordinate cos(α) and Y coordinate sin(α).][sine_cosine_unit_circle]
	///
	/// # Examples:
	/// ```
	/// use std::f64::consts::TAU;
	/// use apex::Angle;
	/// let thirty_degrees_sine = Angle::radians(TAU / 12.0).sin(); //30 degrees.
	/// use assert_float_eq::assert_float_absolute_eq;
	/// assert_float_absolute_eq!(thirty_degrees_sine, 0.5); //The sine of 30 degrees is 1/2.
	/// ```
	pub fn sin(&self) -> f64 {
		self.value.sin()
	}
}

impl Into<f64> for Angle {
	/// Convert an angle into a floating point number, in radians.
	fn into(self) -> f64 {
		self.value
	}
}

impl From<f64> for Angle {
	/// Convert a floating point number in radians into an angle.
	///
	/// # Arguments:
	/// * `radians`: The angle that needs to be represented, in radians.
	///
	/// # Examples:
	/// ```
	/// use std::f64::consts::TAU;
	/// use apex::Angle;
	/// let quarter_turn = Angle::from(TAU * 0.25); //90 degrees, or a "straight angle".
	/// assert_eq!(quarter_turn, TAU * 0.25);
	/// let half_turn = Angle::from(TAU * 0.5); //180 degrees.
	/// assert_eq!(half_turn, TAU * 0.5);
	/// let zero_angle = Angle::from(0.0); //0 degrees.
	/// assert_eq!(zero_angle, 0.0);
	/// let full_turn = Angle::from(TAU); //360 degrees, which gets stored as 0.
	/// assert_eq!(full_turn, 0.0);
	/// let negative_quarter_turn = Angle::from(-TAU * 0.25); //-90 degrees, which gets stored as 270.
	/// assert_eq!(negative_quarter_turn, TAU * 0.75);
	/// let overturn = Angle::from(-TAU * 1.75); //-630 degrees, which gets stored as 90.
	/// assert_eq!(overturn, TAU * 0.25);
	/// ```
	fn from(radians: f64) -> Angle {
		Angle::radians(radians)
	}
}

impl fmt::Debug for Angle {
	/// A reference string representing the angle.
	///
	/// For human-readability, the angle is represented as degrees, rather than radians.
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "{}°", self.value * 360.0 / TAU)
	}
}

impl PartialEq<f64> for Angle {
	/// Compare equality of angles to a number of radians.
	///
	/// # Arguments
	/// * `other` - The radian value to compare to.
	///
	/// # Examples
	/// ```
	/// use apex::Angle;
	/// use std::f64::consts::TAU;
	/// let is_equal = Angle::radians(3.0) == 3.0;
	/// assert!(is_equal);
	/// let is_not_equal = Angle::radians(1.2) == 2.4;
	/// assert!(!is_not_equal);
	/// let is_equal_modulo = Angle::radians(2.0 + TAU) == 2.0;
	/// ```
	fn eq(&self, other: &f64) -> bool {
		self.value == *other
	}
}

#[cfg(test)]
mod tests {
	use assert_float_eq::assert_float_absolute_eq;
	use test_case::test_case;
	use super::*;

	/// Test converting radians into angles when the radians are within the 0 - τ clipping range.
	#[test_case(0.0; "Zero")]
	#[test_case(1.0; "One")]
	#[test_case(6.2; "Almost tau")]
	fn within_clipping_range(value: f64) {
		//Convert to Angle and back and see if it remained the same.
		let angle = Angle::radians(value);
		let converted: f64 = angle.into();
		assert_float_absolute_eq!(value, converted);
	}

	/// Test converting radians into angles when the radians are outside of the 0 - τ clipping
	/// range.
	///
	/// The value should be clipped to be within that range then.
	#[test_case(TAU, 0.0; "Tau")]
	#[test_case(-TAU / 4.0, TAU * 0.75; "Negative turn")]
	#[test_case(TAU * 1000.0 + 1.0, 1.0; "Big positive")]
	#[test_case(-TAU * 900.0 + 1.0, 1.0; "Big negative")]
	#[test_case(TAU * 1.5, TAU * 0.5; "Turn-and-a-half")]
	fn outside_clipping_range(value: f64, expected: f64) {
		//Convert to Angle and back and see if it became the expected value.
		let angle = Angle::radians(value);
		let converted: f64 = angle.into();
		assert_float_absolute_eq!(expected, converted);
	}

	/// Test converting degrees into angles.
	#[test_case(0.0, 0.0; "Zero")]
	#[test_case(30.0, TAU / 12.0; "Small angle")]
	#[test_case(90.0, TAU / 4.0; "Right angle")]
	#[test_case(180.0, TAU / 2.0; "Straight angle")]
	#[test_case(360.0, 0.0; "Full turn")]
	#[test_case(-90.0, TAU * 0.75; "Right angle negative")]
	#[test_case(630.0, TAU * 0.75; "Turn and three quarters")]
	#[test_case(-630.0, TAU / 4.0; "Negative turn and three quarters")]
	fn from_degrees(degrees: f64, radians: f64) {
		//Convert to Angle and then to radians and see if it became the expected value.
		let angle = Angle::degrees(degrees);
		let converted: f64 = angle.into();
		assert_float_absolute_eq!(radians, converted);
	}

	#[test_case(0.0, 1.0; "Zero")]
	#[test_case(TAU / 4.0, 0.0; "Quarter turn")]
	#[test_case(TAU / 2.0, -1.0; "Half turn")]
	#[test_case(TAU * 0.75, 0.0; "Three quarter turn")]
	#[test_case(TAU / 6.0, 0.5; "60 degrees")]
	#[test_case(TAU / 8.0, 1.0 / 2.0_f64.sqrt(); "45 degrees")]
	#[test_case(TAU / 12.0, 0.5 * 3.0_f64.sqrt(); "30 degrees")]
	/// Test getting the cosine of an angle.
	fn cosine(radians: f64, expected: f64) {
		let angle = Angle::radians(radians);
		let cosine = angle.cos();
		assert_float_absolute_eq!(expected, cosine);
	}

	#[test_case(0.0, 0.0; "Zero")]
	#[test_case(TAU / 4.0, 1.0; "Quarter turn")]
	#[test_case(TAU / 2.0, 0.0; "Half turn")]
	#[test_case(TAU * 0.75, -1.0; "Three quarter turn")]
	#[test_case(TAU / 6.0, 0.5 * 3.0_f64.sqrt(); "60 degrees")]
	#[test_case(TAU / 8.0, 1.0 / 2.0_f64.sqrt(); "45 degrees")]
	#[test_case(TAU / 12.0, 0.5; "30 degrees")]
	/// Test getting the cosine of an angle.
	fn sine(radians: f64, expected: f64) {
		let angle = Angle::radians(radians);
		let sine = angle.sin();
		assert_float_absolute_eq!(expected, sine);
	}
}
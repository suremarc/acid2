#![feature(const_float_bits_conv)]
#![feature(const_trait_impl)]
#![feature(bigint_helper_methods)]
#![feature(const_bigint_helper_methods)]
#![cfg_attr(not(test), no_std)]

use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
/// Double-precision floating-point 2-adic values.
///
/// Binary layout:
///
/// ```none
///     11111111 11100000 00000000 00000000 00000000 00000000 00000000 00000000
///     ^^^^^^^^^^^^
///           |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
///           |                                 |
/// This is the negative exponent (11 bits)     |
///                                         This is the significand (53 bits)
/// ```
///
/// In many ways, p-adic floating-point arithmetic is simpler than real floating-point arithmetic.
/// For one, we don't require a sign bit, since all "negative" integers have a convergent representation
/// as the limit of a sequence of "positive" integers. Really, there is no such thing as a negative number
/// in the p-adic universe; there are negatives of numbers, but there are no numbers that are negative, per se.
/// In any case, this means we have room for one more bit (53) in the significand.
///
/// The "negative exponent" is the negative of the 2-adic valuation. Because in the 2-adic universe a high
/// power of 2 makes a number smaller, the negative exponent is more similar to the regular floating-point exponent.
///
/// The least significant bit of the significand is always 1, except for infinity and zero.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64(u64);

const MASK_SIGNIFICAND: u64 = 0x1fffffffffffff;
const MASK_EXPONENT: u64 = !MASK_SIGNIFICAND;
const NEG_EXPONENT_MAX: i16 = 1024;
const NEG_EXPONENT_UNSIGNED_MAX: u16 = 2047;
const NEG_EXPONENT_UNSIGNED_ZERO: u16 = 1023;

impl F64 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self((NEG_EXPONENT_UNSIGNED_ZERO as u64) << 53 | 1);
    pub const INFINITY: Self = Self((NEG_EXPONENT_UNSIGNED_MAX as u64) << 53);
    pub const NAN: Self = Self((NEG_EXPONENT_UNSIGNED_MAX as u64) << 53 | 1);

    #[inline(always)]
    const fn neg_exponent(self) -> i16 {
        self.neg_exponent_unsigned() as i16 - NEG_EXPONENT_UNSIGNED_ZERO as i16
    }

    const fn neg_exponent_unsigned(self) -> u16 {
        (self.0 >> 53) as u16
    }

    /// The exponent of this 2-adic number.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let f = F64::from(8);
    ///
    /// assert_eq!(f.exponent(), 3);
    /// ```
    #[inline(always)]
    pub const fn exponent(self) -> i16 {
        -self.neg_exponent()
    }

    /// The part of this number coprime to 2.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let f = F64::from(36);
    ///
    /// assert_eq!(f.significand(), 9);
    /// ```
    #[inline(always)]
    pub const fn significand(self) -> u64 {
        self.0 & MASK_SIGNIFICAND
    }

    const fn neg_exponent_and_significand(self) -> (i16, u64) {
        (self.neg_exponent(), self.significand())
    }

    const fn neg_exponent_unsigned_and_significand(self) -> (u16, u64) {
        (self.neg_exponent_unsigned(), self.significand())
    }

    /// The exponent and significand of this number, packed together in a tuple.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let f = F64::from(12);
    ///
    /// assert_eq!(f.exponent_and_significand(), (2, 3));
    /// ```
    #[inline(always)]
    pub const fn exponent_and_significand(self) -> (i16, u64) {
        (self.exponent(), self.significand())
    }

    /// The 2-adic absolute value of this number.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let f = F64::from(72);
    ///
    /// assert_eq!(f.abs(), 2f64.powi(-3));
    /// assert_eq!(F64::ZERO.abs(), 0.0);
    /// assert_eq!(F64::INFINITY.abs(), f64::INFINITY);
    /// assert!(F64::NAN.abs().is_nan());
    /// ```
    #[inline(always)]
    pub const fn abs(self) -> f64 {
        f64::from_bits((self.0 & MASK_EXPONENT) >> 1 | self.is_nan() as u64)
    }

    /// The fractional part of this 2-adic number, representation as a real floating-point number.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let f = F64::from(13) / F64::from(8);
    ///
    /// assert_eq!(f.fract(), 0.625);
    /// ```
    #[inline]
    pub fn fract(self) -> f64 {
        ((self.significand() as f64) / self.abs()).fract()
    }

    /// Whether or not this number is NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let x = F64::ONE;
    /// let y = F64::NAN;
    /// let z = F64::INFINITY;
    ///
    /// assert!(!x.is_nan());
    /// assert!(y.is_nan());
    /// assert!(!z.is_nan());
    /// ```
    #[inline(always)]
    pub const fn is_nan(self) -> bool {
        self.0 & MASK_EXPONENT == MASK_EXPONENT && self.0 & MASK_SIGNIFICAND != 0
    }

    /// Whether or not this number is infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let x = F64::ONE;
    /// let y = F64::NAN;
    /// let z = F64::INFINITY;
    ///
    /// assert!(!x.is_infinite());
    /// assert!(!y.is_infinite());
    /// assert!(z.is_infinite());
    /// ```
    #[inline(always)]
    pub const fn is_infinite(self) -> bool {
        self.0 == Self::INFINITY.0
    }

    /// Whether or not this number is finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let x = F64::ONE;
    /// let y = F64::NAN;
    /// let z = F64::INFINITY;
    ///
    /// assert!(x.is_finite());
    /// assert!(!y.is_finite());
    /// assert!(!z.is_finite());
    /// ```
    #[inline(always)]
    pub const fn is_finite(self) -> bool {
        self.0 & MASK_EXPONENT != MASK_EXPONENT
    }

    /// Computes the 2-adic square root of this number.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let f = F64::from(81);
    /// let sqrt = f.sqrt();
    ///
    /// assert!((sqrt - F64::from(9)).abs() < 1e-15); // this won't always be true for perfect squares, but in this case it is
    /// assert!((sqrt * sqrt - f).abs() < 1e-7);
    /// ```
    // FIXME: investigate why this doesn't give exact results
    pub const fn sqrt(self) -> Self {
        let (e, s) = self.neg_exponent_and_significand();
        assert!(e % 2 == 0);
        assert!(s % 8 == 1);

        let two_b = self.significand() >> 2;
        let mut x = two_b; // mod 4; we proceed by doing a hensel lift
        let mut two_xp1 = x.wrapping_shl(1).wrapping_add(1);
        let mut i = 2;
        while i < 6 {
            // x' = x - (x^2 + x - 2b) / (2x + 1)
            x = x.wrapping_sub(
                x.wrapping_mul(x)
                    .wrapping_add(x)
                    .wrapping_sub(two_b)
                    .wrapping_mul(invert(two_xp1, i)),
            );
            two_xp1 = x.wrapping_shl(1) | 1;
            i += 1;
        }

        // debug_assert!(two_xp1.wrapping_mul(two_xp1) & MASK_SIGNIFICAND == s);

        Self(
            (((e / 2 + NEG_EXPONENT_UNSIGNED_ZERO as i16) as u64) << 53)
                | (two_xp1 & MASK_SIGNIFICAND),
        )
    }

    /// Computes the 2-adic reciprocal of this number.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let f = F64::from(13);
    /// let recip = f.recip();
    ///
    /// assert!((f * recip - F64::ONE).abs() < 1e-15);
    /// ```
    pub const fn recip(self) -> Self {
        let (e, s) = self.neg_exponent_unsigned_and_significand();

        let exponent = // short circuit to INF if equal to zero
            2046u16.wrapping_sub(e) | (((self.0 == 0) as u16) * NEG_EXPONENT_UNSIGNED_MAX);
        Self((exponent as u64) << 53 | invert(s, 5) & MASK_SIGNIFICAND)
    }
}

// find the multiplicative inverse modulo 2^(2^(N+1)), where N is num_iterations
// formula sourced from "Modern Computer Arithmetic" version 0.5.9, pg. 66
const fn invert(x: u64, num_iterations: u8) -> u64 {
    let mut i = 0;
    let mut inverse = x;
    while i < num_iterations {
        inverse = inverse.wrapping_mul(2u64.wrapping_sub(x.wrapping_mul(inverse)));
        i += 1;
    }

    inverse
}

impl const Neg for F64 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        let (e, s) = self.neg_exponent_unsigned_and_significand();
        Self(((e as u64) << 53) | (s.wrapping_neg() & MASK_SIGNIFICAND))
    }
}

impl const Add for F64 {
    type Output = Self;

    /// Computes the 2-adic sum of two numbers, truncating on the left side where necessary.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let x = F64::from(13);
    /// let y = F64::from(11);
    ///
    /// assert!((x + y - F64::from(24)).abs() < 1e-15);
    /// ```
    // TODO: investigate if this works for non-normal numbers
    fn add(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.neg_exponent_unsigned_and_significand();
        let (e1, s1) = rhs.neg_exponent_unsigned_and_significand();

        let max_e = e0.max(e1);
        let s2 = (s0.wrapping_shl((max_e - e0) as u32))
            .wrapping_add(s1.wrapping_shl((max_e - e1) as u32));
        let l = s2.trailing_zeros() as u16;
        let e2 = max_e.saturating_sub(l)
            | ((e0 == NEG_EXPONENT_UNSIGNED_MAX || e1 == NEG_EXPONENT_UNSIGNED_MAX) as u16
                * NEG_EXPONENT_UNSIGNED_MAX);

        Self((e2 as u64) << 53 | ((s2 >> l) & MASK_SIGNIFICAND))
    }
}

impl AddAssign for F64 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl const Mul for F64 {
    type Output = Self;

    /// Computes the 2-adic product of two numbers, truncating on the left side where necessary.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let x = F64::from(14);
    /// let y = F64::from(6);
    ///
    /// assert!((x * y - F64::from(84)).abs() < 1e-15);
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.neg_exponent_and_significand();
        let (e1, s1) = rhs.neg_exponent_and_significand();

        let mut e2 = ((e0 + e1 + NEG_EXPONENT_UNSIGNED_ZERO as i16) as u64)
            .min(NEG_EXPONENT_UNSIGNED_MAX as u64);

        // handle infinity or nan
        e2 |= (e0 == NEG_EXPONENT_MAX || e1 == NEG_EXPONENT_MAX) as u64
            * NEG_EXPONENT_UNSIGNED_MAX as u64;
        let mut s2 = s0.wrapping_mul(s1) & MASK_SIGNIFICAND;
        s2 |= (self.is_nan() || rhs.is_nan()) as u64;

        Self(e2 << 53 | s2)
    }
}

impl MulAssign for F64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl const Sub for F64 {
    type Output = Self;

    /// Computes the 2-adic difference of two numbers, truncating on the left side where necessary.
    ///
    /// `x - y` is exactly equivalent to `x + (-y)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let x = F64::from(13);
    /// let y = F64::from(11);
    ///
    /// assert!((x - y - F64::from(2)).abs() < 1e-15);
    /// ```
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl SubAssign for F64 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}

impl const Div for F64 {
    type Output = Self;

    /// Computes the 2-adic difference of two numbers, truncating on the left side where necessary.
    ///
    /// `x / y` is exactly equivalent to `x * y.recip()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use acid2::F64;
    ///
    /// let x = F64::from(18);
    /// let y = F64::from(6);
    ///
    /// assert!((x / y - F64::from(3)).abs() < 1e-15);
    /// ```
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)] // lol
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.recip()
    }
}

impl DivAssign for F64 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs
    }
}

impl const From<u32> for F64 {
    #[inline(always)]
    fn from(x: u32) -> Self {
        let l = x.trailing_zeros() as u16;
        Self(((NEG_EXPONENT_UNSIGNED_ZERO - l) as u64) << 53 | (x as u64) >> l)
    }
}

impl Debug for F64 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("F64")
            .field(&self.neg_exponent())
            .field(&self.significand())
            .finish()
    }
}

#[cfg(feature = "rand")]
impl rand::distributions::Distribution<F64> for rand::distributions::Standard {
    // Sample from the open disk |x| < 1, with frequency given by the 2-adic Haar measure.
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> F64 {
        let scale = rand_distr::StandardGeometric.sample(rng);
        let mut significand: u64 = self.sample(rng);
        significand |= 1; // make sure it's odd
        F64((NEG_EXPONENT_UNSIGNED_ZERO as u64 - scale) << 53 | (significand & MASK_SIGNIFICAND))
    }
}

#[cfg(test)]
mod tests {}

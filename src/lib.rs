#![feature(const_float_bits_conv)]
#![feature(const_trait_impl)]
#![feature(bigint_helper_methods)]
#![feature(const_bigint_helper_methods)]
// #![cfg_attr(not(test), no_std)]

use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F64(u64);

const MASK_SIGNIFICAND: u64 = 0x1fffffffffffff;
const MASK_EXPONENT: u64 = !MASK_SIGNIFICAND;
const EXPONENT_MAX: i16 = 1024;
const EXPONENT_UNSIGNED_MAX: u16 = 2047;
const EXPONENT_UNSIGNED_ZERO: u16 = 1023;

impl F64 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self((EXPONENT_UNSIGNED_ZERO as u64) << 53 | 1);
    pub const INFINITY: Self = Self((EXPONENT_UNSIGNED_MAX as u64) << 53);
    pub const NAN: Self = Self((EXPONENT_UNSIGNED_MAX as u64) << 53 | 1);

    #[inline(always)]
    pub const fn exponent(self) -> i16 {
        self.exponent_unsigned() as i16 - EXPONENT_UNSIGNED_ZERO as i16
    }

    const fn exponent_unsigned(self) -> u16 {
        (self.0 >> 53) as u16
    }

    #[inline(always)]
    pub const fn significand(self) -> u64 {
        self.0 & MASK_SIGNIFICAND
    }

    #[inline(always)]
    pub const fn split(self) -> (i16, u64) {
        (self.exponent(), self.significand())
    }

    const fn split_unsigned(self) -> (u16, u64) {
        (self.exponent_unsigned(), self.significand())
    }

    #[inline(always)]
    pub const fn abs(self) -> f64 {
        f64::from_bits((self.0 & MASK_EXPONENT) >> 1 | self.is_nan() as u64)
    }

    // FIXME: this is completely broken for most inputs
    #[inline]
    pub fn fract(self) -> f64 {
        let (e, s) = self.split_unsigned();
        let significand = (s >> 1).reverse_bits() >> 12;

        f64::from_bits(((EXPONENT_UNSIGNED_MAX - e) as u64 & 0x7ff) << 52 | significand).fract()
    }

    #[inline(always)]
    pub const fn is_nan(self) -> bool {
        self.0 & MASK_EXPONENT == MASK_EXPONENT && self.0 & MASK_SIGNIFICAND != 0
    }

    #[inline(always)]
    pub const fn is_infinite(self) -> bool {
        self.0 == Self::INFINITY.0
    }

    #[inline(always)]
    pub const fn is_finite(self) -> bool {
        self.0 & MASK_EXPONENT != MASK_EXPONENT
    }

    // FIXME: investigate why this doesn't give exact results
    pub const fn sqrt(self) -> Self {
        let (e, s) = self.split();
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
            (((e / 2 + EXPONENT_UNSIGNED_ZERO as i16) as u64) << 53) | (two_xp1 & MASK_SIGNIFICAND),
        )
    }

    pub const fn recip(self) -> Self {
        let (e, s) = self.split_unsigned();

        let exponent = // short circuit to INF if equal to zero
            2046u16.wrapping_sub(e) | (((self.0 == 0) as u16) * EXPONENT_UNSIGNED_MAX);
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
        let (e, s) = self.split_unsigned();
        Self(((e as u64) << 53) | (s.wrapping_neg() & MASK_SIGNIFICAND))
    }
}

impl const Add for F64 {
    type Output = Self;

    // TODO: investigate if this works for non-normal numbers
    fn add(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.split_unsigned();
        let (e1, s1) = rhs.split_unsigned();

        let max_e = e0.max(e1);
        let s2 = (s0.wrapping_shl((max_e - e0) as u32))
            .wrapping_add(s1.wrapping_shl((max_e - e1) as u32));
        let l = s2.trailing_zeros() as u16;
        let e2 = max_e.saturating_sub(l)
            | ((e0 == EXPONENT_UNSIGNED_MAX || e1 == EXPONENT_UNSIGNED_MAX) as u16
                * EXPONENT_UNSIGNED_MAX);

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

    fn mul(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.split();
        let (e1, s1) = rhs.split();

        let mut e2 =
            ((e0 + e1 + EXPONENT_UNSIGNED_ZERO as i16) as u64).min(EXPONENT_UNSIGNED_MAX as u64);

        // handle infinity or nan
        e2 |= (e0 == EXPONENT_MAX || e1 == EXPONENT_MAX) as u64 * EXPONENT_UNSIGNED_MAX as u64;
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
        Self(((EXPONENT_UNSIGNED_ZERO - l) as u64) << 53 | (x as u64) >> l)
    }
}

impl Debug for F64 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("F64")
            .field(&self.exponent())
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
        F64((EXPONENT_UNSIGNED_ZERO as u64 - scale) << 53 | (significand & MASK_SIGNIFICAND))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-15;

    #[test]
    fn it_works() {
        let x = F64::from(25) - F64::from(39);
        println!("{}", x.abs());
        let y = F64::from(39) - F64::from(25);
        println!("{}", y.abs());
        let z = x + y;
        println!("{}", z.abs());
        assert!(z.abs() < EPSILON);
        let w = F64::from(2) * F64::from(4);
        assert_eq!(w.0, F64::from(8).0);

        let frac = F64::from(13) / F64::from(11) / F64::from(11);
        println!("{:?}", frac);
        let thirteen = frac * F64::from(121);
        println!("{:?}", thirteen);
        assert!((thirteen - F64::from(13)).abs() < EPSILON);

        println!("{:?}", (F64::from(2) * F64::INFINITY));
        assert!((F64::from(2) * F64::INFINITY).is_infinite());
        println!("{:?}", F64::INFINITY.abs());
        assert!(F64::INFINITY.abs().is_infinite());

        let one = F64::from(3) / F64::from(8) + F64::from(5) / F64::from(8);
        println!("{:?}", one);
        assert_eq!(one.0, F64::ONE.0);

        println!("{:?}", (F64::NAN * F64::INFINITY).abs());

        let n = F64::from(65);
        let sqrt = n.sqrt();
        println!("{:?}", sqrt);
        println!("{:?}", sqrt * sqrt);
        println!("{:?}", (sqrt * sqrt - n));
        assert!((sqrt * sqrt - n).abs() < EPSILON.sqrt());

        println!("{}", (F64::from(3) / F64::from(8)).fract());
    }
}

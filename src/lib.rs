#![feature(const_float_bits_conv)]
#![feature(const_trait_impl)]

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct F64(u64);

const MASK_SIGNIFICAND: u64 = 0x1fffffffffffff;
const MASK_VALUATION: u64 = !MASK_SIGNIFICAND;
const VALUATION_MAX: i16 = 1024;
const VALUATION_UNSIGNED_MAX: u16 = 2047;
const VALUATION_UNSIGNED_ZERO: u16 = 1023;

impl F64 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self((VALUATION_UNSIGNED_ZERO as u64) << 53 | 1);
    pub const INFINITY: Self = Self((VALUATION_UNSIGNED_MAX as u64) << 53);
    pub const NAN: Self = Self((VALUATION_UNSIGNED_MAX as u64) << 53 | 1);

    #[inline(always)]
    pub const fn valuation(self) -> i16 {
        self.valuation_unsigned() as i16 - VALUATION_UNSIGNED_ZERO as i16
    }

    const fn valuation_unsigned(self) -> u16 {
        (self.0 >> 53) as u16
    }

    #[inline(always)]
    pub const fn significand(self) -> u64 {
        self.0 & MASK_SIGNIFICAND
    }

    #[inline(always)]
    pub const fn split(self) -> (i16, u64) {
        (self.valuation(), self.significand())
    }

    const fn split_unsigned(self) -> (u16, u64) {
        (self.valuation_unsigned(), self.significand())
    }

    #[inline(always)]
    pub const fn exponent(self) -> i16 {
        -self.valuation()
    }

    #[inline(always)]
    pub const fn abs(self) -> f64 {
        f64::from_bits((self.0 & MASK_VALUATION) >> 1 | self.is_nan() as u64)
    }

    #[inline(always)]
    pub const fn is_nan(self) -> bool {
        self.0 & MASK_VALUATION == MASK_VALUATION && self.0 & MASK_SIGNIFICAND != 0
    }

    #[inline(always)]
    pub const fn is_infinite(self) -> bool {
        self.0 == Self::INFINITY.0
    }

    #[inline(always)]
    pub const fn is_finite(self) -> bool {
        self.0 & MASK_VALUATION != MASK_VALUATION
    }

    #[inline(always)]
    pub const fn exp4(self) -> Self {
        todo!()
    }
}

impl const Neg for F64 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        let (v, s) = self.split_unsigned();
        Self(((v as u64) << 53) | (s.wrapping_neg() & MASK_SIGNIFICAND))
    }
}

impl const Add for F64 {
    type Output = Self;

    // TODO: investigate if this works for non-normal numbers
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let (v0, s0) = self.split_unsigned();
        let (v1, s1) = rhs.split_unsigned();

        let min_v = v0.min(v1);
        let s2 = (s0 << ((v0 - min_v) as u64)) + (s1 << ((v1 - min_v) as u64));
        let l = s2.trailing_zeros() as u16;
        let v2 = min_v.saturating_sub(l)
            | ((v0 == VALUATION_UNSIGNED_MAX || v1 == VALUATION_UNSIGNED_MAX) as u16
                * VALUATION_UNSIGNED_MAX);

        Self((v2 as u64) << 53 | ((s2 >> l) & MASK_SIGNIFICAND))
    }
}

impl const Mul for F64 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let (v0, s0) = self.split();
        let (v1, s1) = rhs.split();

        let mut v2 =
            ((v0 + v1 + VALUATION_UNSIGNED_ZERO as i16) as u64).min(VALUATION_UNSIGNED_MAX as u64);

        // handle infinity or nan
        v2 |= (v0 == VALUATION_MAX || v1 == VALUATION_MAX) as u64 * VALUATION_UNSIGNED_MAX as u64;
        let mut s2 = s0.wrapping_mul(s1) & MASK_SIGNIFICAND;
        s2 |= (self.is_nan() || rhs.is_nan()) as u64;

        Self(v2 << 53 | s2)
    }
}

impl const Sub for F64 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl const Div for F64 {
    type Output = Self;

    // FIXME: investigate behavior for nan/inf/subnormals (most definitely does not work)
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let (v0, s0) = self.split();
        let (v1, s1) = rhs.split();

        let v2 = v0 + VALUATION_UNSIGNED_ZERO as i16 - v1;

        let (mut t0, mut t1) = (0u64, 1u64);
        let (mut r0, mut r1) = (1u64 << 54, s1);

        while r1 != 0 {
            let q = r0 / r1;
            (t0, t1) = (t1, t0.wrapping_sub(q.wrapping_mul(t1)));
            (r0, r1) = (r1, r0.wrapping_sub(q.wrapping_mul(r1)));
        }

        let inv_s1 = t0;

        Self((v2 as u64) << 53 | (inv_s1.wrapping_mul(s0) & MASK_SIGNIFICAND))
    }
}

impl const From<u32> for F64 {
    #[inline(always)]
    fn from(x: u32) -> Self {
        let l = x.trailing_zeros() as u16;
        Self(((VALUATION_UNSIGNED_ZERO - l) as u64) << 53 | (x as u64) >> l)
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
        println!("{:?}", frac.split());
        let thirteen = frac * F64::from(121);
        println!("{:?}", thirteen.split());
        assert!((thirteen - F64::from(13)).abs() < EPSILON);

        println!("{:?}", (F64::from(2) * F64::INFINITY).split());
        assert!((F64::from(2) * F64::INFINITY).is_infinite());
        println!("{:?}", F64::INFINITY.abs());
        assert!(F64::INFINITY.abs().is_infinite());

        let one = F64::from(3) / F64::from(8) + F64::from(5) / F64::from(8);
        println!("{:?}", one.split());
        assert_eq!(one.0, F64::ONE.0);

        println!("{:?}", (F64::NAN * F64::INFINITY).abs());
    }
}

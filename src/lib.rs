#![feature(const_float_bits_conv)]

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct F64(u64);

const MASK_SIGNIFICAND: u64 = 0x7ffffffffffff;
const MASK_EXPONENT: u64 = !MASK_SIGNIFICAND;
const VALUATION_MAX: i16 = 1024;
const VALUATION_MIN: i16 = -1023;
const VALUATION_UNSIGNED_MAX: u16 = 2047;
const VALUATION_UNSIGNED_ZERO: u16 = 1023;

impl F64 {
    #[inline(always)]
    pub const fn zero() -> Self {
        Self(0)
    }

    #[inline(always)]
    pub const fn one() -> Self {
        Self((VALUATION_UNSIGNED_ZERO as u64) << 53 | 1)
    }

    #[inline(always)]
    pub const fn infinity() -> Self {
        Self((VALUATION_UNSIGNED_MAX as u64) << 53 | 1)
    }

    #[inline(always)]
    pub const fn nan() -> Self {
        Self((VALUATION_UNSIGNED_MAX as u64) << 53)
    }

    #[inline(always)]
    pub const fn valuation(self) -> i16 {
        self.valuation_unsigned() as i16 - VALUATION_UNSIGNED_ZERO as i16
    }

    const fn valuation_unsigned(self) -> u16 {
        ((self.0 & MASK_EXPONENT) >> 53) as u16
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
    pub const fn abs(self) -> f64 {
        f64::from_bits((self.valuation_unsigned() as u64) << 52)
    }

    #[inline]
    pub const fn normalize(self) -> Self {
        let (e, s) = self.split_unsigned();

        if s == 0 {
            self
        } else if e == 0 {
            Self::infinity()
        } else {
            let l = s.trailing_zeros();
            Self(((e as u64 + l as u64) << 53) | (s >> l))
        }
    }

    #[inline]
    pub const fn is_nan(self) -> bool {
        self.normalize().0 == 0
    }
}

impl Neg for F64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let (e, s) = self.split_unsigned();
        Self(((e as u64) << 53) | (s.wrapping_neg() & MASK_SIGNIFICAND))
    }
}

impl Add for F64 {
    type Output = Self;

    // TODO: investigate if this works for non-normal numbers
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.split_unsigned();
        let (e1, s1) = rhs.split_unsigned();

        let (e2, s2) = (e0.min(e1), s0 + s1);
        let l = s2.trailing_zeros() as u16;

        Self((e2.saturating_sub(l) as u64) << 53 | ((s2 >> l) & MASK_SIGNIFICAND))
    }
}

impl Mul for F64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.split();
        let (e1, s1) = rhs.split();

        let mut e2 =
            ((e0 + e1 + VALUATION_UNSIGNED_ZERO as i16) as u64).min(VALUATION_UNSIGNED_MAX as u64);

        // handle infinity or nan
        // FIXME: broken after changing internal representation of exponent
        e2 &= !(e0 == VALUATION_MAX || e1 == VALUATION_MAX) as u64 * VALUATION_UNSIGNED_MAX as u64;

        Self(e2 << 53 | (s0.wrapping_mul(s1) & MASK_SIGNIFICAND))
    }
}

impl Sub for F64 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Div for F64 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.split();
        let (e1, s1) = rhs.split();

        let e2 = e0 + VALUATION_UNSIGNED_ZERO as i16 - e1;

        let (mut t0, mut t1) = (0u64, 1u64);
        let (mut r0, mut r1) = (1u64 << 54, s1);

        while r1 != 0 {
            let q = r0 / r1;
            (t0, t1) = (t1, t0.wrapping_sub(q.wrapping_mul(t1)));
            (r0, r1) = (r1, r0.wrapping_sub(q.wrapping_mul(r1)));
        }

        let inv_s1 = t0;

        Self((e2 as u64) << 53 | (inv_s1.wrapping_mul(s0) & MASK_SIGNIFICAND))
    }
}

impl From<u32> for F64 {
    #[inline(always)]
    fn from(x: u32) -> Self {
        let l = x.trailing_zeros() as u16;
        Self(((VALUATION_UNSIGNED_ZERO - l) as u64) << 53 | (x as u64) >> l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let x = F64::from(25) - F64::from(39);
        println!("{}", x.abs());
        let y = F64::from(39) - F64::from(25);
        println!("{}", y.abs());
        let z = x + y;
        println!("{}", z.abs());
        let w = F64::from(2) * F64::from(4);
        println!("{}", w.abs());

        let frac = F64::from(13) / F64::from(11) / F64::from(11);
        println!("{:?}", frac.split());
        println!("{:?}", (frac * F64::from(121)).split());

        println!("{:?}", (F64::from(2) * F64::infinity()).split());
        println!("{:?}", F64::infinity().abs());
    }
}

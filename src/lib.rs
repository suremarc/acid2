#![feature(const_float_bits_conv)]

use std::ops::{Add, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct F64(u64);

const MASK_SIGNIFICAND: u64 = 0x7ffffffffffff;
const MASK_EXPONENT: u64 = !MASK_SIGNIFICAND;
const EXPONENT_RAW_MAX: u16 = 2047;
const EXPONENT_ZERO: u16 = 1023;

impl F64 {
    #[inline(always)]
    pub const fn zero() -> Self {
        Self((EXPONENT_RAW_MAX as u64) << 53)
    }

    #[inline(always)]
    pub const fn infinity() -> Self {
        Self(1)
    }

    #[inline(always)]
    pub const fn nan() -> Self {
        Self(0)
    }

    #[inline(always)]
    pub const fn exponent(self) -> i16 {
        self.exponent_raw() as i16 - EXPONENT_ZERO as i16
    }

    const fn exponent_raw(self) -> u16 {
        ((self.0 & MASK_EXPONENT) >> 53) as u16
    }

    #[inline(always)]
    pub const fn significand(self) -> u64 {
        self.0 & MASK_SIGNIFICAND
    }

    #[inline(always)]
    pub const fn split(self) -> (i16, u64) {
        (self.exponent(), self.significand())
    }

    const fn split_raw(self) -> (u16, u64) {
        (self.exponent_raw(), self.significand())
    }

    #[inline(always)]
    pub const fn abs(self) -> f64 {
        f64::from_bits(2046u64.saturating_sub(self.exponent_raw() as u64) << 52)
    }

    #[inline]
    pub const fn normalize(self) -> Self {
        let (e, s) = self.split_raw();

        if s == 0 {
            self
        } else if e == 0 {
            Self::infinity()
        } else {
            let l = s.trailing_zeros();
            Self(((e as u64 + l as u64) << 53) | (s >> l))
        }
    }
}

impl Neg for F64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let (e, s) = self.split_raw();
        Self(((e as u64) << 53) | (s.wrapping_neg() & MASK_SIGNIFICAND))
    }
}

impl Add for F64 {
    type Output = Self;

    // TODO: investigate if this works for non-normal numbers
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.split_raw();
        let (e1, s1) = rhs.split_raw();

        let (e2, s2) = (e0.max(e1), s0 + s1);
        let l = s2.trailing_zeros() as u16;

        Self((((e2 + l).min(EXPONENT_RAW_MAX) as u64) << 53) | ((s2 >> l) & MASK_SIGNIFICAND))
    }
}

impl Mul for F64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.split();
        let (e1, s1) = rhs.split();

        Self(
            ((e0 + e1 + EXPONENT_ZERO as i16) as u64) << 53
                | (s0.wrapping_mul(s1) & MASK_SIGNIFICAND),
        )
    }
}

impl Sub for F64 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl From<u32> for F64 {
    fn from(x: u32) -> Self {
        let l = x.trailing_zeros() as u16;
        // println!("{l}");
        let mut exponent = (l + EXPONENT_ZERO) as u64;
        exponent |= (x == 0) as u64 * EXPONENT_RAW_MAX as u64;
        Self(exponent << 53 | (x as u64) >> l)
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

        println!("{:?}", F64((EXPONENT_RAW_MAX as u64 - 1) << 53).abs());
    }
}

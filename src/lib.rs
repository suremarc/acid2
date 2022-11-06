#![feature(const_float_bits_conv)]
#![feature(const_trait_impl)]
#![feature(bigint_helper_methods)]
#![feature(const_bigint_helper_methods)]

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy)]
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

    // using assignment requires const support for mutable references
    #[allow(clippy::assign_op_pattern)]
    pub const fn powi(self, pow: u32) -> Self {
        let mut res = Self::ONE;
        let mut i = 0;
        while i < pow {
            i += 1;
            res = res * self;
        }

        res
    }

    #[inline(always)]
    pub const fn recip(self) -> Self {
        let (e, s) = self.split_unsigned();

        let mut x = s.wrapping_mul(2u64.wrapping_sub(s.wrapping_mul(s)));
        x = x.wrapping_mul(2u64.wrapping_sub(s.wrapping_mul(x)));
        x = x.wrapping_mul(2u64.wrapping_sub(s.wrapping_mul(x)));
        x = x.wrapping_mul(2u64.wrapping_sub(s.wrapping_mul(x)));
        x = x.wrapping_mul(2u64.wrapping_sub(s.wrapping_mul(x)));

        Self(2046u64.wrapping_sub(e as u64) << 53 | x & MASK_SIGNIFICAND)
    }
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
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.split_unsigned();
        let (e1, s1) = rhs.split_unsigned();

        let max_e = e0.max(e1);
        let mut s2 = (s0.wrapping_shl((max_e - e0) as u32))
            .wrapping_add(s1.wrapping_shl((max_e - e1) as u32));
        let l = s2.trailing_zeros() as u16;
        s2 >>= l;
        let e2 = max_e.saturating_sub(l)
            | ((e0 == EXPONENT_UNSIGNED_MAX || e1 == EXPONENT_UNSIGNED_MAX) as u16
                * EXPONENT_UNSIGNED_MAX);

        Self((e2 as u64) << 53 | (s2 & MASK_SIGNIFICAND))
    }
}

impl AddAssign for F64 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl const Mul for F64 {
    type Output = Self;

    #[inline(always)]
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

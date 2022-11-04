use std::ops::{Add, Neg, Sub};

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct Float64(u64);

const MASK_SIGNIFICAND: u64 = 0xffffffffffff;
const MASK_EXPONENT: u64 = !MASK_SIGNIFICAND;
const EXPONENT_RAW_MAX: u16 = 4094;
const EXPONENT_ZERO: u16 = 2047;

impl Float64 {
    #[inline(always)]
    pub const fn zero() -> Self {
        Self(0)
    }

    #[inline(always)]
    pub const fn infinity() -> Self {
        Self((EXPONENT_RAW_MAX as u64) << 48)
    }

    #[inline(always)]
    pub const fn nan() -> Self {
        Self((EXPONENT_RAW_MAX as u64) << 48 | 1)
    }

    #[inline(always)]
    pub const fn exponent(self) -> i16 {
        self.exponent_raw() as i16 - 2047
    }

    #[inline(always)]
    const fn exponent_raw(self) -> u16 {
        ((self.0 & MASK_EXPONENT) >> 48) as u16
    }

    #[inline(always)]
    pub const fn significand(self) -> u64 {
        self.0 & MASK_SIGNIFICAND
    }

    #[inline(always)]
    pub const fn split(self) -> (i16, u64) {
        (self.exponent(), self.significand())
    }

    #[inline(always)]
    pub const fn split_raw(self) -> (u16, u64) {
        (self.exponent_raw(), self.significand())
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
            Self(((e as u64 + l as u64) << 48) | (s >> l))
        }
    }
}

impl Neg for Float64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let (e, s) = self.split_raw();
        Self(((e as u64) << 48) | (s.wrapping_neg() & MASK_SIGNIFICAND))
    }
}

impl Add for Float64 {
    type Output = Self;

    // TODO: investigate if this works for non-normal numbers
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.split_raw();
        let (e1, s1) = rhs.split_raw();

        let (e2, s2) = (e0.max(e1), s0 + s1);
        let l = s2.trailing_zeros() as u16;

        Self(((e2 as u64 + l as u64) << 48) | ((s2 & MASK_SIGNIFICAND) >> l))
    }
}

impl Sub for Float64 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl From<u32> for Float64 {
    fn from(x: u32) -> Self {
        Self(((EXPONENT_ZERO as u64) << 48) | x as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut result = Float64::from(25) - Float64::from(39);
        println!("{:?}", result.split());
        result = result + (Float64::from(39) - Float64::from(25));
        println!("{:?}", result.split());
    }
}

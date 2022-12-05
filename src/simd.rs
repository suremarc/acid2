use crate::core::F64;
use core::{
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    simd::{
        LaneCount, Mask, Simd, SimdFloat, SimdOrd, SimdPartialEq, SimdUint, SupportedLaneCount,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct SimdF64<const LANES: usize>(Simd<u64, LANES>)
where
    LaneCount<LANES>: SupportedLaneCount;

impl<const LANES: usize> SimdF64<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    pub fn from_array(arr: [F64; LANES]) -> Self {
        Self(Simd::from_array(unsafe { core::mem::transmute_copy(&arr) }))
    }

    #[inline]
    fn neg_exponent(self) -> Simd<i16, LANES> {
        (self.neg_exponent_unsigned() - Simd::splat(F64::NEG_EXPONENT_UNSIGNED_ZERO)).cast()
    }

    #[inline]
    fn neg_exponent_unsigned(self) -> Simd<u16, LANES> {
        (self.0 >> Simd::splat(53)).cast()
    }

    #[inline]
    pub fn exponent(self) -> Simd<i16, LANES> {
        -self.neg_exponent()
    }

    #[inline]
    pub fn significand(self) -> Simd<u64, LANES> {
        self.0 & Simd::splat(F64::MASK_SIGNIFICAND)
    }

    #[inline]
    fn neg_exponent_and_significand(self) -> (Simd<i16, LANES>, Simd<u64, LANES>) {
        (self.neg_exponent(), self.significand())
    }

    #[inline]
    fn neg_exponent_unsigned_and_significand(self) -> (Simd<u16, LANES>, Simd<u64, LANES>) {
        (self.neg_exponent_unsigned(), self.significand())
    }

    #[inline]
    pub fn abs(self) -> Simd<f64, LANES> {
        Simd::<f64, LANES>::from_bits(
            (self.0 & Simd::splat(F64::MASK_EXPONENT)) >> Simd::splat(1)
                | self.is_nan().to_int().cast(),
        )
    }

    #[inline]
    pub fn is_nan(self) -> Mask<i64, LANES> {
        (self.0 & Simd::splat(F64::MASK_EXPONENT)).simd_eq(Simd::splat(F64::MASK_EXPONENT))
            & (self.0 & Simd::splat(F64::MASK_SIGNIFICAND)).simd_ne(Simd::splat(0))
    }

    #[inline]
    pub fn is_infinite(self) -> Mask<i64, LANES> {
        self.0.simd_eq(Simd::splat(F64::INFINITY.to_bits()))
    }

    #[inline]
    pub fn is_finite(self) -> Mask<i64, LANES> {
        (self.0 & Simd::splat(F64::MASK_EXPONENT)).simd_ne(Simd::splat(F64::MASK_EXPONENT))
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        let (e, s) = self.neg_exponent_and_significand();
        assert!((e & Simd::splat(0b1)).simd_eq(Simd::splat(0)).all());
        assert!((s & Simd::splat(0b111)).simd_eq(Simd::splat(1)).all());

        let two_b = s >> Simd::splat(2);
        let mut x = two_b;
        let mut two_xp1 = (x << Simd::splat(1u64)) + Simd::splat(1);
        for i in 2..6 {
            // x' = x - (x^2 + x - 2b) / (2x + 1)
            x -= (x * x + x - two_b) * invert(two_xp1, i);
            two_xp1 = x << Simd::splat(1) | Simd::splat(1);
        }

        Self(
            (e / Simd::splat(2) + Simd::splat(F64::NEG_EXPONENT_UNSIGNED_ZERO).cast()).cast()
                << Simd::splat(53u64)
                | (two_xp1 & Simd::splat(F64::MASK_SIGNIFICAND)),
        )
    }

    #[inline]
    pub fn recip(self) -> Self {
        let (e, s) = self.neg_exponent_unsigned_and_significand();

        let exponent = (Simd::splat(2046u16) - e).cast::<u64>() << Simd::splat(53)
            | (self
                .0
                .simd_eq(Simd::splat(0))
                .select(Simd::splat(F64::MASK_EXPONENT), Simd::splat(0)));

        Self(exponent | invert(s, 5) & Simd::splat(F64::MASK_SIGNIFICAND))
    }
}

// find the multiplicative inverse modulo 2^(2^(N+1)), where N is num_iterations
// formula sourced from "Modern Computer Arithmetic" version 0.5.9, pg. 66
fn invert<const LANES: usize>(x: Simd<u64, LANES>, num_iterations: u8) -> Simd<u64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut inverse = x;
    for _ in 0..num_iterations {
        inverse *= Simd::splat(2u64) - x * inverse;
    }

    inverse
}

impl<const LANES: usize> Neg for SimdF64<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        let (e, s) = self.neg_exponent_unsigned_and_significand();
        Self(
            e.cast::<u64>() << Simd::splat(53)
                | ((Simd::splat(0) - s) & Simd::splat(F64::MASK_SIGNIFICAND)),
        )
    }
}

impl<const LANES: usize> Add for SimdF64<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.neg_exponent_unsigned_and_significand();
        let (e1, s1) = rhs.neg_exponent_unsigned_and_significand();

        let max_e = e0.simd_max(e1);
        let s2 = (s0 << ((max_e - e0).cast())) + (s1 << ((max_e - e1).cast()));

        // this part is slow
        let l = Simd::from_array(s2.to_array().map(u64::trailing_zeros)).cast();

        let e2 = max_e.saturating_sub(l)
            | (e0.simd_eq(Simd::splat(F64::NEG_EXPONENT_UNSIGNED_MAX))
                | e1.simd_eq(Simd::splat(F64::NEG_EXPONENT_UNSIGNED_MAX)))
            .select(Simd::splat(F64::NEG_EXPONENT_UNSIGNED_MAX), Simd::splat(0));

        Self(
            e2.cast::<u64>() << Simd::splat(53)
                | ((s2 >> l.cast()) & Simd::splat(F64::MASK_SIGNIFICAND)),
        )
    }
}

impl<const LANES: usize> AddAssign for SimdF64<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const LANES: usize> Mul for SimdF64<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.neg_exponent_and_significand();
        let (e1, s1) = rhs.neg_exponent_and_significand();

        let mut e2 = core::simd::SimdOrd::simd_min(
            (e0 + e1 + Simd::splat(F64::NEG_EXPONENT_UNSIGNED_ZERO as i16)).cast(),
            Simd::splat(F64::NEG_EXPONENT_UNSIGNED_MAX as u64),
        );
        // handle infinity or nan
        e2 |= (e0.simd_eq(Simd::splat(F64::NEG_EXPONENT_MAX))
            | e1.simd_eq(Simd::splat(F64::NEG_EXPONENT_MAX)))
        .cast()
        .select(
            Simd::splat(F64::NEG_EXPONENT_UNSIGNED_MAX as u64),
            Simd::splat(0),
        );
        let mut s2 = (s0 * s1) & Simd::splat(F64::MASK_SIGNIFICAND);
        s2 = (self.is_nan() | rhs.is_nan()).select(s2, Simd::splat(0));

        Self(e2 << Simd::splat(53) | s2)
    }
}

impl<const LANES: usize> MulAssign for SimdF64<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const LANES: usize> Sub for SimdF64<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl<const LANES: usize> SubAssign for SimdF64<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const LANES: usize> Div for SimdF64<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)] // lol
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.recip()
    }
}

impl<const LANES: usize> DivAssign for SimdF64<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs
    }
}

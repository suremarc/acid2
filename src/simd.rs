use crate::core::F64;
use core::{
    ops::{Add, Mul},
    simd::{LaneCount, Mask, Simd, SimdFloat, SimdPartialEq, SimdUint, SupportedLaneCount},
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

        let max_e = e0.max(e1);
        let s2 = (s0 << ((max_e - e0).cast())) + (s1 << ((max_e - e1).cast()));

        // this part is slow
        let l = Simd::from_array(s2.to_array().map(u64::trailing_zeros)).cast();

        let e2 = max_e.saturating_sub(l);
        // TODO: handle NaN/inf
        // | ((e0 == F64::NEG_EXPONENT_UNSIGNED_MAX || e1 == F64::NEG_EXPONENT_UNSIGNED_MAX)
        //     as u16
        //     * F64::NEG_EXPONENT_UNSIGNED_MAX);

        Self(
            e2.cast::<u64>() << Simd::splat(53)
                | ((s2 >> l.cast()) & Simd::splat(F64::MASK_SIGNIFICAND)),
        )
    }
}

impl<const LANES: usize> Mul for SimdF64<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (e0, s0) = self.neg_exponent_and_significand();
        let (e1, s1) = rhs.neg_exponent_and_significand();

        let mut e2 = ((e0 + e1 + Simd::splat(F64::NEG_EXPONENT_UNSIGNED_ZERO as i16)).cast())
            .min(Simd::splat(F64::NEG_EXPONENT_UNSIGNED_MAX as u64));
        // TODO: handle infinity or nan
        // e2 |= (e0 == Self::NEG_EXPONENT_MAX || e1 == Self::NEG_EXPONENT_MAX) as u64
        //     * Self::NEG_EXPONENT_UNSIGNED_MAX as u64;
        let mut s2 = (s0 * s1) & Simd::splat(F64::MASK_SIGNIFICAND);
        // s2 |= self.is_nan() as u64 | rhs.is_nan() as u64;

        Self(e2 << Simd::splat(53) | s2)
    }
}

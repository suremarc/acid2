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
    pub fn splat(x: F64) -> Self {
        Self(Simd::splat(x.to_bits()))
    }

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

    /// The exponent of this 2-adic number.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::Simd;
    ///
    /// let f = SimdF64::<8>::splat(F64::from(8));
    ///
    /// assert_eq!(f.exponent(), Simd::splat(3));
    /// ```
    #[inline]
    pub fn exponent(self) -> Simd<i16, LANES> {
        -self.neg_exponent()
    }

    /// The part of this number coprime to 2.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::Simd;
    ///
    /// let f = SimdF64::<8>::splat(F64::from(8));
    ///
    /// let f = SimdF64::<8>::splat(F64::from(36));
    ///
    /// assert_eq!(f.significand(), Simd::splat(9));
    /// ```
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

    /// The exponent and significand of this number, packed together in a tuple.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::Simd;
    ///
    /// let f = SimdF64::<8>::splat(F64::from(12));
    ///
    /// assert_eq!(f.exponent_and_significand(), (Simd::splat(2), Simd::splat(3)));
    /// ```
    #[inline]
    pub fn exponent_and_significand(self) -> (Simd<i16, LANES>, Simd<u64, LANES>) {
        (self.exponent(), self.significand())
    }

    /// The 2-adic absolute value of this number.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::{Simd, SimdFloat};
    ///
    /// let f = SimdF64::<8>::splat(F64::from(72));
    ///
    /// assert_eq!(f.abs(), Simd::splat(2f64.powi(-3)));
    /// assert_eq!(SimdF64::<8>::splat(F64::ZERO).abs(), Simd::splat(0.0));
    /// assert_eq!(SimdF64::<8>::splat(F64::INFINITY).abs(), Simd::splat(f64::INFINITY));
    /// assert!(SimdF64::<8>::splat(F64::NAN).abs().is_nan().all());
    /// ```
    #[inline]
    pub fn abs(self) -> Simd<f64, LANES> {
        Simd::<f64, LANES>::from_bits(
            (self.0 & Simd::splat(F64::MASK_EXPONENT)) >> Simd::splat(1)
                | self.is_nan().to_int().cast(),
        )
    }

    /// Whether or not this number is NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::Simd;
    ///
    /// let x = SimdF64::<8>::splat(F64::ONE);
    /// let y = SimdF64::<8>::splat(F64::NAN);
    /// let z = SimdF64::<8>::splat(F64::INFINITY);
    ///
    /// assert!(!x.is_nan().any());
    /// assert!(y.is_nan().all());
    /// assert!(!z.is_nan().any());
    /// ```
    #[inline]
    pub fn is_nan(self) -> Mask<i64, LANES> {
        (self.0 & Simd::splat(F64::MASK_EXPONENT)).simd_eq(Simd::splat(F64::MASK_EXPONENT))
            & (self.0 & Simd::splat(F64::MASK_SIGNIFICAND)).simd_ne(Simd::splat(0))
    }

    /// Whether or not this number is infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::Simd;
    ///
    /// let x = SimdF64::<8>::splat(F64::ONE);
    /// let y = SimdF64::<8>::splat(F64::NAN);
    /// let z = SimdF64::<8>::splat(F64::INFINITY);
    ///
    /// assert!(!x.is_infinite().any());
    /// assert!(!y.is_infinite().any());
    /// assert!(z.is_infinite().all());
    /// ```
    #[inline]
    pub fn is_infinite(self) -> Mask<i64, LANES> {
        self.0.simd_eq(Simd::splat(F64::INFINITY.to_bits()))
    }

    /// Whether or not this number is finite.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::Simd;
    ///
    /// let x = SimdF64::<8>::splat(F64::ONE);
    /// let y = SimdF64::<8>::splat(F64::NAN);
    /// let z = SimdF64::<8>::splat(F64::INFINITY);
    ///
    /// assert!(x.is_finite().all());
    /// assert!(!y.is_finite().any());
    /// assert!(!z.is_finite().any());
    /// ```
    #[inline]
    pub fn is_finite(self) -> Mask<i64, LANES> {
        (self.0 & Simd::splat(F64::MASK_EXPONENT)).simd_ne(Simd::splat(F64::MASK_EXPONENT))
    }

    /// Computes the 2-adic square root of this number.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::{Simd, SimdPartialOrd};
    ///
    /// let f = SimdF64::<8>::splat(F64::from(81));
    /// let sqrt = f.sqrt();
    ///
    /// assert!((sqrt - SimdF64::<8>::splat(F64::from(9))).abs().simd_le(Simd::splat(1e-15)).all()); // this won't always be true for perfect squares, but in this case it is
    /// assert!((sqrt * sqrt - f).abs().simd_le(Simd::splat(1e-7)).all());
    /// ```
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

    /// Computes the 2-adic reciprocal of this number.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::{Simd, SimdPartialOrd};
    ///
    /// let f = SimdF64::<8>::splat(F64::from(13));
    /// let recip = f.recip();
    ///
    /// assert!((f * recip - SimdF64::<8>::splat(F64::ONE)).abs().simd_le(Simd::splat(1e-15)).all());
    /// ```
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

    /// Computes the 2-adic sum of two numbers, truncating on the left side where necessary.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::{Simd, SimdPartialOrd};
    ///
    /// let x = SimdF64::<8>::splat(F64::from(13));
    /// let y = SimdF64::<8>::splat(F64::from(11));
    ///
    /// assert!((x + y - SimdF64::<8>::splat(F64::from(24))).abs().simd_le(Simd::splat(1e-15)).all());
    /// ```
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

    /// Computes the 2-adic product of two numbers, truncating on the left side where necessary.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::{Simd, SimdPartialOrd};
    ///
    /// let x = SimdF64::<8>::splat(F64::from(14));
    /// let y = SimdF64::<8>::splat(F64::from(6));
    ///
    /// assert!((x * y - SimdF64::<8>::splat(F64::from(84))).abs().simd_le(Simd::splat(1e-15)).all());
    /// ```
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
        s2 |= (self.is_nan() | rhs.is_nan()).select(Simd::splat(1), Simd::splat(0));

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

    /// Computes the 2-adic difference of two numbers, truncating on the left side where necessary.
    ///
    /// `x - y` is exactly equivalent to `x + (-y)`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::{Simd, SimdPartialOrd};
    ///
    /// let x = SimdF64::<8>::splat(F64::from(13));
    /// let y = SimdF64::<8>::splat(F64::from(11));
    ///
    /// assert!((x - y - SimdF64::<8>::splat(F64::from(2))).abs().simd_le(Simd::splat(1e-15)).all());
    /// ```
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

    /// Computes the 2-adic difference of two numbers, truncating on the left side where necessary.
    ///
    /// `x / y` is exactly equivalent to `x * y.recip()`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use acid2::{core::F64, simd::SimdF64};
    /// use core::simd::{Simd, SimdPartialOrd};
    ///
    /// let x = SimdF64::<8>::splat(F64::from(18));
    /// let y = SimdF64::<8>::splat(F64::from(6));
    /// assert!((x / y - SimdF64::<8>::splat(F64::from(3))).abs().simd_le(Simd::splat(1e-15)).all());
    /// ```
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

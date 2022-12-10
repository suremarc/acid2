#![feature(const_float_bits_conv)]
#![feature(const_trait_impl)]
#![feature(doc_cfg)]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(not(test), no_std)]

mod macros;

/// Basic floating-point types.
pub mod core;

#[cfg(feature = "simd")]
#[doc(cfg(feature = "simd"))]
/// SIMD-accelerated types for additional performance.
pub mod simd;

#![feature(const_float_bits_conv)]
#![feature(const_trait_impl)]
#![feature(doc_cfg)]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(not(test), no_std)]

mod macros;

pub mod core;

#[cfg(any(feature = "simd", doc))]
#[doc(cfg(feature = "simd"))]
pub mod simd;

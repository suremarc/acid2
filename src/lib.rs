#![feature(const_float_bits_conv)]
#![feature(const_trait_impl)]
#![feature(doc_cfg)]
#![cfg_attr(not(test), no_std)]

mod macros;

pub mod core;
pub mod simd;

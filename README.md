# acid2

[![Build Status][actions-badge]][actions-url]
![License][license-badge]
[![Crates.io][crates-badge]][crates-url]
[![API][docs-badge]][docs-url]

[actions-badge]: https://github.com/suremarc/acid2/workflows/build/badge.svg?event=push
[actions-url]: https://github.com/suremarc/acid2/actions?query=workflow%3Abuild+branch%3Amaster
[docs-badge]: https://docs.rs/acid2/badge.svg
[docs-url]: https://docs.rs/acid2
[license-badge]: https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg
[crates-badge]: https://img.shields.io/crates/v/acid2.svg
[crates-url]: https://crates.io/crates/acid2

2-adic floating-point implementation, for maximum hardware affinity and performance.

This project is in its _very_ early stages. No guarantees can be made about its correctness until the testing infrastructure is more robust.

## Goals

The goal of this project is to implement 2-adic floating point arithmetic with maximum performance. Currently, the only known use case is for approximating p-adic integrals, where exactness is not a hard requirement.

Explicitly branchless code is preferred where possible, at the expense of some readability. However, since 2-adic addition requires counting trailing zeros, this may not be possible for some architectures. On some x86 platforms, LLVM's `cttz` intrinsic is implemented using a bitscan (`bsfl`) and a conditional jump.

## Feature wishlist

### More tests

Having numerous, table-driven unit tests with cases covering over/underflow and inf/NaN would be ideal. Manually constructed, hand-worked test cases are probably still useful, even for something as complex as floating-point arithmetic.

Fuzz-testing might also be useful for detecting broken invariants. Currently, the only invariant is that the significand is odd. This invariant may disappear if we change the internal representation to have an implicit lowest bit of 1.

### numpy support

Python support would make experimentation with this library extremely quick. More investigation is needed as to what would be involved here (read: I have no idea how to do this).

### String conversions

Being able to convert to/from string representations of p-adic numbers would be very helpful in debugging -- not just for the author, but presumably for consumers of this library as well. Currently there is a placeholder implementation of `fmt::Debug` that just shows the exponent and significand as a tuple.

### Subnormals

This is less of a wishlist item, and more a note that that not much thought has been given to subnormal numbers and how they're handled, as of now.

## Comparing p-adic vs. real floating-point numbers

### p-adic floating-point numbers don't have a sign bit

The additive inverse of any integer in the p-adic numbers is the limit of a convergent sequence of positive integers under the p-adic metric. Hence, there is no need for a sign bit.

### There are no exact additive inverses (except for 0)

As a corollary of the above, since negatives of integers have infinitely many digits, they cannot be represented exactly. Consider the number -1. Its 2-adic representation is the following:

```none
...1111111111.0
```

Because there are infinitely many digits, the floating-point representation is truncated to 53 bits. A corollary of this is that self-subtraction does not result in zero: given a p-adic floating-point value `x`, we have that `(x - x).abs() != 0`.

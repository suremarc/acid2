# acid2

2-adic floating-point implementation, for maximum hardware affinity and performance.

This project is in its _very_ early stages. No guarantees can be made about its correctness until the testing infrastructure is more robust.

## Goals

The goal of this project is to implement 2-adic floating point arithmetic with maximum performance. Currently, the only known use case is for approximating p-adic integrals, where exactness is not a hard requirement.

Explicitly branchless code is preferred where possible, at the expense of some readability. However, since 2-adic addition requires counting trailing zeros, this may not be possible for some architectures. On some x86 platforms, LLVM's `cttz` intrinsic is implemented using a bitscan (`bsfl`) and a conditional jump.

## Feature wishlist

### More tests

Having numerous, table-driven unit tests with cases covering over/underflow and inf/NaN would be ideal. Manually constructed, hand-worked test cases are probably still useful, even for something as complex as floating-point arithmetic.

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

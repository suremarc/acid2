# acid2

2-adic floating-point implementation, for maximum hardware affinity and performance.

## Floating-point arithmetic: comparing p-adic vs. real numbers

### p-adic floating-point numbers don't have a sign bit

The additive inverse of any integer in the p-adic numbers is the limit of a convergent sequence of positive integers under the p-adic metric. Hence, there is no need for a sign bit.

### There are no exact additive inverses (except for 0)

As a corollary of the above, since negatives of integers have infinitely many digits, they cannot be represented exactly. Consider the number -1. Its 2-adic representation is the following:

```none
...1111111111.0
```

Because there are infinitely many digits, the floating-point representation is truncated to 53 bits. A corollary of this is that self-subtraction does not result in zero: given a p-adic floating-point value `x`, we have that `(x - x).abs() != 0`.

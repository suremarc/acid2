# acid2
2-adic floating-point implementation, for maximum hardware affinity and performance. 

## Floating-point arithmetic: comparing p-adic vs. real numbers

### There are no exact additive inverses (except for 0)

Consider the number -1. Its 2-adic representation is the following:

```
...1111111111.0
```

Because there are infinitely many digits, the floating-point representation is truncated to 53 bits. A corollary of this is that self-subtraction does not result in zero: given a p-adic floating-point value `x`, we have that `(x - x).abs() != 0`.

# Precision Guarantees

`rustebra` implements `sqrt`, `sin`, and `cos` itself rather than depending on `std`'s
floating-point intrinsics, since the crate is `no_std` by default. Each uses a
**fixed-iteration** approximation rather than a convergence-checked loop:

- `sqrt` uses Newton-Raphson (Babylonian) iteration, `x_{n+1} = (x_n + value / x_n) / 2`,
  run for a fixed 50 iterations.
- `sin`/`cos` use a Taylor series expansion around zero (after range-reducing the input to
  `[-pi, pi]`), run for a fixed 20 iterations.

In both cases the iteration count is fixed instead of stopping once successive iterates
stop changing, so the amount of work done is predictable and independent of the input —
important in `no_std` contexts, where there's no easy way to bound worst-case iteration
count for a convergence-checked loop otherwise. The trade-off is that precision isn't
uniform across the input domain: iteration counts were chosen generously enough to converge
for the normal range of inputs these functions expect (vector norms and angles typical of
linear algebra work), but may lose precision at the extreme ends of a type's exponent range,
where more iterations would be needed to leave the initial guess's slow-convergence region.

```rust
use rustebra::scalar::Scalar;

let result = Scalar::sqrt(2.0_f64);
assert!((result - core::f64::consts::SQRT_2).abs() < 1e-9);
```

Comparing two floating-point results for approximate equality — accounting for the
rounding error these iterative approximations (and floating-point arithmetic generally)
introduce — is what `FloatTolerance::epsilon()` is for; see
[Scalars & Numeric Types](../02-scalars.md).

## Gotchas

- These functions don't document a specific worst-case error bound (e.g. "accurate to N
  ULPs") — the guarantee is the *mechanism* (a fixed, generously-sized iteration budget
  tuned for typical inputs), not a numeric precision figure. Don't assume bit-for-bit
  parity with `std`'s `f64::sqrt`/`sin`/`cos` for values at the edges of the valid range.

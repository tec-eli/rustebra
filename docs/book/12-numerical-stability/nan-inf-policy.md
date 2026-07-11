# NaN/Inf Policy

Outside of `rustebra::krylov`, the crate does not check for `NaN` or infinite values — it
lets IEEE-754 arithmetic propagate them the way plain `f32`/`f64` operations naturally do.
An operation that receives or produces a non-finite value doesn't return an error; the
non-finite value just flows through.

```rust
let x = 0.0_f64 / 0.0;
assert!(x.is_nan());
// rustebra's arithmetic (add/sub/mul/div, dot products, norms, ...) doesn't intercept
// this — a NaN input produces a NaN output, the same as raw f64 arithmetic would.
```

Krylov methods (`power_iteration`, `inverse_power_iteration`) are the deliberate
exception: they loop, so a poisoned iterate doesn't just produce one bad result — it burns
the entire remaining iteration budget computing on garbage. On embedded targets that budget
is bounded and can't be spent elsewhere, so these functions detect a non-finite iterate and
stop early with `ConvergenceError::NonFinite`, rather than paying for the full iteration
count on a chain of `NaN` arithmetic.

## Gotchas

- Don't rely on a `NaN` result from most `rustebra` operations to signal an error condition
  the way `Result` does elsewhere in the crate — only the Krylov methods actively check for
  and report non-finite values. Everywhere else, a `NaN`/`Inf` input or intermediate result
  is expected to propagate silently, same as raw floating-point arithmetic.
- `Scalar::sqrt` of a negative number is a deliberate exception to *this* exception: it
  returns `0`, not `NaN` — see [Scalars & Numeric Types](../02-scalars.md).

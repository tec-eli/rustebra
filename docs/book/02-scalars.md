# Scalars & Numeric Types

Every algorithm in `rustebra` is generic over a `Scalar` type rather than hard-coded to
`f32` or `f64`. `Scalar` (defined in `rustebra::scalar`) is the minimal arithmetic surface
the rest of the crate depends on: the additive and multiplicative identities (`zero`,
`one`), the four basic operations (`add`, `sub`, `mul`, `div`), and the transcendental
functions needed by the algorithm layer (`sqrt`, `sin`, `cos`). `f32` and `f64` are the
two implementors provided today.

```rust
use rustebra::scalar::Scalar;

fn double<T: Scalar>(x: T) -> T {
    x.add(x)
}
```

Note the call style for the transcendental methods: `Scalar::sqrt(4.0f64)` rather than
`4.0f64.sqrt()`. `f64`/`f32` already have an inherent `sqrt` from `std`, which would shadow
the trait method if called with dot syntax, so code that needs to stay generic over `T:
Scalar` (or that wants the trait's specific implementation) calls it as an associated
function instead.

## How `sqrt`, `sin`, and `cos` are computed

`Scalar` is implemented for `f32`/`f64` without relying on `std`'s floating-point
intrinsics for these three methods, since the crate is `no_std` by default:

- `sqrt` uses a fixed-iteration Newton-Raphson (Babylonian) iteration:
  `x_{n+1} = (x_n + self / x_n) / 2`. `self == 0` short-circuits to `0` rather than
  iterating (the formula would otherwise divide by zero on the first step), and negative
  inputs return `0` rather than `NaN` or a panic, since `Scalar` is an infallible trait
  that must also support future non-float implementors with no `NaN` representation.
- `sin` and `cos` use a fixed-iteration Taylor series expansion around zero, after first
  reducing the input to `[-pi, pi]` by subtracting the nearest multiple of `2*pi`.

In all three cases the iteration count is fixed rather than convergence-checked, so the
amount of work done is predictable and independent of the input — important in `no_std`
contexts where there's no easy way to bound worst-case iteration count otherwise.

## `FloatTolerance`

Comparing floating-point results for approximate equality requires a tolerance, and
`rustebra` doesn't hard-code one into `Scalar` itself. Instead, `FloatTolerance` is a
separate trait (also implemented for `f32`/`f64`) that adds a single method, `epsilon()`,
reporting the type's machine epsilon — the smallest positive value `e` such that
`T::one().add(e) != T::one()`. It's kept separate from `Scalar` so that a hypothetical
future `Scalar` implementor with no meaningful notion of machine epsilon (e.g. a
fixed-point or exact-rational type) isn't forced to implement it just to satisfy `Scalar`.

```rust
use rustebra::scalar::FloatTolerance;

assert_eq!(f64::epsilon(), f64::EPSILON);
```

## Gotchas

- Call the transcendental methods as `Scalar::sqrt(x)` / `Scalar::sin(x)` / `Scalar::cos(x)`
  when `x`'s type isn't pinned to a concrete float — dot-call syntax (`x.sqrt()`) resolves
  to `std`'s inherent method on concrete `f32`/`f64` values, not the trait method, and won't
  compile at all in a generic `T: Scalar` context.
- `Scalar::sqrt` of a negative number returns `0`, not `NaN` — don't rely on `NaN`
  propagation to detect a negative input; check the sign yourself if that matters to your
  code.

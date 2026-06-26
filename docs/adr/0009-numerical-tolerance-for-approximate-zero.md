
# ADR 0009: Numerical Tolerance for Approximate-Zero Comparisons

## Status
Accepted

## Context
Several algorithms in `algorithm::matrix` need to decide whether a computed floating-point
value counts as zero: `rank` (counting rows that survive elimination), `svd`/
`svd_qr_iteration` and `condition_number`/`condition_number_svd` (deciding whether a singular
value is negligible, and therefore whether a matrix is singular), and `cholesky`/
`cholesky_decompose` (deciding whether a value that should be non-negative before a square
root is negative only due to rounding noise, or genuinely violates positive-definiteness).

Floating-point arithmetic accumulates rounding error proportional to machine epsilon and the
magnitude of the values involved. Comparing a computed value to exact `0.0` — which is what
all the functions above did, alongside `lu_partial_pivot` and `qr_householder`/
`qr_gram_schmidt` — only behaves correctly when the input happens to reduce to an exactly
representable zero. Toy/integer-valued matrices reliably do; matrices with irrational
entries, measured/noisy data, or results of prior floating-point computation essentially
never do. Exact comparison in that case isn't a rare edge case — it's a near-total blind
spot, silently producing a different (wrong) answer rather than erroring or returning a
visibly approximate result.

Not every exact-zero comparison in this set represents the same kind of decision, though.
Two categories exist:

1. **"Is there truly nothing left to act on"** — an algebraic fact about the matrix, correctly
   decided by comparing to exact `0.0`, *provided* the algorithm is already choosing the best
   available candidate (i.e. true partial pivoting, not "first nonzero"). `lu_partial_pivot`'s
   pivot search and `qr_householder`/`qr_gram_schmidt`'s reflector/normalization
   zero-norm checks fall here: a tiny-but-nonzero pivot or column norm is still mathematically
   valid to use, and the resulting error amplification is the honest consequence of an
   ill-conditioned input, not a defect a threshold should mask. LAPACK's own dense LU/QR
   kernels (`dgetrf`/`dgeqrf`) take no tolerance parameter for exactly this reason — they
   compute the decomposition the input actually has.
2. **"Is this small enough not to count"** — an inherently approximate judgment with no exact
   answer once real (non-toy) floating-point data is involved: numerical rank, whether a
   singular value is negligible, and whether a matrix that should be positive-definite is
   only failing that test due to rounding noise. Every serious numerical library (LAPACK,
   NumPy, Eigen) thresholds these; there is no alternative that avoids picking a tolerance.

Per [ADR 0008](0008-two-level-public-api.md), this library serves two kinds of users: general
users who want a correct result without thinking about algorithm- or precision-level detail,
and mathematical users who want explicit control. A caller-supplied `tolerance` for
category-2 functions serves the second group well, but leaves the first group with no way to
call `rank`/`svd`/`condition_number`/`cholesky` at all without first knowing what a
reasonable tolerance for their data looks like — a question that requires exactly the
numerical judgment those callers don't have. Computing a sensible default needs some notion
of "how much rounding error can this `Scalar` type's arithmetic actually produce" (machine
epsilon), which `Scalar` itself doesn't expose, deliberately: ADR 0005 keeps it to the
minimal arithmetic surface the algorithm layer needs, and a hypothetical future `Scalar`
implementor (e.g. a fixed-point or exact-rational type) shouldn't be forced to answer a
question that may not apply to it just to satisfy a convenience feature it may never use.

## Decision
Functions answering a category-2 question gain an explicit, caller-supplied `tolerance: T`
parameter: `rank`, `svd`/`svd_qr_iteration`, `condition_number`/`condition_number_svd`, and
`cholesky`/`cholesky_decompose`. The caller chooses the value (per ADR 0008, this preserves
"mathematical user" control rather than the library guessing a magic constant on their
behalf):

* Where a meaningful scale is already computed as a side effect of the algorithm,
  `tolerance` is treated as *relative* to it: `svd_qr_iteration`/`condition_number_svd`
  compare a singular value against `tolerance * sigma_max`, matching the convention LAPACK
  and NumPy use for numerical rank and singular-value negligibility.
* Where no such scale exists without adding work that wouldn't actually make the comparison
  more correct (`rank`'s elimination, which only ever has whatever sub-block remains at each
  step to look at), `tolerance` is compared directly as an absolute threshold, documented at
  the function as something the caller should choose relative to their own data's scale.
* `cholesky_decompose`'s positive-definiteness check tolerates a value in `[-tolerance, 0)`
  under the square root as positive-*semi*-definite (rounding noise) rather than erroring,
  only treating values below `-tolerance` as genuinely not positive-definite.

Functions answering a category-1 question do **not** gain a tolerance parameter. Their actual
defect — selecting the first nonzero pivot/column rather than the largest-magnitude one — is
fixed on its own terms instead: `lu`/`lu_partial_pivot` move to true partial pivoting
(largest-magnitude candidate), after which the existing exact-`0.0` check correctly identifies
only genuinely, entirely-zero columns. `qr_householder`/`qr_gram_schmidt`'s zero-norm checks
are left as exact comparisons for the same reason.

A caller who needs to know whether an `lu`/`qr` decomposition was computed from a dangerously
ill-conditioned input should call `condition_number` rather than expect `lu`/`qr` to make that
judgment internally — this keeps the decomposition functions' output mathematically faithful
to their input, rather than silently approximating it differently depending on a threshold
the caller may not even know was applied.

### Auto-computed tolerance for general users

Each category-2 operation gets two entry points, rather than one function with an
`Option<T>` tolerance: a single function with one signature would need one trait bound
covering every call, which would force the auto-computing capability's trait requirement (see
below) onto callers who already supply their own tolerance and never needed it — exactly the
coupling this section avoids by splitting the two paths into separate functions instead.

* `rank`, `cholesky`, `svd`, and `condition_number` keep their existing names as the
  general-user entry point: no `tolerance` parameter, a sensible default is computed
  internally.
* The previously-existing explicit names (`cholesky_decompose`, `svd_qr_iteration`,
  `condition_number_svd`) become the mathematical-user entry point, gaining the
  caller-supplied `tolerance` parameter described above. `rank` has no second name to
  reuse this way (it has only one algorithm), so a new explicit function,
  `rank_with_tolerance`, is added alongside it.

Computing a default tolerance needs machine epsilon for the concrete `Scalar` type in use.
Rather than adding this to `Scalar` itself (reopening the question above), a separate,
narrower trait holds it:

```rust
trait FloatTolerance: Scalar {
    /// The smallest value `e` such that `Self::one().add(e) != Self::one()` —
    /// i.e. machine epsilon for this type.
    fn epsilon() -> Self;
}
```

implemented only by `f32` and `f64` (returning `f32::EPSILON`/`f64::EPSILON`). The
auto-computing entry points (`rank`, `cholesky`, `svd`, `condition_number`) are bound by
`T: Scalar + FloatTolerance`; the explicit ones (`rank_with_tolerance`,
`cholesky_decompose`, `svd_qr_iteration`, `condition_number_svd`) stay bound by `T: Scalar`
alone, exactly as before this section — a future non-float `Scalar` implementor that has no
meaningful epsilon simply never implements `FloatTolerance`, and loses nothing except the
auto-computing convenience: it can still call every explicit function by supplying its own
tolerance.

The default itself follows the same relative-where-it's-free, absolute-elsewhere split as the
caller-supplied case:

* `svd`/`condition_number` default to `n * QR_ITERATIONS * epsilon()`, compared as
  `tolerance * sigma_max` inside `svd_qr_iteration`/`condition_number_svd` exactly as a
  caller-supplied value would be (`n` is `max(rows, cols)`). Both compute singular values by
  eigendecomposing `aᵗ * a` over a fixed number of QR sweeps (`QR_ITERATIONS`), each
  contributing its own rounding error; the achievable precision floor empirically sits well
  above a plain `n * epsilon()` once that accumulation is accounted for, so the default needs
  the extra factor to ever actually classify a singular value as negligible on this
  algorithm's own output. Without it, a true singular value far below `n * epsilon()` (e.g.
  `1e-20` against a largest singular value of `1`) would never read back as negligible,
  because the noise floor from 100 QR sweeps sits around `1e-15`, well above the unmultiplied
  default.
* `rank` defaults to `n * epsilon() * scale`, where `scale` is the largest-magnitude entry in
  `a` (one cheap additional pass, the only practical proxy available without a column-by-
  column matrix norm).
* `cholesky` defaults to `n * epsilon() * scale`, where `scale` is the largest-magnitude
  diagonal entry in `a` (the quantities the positive-definiteness check itself compares
  against).

`n` (a `usize`) is converted to `T` by repeated addition from `T::one()` — `Scalar` already
has everything this needs (`zero`, `one`, `add`); no numeric-conversion method is added to
either trait for it.

This ADR is additive to [ADR 0004](0004-per-module-error-handling.md): it does not change how
failures are reported (still `Result`, still per-module error types), only what counts as a
failure-worthy condition in the functions listed above. ADR 0004 is unaffected and remains
`Accepted`.

## Consequences

### Positive
* Decompositions stay mathematically faithful: `lu`/`qr` always compute the decomposition the
  input actually has, never a thresholded approximation of it.
* Functions whose entire purpose is a threshold judgment (`rank`, `condition_number`, `svd`'s
  singular-value negligibility, Cholesky's positive-definiteness check) stop being silently
  wrong on any input that doesn't reduce to an exact floating-point zero — true of
  essentially all non-toy inputs before this ADR.
* Matches established practice (LAPACK, NumPy, Eigen) for where tolerance does and doesn't
  belong, rather than inventing a project-specific convention from scratch.
* General users (ADR 0008's first category) get a working `rank`/`svd`/`condition_number`/
  `cholesky` with no tolerance to reason about, without `Scalar` itself growing a method only
  some implementors could give a meaningful answer to.
* `StaticMatrix`/`DynamicMatrix`'s existing `rank()` and `determinant()` methods need no
  signature change at all: `rank()` keeps calling the now-auto-computing `rank` free
  function, and `determinant` is untouched by this ADR entirely (it only ever reaches
  `lu_partial_pivot`, which doesn't take a tolerance — see above).

### Negative / Trade-offs
* Breaking change to the *explicit* function signatures only: `rank_with_tolerance` (new),
  `svd_qr_iteration`, `condition_number_svd`, and `cholesky_decompose` gain a required
  `tolerance` parameter. The general-user-facing names (`rank`, `svd`, `condition_number`,
  `cholesky`) keep their existing signatures. Acceptable pre-1.0 per `TODO.md`'s stated
  versioning policy regardless.
* Two different tolerance semantics exist side by side (relative-to-`sigma_max` for
  `svd`/`condition_number`, absolute for `rank`), which must be documented clearly at each
  function so callers don't assume one implies the other.
* A second trait (`FloatTolerance`) enters the crate's vocabulary alongside `Scalar`, and the
  rule for which functions need it (auto-computing ones only) has to be learned once rather
  than being self-evident from the function name alone.
* `rank`/`rank_with_tolerance` is a naming asymmetry against `cholesky`/`svd`/
  `condition_number`'s existing high-level/explicit pairs (which name an *algorithm*, not a
  *tolerance-handling difference*) — accepted because `rank` has only one algorithm to name a
  pair after.
* Callers who want defense-in-depth diagnostics on `lu`/`qr` (e.g. "warn me if a pivot was
  small") get no built-in mechanism from this ADR; left as a possible future addition (e.g.
  returning the smallest pivot/norm magnitude encountered) rather than solved here.

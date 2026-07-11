# Numerical Stability and Precision

This document records the *measured* precision and stability characteristics of rustebra,
as a companion to the policy decision in [ADR 0013](archived/adr/0013-numerical-stability-and-error-bounds-policy.md).
That ADR states the target bounds and the NaN/Inf policy; this document records what was
actually verified, how, and where the measured behavior falls short of the stated target.

## Elementary functions (`Scalar::sqrt`/`sin`/`cos`)

### Target

| Type | Target relative error | Domain |
|------|-----------------------|------------|
| f64  | < 1e-14               | `[-2π, 2π]` |
| f32  | < 1e-6                | `[-2π, 2π]` |

Outside `[-2π, 2π]`, behavior is degraded and untested, not bounded. There is no reduced-
precision fast path for smaller devices.

### Verification method

`sqrt` is a fixed-iteration Newton-Raphson (Babylonian) iteration; `sin`/`cos` are fixed-
iteration Taylor series expansions around zero with **no range reduction**
(`src/scalar/newton_raphson.rs`, `src/scalar/taylor.rs`, `src/scalar/trigonometry.rs`).

Accuracy is checked in `tests/integration/scalar/precision.rs` against two pre-generated,
static reference fixtures (`tests/integration/scalar/fixtures/elementary_reference_{f64,f32}.csv`),
1000 evenly spaced samples per function:

- `sqrt`: `[0, 2π]` — negative inputs are contract-defined to return `0` (see
  `Scalar::sqrt`'s doc comment), not a precision question, so they're out of scope for an
  accuracy fixture.
- `sin`/`cos`: `[-2π, 2π]`.

The reference values were generated **offline**, by a standalone Rust program using the host
platform's `f64` libm (`f64::sqrt`/`sin`/`cos`, correctly rounded to within about 1 ulp,
~2e-16 relative error) — not the crate's own fixed-iteration implementations, and not part of
the test binary. That reference is ~100x tighter than the f64 target and ~1e10x tighter than
the f32 target, so it is precise enough to certify both bounds without requiring an
arbitrary-precision tool. The f32 fixture's inputs are pre-rounded to `f32` before the
reference is computed, so each expected value matches the *exact* f32 input the f32 test
feeds in.

**Deviation from the original verification plan:** the task that produced this document
called for generating the reference via Python's `mpmath` (or a similar arbitrary-precision
tool). Python was not available in the environment this was verified in, so the reference
was instead generated from the host's double-precision libm as described above. This is
still a valid reference for certifying both target bounds (its own error is over two orders
of magnitude below the f64 target, and about ten orders of magnitude below the f32 target),
but it is not the arbitrary-precision fixture originally specified — worth regenerating with
`mpmath` if bit-for-bit independence from any platform's libm is later required.

Near zero crossings (`sin(0)`, `sin(±π)`, `cos(±π/2)`, ...), relative error is undefined or
dominated by rounding noise in the reference itself; the test falls back to absolute error
(against the same numeric tolerance) whenever `|expected| < 1e-9`.

### Result: the f64 and f32 targets are not currently met over the full stated domain

Running the fixture test today:

- **f64: 75 of 3000 samples (2.5%) exceed the 1e-14 target.** All failures are `sin`/`cos`
  samples in the outer portion of the domain (roughly `|x| > 6.1`, i.e. within about 0.18
  radians of the `±2π` boundary). Worst observed error is ~6.2e-13, about 60x over target.
  Every `sqrt` sample and every `sin`/`cos` sample closer to zero passes.
- **f32: 420 of 3000 samples (14%) exceed the 1e-6 target.** Failures are concentrated in
  `cos` for `x` roughly above `4.6` in magnitude, growing in both frequency and size toward
  the domain boundary. Worst observed error is ~2.1e-5, about 21x over target.

This is a real precision gap, not a fixture or test-harness artifact — the target in ADR 0013
is **not adjusted** to match it. The cause is exactly what the existing source comments on
`Scalar::sin`/`Scalar::cos` already flag: the Taylor expansion runs a fixed 20 iterations
around zero with no range reduction (no reduction modulo `2π` before evaluating the series),
so truncation error grows with `|x|` and is worst at the domain's own boundary. `sqrt`'s
Newton-Raphson iteration (50 fixed iterations) meets its target everywhere in `[0, 2π]` with
comfortable margin in this sampling.

Practical implication: callers needing the full `1e-14`/`1e-6` guarantee near `±2π` should
range-reduce their own inputs closer to zero before calling `sin`/`cos` (e.g. into
`[-π, π]`), where measured error is comfortably within target. Closing this gap in the
library itself would mean adding range reduction to `sin`/`cos`, which is out of scope for
this verification task.

## Decomposition precision limits

No dedicated accuracy fixture exists yet for `algorithm::matrix` decompositions (LU, QR,
Cholesky, SVD). The only precision-relevant behavior currently encoded in the library is the
tolerance system from [ADR 0009](archived/adr/0009-numerical-tolerance-for-approximate-zero.md):

- `svd`/`condition_number` (auto-tolerance entry points) default their negligibility
  threshold to `n * QR_ITERATIONS * epsilon()`, where `n = max(rows, cols)` and
  `QR_ITERATIONS = 100` (`src/algorithm/matrix/svd.rs`). This factor exists because the
  fixed-iteration QR eigendecomposition these functions use accumulates its own rounding
  error across 100 sweeps, empirically settling at a noise floor around `1e-15` — well above
  a plain `n * epsilon()` — so a true singular value below that floor still needs the scaled
  threshold to read back as negligible.
- `rank`/`cholesky` default to `n * epsilon() * scale`, where `scale` is the largest-magnitude
  relevant entry (the full matrix for `rank`, the diagonal for `cholesky`).
- `lu_partial_pivot`, `qr_householder`, and `qr_gram_schmidt` take no tolerance at all and
  compute the decomposition the input mathematically has, including on ill-conditioned
  input — per ADR 0009, faithfulness to the input takes priority over silently
  approximating it. A caller wanting to know whether a decomposition was computed from a
  dangerously ill-conditioned matrix should call `condition_number` separately.

No measured error bound (e.g. "LU residual stays under X for condition numbers up to Y") has
been established for these decompositions; that would require a dedicated stress fixture
analogous to the elementary-function one above, which is out of scope here.

## Krylov iterative methods

### Convergence assumptions

- **Power iteration** (`src/krylov/power_iteration.rs`): eigenvector-direction error shrinks
  by a factor of roughly `|λ2 / λ1|` per iteration (ratio of second-largest to largest
  eigenvalue magnitude). Converges quickly when the dominant eigenvalue is well separated;
  slowly as `|λ2| → |λ1|`; not at all when `|λ2| == |λ1|` with `λ2 != λ1` (e.g. `λ2 == -λ1`,
  or a complex-conjugate dominant pair), in which case the call returns
  `ConvergenceError::MaxIterationsExceeded` rather than looping forever undetected. A
  *repeated* dominant eigenvalue (`λ2 == λ1`) converges normally onto one vector in the
  shared eigenspace.
- **Inverse power iteration** (`src/krylov/inverse_power_iteration.rs`): same convergence
  shape, applied to the shifted-and-inverted spectrum, so its rate is governed by how close
  the shift sits to the target eigenvalue relative to the next-nearest one.
- Both stop once **two** convergence measures fall within the caller's `tol` (relative to the
  current eigenvalue estimate): eigenvalue stabilization (`|λ_k - λ_{k-1}| <= tol * |λ_k|`)
  and eigenvector-residual stabilization (`‖a·v_k - λ_k·v_k‖ <= tol * |λ_k|`). The first check
  needs a prior estimate, so `max_iter < 2` can never converge.

### NaN/Inf policy

Per ADR 0013:

- **Single arithmetic operations and elementary functions propagate non-finite values**, per
  IEEE 754 semantics, undocumented per call. This matches the underlying hardware/software
  behavior and adds no branching cost to the common path.
- **Krylov iterative methods are the documented exception**: `power_iteration` and
  `inverse_power_iteration` explicitly detect a non-finite iterate mid-loop and return
  `Err(ConvergenceError::NonFinite)` instead of continuing to spend the iteration budget on
  poisoned state. This isn't a contradiction of the propagate-by-default rule — a single
  operation costs one evaluation regardless of outcome, while an iterative solver can burn
  the entire `max_iter` budget converging toward nothing once a `NaN`/`Inf` enters the state,
  a cost that matters on embedded targets with a genuinely bounded compute budget.
- `ConvergenceError::NonFinite` is distinct from `ConvergenceError::ZeroVector` (an iterate
  that is finite but has no direction to normalize) and from
  `ConvergenceError::SingularShift` (the shifted matrix in inverse iteration is singular, or
  within `singular_tol` of it — reported as a hard error rather than iterating on amplified
  noise).

### Tolerance defaults and condition-number thresholds

Krylov methods take `tol` (convergence) and, for inverse iteration, `singular_tol`
(singularity of the shifted matrix) as required caller-supplied parameters — there is no
auto-computed default at this layer, unlike the `algorithm::matrix` category-2 functions in
ADR 0009. Callers choose these relative to their own problem scale; the property/stress test
suite (`tests/property/krylov/`, `tests/numerical_stress.rs`) exercises `tol` values around
`1e-10` (f64) and `sqrt(f32::EPSILON)` (f32), with assertion tolerances two orders of
magnitude looser to account for the gap between the requested convergence tolerance and
actually achieved accuracy.

Condition number itself (`algorithm::matrix::condition_number`) is not consumed internally by
the Krylov solvers as a preconditioning signal; it exists as a caller-facing diagnostic (see
`docs/algorithms/condition-number.md`) for deciding, before or after a solve, how much
precision loss to expect.

## Chained-operation error behavior

No dedicated fixture exists yet measuring error accumulation across chained operations (e.g.
several matmuls followed by a decomposition). The `tests/numerical_stress.rs` suite covers
one related case: Krylov methods retain correct dominant/target eigenpairs on matrices with a
condition number as large as `1e8` (`fixed_similarity_3`, see that file), provided the
targeted eigenvalue is itself well separated in the relevant sense (dominant-ratio for power
iteration, shift-distance for inverse iteration) — conditioning of the matrix as a whole does
not by itself disturb a well-separated target, but Krylov's convergence-rate assumptions
above still apply and govern how many iterations that accuracy costs.

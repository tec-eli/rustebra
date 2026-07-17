# GMRES(m)

`gmres` solves the general (possibly non-symmetric) linear system `A x = b` via restarted
GMRES, GMRES(`M`): unlike [Conjugate Gradient](conjugate-gradient.md), `A` need not be
symmetric positive-definite. Each restart cycle runs [Arnoldi Iteration](arnoldi.md) from the
current residual to build an `M`-dimensional Krylov basis, solves the resulting small
least-squares problem via Givens rotations, and updates `x` — restarting from the improved
iterate until either the residual meets `tol` or `max_restarts` cycles are exhausted.

`A` is supplied as a sparse linear operator rather than a dense matrix: applying it never
allocates, so restart cycles reuse the same workspace (`out_x`, `basis`, `scratch`) throughout.

```rust
{{#include ../../../examples/krylov/gmres.rs}}
```

## Algorithm

Each restart cycle:

1. Computes the residual `r = b - A x` and its norm `β = ‖r‖`, returning `Ok` immediately if
   `β <= tol`.
2. Runs Arnoldi iteration from `q_0 = r / β`, building an orthonormal basis `Q` of up to `M`
   vectors and the upper Hessenberg projection `H`, stopping early (before `M` steps) on
   breakdown — an invariant subspace found before the basis filled up, the same
   "success, not failure" case documented on [Arnoldi Iteration](arnoldi.md).
3. Solves `min_y ‖β e_1 - H y‖` via incremental Givens rotations, then updates `x <- x + Q y`.

## Convergence

GMRES's residual norm decreases monotonically within a cycle (each additional basis vector can
only improve the least-squares fit) and never increases across a restart, because restarting
recomputes the same residual the next cycle continues from. It is not guaranteed to decrease
*strictly* every cycle, though: a starting vector aligned with an invariant subspace the
operator doesn't expand (breakdown on the very first Arnoldi step) leaves `x` unchanged, and the
iteration stagnates. Restarting also discards the larger Krylov subspace full (non-restarted)
GMRES would have kept building, so GMRES(`M`) can converge slower, or stagnate on problems full
GMRES would resolve — the restart budget `max_restarts` bounds the cost of that risk rather than
eliminating it.

## Gotchas

- The restart size `M` is a `const` generic on the caller's `Basis` buffer, not a runtime
  parameter — see
  [Krylov Basis-Size Const-Generic Convention](../../specs/krylov-basis-size-const-generics.md).
  `M > n` (the operator's dimension) is a `DimensionMismatch`.
- `M == 0` never reduces a nonzero residual: no basis vector can be built, so every restart
  cycle is a no-op check against `tol`, and a nonzero residual exhausts `max_restarts` without
  ever moving `x`.
- `tol` has no auto-computed default — see
  [Krylov Tolerance and Convergence Criteria](../../specs/krylov-tolerance-and-convergence.md).
- Non-finite residuals or Arnoldi iterates (`NaN` or infinite, including values that overflow
  `f64` mid-computation) return `ConvergenceError::NonFinite` rather than silently producing a
  wrong answer.
- A zero pivot in the small least-squares system built from `H` — a coincidental exact
  singularity in the projected system that Arnoldi's own breakdown test didn't already catch —
  returns `ConvergenceError::Breakdown`.

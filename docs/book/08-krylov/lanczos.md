# Lanczos Iteration

`lanczos` tridiagonalizes a symmetric `n x n` matrix `a` over a `K`-dimensional Krylov
subspace: starting from a normalized `v0`, it builds an orthonormal basis `Q` of
`span{v0, a*v0, ..., a^{K-1}*v0}` and returns the projection `T = Qᵗ * a * Q`, a symmetric
`K x K` tridiagonal matrix. With `K == n` and no breakdown, `T` is orthogonally similar to
`a` and therefore has exactly its spectrum; with `K < n`, `T`'s eigenvalues (the Ritz values)
approximate the extreme eigenvalues of `a`, generally converging well before `K` reaches `n`.
Unlike `power_iteration` and `inverse_power_iteration`, which each refine a single eigenpair,
Lanczos builds a whole subspace at once — the basis for later extracting several eigenvalues,
or for feeding other Krylov solvers (CG, MINRES) that need the same tridiagonal projection.

Each step computes `w = a * q_j`, records the diagonal entry `α_j = q_jᵗ * w`, subtracts off
the components along `q_j` and the previous basis vector `q_{j-1}` (the three-term
recurrence), and normalizes what remains into `q_{j+1}`, recording its length as the
off-diagonal entry `β_j`. In floating point, rounding error erodes the basis's orthogonality
as the Ritz values converge, so every step also re-orthogonalizes `w` against *every* basis
vector built so far (full reorthogonalization) rather than relying on the three-term
recurrence alone.

```rust
{{#include ../../../examples/krylov/lanczos.rs}}
```

## Breakdown

When the candidate for the next basis vector has (numerically) zero norm relative to `‖a *
q_j‖`, the Krylov subspace is invariant: there's no new direction to extend the basis with,
and the call fails with `ConvergenceError::Breakdown` rather than dividing by a vanishing
norm. This is not a numerical failure to work around — it's a structural property of the
pairing of `a` and `v0`. A repeated eigenvalue can contribute at most one basis vector to the
Krylov subspace no matter how large `K` is (the identity matrix breaks down immediately for
any `v0`, since `a * v0` never points anywhere new), and a `v0` that happens to lie in a
proper invariant subspace of `a` breaks down as soon as that subspace is exhausted, even with
a fully distinct spectrum. The remedy is different from a plain convergence failure: retry
with a smaller `K`, or a different `v0`.

## Gotchas

- `a` is *assumed* symmetric, never verified. For a non-symmetric input the projection isn't
  actually tridiagonal, and `T` silently misrepresents it — Lanczos has no equivalent of the
  dimension or non-finite checks for this assumption, since checking symmetry itself would
  cost as much as the decomposition it's protecting.
- `tol` has no auto-computed default, the same as `power_iteration` and
  `inverse_power_iteration` — see
  [Krylov Tolerance and Convergence Criteria](../../specs/krylov-tolerance-and-convergence.md).
  A `tol` of `0` detects only exact breakdown.
- The basis size `K` is a `const` generic on the caller's `Basis` buffer, not a runtime
  parameter — see
  [Krylov Basis-Size Const-Generic Convention](../../specs/krylov-basis-size-const-generics.md).
  `K > n` is a `DimensionMismatch`: an `n`-dimensional space has no `K` orthonormal directions
  to find.
- Both `ConvergenceError::ZeroVector` and `ConvergenceError::NonFinite` on `v0` are checked
  up front, even when `K == 0` means no basis vector is ever written — a `K == 0` call is not
  a shortcut around input validation, only around the iteration itself.

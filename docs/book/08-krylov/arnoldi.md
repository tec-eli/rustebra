# Arnoldi Iteration

`arnoldi` reduces a general, row-major `n x n` matrix `a` to upper Hessenberg form over a
`K`-dimensional Krylov subspace: starting from a normalized `v0`, it builds an orthonormal
basis `Q` of `span{v0, a*v0, ..., a^{K-1}*v0}` and returns the projection `H = Qᵗ * a * Q`,
which is upper Hessenberg. Unlike [Lanczos Iteration](lanczos.md), `a` need not be symmetric —
there is no three-term recurrence to exploit, so every step orthogonalizes the candidate
vector against the *entire* basis built so far, by modified Gram-Schmidt, rather than just the
two previous vectors.

```rust
{{#include ../../../examples/krylov/arnoldi.rs}}
```

## Orthogonalization

Modified Gram-Schmidt (subtracting each projection immediately, rather than computing all
projections against the original candidate and subtracting them at the end) is the standard
trade for Arnoldi: markedly more stable than classical Gram-Schmidt at the same `O(K * n)`
per-step cost, though still less stable than Householder Arnoldi, which trades that extra
stability for losing the explicit basis vectors the Krylov projection needs. No
reorthogonalization pass is added on top, unlike Lanczos's full reorthogonalization — Arnoldi's
every-vector orthogonalization does not erode as quickly as Lanczos's three-term recurrence
does, so a second pass doesn't earn its keep the same way.

## Breakdown is success, not failure

When the candidate vector's norm falls to (numerically) zero relative to `‖a * q_j‖` after
orthogonalization, `q_0, ..., q_j` already span an invariant subspace of `a`: there's no new
direction to extend the basis with, but the vectors and the leading block of `H` already built
are exact and useful. This is reported as `Ok((h, reached))` with `reached < K`, not an error —
unlike Lanczos, which reports the analogous condition as `ConvergenceError::Breakdown`. The
difference is what the caller does next: a Lanczos caller that requested `K` vectors and got
fewer has nothing it can use without changing its request, while GMRES, built on top of
Arnoldi, can solve directly in the smaller subspace `Ok` reports — the exact subspace is often
exactly where the true solution already lives. Callers that do need the full `K` vectors
distinguish this case from a complete run by checking `reached < K`.

## Gotchas

- `K` is a `const` generic on the caller's `Basis` buffer, not a runtime parameter — see
  [Krylov Basis-Size Const-Generic Convention](../../specs/krylov-basis-size-const-generics.md).
  `K > n` is a `DimensionMismatch`.
- Both `ConvergenceError::ZeroVector` and `ConvergenceError::NonFinite` on `v0` are checked up
  front, even when `K == 0` means no basis vector is ever written.
- `tol` has no auto-computed default — see
  [Krylov Tolerance and Convergence Criteria](../../specs/krylov-tolerance-and-convergence.md).
  A `tol` of `0` detects only exact breakdown.

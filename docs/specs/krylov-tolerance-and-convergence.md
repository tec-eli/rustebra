# Krylov Tolerance and Convergence Criteria

## Summary

Krylov iterative methods take `tol` (convergence) and, for inverse iteration, `singular_tol`
(shifted-matrix singularity) as required caller-supplied parameters, with no auto-computed
default. Convergence is only declared once two independent measures — eigenvalue
stabilization and eigenvector-residual stabilization — both fall within `tol`.

## Scope

Applies to `power_iteration`, `inverse_power_iteration`, and `lanczos`, and to any future
Krylov solver (CG, Arnoldi, GMRES(m)) that shares the same iterative-refinement shape. Does not
apply to the `algorithm::matrix` tolerance-taking functions (rank, SVD, condition-number
estimation, Cholesky decomposition), which are covered by [[approximate-zero-tolerance]] and
get an auto-computed default under [[auto-tolerance-defaults]] instead.

## Decision

`tol`, and `singular_tol` for inverse iteration, are required parameters with no
general-user, default-computing entry point, unlike the `algorithm::matrix` category-2
functions. Those functions can derive a default from a scale the algorithm already computes
(the largest singular value, the largest-magnitude diagonal entry) or from machine epsilon
and problem size alone. A Krylov solver has no equivalent problem-independent quantity before
it starts: its natural scale is the eigenvalue estimate being refined, which doesn't exist
until at least one iteration has run. Callers choose `tol` relative to their own problem scale
and the accuracy their use case needs.

Convergence requires two measures to independently fall within `tol` (relative to the current
eigenvalue estimate) before an iterate is accepted:

- eigenvalue stabilization: `|λ_k - λ_{k-1}| <= tol * |λ_k|`
- eigenvector-residual stabilization: `‖A·v_k - λ_k·v_k‖ <= tol * |λ_k|`

Neither measure is sufficient alone. An eigenvalue estimate can plateau prematurely while the
eigenvector direction is still rotating toward its limit, and a small residual computed from
an estimate that hasn't yet stabilized can understate how much the eigenvalue itself is still
moving. Requiring both closes each measure's blind spot with the other. Because the
eigenvalue-stabilization check needs a prior estimate to compare against, at least one full
iteration must run before convergence can be declared: `max_iter < 2` can never converge.

`inverse_power_iteration` additionally takes `singular_tol`, governing how close the shifted
matrix `a - shift * I` may be to singular before the solver reports `SingularShift` instead of
continuing to iterate on amplified noise. See [[nan-inf-policy]] for how `SingularShift`
relates to the `NonFinite` and `ZeroVector` failure modes of the same solvers.

`condition_number` is not consumed internally by the Krylov solvers as a preconditioning
signal; it exists purely as a caller-facing diagnostic for deciding, before or after a solve,
how much precision loss to expect.

## Constraints

- `tol` and `singular_tol` never get an auto-computed default; every Krylov entry point
  requires the caller to supply them directly.
- Convergence must never be declared on a single measure — both eigenvalue stabilization and
  residual stabilization must hold before an iterate is accepted.
- `max_iter < 2` must never report convergence, since the eigenvalue-stabilization check
  requires a prior estimate that doesn't exist before the first iteration completes.
- `condition_number` remains a caller-facing diagnostic only; no Krylov solver may consume it
  internally as a preconditioning signal.

## Status

Implemented. `power_iteration` requires `tol`; `inverse_power_iteration` requires both `tol`
and `singular_tol`. Both check eigenvalue and residual stabilization before declaring
convergence, and neither exposes an auto-computed default. `lanczos` also requires `tol`,
with no default, but as a basis-breakdown threshold rather than an eigenvalue/residual
convergence check: it has no eigenvalue estimate to stabilize against, only a candidate basis
vector's norm to compare against the local matrix-vector scale.

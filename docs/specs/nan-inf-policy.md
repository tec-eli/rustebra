# NaN/Inf Policy

## Summary

The library propagates non-finite values through single arithmetic operations and
elementary functions by default, following IEEE 754 semantics. Iterative Krylov solvers
are the one documented exception: they detect a non-finite iterate mid-loop and return
an error instead of continuing to spend iterations on a poisoned state.

## Scope

Applies library-wide to scalar arithmetic and elementary functions (`sqrt`, `sin`, `cos`,
and similar) on every scalar type. Separately, applies to the Krylov iterative solvers —
power iteration, inverse power iteration, and future additions such as CG, Lanczos,
Arnoldi, and GMRES(m) — where the exception described below takes over.

## Decision

For a single arithmetic operation or elementary function call, a `NaN` or `Inf` operand
produces a `NaN` or `Inf` result per standard IEEE 754 rules, with no special detection,
no branch, and no documentation obligation at each call site — the behavior is exactly
what the underlying floating-point hardware already does.

Krylov iterative solvers do not follow this rule. Each iteration explicitly checks
whether the current iterate has become non-finite; if it has, the solver stops
immediately and reports the failure through its error type rather than continuing to
loop. This is a deliberate carve-out, not an inconsistency: a single operation always
costs exactly one evaluation no matter what value it produces, but an iterative solver
that keeps looping on a poisoned (`NaN`/`Inf`) state can burn its entire iteration budget
converging toward nothing, producing a wrong or meaningless answer only after paying the
full cost of every remaining iteration.

## Constraints

- The common path (a single scalar operation) must not carry extra branching cost for
  non-finite detection — propagation is free precisely because it does nothing beyond
  what hardware floating-point already does.
- On resource-constrained targets, the compute budget spent inside an iteration loop is
  not negligible; a solver that silently iterates to `max_iter` on poisoned state wastes
  a real, bounded resource rather than just wall-clock time.
- Failure reporting must go through the existing `Result`-based error model — no panics,
  no silent truncation of the iteration, and no ambiguity between "converged" and
  "gave up on garbage input."
- The non-finite failure must be distinguishable from other convergence failures (for
  example, an iterate that is finite but has no direction to normalize, or a shift that
  makes an inner matrix singular) so callers can tell *why* a solve stopped early.

## Status

Implemented. Single operations and elementary functions propagate non-finite values with
no extra checks. Krylov solvers detect non-finite iterates mid-loop and return a
dedicated error variant distinct from other convergence-failure variants.

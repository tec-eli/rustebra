# ADR 0013: Numerical Stability and Error-Bounds Policy

## Status
Accepted

## Context
rustebra targets embedded and resource-constrained systems where numerical failures
(NaN/Inf propagation, unbounded iteration on poisoned state, undocumented precision
degradation) carry a different cost than in desktop/server contexts: iterating over
garbage state in a tight loop wastes a real, bounded compute budget, not just wall-clock
time. This ADR establishes a single stated policy for how the library handles
non-finite values and how it documents the precision it can and cannot guarantee.

## Decision

### NaN/Inf policy
The library's default policy for single arithmetic operations and elementary
functions (sqrt, sin, cos, etc.) is to **propagate** non-finite values per IEEE 754
semantics, undocumented per-call. This matches the behavior of the underlying
hardware/software operations and imposes no extra branching cost on the common path.

**Krylov iterative methods are a documented exception.** Power iteration, inverse
power iteration, and future Krylov solvers (CG, Lanczos, Arnoldi, GMRES(m)) explicitly
detect non-finite values mid-loop and return `Err(ConvergenceError::NonFinite)` rather
than continuing to iterate on poisoned state. This is not a contradiction of the
general propagate policy — it reflects a different context: a single operation costs
one evaluation regardless of outcome, while an iterative solver can burn `max_iter`
evaluations converging toward nothing once a NaN or Inf enters the state. On embedded
targets, that cost is not negligible.

### Elementary function precision
Precision targets are tiered by scalar type, not by target device:

| Type | Target relative error | Domain |
|------|----------------------|--------|
| f64  | < 1e-14               | [-2π, 2π] |
| f32  | < 1e-6                | [-2π, 2π] |

Outside `[-2π, 2π]`, behavior is documented as **degraded and untested**, not bounded.
The library does not provide a separate reduced-precision fast path for smaller devices;
no evidence of demand justifies that added complexity, and it would compete directly
with higher-priority Krylov work.

Tolerance values used in threshold judgments elsewhere in the library (e.g.
`singular_tol`, rank/condition-number decisions) remain caller-configurable per the
existing tolerance-system design (ADR 0009) and are out of scope for this ADR.

## Consequences
- `ConvergenceError` gains a `NonFinite` variant, distinct from `ZeroVector`, which
  today conflates zero-vector initialization failure with non-finite poisoning.
  Existing Krylov code must be updated to detect and report `NonFinite` explicitly.
- Elementary-function implementations (sqrt, sin, cos) must be verified against a
  high-precision reference over `[-2π, 2π]`.
- **`docs/NUMERICAL_STABILITY.md` does not yet exist and must be created** as a
  companion document recording: elementary-fn precision bounds and verification method,
  per-decomposition precision limits, iterative-method convergence assumptions,
  tolerance defaults, condition-number thresholds, chained-operation error behavior,
  and the NaN/Inf policy stated above. This ADR states policy; that document records
  verified numbers and per-module detail.
- No fast-path variants of elementary functions will be added without a concrete,
  evidenced use case.

## References
- ADR 0004 (per-module Result-based error handling)
- ADR 0009 (tolerance system for approximate-zero comparisons)
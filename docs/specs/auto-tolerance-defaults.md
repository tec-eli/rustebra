# Auto-Computed Tolerance Defaults

## Summary

Operations that require an explicit tolerance (rank, singular value decomposition,
condition-number estimation, Cholesky decomposition) each expose a second, general-user
entry point that computes a sensible default tolerance internally, so a caller who doesn't
know what tolerance to pick can still call the operation.

## Scope

Applies to every operation covered by [[approximate-zero-tolerance]]'s second category — the
functions that already take an explicit, caller-supplied tolerance — and to how a default
value for that tolerance is computed and exposed for callers who don't want to supply one.

## Decision

Each tolerance-taking operation gets two entry points rather than a single function with an
optional tolerance: a general-user name with no tolerance parameter, and the existing
explicit name that takes a caller-supplied tolerance. A single function with one signature
would need one trait bound covering every call, which would force the auto-computing
capability's trait requirement onto callers who already supply their own tolerance and never
needed it — splitting the two paths into separate functions avoids that coupling.

Computing a default requires machine epsilon for the concrete scalar type in use. Rather
than adding an epsilon method to the crate's core scalar abstraction — which would force
every scalar implementation, including hypothetical future ones with no meaningful notion of
machine epsilon, to answer a question that may not apply to it — a separate, narrower trait
holds it, implemented only by the concrete floating-point types. The general-user entry
points require both the core scalar trait and this narrower one; the explicit,
caller-supplied-tolerance entry points require only the core scalar trait, exactly as
before. A future scalar type with no meaningful epsilon simply never implements the narrower
trait, and loses only the auto-computing convenience — every explicit function remains
callable by supplying a tolerance directly.

The default itself follows the same relative-where-it's-free, absolute-elsewhere split used
for caller-supplied tolerance: where an operation already computes a natural scale as a side
effect (such as the largest singular value, or the largest-magnitude diagonal entry), the
default is derived relative to that scale; where no such scale exists without adding work
that wouldn't make the comparison more correct, the default is an absolute value derived
from machine epsilon and the problem size alone.

## Constraints

- No auto-computing capability may be added to the core scalar trait itself; it must live
  behind a separate, narrower trait that only floating-point-like types need implement.
- A caller who already supplies their own tolerance must never be forced to satisfy a trait
  bound that exists only to support the auto-computing default path.
- The default tolerance for a given operation must be derived either relative to a scale the
  operation already computes, or as an absolute value tied to machine epsilon and problem
  size — never an arbitrary constant unrelated to either.
- A scalar type that cannot meaningfully provide machine epsilon must still be able to call
  every explicit, tolerance-taking function by supplying its own value; it only loses access
  to the general-user, default-computing entry point.

## Status

Implemented. Rank, singular value decomposition, condition-number estimation, and Cholesky
decomposition each expose a general-user entry point with an auto-computed default, backed
by a narrow epsilon-exposing trait implemented by the floating-point scalar types, alongside
their existing explicit, caller-supplied-tolerance entry points.

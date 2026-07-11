# Elementary Function Precision

## Summary

Precision targets for elementary functions are tiered by scalar type, not by target device:
`f64` targets a relative error below `1e-14`, `f32` targets a relative error below `1e-6`,
both measured over the primary domain `[-2π, 2π]`.

## Scope

Applies to every elementary function the scalar type provides — trigonometric functions,
exponential, logarithm, and similar — over the domain `[-2π, 2π]`. Behavior outside that
domain is explicitly out of scope for any precision guarantee.

## Decision

Two fixed precision tiers exist, keyed only to which concrete floating-point type is in use,
with no per-device variation: every target using `f64` gets the same precision target as
every other target using `f64`, regardless of whether that target is a desktop machine or a
microcontroller, and likewise for `f32`.

No separate, reduced-precision fast path is provided for smaller or more constrained
devices. That option was considered and rejected: there is no concrete evidence of demand
for it, and building it would compete directly with higher-priority work (Krylov solver
development) for the same engineering time.

## Constraints

- Outside the primary domain `[-2π, 2π]`, elementary function behavior is documented as
  degraded and untested — not bounded by any stated precision guarantee. Callers operating
  outside this domain get no promise about the result's accuracy.
- Precision targets are a function of scalar type alone; no code path may vary an elementary
  function's precision based on which target device it is compiled for.
- A reduced-precision fast path must not be added without a concrete, evidenced use case —
  the absence of demand today is a real constraint on scope, not an oversight to be filled
  in speculatively.

## Status

Implemented. Elementary function implementations are verified against a high-precision
reference over `[-2π, 2π]`, meeting the stated per-type targets. No fast-path variant
exists.

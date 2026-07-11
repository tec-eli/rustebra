# Minimal Shared Storage Trait Between Static and Dynamic Types

## Summary

Stack-based (compile-time-sized) and heap-based (runtime-sized) storage share a small,
narrowly scoped trait so that generic algorithms can be written once and run over either
kind of storage, without unifying the two families' public-facing types.

## Scope

Applies to the internal seam between the algorithm layer and the two storage strategies the
crate offers. Does not apply to the public types themselves (the stack-allocated and
heap-allocated vector/matrix types remain separate, independently designed types for direct
use by callers) — this decision only governs how generic algorithm code accesses storage
internally.

## Decision

A minimal storage trait exposes only what generic algorithms actually need — element access
and dimensions — and nothing more. It is not designed upfront in full; new capability is
added to the trait only at the point some concrete algorithm needs it, not speculatively.

This trait exists purely as an internal seam for the algorithm layer. It does not replace or
merge the public stack-allocated and heap-allocated types, which remain distinct types with
their own ergonomics, aimed at their own use cases. A caller using either type directly sees
no difference in their public API as a result of this trait existing.

Expensive algorithms — Krylov methods among them — are written once against this trait and
work unmodified over both storage strategies, rather than requiring a duplicate
implementation per strategy.

## Constraints

- The trait's surface must only grow when a specific, real algorithm needs a capability it
  doesn't yet expose — no capability is added speculatively ahead of a concrete caller.
- The trait must not leak the dynamic-size assumptions of heap-backed storage into the
  compile-time-size guarantees the stack-backed storage relies on, or vice versa.
- The public-facing stack-allocated and heap-allocated types must remain unaffected by this
  trait's existence — callers using them directly should observe no API change.

## Status

Implemented. The trait is scoped to element access and dimensions today; expensive
algorithms such as Krylov methods are already written once against it and operate over both
storage strategies without duplicated implementations.

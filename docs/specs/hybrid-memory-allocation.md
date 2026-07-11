# Hybrid Memory Allocation Strategy

## Summary

The library is built around two tiers of memory use: a strict, allocation-free core that
is the default build, and an optional feature flag that unlocks heap-backed dynamic data
structures and algorithms for callers who have an allocator available.

## Scope

Applies to every data layout and elementary operation in the crate. The baseline tier
covers fixed-size vectors, buffers, and matrices whose dimensions are known at compile
time. The optional tier covers growable, `Vec`-backed matrices and vectors, and any
algorithm whose workspace size cannot be known until runtime — sparse matrix formats and
iterative solvers being the primary examples.

## Decision

By default, the crate builds against no allocator at all. Baseline types use const
generics (`<const N: usize>`) to fix the size of every vector, buffer, and matrix at
compile time, and live entirely on the stack.

A single conditional-compilation feature unlocks a second tier: dynamic, `Vec`-backed
data structures and the algorithms that need a workspace whose size depends on runtime
input rather than a compile-time constant. Enabling this feature is opt-in and additive —
it does not change the behavior or requirements of the baseline tier.

Static (compile-time-sized) and dynamic (heap-backed) types are kept as distinct, separate
type families rather than unified behind one shared public API. How much internal
behavior the two families share at the implementation level is treated as its own,
separate design question, not settled by this decision.

## Constraints

- The baseline build must run unmodified on microcontrollers with only a few kilobytes of
  RAM and no global memory allocator — this is the floor the strict tier is designed
  against, not an aspirational target.
- In the baseline tier, internal memory overflow and heap-fragmentation failures must be
  impossible by construction (no allocator exists to fragment or exhaust), not merely
  avoided by discipline or testing.
- Baseline, stack-allocated operations must have deterministic, bounded latency, with no
  allocator-dependent jitter — this rules out any hidden heap use inside the default
  build.
- Some algorithms — sparse matrix multiplication and iterative solvers among them —
  fundamentally need a workspace whose size is not known until runtime; a purely
  stack-only design cannot express them, which is why the optional tier exists rather
  than trying to force every algorithm into the const-generic model.

## Status

Implemented (Revised). The baseline crate remains allocator-free with const-generic,
stack-only storage; the optional feature gates every `Vec`-backed dynamic type and every
workspace-heavy algorithm. The accepted trade-offs are additional signature complexity in
the baseline tier from const generics, and the maintenance of static and dynamic types as
two separate, non-unified families.

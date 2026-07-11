# Specs Index

This index lists every spec in `docs/specs/`, ordered from foundational decisions to the
ones that build on them. Read top to bottom for the full picture; jump directly to a spec
if you already know the area you're touching.

## Crate shape and layering

The decisions everything else sits on top of: how the crate is distributed, how it's
layered internally, and what the core scalar and storage abstractions look like.

- [Single Crate with Incremental Feature Flags](single-crate-feature-flags.md)
- [Trait-Based Generic Layered Architecture](layered-generic-architecture.md)
- [Generic Scalar Type](generic-scalar-type.md)
- [Hybrid Memory Allocation Strategy](hybrid-memory-allocation.md)
- [Minimal Shared Storage Trait Between Static and Dynamic Types](static-dynamic-storage-trait.md)

## Cross-cutting conventions

Conventions that apply across every module, independent of which mathematical domain
they show up in.

- [Per-Module Error Handling with Result](per-module-error-handling.md)
- [NaN/Inf Policy](nan-inf-policy.md)

## Scalar functions and numerical tolerance

How elementary functions are implemented and how the library decides when a computed
value is small enough to treat as zero.

- [Public Elementary Scalar Functions, Scoped to Internal Needs](elementary-scalar-functions.md)
- [Elementary Function Precision](elementary-function-precision.md)
- [Numerical Tolerance for Approximate-Zero Comparisons](approximate-zero-tolerance.md)
- [Auto-Computed Tolerance Defaults](auto-tolerance-defaults.md)

## Dense matrix API

- [Two-Level Public API for Matrix Operations](two-level-matrix-api.md)

## Sparse matrix API

Sparse-specific decisions, ordered from the underlying storage representation up to the
operations built on it.

- [Sparse Index Type (u32)](sparse-index-type.md)
- [Sparse Matrix Public API Shape](sparse-matrix-api-shape.md)
- [SortedCsrMatrix — Type-Level Column-Sort Invariant](sorted-csr-matrix.md)
- [Sparse Entry Pruning and Explicit Zero Semantics](sparse-entry-pruning.md)

## Krylov subspace methods

- [Krylov Basis-Size Const-Generic Convention](krylov-basis-size-const-generics.md)

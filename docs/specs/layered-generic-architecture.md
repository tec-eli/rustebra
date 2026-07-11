# Trait-Based Generic Layered Architecture

## Summary

The crate is organized into four dependency-ordered layers — scalar, storage, algorithm,
and public API — each depending only on the layers before it, following the architecture
pattern used by established Rust linear algebra libraries.

## Scope

Applies to the overall organization of the entire crate: where a new type, trait, or
function belongs, and which other parts of the codebase it is allowed to depend on.

## Decision

The crate has four layers:

1. **Scalar layer.** Defines what it means to be a number the library can operate on —
   arithmetic and the relevant elementary functions. Depends on nothing else in the crate.
2. **Storage layer.** A minimal trait describing how to access the elements and dimensions
   of a vector or matrix, independent of whether the underlying memory is stack- or
   heap-based. Implemented by both storage strategies the crate offers.
3. **Algorithm layer.** Mathematical algorithms — arithmetic operations, norms,
   decompositions, Krylov methods, and so on — written generically against the scalar and
   storage layers. Each algorithm is implemented once and works over any type satisfying the
   relevant traits, regardless of which storage strategy backs it.
4. **Public API layer.** The concrete, ergonomic types a caller actually uses. These wire a
   specific storage strategy together with the algorithm layer, so that callers interact
   with concrete types rather than generics or trait bounds directly.

Each layer depends only on the layers listed before it; there is no dependency running in
the other direction.

## Constraints

- The storage layer's trait must stay minimal, growing only as real algorithms need new
  capability from it — a broad, leaky storage abstraction would undermine the entire
  layering.
- Every mathematical algorithm belongs in the algorithm layer and must be written against
  the scalar and storage traits, never against a concrete storage type directly, so a single
  implementation always covers every storage strategy.
- The public API layer must remain the only layer an ordinary caller needs to understand;
  generics and trait bounds from the lower layers should not leak into it unnecessarily.

## Status

Implemented. The four layers are in place: the scalar and storage traits at the base, a
generic algorithm layer built on top of them, and the concrete public-facing vector and
matrix types assembled from that foundation.

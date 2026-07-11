
# ADR 0006: Trait-Based Generic Layered Architecture

## Status
Accepted

## Context
The project needed an overall architecture: a way of organizing how the generic scalar type
(ADR 0005), the storage strategies (ADR 0001), and the mathematical algorithms relate to each
other, so that algorithms are not duplicated across storage strategies and the system has a
clear, well-understood shape.

Hexagonal architecture (ports and adapters) was considered first, framing storage strategies
as external actors behind a port. This was rejected: hexagonal architecture is intended for
systems with genuine external actors — databases, network services, multiple delivery
mechanisms — that need to be swapped without touching business logic. This project has no
such actors; "static storage" and "dynamic storage" are not external systems, they are
internal representation choices within the same domain. Applying hexagonal architecture here
would have been a forced metaphor rather than a genuine fit.

Instead, the architecture used by established Rust linear algebra libraries (such as
`nalgebra` and `ndarray`) was adopted as a precedent: a small set of core traits, with
algorithms written generically against them, and concrete public-facing types assembled on
top for ergonomic use.

## Decision
The crate is organized into four layers, each depending only on the layers before it:

1. **Scalar layer.** Defines what it means to be a number the library can operate on
   (arithmetic, relevant elementary functions). Depends on nothing else in the crate.
   Corresponds to ADR 0005.
2. **Storage layer.** A minimal trait describing how to access the elements and dimensions of
   a vector or matrix, independent of whether the underlying memory is stack- or heap-based.
   Implemented by the static (const-generic) and dynamic (`alloc`-gated) storage strategies
   from ADR 0001. The trait is scoped to ADR 0003.
3. **Algorithm layer.** Mathematical algorithms (arithmetic operations, norms, decompositions,
   Krylov methods, etc.) written generically against the Scalar and Storage layers. An
   algorithm is implemented once and works over any type that satisfies the relevant traits,
   regardless of storage strategy.
4. **Public API layer.** The concrete types a caller actually uses (e.g. a stack-allocated
   vector type, a heap-allocated vector type). These wire together a specific storage strategy
   with the algorithm layer into an ergonomic, concrete interface, so that callers do not need
   to interact with generics or trait bounds directly.

## Consequences

### Positive
* **No forced metaphor.** The architecture matches what the project actually has — internal
  representation choices, not external systems — instead of adapting a pattern designed for a
  different kind of problem.
* **Algorithms written once.** Layer 3 depends only on traits, never on a concrete storage
  type, so a single implementation of an algorithm works across storage strategies.
* **Established precedent.** This matches the architecture used by existing, mature Rust
  numerical libraries, rather than an untested design.
* **Clear dependency direction.** Each layer only depends on the ones before it; there is no
  ambiguity about where a new piece of code belongs.

### Negative / Trade-offs
* **Trait design discipline required.** The Storage trait (layer 2) must stay minimal and
  grow only as real algorithms need it, or it risks becoming the kind of broad, leaky
  abstraction earlier decisions deliberately avoided.
* **Indirection for new contributors.** Understanding why a function lives in the algorithm
  layer instead of directly on a public type requires understanding the layering, which is an
  upfront learning cost.

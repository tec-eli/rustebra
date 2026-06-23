---
layout: default
title: "ADR 0003: Minimal Shared Storage Trait Between Static and Dynamic Types"
---

# ADR 0003: Minimal Shared Storage Trait Between Static and Dynamic Types

## Status
Accepted (Revised — see Revision History)

## Context
ADR 0001 originally proposed that static (stack-based, const-generic) and dynamic
(heap-based, `alloc`-gated) types would share a common set of traits, so that algorithms
requiring no internal allocation could operate identically over either kind of storage.

Designing such a shared trait correctly is non-trivial: it must be general enough to cover
both storage strategies without leaking the dynamic-size assumptions of one into the
compile-time-size guarantees of the other. A first, conservative version of this decision
(see Revision History) rejected any shared trait at all, to avoid premature abstraction
before there was real code to learn from.

Since then, the project adopted a layered, trait-based generic architecture (see ADR 0006).
That architecture only works if algorithms can be written once against a minimal,
purpose-built abstraction, rather than against each concrete storage type. Without some
shared trait, every algorithm — including expensive ones like Lanczos or Arnoldi — would need
to be implemented twice, which is the exact duplication ADR 0001 originally hoped to avoid.

## Decision
Static and dynamic types share a **minimal storage trait**, scoped only to what generic
algorithms actually need (such as element access and dimensions). This trait is not designed
upfront in full; it grows only when a specific algorithm needs a capability it doesn't yet
expose.

This shared trait governs only how generic algorithms (ADR 0006, layer 3) interact with
storage. It does **not** unify the public-facing API: `Static*` and `Dynamic*` types remain
separate, independently designed types for direct use by callers, as originally intended.
The shared trait is an internal seam for the algorithm layer, not a replacement for the
public types.

## Consequences

### Positive
* **No duplicated algorithm logic.** Expensive algorithms (e.g. Krylov methods) are written
  once, against the trait, and work over both storage strategies.
* **Scoped abstraction.** Because the trait only grows as algorithms need it, it avoids the
  original risk of a large, prematurely-generalized interface.
* **Public API unaffected.** Callers using `Static*` or `Dynamic*` types directly see no
  difference; the trait is an implementation detail of the algorithm layer.

### Negative / Trade-offs
* **An abstraction now exists where there previously was none.** Its boundaries must be
  actively maintained to avoid growing back into the large, leaky trait this ADR originally
  avoided.
* **Two things to keep mentally separate.** Contributors need to understand that the shared
  trait and the public types are not the same axis of design — one governs algorithm reuse,
  the other governs the user-facing API.

## Revision History
* **Original version ("Separated Traits for Static and Dynamic Types"):** decided that static
  and dynamic types would share no trait at all, with any dual-mode algorithm implemented
  twice. This was revised once the project adopted a trait-based generic architecture
  (ADR 0006), which requires a minimal shared trait at the algorithm layer to avoid that
  duplication. The decision to keep the public-facing types themselves separate is preserved
  from the original version.
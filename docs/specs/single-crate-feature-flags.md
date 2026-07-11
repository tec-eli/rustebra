# Single Crate with Incremental Feature Flags

## Summary

The library ships as one crate, not a multi-crate workspace, and feature flags are added
one at a time, only when the code they gate actually exists.

## Scope

Applies to the crate's overall distribution shape — how the mathematical domains it covers
(dense linear algebra, sparse matrices, Krylov methods, and other domains added over time)
are packaged, versioned, and gated behind Cargo features.

## Decision

All mathematical domains live in a single published crate rather than being split across a
workspace of per-domain crates. New feature flags are introduced incrementally, in the same
change that introduces the code they gate — never added ahead of time in anticipation of a
module that doesn't exist yet.

A workspace split is not ruled out permanently; it remains available later if a specific
domain grows large enough on its own to justify independent versioning and release cycles.
It is simply not the starting point, because most domains this crate will eventually cover
don't exist yet, and designing a multi-crate boundary around code that hasn't been written
is exactly the kind of upfront speculation the project avoids elsewhere.

## Constraints

- A single manifest, changelog, and version number must be enough to describe the whole
  crate's public surface at any point in time — callers should never need to reason about
  which of several crate versions is compatible with which.
- A feature flag may only be added in the same change that introduces the functionality it
  gates. No flag is reserved or stubbed in for a domain that is still unimplemented.
- Splitting into a workspace later must remain structurally possible; nothing in the current
  module layout should make a future split disproportionately expensive.

## Status

Implemented. At present the crate carries a single feature flag, gating the optional
heap-backed storage tier; further flags are expected to arrive one per domain as those
domains are implemented, each introduced alongside its own code rather than in advance.

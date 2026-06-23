---
layout: default
title: "ADR 0002: Single Crate with Incremental Feature Flags"
---

# ADR 0002: Single Crate with Incremental Feature Flags

## Status
Accepted

## Context
ADR 0001 established a hybrid memory strategy gated by an `alloc` feature flag. Beyond that
split, the project will eventually cover multiple distinct mathematical domains (dense linear
algebra, sparse matrices, Krylov methods, numerical calculus, dynamical systems, signal
processing). A decision is needed on how this code is distributed and versioned: as a single
crate, or as a multi-crate workspace with one crate per domain.

A multi-crate workspace would let each mathematical domain version and release
independently, at the cost of additional maintenance overhead (multiple manifests, multiple
changelogs, coordinating any shared base types across crate boundaries).

A single crate keeps maintenance simple at this stage, since most domains do not exist yet
and a workspace split can be introduced later without changing how the mathematics itself is
implemented.

## Decision
The project will be published as a single crate. Feature flags are added incrementally, only
when the code they gate actually exists — not in advance. At the time of this decision, the
only feature flag is `alloc`, as established in ADR 0001. Additional feature flags (one per
mathematical domain, or otherwise) will be introduced as their corresponding modules are
implemented, each addressed in its own future ADR or changelog entry rather than decided
upfront.

## Consequences

### Positive
* **Simple distribution.** A single `cargo add` gives access to the whole library; a single
  manifest, a single changelog, and a single version number to reason about.
* **No upfront design burden.** Feature flags are designed against real code, not against
  speculation about future modules.
* **Reversible.** Splitting into a workspace later remains possible if a specific domain
  grows large enough to warrant independent versioning.

### Negative / Trade-offs
* **Coupled versioning.** A change to one mathematical domain forces a version bump of the
  entire crate, even if unrelated domains are unaffected.
* **Single compilation unit.** All enabled code is compiled together; users cannot depend on
  one domain without the crate's manifest including the others, even if disabled via
  features.
* **Revisitable later.** If maintenance friction grows once several domains exist
  simultaneously, this decision may be superseded by a workspace-based ADR.
---
layout: default
title: "ADR 0007: Public Elementary Scalar Functions, Scoped to Internal Needs"
---

# ADR 0007: Public Elementary Scalar Functions, Scoped to Internal Needs

## Status
Accepted

## Context
The `Scalar` trait requires elementary functions (starting with `sqrt`, extending to `sin`,
`cos`, and others) that `#![no_std]` does not provide. These are implemented from scratch
rather than depending on the existing, actively maintained `libm` crate, consistent with the
project's preference for minimal external dependencies.

Since these functions already exist internally, a decision was needed on whether to expose
them as part of the public API, and how broad that exposure should be.

## Decision
Elementary functions on `Scalar` are public. They are implemented only as needed by the
library's own operations (vectors, matrices, and later groups) — not as a complete,
general-purpose replacement for `libm`. A user depending on this crate for its math
operations does not need a separate `libm` dependency for any function this crate already
implements for its own use.

## Consequences

### Positive
* Users avoid an extra dependency for functions this crate already needs internally.
* No added implementation cost — the functions are already being written either way.

### Negative / Trade-offs
* Coverage is incomplete by design; functions this crate doesn't internally need (e.g.
  special functions) are out of scope and won't be added solely for external use.
* Precision/edge-case guarantees are whatever this crate's own implementations provide, not
  necessarily matching `libm`'s.

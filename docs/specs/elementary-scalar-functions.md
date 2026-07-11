# Public Elementary Scalar Functions, Scoped to Internal Needs

## Summary

Elementary functions on the scalar type — `sqrt`, `sin`, `cos`, and others — are implemented
from scratch, are exposed publicly, and are scoped only to what the library's own operations
actually need, rather than aiming to be a complete replacement for an external math library.

## Scope

Applies to every elementary function required by the scalar trait and its implementations,
and to the decision of whether and how those functions are exposed to callers of the crate.

## Decision

Elementary functions are implemented internally rather than pulled in from an external
dependency, consistent with the library's preference for a minimal dependency footprint.
Because these functions already have to exist for the library's own internal use, they are
also exposed as public API — a caller depending on this crate for its math operations does
not need a separate math dependency for any function this crate already implements for
itself.

Coverage is deliberately incomplete: only the functions the library's own vector, matrix,
and future group operations actually require are implemented. Functions with no internal
caller — special functions, for instance — are out of scope and are not added purely to
serve external callers who might want a general-purpose function library.

## Constraints

- No function is added to this surface solely for external convenience; every elementary
  function implemented here must have at least one internal caller within the crate's own
  operations.
- Precision and edge-case behavior are whatever this crate's own implementations provide;
  there is no promise of matching any particular external math library's behavior bit for
  bit.

## Status

Implemented. Elementary functions required internally are implemented and exposed publicly;
no functions exist in this surface purely for external use.

# Two-Level Public API for Matrix Operations

## Summary

Any matrix operation with more than one available algorithm exposes two levels of public
API: a high-level function that picks an algorithm automatically, and one explicit function
per algorithm that always runs the named algorithm regardless of input.

## Scope

Applies to every matrix operation where multiple algorithms exist to compute the same
result — solving a linear system, computing a determinant, decomposing a matrix — and where
those algorithms differ in cost, numerical stability, or precondition (for example, one
algorithm requiring symmetric positive-definite input while another has no such
requirement).

## Decision

For each such operation, two entry points are published:

1. A **high-level function** (for example, a general `determinant` or `solve`) that selects
   an algorithm automatically based on observable properties of the input — size, symmetry,
   detectable positive-definiteness. Which specific algorithm runs underneath is not part of
   the function's public contract; it may change between versions without being a breaking
   change, as long as the mathematical result stays correct.
2. One **explicit algorithm function** per algorithm (for example, a cofactor-expansion
   determinant and an LU-based determinant) that always runs the named algorithm, regardless
   of what the input looks like. These functions are stable in the stronger sense: the named
   algorithm never changes underneath them.

Both levels are public and documented; neither is treated as internal or discouraged.

## Constraints

- General users who don't want to choose an algorithm must have a single, ergonomic entry
  point per operation that always returns a correct result.
- Mathematical users who need to study or rely on a specific algorithm's exact behavior must
  be able to call it by name and trust that name never silently starts running a different
  algorithm.
- The heuristics a high-level function uses to select an algorithm must be documented well
  enough that a caller can predict what will run, even though that choice is not a formal,
  version-stable guarantee.

## Status

Implemented. Operations with multiple available algorithms — determinant and solve among
them — expose both a high-level automatic entry point and named explicit-algorithm
functions.

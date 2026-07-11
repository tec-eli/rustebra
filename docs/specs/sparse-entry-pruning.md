# Sparse Entry Pruning and Explicit Zero Semantics

## Summary

A pruning function drops sparse-matrix entries whose magnitude falls at or below a
caller-supplied threshold, using only ordering comparisons rather than requiring the scalar
type to provide an absolute-value method, and the threshold is always supplied by the
caller — no auto-computed default exists for this operation.

## Scope

Applies to CSR sparse matrices accumulating numerically negligible entries over the course
of iterative computation — fill-in from products, cancellation from additions, ordinary
floating-point rounding — where those entries are structurally present but too small to be
worth the memory and arithmetic cost of keeping.

## Decision

An entry with value `v` is dropped when its magnitude is at or below a threshold `τ > 0`
(equivalently, it is kept when `v` is strictly greater than `τ` or strictly less than `-τ`).
This matches the standard definition of threshold-based pruning used elsewhere in numerical
software.

Because the scalar type deliberately exposes no absolute-value or ordering methods of its
own — keeping its required surface minimal rather than assuming every possible scalar
implementation has a meaningful notion of magnitude — the pruning function adds the ordering
bound it needs directly to its own signature instead. The magnitude comparison is then
expressed without an absolute-value method at all, by comparing against both the threshold
and its negation using only ordering and the scalar type's existing arithmetic.

The threshold is always supplied by the caller. Unlike operations where a machine-epsilon-
derived default is a sensible baseline, the "right" pruning threshold depends entirely on
the caller's own domain — the physical scale being modeled, the acceptable accuracy loss
downstream, whether the goal is memory compression or cancellation-noise removal — and no
neutral default is correct across those cases. A caller who wants no pruning at all simply
passes a zero threshold.

The function returns the plain CSR format, not the sorted-invariant wrapper type, even
though pruning only removes entries and never reorders the ones that remain: if the input
was already known to be sorted, the output is too, but expressing that in the return type
would require a second function or an overload, which is not justified without a concrete
caller needing it. A caller who needs the sorted wrapper type after pruning applies the
public sorting constructor to the result.

Only the row-oriented format is covered. The column-oriented equivalent is not implemented,
because no current caller needs it — it would be structurally identical with rows and
columns swapped, and is deferred until a real use case exists rather than added
speculatively.

## Constraints

- The pruning function must not require any capability beyond ordering comparisons and the
  scalar type's existing arithmetic — no absolute-value method is added to the core scalar
  abstraction to support it.
- No auto-computed default threshold is offered; every call must supply an explicit value,
  because no domain-independent default is correct.
- Pruning must never reorder surviving entries — only remove entries, preserving whatever
  ordering the input already had.

## Status

Implemented for the row-oriented format. The column-oriented equivalent is deferred until a
concrete caller needs it; callers working in that format today convert to the row-oriented
format, prune, and convert back.

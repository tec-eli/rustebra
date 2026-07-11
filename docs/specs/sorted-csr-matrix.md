# SortedCsrMatrix — Type-Level Column-Sort Invariant

## Summary

A newtype wrapper around the CSR format carries a compile-time guarantee that every row's
column indices are stored in ascending order, so algorithms that require this ordering can
state the requirement in their signature instead of checking or re-sorting at runtime.

## Scope

Applies to any CSR matrix produced by an operation that already naturally yields sorted
column indices per row, and to any current or future algorithm — sparse triangular solve and
sparse Cholesky factorization among them — that requires that ordering as a precondition.

## Decision

A wrapper type around the plain CSR format restricts construction so that only genuinely
sorted matrices can exist as values of the wrapped type. Two construction paths exist: a
public constructor that sorts an arbitrary CSR matrix's rows in place before wrapping it,
safe for any input; and a second, module-internal constructor that asserts the invariant
without independently verifying it, used only by the specific conversion and multiplication
operations that already guarantee sorted output as a natural consequence of how they work
(so no separate sort pass is needed in those two cases).

The wrapper transparently exposes every read-only method of the underlying plain CSR type
through automatic dereferencing, so existing code that reads fields or calls accessors on a
result that happens to be sorted needs no changes. A consuming conversion lets a caller
discard the sorted guarantee explicitly when they need to pass the value to code that only
accepts the plain format.

Operations that already produce sorted output by construction return the sorted wrapper type
directly, communicating the guarantee to their callers at the type level rather than by
convention or documentation alone. Ordered COO → CSR → CSC by format/flow rather than by
release history (all six ship together, so no chronological ordering applies), these are:
`coo_to_csr`, `csr_to_csc`, `csc_to_csr`, `spmm`, `add_csr`, and `add_csc`.

## Constraints

- The module-internal, invariant-skipping constructor must only be reachable from code that
  can actually guarantee sorted output as a structural consequence of its own algorithm —
  never exposed as a general-purpose "trust me" escape hatch.
- The wrapper must add no runtime cost beyond what plain construction already costs; the
  guarantee is enforced once at construction, not re-checked on every subsequent read.
- Existing code operating on the plain format's read-only accessors must continue to compile
  unchanged when handed a value of the wrapped type.

## Status

Implemented. The wrapper type is used as the return type of the six operations that produce
sorted output as a natural consequence of their algorithm — `coo_to_csr`, `csr_to_csc`,
`csc_to_csr`, `spmm`, `add_csr`, and `add_csc` — and a public, verifying constructor is
available for wrapping an arbitrary CSR or CSC matrix from any other source.

# Sparse Index Type (u32)

## Summary

Every sparse matrix storage format — COO, CSR, and CSC — represents row indices, column
indices, and row/column pointer arrays as `u32`, not `usize`. Any value that would not
fit in a `u32` is rejected through the format's own error type at the point it would be
stored, rather than being truncated or silently wrapped.

## Scope

Applies to the internal index storage of `CooMatrix`, `CsrMatrix`, and `CscMatrix`
(`row_indices`/`col_indices`, `row_ptr`/`col_ptr`), and to every operation that produces
or consumes those arrays: format construction, COO/CSR/CSC conversion, `spmm`, `matmat`,
`add`, entry pruning, and validation. It does not apply to matrix dimensions themselves
(`rows`, `cols`), which remain `usize`, or to scalar values stored in the matrix.

## Decision

Index and pointer arrays are `Vec<u32>` (owned) and `&[u32]` (borrowed) throughout the
sparse module. Any point where a `usize` value — a dimension, a running count, a computed
offset — must be written into one of these arrays goes through a checked conversion
(`u32::try_from`) or checked arithmetic (`checked_add`, `checked_mul`), and a failure
there is surfaced as the format's `DimensionOverflow` error variant. Constructors and
conversions never truncate a `usize` into a `u32` with an `as` cast where the value could
plausibly exceed `u32::MAX`.

## Constraints

- Index arrays, not values, dominate memory use for sparse formats at typical sparsity
  levels — a CSR matrix with more rows/columns than nonzeros spends more memory on
  `row_ptr` and `col_indices` than on the stored values themselves. Halving the width of
  every index entry (`u32` versus a 64-bit `usize` on the platforms this library
  primarily targets) is a direct, unconditional reduction in that footprint, which
  matters most on RAM-constrained targets.
- No operation may panic or silently wrap on a value that doesn't fit — every narrowing
  conversion from `usize` to `u32` must be checked and reported through the crate's
  existing `Result`-based error model.
- `u32::MAX` (4,294,967,295) rows or columns is far beyond what a sparse matrix on a
  memory-constrained target could hold in the first place, so the narrower range is not
  a realistic ceiling in practice, while the memory savings apply to every matrix
  regardless of size.

## Status

Implemented. All format constructors, conversions, and sparse operations (`spmm`,
`matmat`, `add`, prune, validate) that write into an index or pointer array go through a
checked path and return `DimensionOverflow` on failure instead of wrapping.

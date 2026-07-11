# LU

LU decomposition factors a square matrix `a` as `l * u`, where `l` is unit lower
triangular (1s on the diagonal) and `u` is upper triangular, up to a row permutation:
`l * u == p * a`, where `p` is the permutation built by applying the reported row swaps, in
order, to the identity.

`rustebra` computes it via Gaussian elimination with partial pivoting: at each step, the
row with the largest-magnitude entry in the current column is swapped into the pivot
position before elimination proceeds, which avoids dividing by a small or zero pivot. The
low-level `lu` function delegates to `lu_partial_pivot`, which documents the pivoting
strategy in more detail; both are available directly, alongside the ergonomic
`StaticMatrix`/`DynamicMatrix::lu()` method used in [Matrix Operations](../04-matrices/operations.md).

```rust
{{#include ../../../examples/algorithm/matrix/lu.rs}}
```

## Gotchas

- The permutation `p` isn't returned as its own matrix — only the number of row swaps
  performed (`swap_count`) is reported. This is enough to recover the permutation's parity
  (used by `determinant`), but not the row ordering itself; there's no direct way to
  recover `p` as a matrix from the public API.
- `lu`/`lu_partial_pivot` are only defined for square matrices — they return
  `Err(DimensionMismatch)` for a non-square input, rather than a rectangular LU variant.

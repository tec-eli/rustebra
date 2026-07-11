# CSR/CSC

Compressed sparse row (CSR) and compressed sparse column (CSC) are the two primary sparse
storage formats `rustebra` operates on. Both store only the non-zero entries, using three
parallel arrays instead of a dense grid.

CSR (`CsrMatrix<T>`) stores:

- `row_ptr` — length `rows + 1`; the range `row_ptr[i]..row_ptr[i + 1]` gives the slice of
  `col_indices`/`values` belonging to row `i`.
- `col_indices` — the column index of each stored entry.
- `values` — the value of each stored entry.

CSC (`CscMatrix<T>`) is the transpose layout: a `col_ptr` of length `cols + 1` plays the
role `row_ptr` plays for CSR, and `row_indices` plays the role `col_indices` plays.

```rust
use rustebra::sparse::CsrMatrix;

// 3x3 identity: row i's single entry is at column i, value 1.0.
let eye = CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0_f64, 1.0, 1.0])
    .unwrap();
assert_eq!(eye.row_ptr(), &[0, 1, 2, 3]);
```

Within a row (CSR) or column (CSC), the stored indices don't have to be sorted — but every
index must be in-bounds, and the pointer array must be non-decreasing, starting at `0` and
ending at the total non-zero count. `CsrMatrix::new`/`CscMatrix::new` validate these
invariants and return `Err` rather than constructing a malformed matrix.

## `SortedCsrMatrix` / `SortedCscMatrix`

`SortedCsrMatrix<T>` and `SortedCscMatrix<T>` wrap a `CsrMatrix`/`CscMatrix` and add the
additional guarantee that indices *within* each row/column are in ascending order. This
enables `O(log(nnz/rows))` binary-search lookup of a specific entry, and is a precondition
some algorithms (sparse triangular solves, certain preconditioners) require. Both types
implement `Deref` to their unsorted counterpart, so every read-only accessor is available
without unwrapping. Construct one directly with `SortedCsrMatrix::from_csr` (which sorts,
paying the cost up front), or get one as the output of `coo_to_csr`/`spmm_csr`, which
produce sorted results as a side effect of their algorithm.

## Gotchas

- A CSR matrix with unsorted column indices is still a *valid* `CsrMatrix` — sortedness is
  an opt-in stronger guarantee via `SortedCsrMatrix`, not a base invariant of `CsrMatrix`
  itself.
- Storing an explicit zero is legal in `CsrMatrix`/`CscMatrix::new` (it doesn't reject
  zero-valued entries), but [`validate_csr`](../05-sparse/operations.md)-style validation
  used elsewhere treats an explicit zero as a violation — don't assume every zero has been
  pruned just because construction succeeded.

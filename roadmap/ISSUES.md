# v0.3.0 Release Issues Report

---

## High Priority — Library API/Consistency

### Missing `prune_csc` (L-3)
The sparse module provides `prune_csr` but not its CSC counterpart, violating the CSR/CSC symmetry principle documented
in ADR 0010. Both formats should have identical operation sets.

### `add_csr` return type inconsistency (L-4)
The function produces sorted output but returns `CsrMatrix<T>` instead of `SortedCsrMatrix<T>`, forcing callers to
re-sort already-sorted data (O(nnz log nnz) overhead). The return type should change to `Result<SortedCsrMatrix<T>, DimensionMismatch>`.

### No examples for sparse module (L-6)
The entire v0.3.0 deliverable (sparse module) has no runnable examples, unlike every other public module. Add
`examples/sparse/main.rs` demonstrating construction, conversion, and core operations.

---

## Medium Priority — Documentation & Firmware Correctness

### Incomplete error documentation (L-5)
The `matmat_csr` and `matmat_csc` `# Errors` sections omit the `x_cols == 0` case, which returns `DimensionMismatch`
when paired with an empty `x` slice.

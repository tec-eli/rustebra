# v0.3.0 Release Issues Report

---

## High Priority — Library API/Consistency

### Missing `prune_csc` (L-3)
The sparse module provides `prune_csr` but not its CSC counterpart, violating the CSR/CSC symmetry principle documented
in ADR 0010. Both formats should have identical operation sets.

## Goes in 0.4.0
### `add_csr` return type inconsistency (L-4)
The function produces sorted output but returns `CsrMatrix<T>` instead of `SortedCsrMatrix<T>`, forcing callers to
re-sort already-sorted data (O(nnz log nnz) overhead). The return type should change to `Result<SortedCsrMatrix<T>, DimensionMismatch>`.

---

┌─────────────────────────────┬─────────────────┬────────┬───────────────────────────────────────────────────────────────────────────────────┐                                                                                                      
│            Issue            │      Type       │ Target │                                      Reason                                       │                                                                                                      
├─────────────────────────────┼─────────────────┼────────┼───────────────────────────────────────────────────────────────────────────────────┤                                                                                                      
│ Missing prune_csc (L-3)     │ New API         │ 0.3.1  │ Non-breaking; adds the missing CSC counterpart for symmetry                       │                                                                                                      
├─────────────────────────────┼─────────────────┼────────┼───────────────────────────────────────────────────────────────────────────────────┤                                                                                                      
│ add_csr return type (L-4)   │ Breaking change │ 0.4.0  │ Changing CsrMatrix<T> → Result<SortedCsrMatrix<T>, …> breaks callers; bumps minor │                                                                                                      
└─────────────────────────────┴─────────────────┴────────┴───────────────────────────────────────────────────────────────────────────────────┘


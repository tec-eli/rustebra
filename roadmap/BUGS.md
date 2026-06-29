# Known Bugs & Design Issues

## Bug Status

┌────────────────────────────────────────────┬────────────────┬────────┬──────────┬────────────────────────────────┐
│              Issue                         │      Type      │ Target │ Severity │         Resolution             │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ add_csr stores cancellation zeros          │ Bug            │ 0.3.2  │ Medium   │ Add zero-check before push     │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ spmm_csr may store exact zero result       │ Potential      │ 0.3.2  │ Low      │ Add zero-check in spmm.rs      │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ scale_by_zero stores zeros                 │ Design Issue   │ 0.3.2  │ Low      │ Prune post-scale or add check  │
└────────────────────────────────────────────┴────────────────┴────────┴──────────┴────────────────────────────────┘

---

## HIGH PRIORITY

*(No high-priority bugs currently identified)*

---

## MEDIUM PRIORITY

### S-1: Sparse Addition Produces Stored Zeros
**Target**: 0.3.2 | **Scope**: add.rs

When two CSR matrices are added and entries at the same position cancel out (e.g., 3.0 + (-3.0)), the result stores an 
explicit 0.0 instead of omitting it. 

**Evidence**: add.rs, lines 70-71: `out_val.push(sum)` with no zero-check  
**Impact**: nnz inflates unnecessarily; users must call `prune_csr` to clean up  
**Example**: `add_csr([3.0], [-3.0])` produces `[0.0]` with nnz=1  
**Workaround**: Apply `prune_csr(result, 0.0)` after addition

- [ ] Add zero-check: `if sum != T::zero()` before pushing to `out_val` in add.rs

---

## LOW PRIORITY

### S-2: Sparse Multiplication (spmm) May Produce Stored Zeros
**Target**: 0.3.2 | **Scope**: spmm.rs

If the inner product of rows in A and columns in B produces an exact zero, it is stored.

**Evidence**: spmm.rs, lines 83-85: emits `dense[col]` without zero-check  
**Impact**: Rare in practice (exact cancellation unlikely), but mathematically possible  
**Example**: `spmm_csr([1, -1], [[1], [1]])` produces `[0.0]`

- [ ] Add zero-check: `if dense[col] != T::zero() { out_col.push(col); out_val.push(dense[col]); }`

### S-3: Scaling by Zero Produces Stored Zeros
**Target**: 0.3.2 | **Scope**: scale.rs

`scale_csr(m, 0.0)` and `scale_csc(m, 0.0)` produce all zeros instead of an empty matrix.

**Evidence**: scale.rs, test line 98: values become `[0.0, 0.0, 0.0]`  
**Impact**: Semantically incorrect (a zero matrix should have nnz=0, not nnz=original)  
**Workaround**: Apply `prune_csr(scaled, 0.0)` to remove stored zeros

- [ ] Check `if scalar != T::zero()` or prune automatically post-scale

---

## Design Notes

**On Stored Zeros**: The crate's philosophy (ADR 0012) is to provide explicit pruning as a separate operation rather
than automatic zero-removal. This gives users fine-grained control (some algorithms benefit from explicit zeros). 
However, operations like addition that *logically* produce zero entries should either:
1. Not emit them in the first place (recommended fix for S-1 and S-2), or
2. Document clearly that callers must prune post-operation

**On Duplicates**: COO's allowance for duplicates is intentional and correct; it's treated as "summing at assembly time"
when converted to CSR. This is efficient and well-documented.

**Comprehensive Sparse Matrix Review (2026-06-28)**: A systematic code review was conducted to assess handling of edge 
cases and numerical issues in the sparse matrix implementation. The review examined: duplicate entries, explicit zeros, 
empty matrices, conversion invariants, NaN/Inf handling, and numerical stability. Properly handled: Empty matrices, 
duplicates, conversions, NaN/Inf.


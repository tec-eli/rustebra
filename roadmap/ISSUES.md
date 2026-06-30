# Rustebra Issues & Roadmap

## All Issues

┌────────────────────────────────────────────┬────────────────┬────────┬──────────┬────────────────────────────────┐
│              Issue                         │      Type      │ Target │ Severity │         Resolution             │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Numerical accuracy of sqrt/sin/cos         │ Verification   │ 0.4.0  │ High     │ Formal convergence & bounds    │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Embedded no_std CI validation              │ CI Gap         │ 0.4.0  │ Medium   │ Test on thumbv7m targets       │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Sparse matrix invariant validation API     │ New Module     │ 0.4.0  │ Medium   │ validate_csr/validate_csc      │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Review silent error handling               │ Code Review    │ 0.4.0  │ Medium   │ Add debug_assert in hot paths  │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Usize arithmetic & overflow checking       │ Safety Review  │ 0.4.0  │ Medium   │ Saturating/checked operations  │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Missing prune_csc                          │ New API        │ 0.3.2  │ Low      │ Implement prune_csc symmetric  │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ add_csr return type inconsistency          │ Breaking API   │ 0.4.0  │ Low      │ Change return to SortedCsrMat  │
└────────────────────────────────────────────┴────────────────┴────────┴──────────┴────────────────────────────────┘

**Note**: Test-related issues have been moved to [roadmap/COVERAGE.md](COVERAGE.md). Bug-related issues are tracked in [roadmap/BUGS.md](BUGS.md). Documentation and ADR items are in [roadmap/DOCS.md](DOCS.md).

---

## HIGH PRIORITY

### N-1: Numerical Accuracy of Elementary Functions
**Target**: 0.4.0 | **Scope**: src/scalar/{sqrt,sin,cos}.rs

Custom `sqrt()`, `sin()`, and `cos()` implementations use simplified algorithms (Taylor series, Newton-Raphson).
Convergence, error bounds, and behavior for very large/small inputs need formal verification.

**Impact**: Silent numerical errors propagate through matrix decompositions (QR, SVD, Cholesky)

- [ ] Compare against IEEE reference implementations
- [ ] Establish error bounds for sqrt function
- [ ] Establish error bounds for sin function
- [ ] Establish error bounds for cos function
- [ ] Document convergence proofs in code
- [ ] Add convergence proof and error bounds to DOCS.md

---

## MEDIUM PRIORITY

### E-1: Embedded no_std Validation on Real Targets
**Target**: 0.4.0 | **Scope**: CI/CD

Current tests run on `x86_64` only. Need to run on actual embedded targets (thumbv7m, Cortex-M4/M7).

- [ ] Add CI tests on `thumbv7m-none-eabi`
- [ ] Add CI tests on `thumbv7em-none-eabihf`
- [ ] Validate stack allocation limits on embedded targets
- [ ] Validate floating-point availability on ARM
- [ ] Test memory layout compatibility

### S-4: Sparse Matrix Invariant Validation
**Target**: 0.4.0 | **Scope**: src/sparse/validate.rs

Add explicit validation module for CSR/CSC invariant checking.

- [ ] Implement `validate_csr<T>(m: &CsrMatrix<T>) -> Result<(), ValidateError>`
- [ ] Implement `validate_csc<T>(m: &CscMatrix<T>) -> Result<(), ValidateError>`
- [ ] Check `row_ptr`/`col_ptr` length consistency
- [ ] Check column indices sorted within each row (or row indices within column for CSC)
- [ ] Check NNZ consistency (`out_col.len() == out_val.len()`)
- [ ] Check index bounds (all indices < n_cols or < n_rows)
- [ ] Check for explicit zeros

### E-3: Review & Harden Error Handling
**Target**: 0.4.0 | **Scope**: Full codebase

Audit all locations where errors are swallowed, panicked, or treated as impossible.

- [ ] Replace `unwrap()` in decomposition algorithms with explicit `debug_assert!`
- [ ] Add invariant checks in hot paths (sparse indexing, matrix bounds)
- [ ] Document invariants guaranteed by callers vs. checked by functions

### S-5: Usize Arithmetic & Allocation Safety
**Target**: 0.4.0 | **Scope**: Sparse operations

Verify no overflow in sparse structures when computing indices, allocations, or bounds.

- [ ] Check `row_ptr[i+1] - row_ptr[i]` overflow protection
- [ ] Check `nnz * sizeof(T)` allocation overflow protection
- [ ] Validate matrix dimensions against system memory limits
- [ ] Add saturating/checked operations in critical paths

---

## LOW PRIORITY

### L-3: Missing `prune_csc`
**Target**: 0.3.1 | **Scope**: src/sparse/

The sparse module provides `prune_csr` but not its CSC counterpart, violating the CSR/CSC symmetry principle (ADR 0010).

- [ ] Implement `prune_csc<T>(m: CscMatrix<T>, tolerance: T) -> CscMatrix<T>` symmetric to `prune_csr`

### L-4: `add_csr` Return Type Inconsistency
**Target**: 0.4.0 | **Scope**: src/sparse/

The function produces sorted output but returns `CsrMatrix<T>` instead of `SortedCsrMatrix<T>`.

- [ ] Change `add_csr` return type to `Result<SortedCsrMatrix<T>, DimensionMismatch>`
- [ ] Apply parallel change to `add_csc`
- [ ] Update type documentation to reflect sorted invariant

---

## Design Notes

**On Duplicates**: COO's allowance for duplicates is intentional and correct; it's treated as "summing at assembly time"
when converted to CSR. This is efficient and well-documented.

**Comprehensive Sparse Matrix Review (2026-06-28)**: A systematic code review was conducted to assess handling of edge 
cases and numerical issues in the sparse matrix implementation. The review examined: duplicate entries, explicit zeros, 
empty matrices, conversion invariants, NaN/Inf handling, and numerical stability. Properly handled: Empty matrices, 
duplicates, conversions, NaN/Inf.

**v0.4.0+ Comprehensive Numerical & Testing Audit**: A mathematician and Rust engineer review of numerical correctness, 
edge-case handling, testing coverage, and embedded safety.

See also: [roadmap/COVERAGE.md](COVERAGE.md) for test coverage roadmap, [roadmap/BUGS.md](BUGS.md) for known bugs, and [roadmap/DOCS.md](DOCS.md) for documentation and ADR items.


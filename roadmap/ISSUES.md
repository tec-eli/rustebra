# Rustebra Issues & Roadmap

## All Issues

┌────────────────────────────────────────────┬────────────────┬────────┬──────────┬────────────────────────────────┐
│              Issue                         │      Type      │ Target │ Severity │         Resolution             │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Numerical accuracy of sqrt/sin/cos         │ Verification   │ 0.4.0  │ High     │ Formal convergence & bounds    │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Missing property-based testing             │ Test Gap       │ 0.4.0  │ High     │ proptest vs nalgebra           │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Differential testing vs nalgebra           │ Test Strategy  │ 0.4.0  │ High     │ Comprehensive cross-checking   │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Comprehensive edge-case testing            │ Test Gap       │ 0.4.0  │ High     │ Singular, 0-size, NaN edge    │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Numerical guarantees documentation         │ Doc Gap        │ 0.4.0  │ High     │ docs/NUMERICAL_STABILITY.md    │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ add_csr stores cancellation zeros          │ Bug            │ 0.3.2  │ Medium   │ Add zero-check before push     │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Fuzz testing for conversions               │ Test Gap       │ 0.4.0  │ Medium   │ cargo-fuzz harness             │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Embedded no_std CI validation              │ CI Gap         │ 0.4.0  │ Medium   │ Test on thumbv7m targets       │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Sparse matrix invariant validation API     │ New Module     │ 0.4.0  │ Medium   │ validate_csr/validate_csc      │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Review silent error handling               │ Code Review    │ 0.4.0  │ Medium   │ Add debug_assert in hot paths  │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Usize arithmetic & overflow checking       │ Safety Review  │ 0.4.0  │ Medium   │ Saturating/checked operations  │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Algorithmic complexity documentation       │ Doc Gap        │ 0.4.0  │ Medium   │ Add O(...) to doc-comments     │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ ADR 0013: Numerical Stability Policy       │ New ADR        │ 0.4.0  │ Medium   │ Create ADR 0013                │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ spmm_csr may store exact zero result       │ Potential      │ 0.3.2  │ Low      │ Add zero-check in spmm.rs      │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ scale_by_zero stores zeros                 │ Design Issue   │ 0.3.2  │ Low      │ Prune post-scale or add check  │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Missing prune_csc                          │ New API        │ 0.3.1  │ Low      │ Implement prune_csc symmetric  │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ add_csr return type inconsistency          │ Breaking API   │ 0.4.0  │ Low      │ Change return to SortedCsrMat  │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Add explicit SVD reconstruction test       │ Improvement    │ 0.4.0  │ Low      │ Test A ≈ UΣVᵀ reconstruction    │
└────────────────────────────────────────────┴────────────────┴────────┴──────────┴────────────────────────────────┘

---

## HIGH PRIORITY

### N-1: Numerical Accuracy of Elementary Functions
**Status**: Review Required | **Target**: 0.4.0 | **Scope**: src/scalar/{sqrt,sin,cos}.rs

Custom `sqrt()`, `sin()`, and `cos()` implementations use simplified algorithms (Taylor series, Newton-Raphson).
Convergence, error bounds, and behavior for very large/small inputs need formal verification.
- **Impact**: Silent numerical errors propagate through matrix decompositions (QR, SVD, Cholesky)
- **Required**: Comparison against IEEE reference implementations; establish error bounds for each function
- **Evidence**: No convergence proofs or error bounds documented in current code
- **Effort**: High

### T-1: Missing Property-Based Testing
**Status**: Test Gap | **Target**: 0.4.0 | **Scope**: tests/

No property tests comparing rustebra results against nalgebra or trusted references.
Critical for LU, QR, SVD, Cholesky, and all sparse operations.
- **Impact**: Correctness bugs only catch when users hit edge cases in production
- **Required**: Differential testing harness comparing outputs against nalgebra for random inputs
- **Suggested Framework**: `proptest` crate with `no_std`-compatible random number generation
- **Coverage**: Decompositions (LU, QR, Cholesky, SVD), sparse add/mul, format conversions
- **Effort**: Very High

### T-2: Differential Testing Against nalgebra
**Status**: New Testing Strategy | **Target**: 0.4.0 | **Scope**: tests/differential/

Cross-check all matrix operations and decompositions against nalgebra on random inputs
to catch numerical inconsistencies, rank deficiency handling, and edge cases.
- **Scope**: LU factorization, QR factorization, Cholesky, SVD, dense matrix ops
- **Scope**: Sparse operations (add_csr, spmm_csr, coo_to_csr, format conversions)
- **Failure Example**: SVD of rank-deficient matrix; nearly-singular matrix with different condition numbers
- **Effort**: Very High

### E-2: Comprehensive Edge-Case Testing
**Status**: Test Gap | **Target**: 0.4.0 | **Scope**: tests/edge_cases/

Formalize and test matrix edge cases that trigger silent failures or subtle bugs.

**Singular & Nearly-Singular Matrices**
- LU factorization with rank < n
- QR on m < n and m > n matrices
- SVD of rank-deficient matrices
- Cholesky on non-positive-definite matrices
- Documentation of rank-computation tolerance (ADR 0009)

**Condition Number Extremes**
- Ill-conditioned matrices (κ >> 1e7)
- Nearly-zero singular values
- Loss of numerical accuracy documentation

**Dimension Edge Cases**
- Zero-sized matrices (0×n, n×0, 0×0)
- 1×1 matrices (scalar operations)
- Rectangular matrices (m >> n, m << n)
- Single row/column matrices

**Sparse Edge Cases**
- Empty sparse matrices (nnz = 0)
- 1×1 sparse matrices
- Diagonal-only sparse matrices
- Fully-dense sparse matrices (all entries non-zero)

**NaN/Inf Handling**
- Matrices containing explicit NaN/Inf (already documented as handled)
- Operations producing NaN/Inf
- Pruning behavior with NaN threshold

- **Effort**: High

### D-1: Numerical Guarantees & Precision Documentation
**Status**: Documentation Gap | **Target**: 0.4.0 | **Scope**: docs/NUMERICAL_STABILITY.md, public doc-comments

Currently no formal documentation of:
- Numerical precision limits for each decomposition (QR, LU, SVD, Cholesky)
- Convergence assumptions for iterative methods
- Machine epsilon relationship and how ADR 0009 tolerance defaults are computed
- Condition number thresholds (when rank is considered < n)
- Accumulated error in chained operations
- How elementary functions (sqrt, sin, cos) accuracy translates to matrix operations

**Required**: New markdown file `docs/NUMERICAL_STABILITY.md` with:
- Accuracy bounds for each algorithm
- When to use explicit `no_std` elementary functions vs. libm equivalents
- Example: "QR of well-conditioned m×n matrix has backward error O(ε‖A‖)"

- **Effort**: High

---

## MEDIUM PRIORITY

### S-1: Sparse Addition Produces Stored Zeros
**Status**: Confirmed Bug | **Target**: 0.3.2 | **Scope**: add.rs

When two CSR matrices are added and entries at the same position cancel out (e.g., 3.0 + (-3.0)), the result stores an 
explicit 0.0 instead of omitting it. 
- **Evidence**: add.rs, lines 70-71: `out_val.push(sum)` with no zero-check
- **Impact**: nnz inflates unnecessarily; users must call `prune_csr` to clean up
- **Example**: `add_csr([3.0], [-3.0])` produces `[0.0]` with nnz=1
- **Workaround**: Apply `prune_csr(result, 0.0)` after addition
- **Fix**: Check `if sum != T::zero()` before pushing to `out_val`

### T-3: Fuzz Testing for Matrix Construction & Conversions
**Status**: New Testing Strategy | **Target**: 0.4.0 | **Scope**: tests/fuzz/

Add fuzzing harness for:
- Matrix construction from random inputs (dense, COO, CSR, CSC)
- Format conversions (COO → CSR, CSR ↔ CSC, dense ↔ sparse)
- Operations on fuzzed matrices (add, mul, scale, decompositions)
- **Tool**: `cargo-fuzz` with `no_std`-compatible targets
- **Effort**: High

### E-1: Embedded no_std Validation on Real Targets
**Status**: CI Gap | **Target**: 0.4.0 | **Scope**: CI/CD

Current tests run on `x86_64` only. Need to run on actual embedded targets (thumbv7m, Cortex-M4/M7).
- **Missing**: Tests on `thumbv7m-none-eabi`, `thumbv7em-none-eabihf`
- **Risk**: Code may not compile or behave differently on ARM without OS
- **Required**: Extend CI matrix to include `cargo test --target thumbv7m-none-eabi`
- **Impact**: Validates stack allocation limits, floating-point availability, memory layout
- **Effort**: Medium

### S-4: Sparse Matrix Invariant Validation
**Status**: Validation Gap | **Target**: 0.4.0 | **Scope**: src/sparse/validate.rs

Add explicit validation module for CSR/CSC invariant checking:
- `row_ptr`/`col_ptr` length consistency (`len == n_rows + 1` or `n_cols + 1`)
- Column indices sorted within each row (or row indices within each column for CSC)
- No duplicate entries (or documented behavior for sparse assembly)
- NNZ consistency (`out_col.len() == out_val.len()`)
- Index bounds checking (all indices < n_cols or < n_rows)
- No explicit zeros unless documented
- **API**: Public `fn validate_csr<T>(m: &CsrMatrix<T>) -> Result<(), ValidateError>`
- **Effort**: Medium

### E-3: Review & Harden Error Handling
**Status**: Code Review | **Target**: 0.4.0 | **Scope**: Full codebase

Audit all locations where errors are swallowed, panicked, or treated as impossible:
- Replace `unwrap()` in decomposition algorithms with explicit `debug_assert!`
- Add invariant checks in hot paths (sparse indexing, matrix bounds)
- Document what invariants are guaranteed by callers vs. checked by functions
- **Effort**: High

### S-5: Usize Arithmetic & Allocation Safety
**Status**: Safety Review | **Target**: 0.4.0 | **Scope**: Sparse operations

Verify no overflow in sparse structures when computing indices, allocations, or bounds:
- `row_ptr[i+1] - row_ptr[i]` can overflow with corrupted input
- `nnz * sizeof(T)` allocation can overflow
- Matrix dimensions checked against system memory limits
- **Validation**: Saturating arithmetic or checked operations in critical paths
- **Effort**: Medium

### D-2: Algorithmic Complexity Documentation
**Status**: Documentation Gap | **Target**: 0.4.0 | **Scope**: Public doc-comments

Document time and space complexity for each operation:
- LU: O(n³) time, O(n²) space
- QR: O(mn²) time, O(n²) space for thin QR
- SVD: O(mn²) or O(m²n) depending on algorithm
- Sparse add/mul: O(nnz_A + nnz_B) or O(nnz_A × nnz_B / m)
- Pruning: O(nnz) time

**Format**: Add `/// Complexity: O(...)` to each public function doc-comment

- **Effort**: Medium

### D-3: ADR 0013 - Numerical Stability & Error Bounds
**Status**: New ADR | **Target**: 0.4.0

Create a new architecture decision record documenting:
- Policy on forward vs. backward error analysis
- Tolerance handling across different matrix scales (ADR 0009 extension)
- When elementary function accuracy matters vs. doesn't
- How custom elementary functions (sqrt, sin, cos) are validated
- Reference to IEEE 754 and backward-error analysis literature
- **Effort**: Medium

---

## LOW PRIORITY

### S-2: Sparse Multiplication (spmm) May Produce Stored Zeros
**Status**: Confirmed Potential | **Target**: 0.3.2 | **Scope**: spmm.rs

If the inner product of rows in A and columns in B produces an exact zero, it is stored.
- **Evidence**: spmm.rs, lines 83-85: emits `dense[col]` without zero-check
- **Impact**: Rare in practice (exact cancellation unlikely), but mathematically possible
- **Example**: `spmm_csr([1, -1], [[1], [1]])` produces `[0.0]`
- **Fix**: Add zero-check: `if dense[col] != T::zero() { out_col.push(col); out_val.push(dense[col]); }`

### S-3: Scaling by Zero Produces Stored Zeros
**Status**: Confirmed Design Issue | **Target**: 0.3.2 | **Scope**: scale.rs

`scale_csr(m, 0.0)` and `scale_csc(m, 0.0)` produce all zeros instead of an empty matrix.
- **Evidence**: scale.rs, test line 98: values become `[0.0, 0.0, 0.0]`
- **Impact**: Semantically incorrect (a zero matrix should have nnz=0, not nnz=original)
- **Workaround**: Apply `prune_csr(scaled, 0.0)` to remove stored zeros
- **Fix**: Check `if scalar != T::zero()` or prune automatically post-scale

### L-3: Missing `prune_csc`
**Status**: New API | **Target**: 0.3.1

The sparse module provides `prune_csr` but not its CSC counterpart, violating the CSR/CSC symmetry principle documented 
in ADR 0010. Both formats should have identical operation sets.
- **Fix**: Implement `prune_csc<T>(m: CscMatrix<T>, tolerance: T) -> CscMatrix<T>` symmetric to `prune_csr`
- **Effort**: ~30 lines (mirror prune.rs logic for columns)

### L-4: `add_csr` Return Type Inconsistency
**Status**: Breaking Change | **Target**: 0.4.0

The function produces sorted output (documented: "The output has sorted, deduplicated column indices within each row") 
but returns `CsrMatrix<T>` instead of `SortedCsrMatrix<T>`, forcing callers to re-sort already-sorted data (O(nnz log nnz) overhead).
- **Current**: `fn add_csr(...) -> Result<CsrMatrix<T>, DimensionMismatch>`
- **Proposed**: `fn add_csr(...) -> Result<SortedCsrMatrix<T>, DimensionMismatch>`
- **Impact**: Eliminates wasted re-sort after addition; type system documents invariant
- **Note**: `add_csc` should have parallel change

### SVD Reconstruction Test
**Status**: Improvement | **Target**: 0.4.0

Current tests verify singular values match determinant. Add explicit test validating A ≈ UΣVᵀ reconstruction accuracy.
- **File**: tests/algorithm/matrix/svd.rs
- **Effort**: ~20 lines

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

**v0.4.0+ Comprehensive Numerical & Testing Audit**: A mathematician and Rust engineer review of numerical correctness, 
edge-case handling, testing coverage, and embedded safety.


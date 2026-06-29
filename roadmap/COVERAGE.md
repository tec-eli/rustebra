# Test Coverage & Property-Based Testing Roadmap

## Test Coverage Status

┌────────────────────────────────────────────┬────────────────┬────────┬──────────┬────────────────────────────────┐
│              Issue                         │      Type      │ Target │ Severity │         Resolution             │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Missing property-based testing             │ Test Gap       │ 0.4.0  │ High     │ proptest vs nalgebra           │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Differential testing vs nalgebra           │ Test Strategy  │ 0.4.0  │ High     │ Comprehensive cross-checking   │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Comprehensive edge-case testing            │ Test Gap       │ 0.4.0  │ High     │ Singular, 0-size, NaN edge     │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Fuzz testing for conversions               │ Test Gap       │ 0.4.0  │ Medium   │ cargo-fuzz harness             │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Add explicit SVD reconstruction test       │ Improvement    │ 0.4.0  │ Low      │ Test A ≈ UΣVᵀ reconstruction   │
└────────────────────────────────────────────┴────────────────┴────────┴──────────┴────────────────────────────────┘

---

## Missing Tests by Priority

### HIGH PRIORITY

#### T-1: Property-Based Testing Infrastructure
**Target**: 0.4.0 | **Scope**: tests/property/

Property tests comparing rustebra's numerical results against invariants and trusted references using `proptest`.

- [x] LU decomposition structural tests (L unit lower triangular, U upper triangular)
- [x] Dev-dependencies added (proptest, nalgebra)
- [x] Property test infrastructure created

**Tests to Implement:**

1. **QR Decomposition** — Verify Q is orthogonal (QᵀQ ≈ I), R is upper triangular, A ≈ QR
   - [ ] Test thin and full QR variants
   - [ ] Verify orthogonality of random matrices
   - [ ] Coverage: 3×3, 4×5, 5×3 matrices with random entries

2. **Cholesky Decomposition** — Requires positive-definite matrices; verify LLᵀ ≈ A, L is lower triangular
   - [ ] Generate positive-definite matrices (XᵀX pattern)
   - [ ] Verify decomposition structural invariants
   - [ ] Test on well-conditioned and ill-conditioned inputs

3. **SVD** — Verify singular values non-negative and decreasing, UΣVᵀ ≈ A, orthogonality of U and V
   - [ ] Generate rank-deficient and full-rank matrices
   - [ ] Verify singular value ordering
   - [ ] Test reconstruction accuracy

4. **Sparse Addition (add_csr/add_csc)** — Generate random sparse matrices, verify addition correctness
   - [ ] Compare against dense equivalents
   - [ ] Test sparsity preservation
   - [ ] Verify no spurious stored zeros

5. **Sparse Multiplication (spmm)** — Generate random sparse pairs, verify spmm_csr correctness
   - [ ] Compare against dense matrix multiply
   - [ ] Test various sparsity patterns

6. **Format Conversions** — COO↔CSR, CSR↔CSC with proptest
   - [ ] Verify round-trip conversions preserve values
   - [ ] Test on matrices with various sparsity patterns

#### T-2: Differential Testing Against nalgebra
**Target**: 0.4.0 | **Scope**: tests/property/

Compare rustebra outputs directly against nalgebra on identical random inputs to catch numerical inconsistencies.
Requires permutation tracking in LU and careful handling of numerical tolerance in SVD.

- [ ] Implement differential testing harness for LU
- [ ] Implement differential testing harness for QR
- [ ] Implement differential testing harness for SVD
- [ ] Implement differential testing harness for Cholesky
- [ ] Implement differential testing harness for sparse operations
- [ ] Document permutation handling and tolerance requirements

#### E-2: Comprehensive Edge-Case Testing
**Target**: 0.4.0 | **Scope**: tests/edge_cases/

Formalize and test matrix edge cases that trigger silent failures or subtle bugs.

**Singular & Nearly-Singular Matrices**
- [ ] LU factorization with rank < n
- [ ] QR on m < n and m > n matrices
- [ ] SVD of rank-deficient matrices
- [ ] Cholesky on non-positive-definite matrices
- [ ] Document rank-computation tolerance (ADR 0009)

**Condition Number Extremes**
- [ ] Ill-conditioned matrices (κ >> 1e7)
- [ ] Nearly-zero singular values
- [ ] Loss of numerical accuracy documentation

**Dimension Edge Cases**
- [ ] Zero-sized matrices (0×n, n×0, 0×0)
- [ ] 1×1 matrices (scalar operations)
- [ ] Rectangular matrices (m >> n, m << n)
- [ ] Single row/column matrices

**Sparse Edge Cases**
- [ ] Empty sparse matrices (nnz = 0)
- [ ] 1×1 sparse matrices
- [ ] Diagonal-only sparse matrices
- [ ] Fully-dense sparse matrices (all entries non-zero)

**NaN/Inf Handling**
- [ ] Matrices containing explicit NaN/Inf
- [ ] Operations producing NaN/Inf
- [ ] Pruning behavior with NaN threshold

### MEDIUM PRIORITY

#### T-3: Fuzz Testing for Matrix Construction & Conversions
**Target**: 0.4.0 | **Scope**: tests/fuzz/

Fuzzing harness for matrix operations and conversions.

- [ ] Set up cargo-fuzz infrastructure
- [ ] Matrix construction from random inputs (dense, COO, CSR, CSC)
- [ ] Format conversions (COO → CSR, CSR ↔ CSC, dense ↔ sparse)
- [ ] Operations on fuzzed matrices (add, mul, scale, decompositions)
- [ ] Ensure `no_std`-compatible targets

### LOW PRIORITY

#### SVD Reconstruction Test
**Target**: 0.4.0 | **Scope**: tests/algorithm/matrix/svd.rs

- [ ] Add explicit test validating A ≈ UΣVᵀ reconstruction accuracy

---

## Testing Framework Guidelines

When implementing property tests:

1. **Value Range Strategy**: Generate matrices in [-100, 100] to avoid numerical underflow/overflow with extreme values
2. **Tolerance Handling**: Use relative tolerance `tol * max(|a|, |b|)` for floating-point comparisons, account for numerical noise (1e-10)
3. **Structural vs. Numerical**: Verify mathematical properties (triangularity, orthogonality) before exact value matching
4. **Permutation Awareness**: LU may permute rows; track swap count and apply inverse permutation when needed
5. **Matrix Size Variation**: Test on multiple matrix sizes (small 3×3, medium 5×5, rectangular)


# Documentation & Architecture Roadmap

## Documentation Status

┌────────────────────────────────────────────┬────────────────┬────────┬──────────┬────────────────────────────────┐
│              Item                          │      Type      │ Target │ Severity │         Resolution             │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Numerical guarantees documentation         │ Doc Gap        │ 0.4.0  │ High     │ docs/NUMERICAL_STABILITY.md    │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ Algorithmic complexity documentation       │ Doc Gap        │ 0.4.0  │ Medium   │ Add O(...) to doc-comments     │
├────────────────────────────────────────────┼────────────────┼────────┼──────────┼────────────────────────────────┤
│ ADR 0013: Numerical Stability Policy       │ New ADR        │ 0.4.0  │ Medium   │ Create ADR 0013                │
└────────────────────────────────────────────┴────────────────┴────────┴──────────┴────────────────────────────────┘

---

## HIGH PRIORITY

### D-1: Numerical Guarantees & Precision Documentation
**Target**: 0.4.0 | **Scope**: docs/NUMERICAL_STABILITY.md, public doc-comments

- [ ] Document numerical precision limits for each decomposition (QR, LU, SVD, Cholesky)
- [ ] Document convergence assumptions for iterative methods
- [ ] Document machine epsilon relationship and ADR 0009 tolerance defaults computation
- [ ] Document condition number thresholds (when rank is considered < n)
- [ ] Document accumulated error in chained operations
- [ ] Document how elementary functions (sqrt, sin, cos) accuracy translates to matrix operations
- [ ] Create new markdown file `docs/NUMERICAL_STABILITY.md` with accuracy bounds for each algorithm
- [ ] Document when to use explicit `no_std` elementary functions vs. libm equivalents
- [ ] Add example: "QR of well-conditioned m×n matrix has backward error O(ε‖A‖)"

---

## MEDIUM PRIORITY

### D-2: Algorithmic Complexity Documentation
**Target**: 0.4.0 | **Scope**: Public doc-comments

- [ ] Document LU complexity: O(n³) time, O(n²) space
- [ ] Document QR complexity: O(mn²) time, O(n²) space for thin QR
- [ ] Document SVD complexity: O(mn²) or O(m²n) depending on algorithm
- [ ] Document sparse add/mul complexity: O(nnz_A + nnz_B) or O(nnz_A × nnz_B / m)
- [ ] Document pruning complexity: O(nnz) time
- [ ] Add `/// Complexity: O(...)` to each public function doc-comment

### D-3: ADR 0013 - Numerical Stability & Error Bounds Policy
**Target**: 0.4.0

- [ ] Document policy on forward vs. backward error analysis
- [ ] Document tolerance handling across different matrix scales (ADR 0009 extension)
- [ ] Document when elementary function accuracy matters vs. doesn't
- [ ] Document how custom elementary functions (sqrt, sin, cos) are validated
- [ ] Add references to IEEE 754 and backward-error analysis literature
- [ ] Create ADR 0013 document with above content

---


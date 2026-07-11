#![cfg(feature = "alloc")]

//! Curated sparse-matrix edge cases the density-agnostic integration suite doesn't focus
//! on: a matrix with zero stored entries, a matrix where every entry is stored (no
//! sparsity at all), and a diagonal-only matrix — checking that `add`, `multiply`
//! (`spmm`/`matvec`), and `prune` all produce correct results without panicking.

use rustebra::sparse::{CsrMatrix, add_csr, matvec_csr, prune_csr, spmm_csr};

// ── nnz = 0 ──────────────────────────────────────────────────────────────────

#[test]
fn nnz_zero_add_multiply_and_prune_all_work() {
    // Two 3×3 matrices with no stored entries at all.
    let a = CsrMatrix::<f64>::new(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let b = CsrMatrix::<f64>::new(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();

    let sum = add_csr(&a, &b).unwrap();
    assert_eq!(sum.nnz(), 0);
    assert_eq!(sum.row_ptr(), &[0, 0, 0, 0]);

    let product = spmm_csr(&a, &b).unwrap();
    assert_eq!(product.nnz(), 0);
    assert_eq!(product.row_ptr(), &[0, 0, 0, 0]);

    let y = matvec_csr(&a, &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(y, vec![0.0, 0.0, 0.0]);

    let pruned = prune_csr(a, 1e-10).unwrap();
    assert_eq!(pruned.nnz(), 0);
    assert_eq!(pruned.rows(), 3);
    assert_eq!(pruned.cols(), 3);
}

#[test]
fn nnz_zero_non_square_add_multiply_and_prune_all_work() {
    // A 2×4 matrix with no stored entries, multiplied against a compatible 4×3 matrix.
    let a = CsrMatrix::<f64>::new(2, 4, vec![0, 0, 0], vec![], vec![]).unwrap();
    let a2 = CsrMatrix::<f64>::new(2, 4, vec![0, 0, 0], vec![], vec![]).unwrap();
    let b = CsrMatrix::<f64>::new(4, 3, vec![0, 0, 0, 0, 0], vec![], vec![]).unwrap();

    let sum = add_csr(&a, &a2).unwrap();
    assert_eq!(sum.nnz(), 0);
    assert_eq!(sum.rows(), 2);
    assert_eq!(sum.cols(), 4);

    let product = spmm_csr(&a, &b).unwrap();
    assert_eq!(product.nnz(), 0);
    assert_eq!(product.rows(), 2);
    assert_eq!(product.cols(), 3);

    let y = matvec_csr(&a, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(y, vec![0.0, 0.0]);

    let pruned = prune_csr(a, 1e-10).unwrap();
    assert_eq!(pruned.nnz(), 0);
    assert_eq!(pruned.rows(), 2);
    assert_eq!(pruned.cols(), 4);
}

// ── fully dense (every entry non-zero) ──────────────────────────────────────

#[test]
fn fully_dense_matrix_add_gives_doubled_entries() {
    // [1 2 3]
    // [4 5 6]
    // [7 8 9]  — every position is a stored entry.
    let m = CsrMatrix::new(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();

    let sum = add_csr(&m, &m).unwrap();
    assert_eq!(sum.nnz(), 9);
    assert_eq!(
        sum.values(),
        &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    );
}

#[test]
fn fully_dense_matrix_matvec_gives_row_sums_for_ones_vector() {
    let m = CsrMatrix::new(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();

    let y = matvec_csr(&m, &[1.0, 1.0, 1.0]).unwrap();
    assert_eq!(y, vec![6.0, 15.0, 24.0]);
}

#[test]
fn fully_dense_matrix_spmm_against_identity_is_unchanged() {
    let m = CsrMatrix::new(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let eye = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 1.0, 1.0],
    )
    .unwrap();

    let product = spmm_csr(&m, &eye).unwrap();
    assert_eq!(product.nnz(), 9);
    assert_eq!(
        product.values(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn fully_dense_matrix_prune_with_zero_tolerance_keeps_every_entry() {
    let m = CsrMatrix::new(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();

    let pruned = prune_csr(m, 0.0).unwrap();
    assert_eq!(pruned.nnz(), 9);
}

// ── diagonal-only ────────────────────────────────────────────────────────────

#[test]
fn diagonal_only_matrix_add_sums_the_diagonals() {
    // diag(2, 3, 4) + diag(2, 3, 4) = diag(4, 6, 8)
    let d = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![2.0_f64, 3.0, 4.0],
    )
    .unwrap();

    let sum = add_csr(&d, &d).unwrap();
    assert_eq!(sum.nnz(), 3);
    assert_eq!(sum.col_indices(), &[0, 1, 2]);
    assert_eq!(sum.values(), &[4.0, 6.0, 8.0]);
}

#[test]
fn diagonal_only_matrix_matvec_scales_each_component() {
    // diag(2, 3, 4) × [1, 2, 3] = [2, 6, 12]
    let d = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![2.0_f64, 3.0, 4.0],
    )
    .unwrap();

    let y = matvec_csr(&d, &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(y, vec![2.0, 6.0, 12.0]);
}

#[test]
fn diagonal_only_matrix_spmm_multiplies_diagonals_elementwise() {
    // diag(2, 3, 4) × diag(2, 3, 4) = diag(4, 9, 16)
    let d = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![2.0_f64, 3.0, 4.0],
    )
    .unwrap();

    let product = spmm_csr(&d, &d).unwrap();
    assert_eq!(product.nnz(), 3);
    assert_eq!(product.col_indices(), &[0, 1, 2]);
    assert_eq!(product.values(), &[4.0, 9.0, 16.0]);
}

#[test]
fn diagonal_only_matrix_prune_removes_only_the_negligible_diagonal_entry() {
    // diag(2, 1e-15, 4): the middle entry is negligible relative to 1e-10.
    let d = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![2.0_f64, 1e-15, 4.0],
    )
    .unwrap();

    let pruned = prune_csr(d, 1e-10).unwrap();
    assert_eq!(pruned.nnz(), 2);
    assert_eq!(pruned.col_indices(), &[0, 2]);
    assert_eq!(pruned.values(), &[2.0, 4.0]);
}

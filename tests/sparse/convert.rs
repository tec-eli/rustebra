use rustebra::sparse::{CooMatrix, CsrMatrix, coo_to_csr, csr_to_coo};

// ── coo_to_csr ──────────────────────────────────────────────────────────────

#[test]
fn coo_to_csr_basic_no_duplicates() {
    // 3×3 diagonal: entries are supplied out of row-order to verify sorting.
    let coo = CooMatrix::new(3, 3, vec![2, 0, 1], vec![2, 0, 1], vec![3.0_f64, 1.0, 2.0]).unwrap();

    let csr = coo_to_csr(coo);
    assert_eq!(csr.rows(), 3);
    assert_eq!(csr.cols(), 3);
    assert_eq!(csr.nnz(), 3);
    assert_eq!(csr.row_ptr(), &[0, 1, 2, 3]);
    assert_eq!(csr.col_indices(), &[0, 1, 2]);
    assert_eq!(csr.values(), &[1.0, 2.0, 3.0]);
}

#[test]
fn coo_to_csr_sums_duplicate_positions() {
    // (0,0) appears three times; the values must be summed.
    let coo = CooMatrix::new(
        2,
        2,
        vec![0, 0, 0, 1],
        vec![0, 0, 0, 1],
        vec![1.0_f64, 2.0, 4.0, 9.0],
    )
    .unwrap();

    let csr = coo_to_csr(coo);
    assert_eq!(csr.nnz(), 2);
    assert_eq!(csr.row_ptr(), &[0, 1, 2]);
    assert_eq!(csr.col_indices(), &[0, 1]);
    assert_eq!(csr.values(), &[7.0, 9.0]); // 1 + 2 + 4 = 7
}

#[test]
fn coo_to_csr_empty_coo_gives_empty_csr() {
    let coo = CooMatrix::<f64>::new(4, 5, vec![], vec![], vec![]).unwrap();
    let csr = coo_to_csr(coo);
    assert_eq!(csr.rows(), 4);
    assert_eq!(csr.cols(), 5);
    assert_eq!(csr.nnz(), 0);
    assert_eq!(csr.row_ptr(), &[0, 0, 0, 0, 0]);
}

#[test]
fn coo_to_csr_zero_row_matrix() {
    let coo = CooMatrix::<f64>::new(0, 3, vec![], vec![], vec![]).unwrap();
    let csr = coo_to_csr(coo);
    assert_eq!(csr.rows(), 0);
    assert_eq!(csr.nnz(), 0);
    assert_eq!(csr.row_ptr(), &[0]);
}

#[test]
fn coo_to_csr_matrix_with_empty_rows() {
    // Row 1 has no entries.
    let coo = CooMatrix::new(3, 3, vec![0, 2], vec![1, 0], vec![5.0_f64, 8.0]).unwrap();
    let csr = coo_to_csr(coo);
    assert_eq!(csr.row_ptr(), &[0, 1, 1, 2]);
    assert_eq!(csr.col_indices(), &[1, 0]);
    assert_eq!(csr.values(), &[5.0, 8.0]);
}

#[test]
fn coo_to_csr_out_of_order_same_row_entries_are_sorted_by_col() {
    // Two entries in row 0: col 2 given before col 0.
    let coo = CooMatrix::new(1, 3, vec![0, 0], vec![2, 0], vec![9.0_f64, 3.0]).unwrap();
    let csr = coo_to_csr(coo);
    assert_eq!(csr.row_ptr(), &[0, 2]);
    assert_eq!(csr.col_indices(), &[0, 2]);
    assert_eq!(csr.values(), &[3.0, 9.0]);
}

// ── csr_to_coo ──────────────────────────────────────────────────────────────

#[test]
fn csr_to_coo_basic() {
    let csr = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 2.0, 3.0],
    )
    .unwrap();

    let coo = csr_to_coo(csr);
    assert_eq!(coo.rows(), 3);
    assert_eq!(coo.cols(), 3);
    assert_eq!(coo.nnz(), 3);
    assert_eq!(coo.row_indices(), &[0, 1, 2]);
    assert_eq!(coo.col_indices(), &[0, 1, 2]);
    assert_eq!(coo.values(), &[1.0, 2.0, 3.0]);
}

#[test]
fn csr_to_coo_with_empty_rows() {
    // Row 1 is empty; row 2 has two entries.
    let csr = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 1, 3],
        vec![0, 1, 2],
        vec![5.0_f64, 6.0, 7.0],
    )
    .unwrap();

    let coo = csr_to_coo(csr);
    assert_eq!(coo.row_indices(), &[0, 2, 2]);
    assert_eq!(coo.col_indices(), &[0, 1, 2]);
    assert_eq!(coo.values(), &[5.0, 6.0, 7.0]);
}

#[test]
fn csr_to_coo_empty_matrix_gives_empty_coo() {
    let csr = CsrMatrix::<f64>::new(3, 4, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let coo = csr_to_coo(csr);
    assert_eq!(coo.nnz(), 0);
    assert_eq!(coo.rows(), 3);
    assert_eq!(coo.cols(), 4);
}

#[test]
fn csr_to_coo_zero_row_matrix() {
    let csr = CsrMatrix::<f64>::new(0, 2, vec![0], vec![], vec![]).unwrap();
    let coo = csr_to_coo(csr);
    assert_eq!(coo.rows(), 0);
    assert_eq!(coo.nnz(), 0);
}

// ── round-trips ─────────────────────────────────────────────────────────────

#[test]
fn round_trip_coo_to_csr_to_coo_no_duplicates() {
    // Starting COO with entries already in (row, col) order; the round-trip must
    // reproduce the same row/col/value arrays.
    let original =
        CooMatrix::new(3, 3, vec![0, 1, 2], vec![2, 0, 1], vec![7.0_f64, 3.0, 9.0]).unwrap();

    let coo2 = csr_to_coo(coo_to_csr(original).into_inner());
    // coo_to_csr sorts by (row,col), so within each row the col indices are ascending.
    assert_eq!(coo2.row_indices(), &[0, 1, 2]);
    assert_eq!(coo2.col_indices(), &[2, 0, 1]);
    assert_eq!(coo2.values(), &[7.0, 3.0, 9.0]);
}

#[test]
fn round_trip_csr_to_coo_to_csr_no_duplicate_cols() {
    // Identity-like CSR (col indices already sorted within rows) must survive the round-trip.
    let csr = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![4.0_f64, 5.0, 6.0],
    )
    .unwrap();
    let expected = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![4.0_f64, 5.0, 6.0],
    )
    .unwrap();

    let csr2 = coo_to_csr(csr_to_coo(csr));
    assert_eq!(csr2, expected);
}

#[test]
fn round_trip_coo_with_duplicates_then_back_to_coo_has_summed_values() {
    // After COO → CSR (which sums duplicates) → COO, duplicates are gone.
    let coo = CooMatrix::new(2, 2, vec![0, 0, 1], vec![1, 1, 0], vec![2.0_f64, 3.0, 8.0]).unwrap();

    let coo2 = csr_to_coo(coo_to_csr(coo).into_inner());
    assert_eq!(coo2.nnz(), 2); // duplicate at (0,1) was merged
    assert_eq!(coo2.row_indices(), &[0, 1]);
    assert_eq!(coo2.col_indices(), &[1, 0]);
    assert_eq!(coo2.values(), &[5.0, 8.0]); // 2 + 3 = 5
}

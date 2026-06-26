use rustebra::sparse::{CscMatrix, CsrMatrix, DimensionMismatch, matvec_csc, matvec_csr};

// ── matvec_csr ───────────────────────────────────────────────────────────────

#[test]
fn matvec_csr_diagonal_matrix() {
    // diag(1, 2, 3) × [4, 5, 6] = [4, 10, 18]
    let m = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 2.0, 3.0],
    )
    .unwrap();
    let y = matvec_csr(&m, &[4.0, 5.0, 6.0]).unwrap();
    assert_eq!(y, vec![4.0, 10.0, 18.0]);
}

#[test]
fn matvec_csr_general_matrix() {
    // [ 1  2 ]   [ 3 ]   [ 11 ]
    // [ 0  5 ] × [ 4 ] = [ 20 ]
    let m = CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 2.0, 5.0]).unwrap();
    let y = matvec_csr(&m, &[3.0, 4.0]).unwrap();
    assert_eq!(y, vec![11.0, 20.0]);
}

#[test]
fn matvec_csr_empty_matrix_returns_zero_vector() {
    let m = CsrMatrix::<f64>::new(3, 2, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let y = matvec_csr(&m, &[1.0, 2.0]).unwrap();
    assert_eq!(y, vec![0.0, 0.0, 0.0]);
}

#[test]
fn matvec_csr_non_square_matrix() {
    // 2×3 matrix × length-3 vector → length-2 output
    let m = CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let y = matvec_csr(&m, &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(y, vec![7.0, 6.0]);
}

#[test]
fn matvec_csr_empty_rows_produce_zero_entries() {
    // Row 1 has no entries (row_ptr[1] == row_ptr[2]).
    let m = CsrMatrix::new(3, 2, vec![0, 1, 1, 2], vec![0, 1], vec![4.0_f64, 6.0]).unwrap();
    let y = matvec_csr(&m, &[1.0, 2.0]).unwrap();
    assert_eq!(y, vec![4.0, 0.0, 12.0]);
}

#[test]
fn matvec_csr_dimension_mismatch_returns_error() {
    let m = CsrMatrix::new(2, 3, vec![0, 1, 1], vec![0], vec![1.0_f64]).unwrap();
    assert_eq!(matvec_csr(&m, &[1.0, 2.0]), Err(DimensionMismatch));
}

#[test]
fn matvec_csr_zero_row_matrix() {
    let m = CsrMatrix::<f64>::new(0, 3, vec![0], vec![], vec![]).unwrap();
    let y = matvec_csr(&m, &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(y, vec![]);
}

// ── matvec_csc ───────────────────────────────────────────────────────────────

#[test]
fn matvec_csc_diagonal_matrix() {
    // diag(1, 2, 3) × [4, 5, 6] = [4, 10, 18]
    let m = CscMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 2.0, 3.0],
    )
    .unwrap();
    let y = matvec_csc(&m, &[4.0, 5.0, 6.0]).unwrap();
    assert_eq!(y, vec![4.0, 10.0, 18.0]);
}

#[test]
fn matvec_csc_general_matrix() {
    // [ 1  2 ]   [ 3 ]   [ 11 ]
    // [ 0  5 ] × [ 4 ] = [ 20 ]
    // CSC: col 0 has [1, 0] at rows [0, 1]; col 1 has [2, 5] at rows [0, 1]
    let m = CscMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0_f64, 0.0, 2.0, 5.0],
    )
    .unwrap();
    let y = matvec_csc(&m, &[3.0, 4.0]).unwrap();
    assert_eq!(y, vec![11.0, 20.0]);
}

#[test]
fn matvec_csc_empty_matrix_returns_zero_vector() {
    let m = CscMatrix::<f64>::new(3, 2, vec![0, 0, 0], vec![], vec![]).unwrap();
    let y = matvec_csc(&m, &[1.0, 2.0]).unwrap();
    assert_eq!(y, vec![0.0, 0.0, 0.0]);
}

#[test]
fn matvec_csc_non_square_matrix() {
    // 2×3 matrix × length-3 vector → length-2 output
    // [ 1  0  2 ]   [ 1 ]   [ 7 ]
    // [ 0  3  0 ] × [ 2 ] = [ 6 ]
    //               [ 3 ]
    let m = CscMatrix::new(
        2,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 0],
        vec![1.0_f64, 3.0, 2.0],
    )
    .unwrap();
    let y = matvec_csc(&m, &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(y, vec![7.0, 6.0]);
}

#[test]
fn matvec_csc_empty_columns_produce_zero_entries() {
    // Column 1 has no entries; rows affected by it contribute zero.
    // [ 4  0 ]   [ 1 ]   [  4 ]
    // [ 0  0 ] × [ 2 ] = [  0 ]
    // [ 6  0 ]           [  6 ]
    let m = CscMatrix::new(3, 2, vec![0, 2, 2], vec![0, 2], vec![4.0_f64, 6.0]).unwrap();
    let y = matvec_csc(&m, &[1.0, 2.0]).unwrap();
    assert_eq!(y, vec![4.0, 0.0, 6.0]);
}

#[test]
fn matvec_csc_dimension_mismatch_returns_error() {
    let m = CscMatrix::new(2, 3, vec![0, 1, 1, 1], vec![0], vec![1.0_f64]).unwrap();
    assert_eq!(matvec_csc(&m, &[1.0, 2.0]), Err(DimensionMismatch));
}

#[test]
fn matvec_csc_zero_row_matrix() {
    let m = CscMatrix::<f64>::new(0, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let y = matvec_csc(&m, &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(y, vec![]);
}

#[test]
fn matvec_csc_single_column() {
    // 3×1 matrix: column vector
    // [ 2 ]   [ 5 ]   [ 10 ]
    // [ 3 ] × [ 1 ] = [ 3  ]
    // [ 1 ]           [ 1  ]
    let m = CscMatrix::new(3, 1, vec![0, 3], vec![0, 1, 2], vec![2.0_f64, 3.0, 1.0]).unwrap();
    let y = matvec_csc(&m, &[5.0]).unwrap();
    assert_eq!(y, vec![10.0, 15.0, 5.0]);
}

#[test]
fn matvec_csc_tall_matrix() {
    // 5×2 matrix (more rows than columns)
    // [ 1  0 ]   [ 2 ]   [  2 ]
    // [ 0  3 ] × [ 4 ] = [ 12 ]
    // [ 2  0 ]           [  4 ]
    // [ 0  1 ]           [  4 ]
    // [ 1  1 ]           [  6 ]
    let m = CscMatrix::new(
        5,
        2,
        vec![0, 3, 6],
        vec![0, 2, 4, 1, 3, 4],
        vec![1.0_f64, 2.0, 1.0, 3.0, 1.0, 1.0],
    )
    .unwrap();
    let y = matvec_csc(&m, &[2.0, 4.0]).unwrap();
    assert_eq!(y, vec![2.0, 12.0, 4.0, 4.0, 6.0]);
}

#[test]
fn matvec_csc_wide_matrix() {
    // 2×4 matrix (more columns than rows)
    // [ 1  2  0  1 ]   [ 1 ]   [ 6 ]
    // [ 3  0  1  0 ] × [ 2 ] = [ 6 ]
    //                   [ 3 ]
    //                   [ 1 ]
    let m = CscMatrix::new(
        2,
        4,
        vec![0, 2, 3, 4, 5],
        vec![0, 1, 0, 1, 0],
        vec![1.0_f64, 3.0, 2.0, 1.0, 1.0],
    )
    .unwrap();
    let y = matvec_csc(&m, &[1.0, 2.0, 3.0, 1.0]).unwrap();
    assert_eq!(y, vec![6.0, 6.0]);
}

#[test]
fn matvec_csc_sparse_pattern() {
    // 4×4 matrix with sparse entries
    // [ 5  0  0  0 ]   [ 1 ]   [  5 ]
    // [ 0  0  0  3 ] × [ 2 ] = [ 12 ]
    // [ 0  7  0  0 ]   [ 2 ]   [ 14 ]
    // [ 0  0  2  0 ]   [ 4 ]   [  4 ]
    let m = CscMatrix::new(
        4,
        4,
        vec![0, 1, 2, 3, 4],
        vec![0, 2, 3, 1],
        vec![5.0_f64, 7.0, 2.0, 3.0],
    )
    .unwrap();
    let y = matvec_csc(&m, &[1.0, 2.0, 2.0, 4.0]).unwrap();
    assert_eq!(y, vec![5.0, 12.0, 14.0, 4.0]);
}

#[test]
fn matvec_csc_dense_pattern() {
    // 3×3 matrix with more entries (denser pattern)
    // [ 1  2  3 ]   [ 2 ]   [ 11 ]
    // [ 4  5  6 ] × [ 3 ] = [ 29 ]
    // [ 7  8  9 ]   [ 1 ]   [ 47 ]
    let m = CscMatrix::new(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![1.0_f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
    )
    .unwrap();
    let y = matvec_csc(&m, &[2.0, 3.0, 1.0]).unwrap();
    assert_eq!(y, vec![11.0, 29.0, 47.0]);
}

#[test]
fn matvec_csc_all_negative_values() {
    // 2×2 matrix with negative entries
    // [ -1  -2 ]   [ 3 ]   [ -7  ]
    // [ -3  -4 ] × [ 2 ] = [ -17 ]
    let m = CscMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![-1.0_f64, -3.0, -2.0, -4.0],
    )
    .unwrap();
    let y = matvec_csc(&m, &[3.0, 2.0]).unwrap();
    assert_eq!(y, vec![-7.0, -17.0]);
}

#[test]
fn matvec_csc_single_row_matrix() {
    // 1×4 matrix (row vector)
    // [ 1  2  3  4 ] × [ 1, 2, 3, 4 ]^T = 30
    let m = CscMatrix::new(
        1,
        4,
        vec![0, 1, 2, 3, 4],
        vec![0, 0, 0, 0],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let y = matvec_csc(&m, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(y, vec![30.0]);
}

#[test]
fn matvec_csc_zero_column_matrix() {
    // 3×3 matrix where all columns are empty
    let m = CscMatrix::<f64>::new(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let y = matvec_csc(&m, &[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(y, vec![0.0, 0.0, 0.0]);
}

#[test]
fn matvec_csc_fractional_values() {
    // 2×2 matrix with fractional values
    // [ 0.5  0.25 ]   [ 4 ]   [ 2.5 ]
    // [ 0.75 0.1  ] × [ 2 ] = [ 3.2 ]
    let m = CscMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![0.5_f64, 0.75, 0.25, 0.1],
    )
    .unwrap();
    let y = matvec_csc(&m, &[4.0, 2.0]).unwrap();
    assert_eq!(y[0], 2.5);
    assert_eq!(y[1], 3.2);
}

#[test]
fn matvec_csc_larger_matrix() {
    // 5×5 diagonal sparse matrix
    let m = CscMatrix::new(
        5,
        5,
        vec![0, 1, 2, 3, 4, 5],
        vec![0, 1, 2, 3, 4],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap();
    let y = matvec_csc(&m, &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    assert_eq!(y, vec![1.0, 4.0, 9.0, 16.0, 25.0]);
}

// ── cross-format consistency ─────────────────────────────────────────────────

#[test]
fn matvec_csr_and_csc_agree_simple_diagonal() {
    // Both formats should produce identical results for a diagonal matrix.
    // [ 1  0  0 ]   [ 2 ]   [ 2 ]
    // [ 0  3  0 ] × [ 4 ] = [ 12 ]
    // [ 0  0  5 ]   [ 1 ]   [ 5 ]
    let x = [2.0_f64, 4.0, 1.0];

    let csr = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 3.0, 5.0],
    )
    .unwrap();

    let csc = CscMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 3.0, 5.0],
    )
    .unwrap();

    let y_csr = matvec_csr(&csr, &x).unwrap();
    let y_csc = matvec_csc(&csc, &x).unwrap();

    assert_eq!(y_csr, vec![2.0, 12.0, 5.0]);
    assert_eq!(y_csc, vec![2.0, 12.0, 5.0]);
    assert_eq!(y_csr, y_csc);
}

#[test]
fn matvec_csr_and_csc_agree_on_general_matrix() {
    // Test a 3×3 general sparse matrix in both CSR and CSC formats.
    // [ 4  0  1 ]
    // [ 0  2  0 ]
    // [ 3  0  5 ]
    let x = [1.0_f64, 2.0, 3.0];

    let csr = CsrMatrix::new(
        3,
        3,
        vec![0, 2, 3, 5],
        vec![0, 2, 1, 0, 2],
        vec![4.0_f64, 1.0, 2.0, 3.0, 5.0],
    )
    .unwrap();

    let csc = CscMatrix::new(
        3,
        3,
        vec![0, 2, 3, 5],
        vec![0, 2, 1, 0, 2],
        vec![4.0_f64, 3.0, 2.0, 1.0, 5.0],
    )
    .unwrap();

    let y_csr = matvec_csr(&csr, &x).unwrap();
    let y_csc = matvec_csc(&csc, &x).unwrap();

    // expected: [4*1 + 1*3, 2*2, 3*1 + 5*3] = [7, 4, 18]
    assert_eq!(y_csr, vec![7.0, 4.0, 18.0]);
    assert_eq!(y_csc, vec![7.0, 4.0, 18.0]);
    assert_eq!(y_csr, y_csc);
}

#[test]
fn matvec_csr_and_csc_agree_on_non_square() {
    // Test a non-square (tall) matrix: 4×3
    // [ 1  2  0 ]   [ 2 ]   [ 6  ]
    // [ 0  3  0 ] × [ 1 ] = [ 3  ]
    // [ 1  0  1 ]   [ 4 ]   [ 6  ]
    // [ 0  0  2 ]           [ 8  ]
    let x = [2.0_f64, 1.0, 4.0];

    let csr = CsrMatrix::new(
        4,
        3,
        vec![0, 2, 3, 5, 6],
        vec![0, 1, 1, 0, 2, 2],
        vec![1.0_f64, 2.0, 3.0, 1.0, 1.0, 2.0],
    )
    .unwrap();

    let csc = CscMatrix::new(
        4,
        3,
        vec![0, 2, 4, 6],
        vec![0, 2, 0, 1, 2, 3],
        vec![1.0_f64, 1.0, 2.0, 3.0, 1.0, 2.0],
    )
    .unwrap();

    let y_csr = matvec_csr(&csr, &x).unwrap();
    let y_csc = matvec_csc(&csc, &x).unwrap();

    assert_eq!(y_csr, vec![4.0, 3.0, 6.0, 8.0]);
    assert_eq!(y_csc, vec![4.0, 3.0, 6.0, 8.0]);
    assert_eq!(y_csr, y_csc);
}

#[test]
fn matvec_csr_and_csc_agree_on_wide_matrix() {
    // Test a wide matrix: 2×4
    // [ 1  0  2  0 ]   [ 1 ]   [ 7  ]
    // [ 0  3  0  1 ] × [ 2 ] = [ 10 ]
    //                   [ 3 ]
    //                   [ 4 ]
    let x = [1.0_f64, 2.0, 3.0, 4.0];

    let csr = CsrMatrix::new(
        2,
        4,
        vec![0, 2, 4],
        vec![0, 2, 1, 3],
        vec![1.0_f64, 2.0, 3.0, 1.0],
    )
    .unwrap();

    let csc = CscMatrix::new(
        2,
        4,
        vec![0, 1, 2, 3, 4],
        vec![0, 1, 0, 1],
        vec![1.0_f64, 3.0, 2.0, 1.0],
    )
    .unwrap();

    let y_csr = matvec_csr(&csr, &x).unwrap();
    let y_csc = matvec_csc(&csc, &x).unwrap();

    assert_eq!(y_csr, vec![7.0, 10.0]);
    assert_eq!(y_csc, vec![7.0, 10.0]);
    assert_eq!(y_csr, y_csc);
}

#[test]
fn matvec_csr_and_csc_agree_with_sparse_pattern() {
    // Test with a highly sparse pattern
    // [ 5  0  0  0 ]   [ 1 ]   [ 5 ]
    // [ 0  0  0  3 ] × [ 2 ] = [ 6 ]
    // [ 0  7  0  0 ]   [ 3 ]   [ 14 ]
    // [ 0  0  2  0 ]   [ 2 ]   [ 4 ]
    let x = [1.0_f64, 2.0, 3.0, 2.0];

    let csr = CsrMatrix::new(
        4,
        4,
        vec![0, 1, 2, 3, 4],
        vec![0, 3, 1, 2],
        vec![5.0_f64, 3.0, 7.0, 2.0],
    )
    .unwrap();

    let csc = CscMatrix::new(
        4,
        4,
        vec![0, 1, 2, 3, 4],
        vec![0, 2, 3, 1],
        vec![5.0_f64, 7.0, 2.0, 3.0],
    )
    .unwrap();

    let y_csr = matvec_csr(&csr, &x).unwrap();
    let y_csc = matvec_csc(&csc, &x).unwrap();

    assert_eq!(y_csr, vec![5.0, 6.0, 14.0, 6.0]);
    assert_eq!(y_csc, vec![5.0, 6.0, 14.0, 6.0]);
    assert_eq!(y_csr, y_csc);
}

#[test]
fn matvec_csr_and_csc_agree_with_zeros_in_vector() {
    let x = [0.0_f64, 0.0, 0.0];

    let csr = CsrMatrix::new(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();

    let csc = CscMatrix::new(
        3,
        3,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![1.0_f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0],
    )
    .unwrap();

    let y_csr = matvec_csr(&csr, &x).unwrap();
    let y_csc = matvec_csc(&csc, &x).unwrap();

    assert_eq!(y_csr, vec![0.0, 0.0, 0.0]);
    assert_eq!(y_csc, vec![0.0, 0.0, 0.0]);
    assert_eq!(y_csr, y_csc);
}

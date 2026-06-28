use rustebra::sparse::{CscMatrix, CsrMatrix, DimensionMismatch, add_csc, add_csr};

#[test]
fn add_csr_disjoint_rows() {
    // a has row 0, b has row 1.
    let a = CsrMatrix::new(2, 2, vec![0, 1, 1], vec![0], vec![3.0_f64]).unwrap();
    let b = CsrMatrix::new(2, 2, vec![0, 0, 1], vec![1], vec![7.0_f64]).unwrap();
    let c = add_csr(&a, &b).unwrap();
    assert_eq!(c.row_ptr(), &[0, 1, 2]);
    assert_eq!(c.col_indices(), &[0, 1]);
    assert_eq!(c.values(), &[3.0, 7.0]);
}

#[test]
fn add_csr_same_positions_are_summed() {
    let a = CsrMatrix::new(1, 3, vec![0, 2], vec![0, 2], vec![1.0_f64, 2.0]).unwrap();
    let b = CsrMatrix::new(1, 3, vec![0, 2], vec![0, 2], vec![10.0_f64, 20.0]).unwrap();
    let c = add_csr(&a, &b).unwrap();
    assert_eq!(c.row_ptr(), &[0, 2]);
    assert_eq!(c.col_indices(), &[0, 2]);
    assert_eq!(c.values(), &[11.0, 22.0]);
}

#[test]
fn add_csr_merged_and_sorted_within_row() {
    // a: col 1; b: col 0.  Output should be sorted: col 0, col 1.
    let a = CsrMatrix::new(1, 3, vec![0, 1], vec![1], vec![5.0_f64]).unwrap();
    let b = CsrMatrix::new(1, 3, vec![0, 1], vec![0], vec![3.0_f64]).unwrap();
    let c = add_csr(&a, &b).unwrap();
    assert_eq!(c.col_indices(), &[0, 1]);
    assert_eq!(c.values(), &[3.0, 5.0]);
}

#[test]
fn add_csr_shape_mismatch_is_an_error() {
    let a = CsrMatrix::<f64>::new(2, 2, vec![0, 0, 0], vec![], vec![]).unwrap();
    let b = CsrMatrix::<f64>::new(2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
    assert_eq!(add_csr(&a, &b), Err(DimensionMismatch));
}

#[test]
fn add_csr_both_empty() {
    let a = CsrMatrix::<f64>::new(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let b = CsrMatrix::<f64>::new(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let c = add_csr(&a, &b).unwrap();
    assert_eq!(c.nnz(), 0);
    assert_eq!(c.row_ptr(), &[0, 0, 0, 0]);
}

#[test]
fn add_csc_same_positions_are_summed() {
    let a = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 5.0]).unwrap();
    let b = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 4.0]).unwrap();
    let c = add_csc(&a, &b).unwrap();
    assert_eq!(c.col_ptr(), &[0, 1, 2]);
    assert_eq!(c.row_indices(), &[0, 1]);
    assert_eq!(c.values(), &[3.0, 9.0]);
}

#[test]
fn add_csc_disjoint_rows_within_column() {
    // Column 0: a has row 0, b has row 1.
    let a = CscMatrix::new(2, 1, vec![0, 1], vec![0], vec![7.0_f64]).unwrap();
    let b = CscMatrix::new(2, 1, vec![0, 1], vec![1], vec![3.0_f64]).unwrap();
    let c = add_csc(&a, &b).unwrap();
    assert_eq!(c.col_ptr(), &[0, 2]);
    assert_eq!(c.row_indices(), &[0, 1]);
    assert_eq!(c.values(), &[7.0, 3.0]);
}

#[test]
fn add_csc_shape_mismatch_is_an_error() {
    let a = CscMatrix::<f64>::new(2, 2, vec![0, 0, 0], vec![], vec![]).unwrap();
    let b = CscMatrix::<f64>::new(3, 2, vec![0, 0, 0], vec![], vec![]).unwrap();
    assert_eq!(add_csc(&a, &b), Err(DimensionMismatch));
}

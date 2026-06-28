use rustebra::sparse::{CsrError, CsrMatrix};

#[test]
fn csr_construction_and_accessors() {
    let m = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 2.0, 3.0],
    )
    .unwrap();

    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 3);
    assert_eq!(m.nnz(), 3);
    assert_eq!(m.row_ptr(), &[0, 1, 2, 3]);
    assert_eq!(m.col_indices(), &[0, 1, 2]);
    assert_eq!(m.values(), &[1.0, 2.0, 3.0]);
}

#[test]
fn csr_empty_matrix_is_valid() {
    let m = CsrMatrix::<f64>::new(100, 200, vec![0; 101], vec![], vec![]).unwrap();
    assert_eq!(m.rows(), 100);
    assert_eq!(m.cols(), 200);
    assert_eq!(m.nnz(), 0);
}

#[test]
fn csr_row_range_returns_correct_slices() {
    // Row 0: empty, row 1: one entry, row 2: two entries.
    let m = CsrMatrix::new(
        3,
        3,
        vec![0, 0, 1, 3],
        vec![2, 0, 1],
        vec![5.0_f64, 3.0, 9.0],
    )
    .unwrap();
    assert_eq!(m.row_range(0), Some(0..0));
    assert_eq!(m.row_range(1), Some(0..1));
    assert_eq!(m.row_range(2), Some(1..3));
    assert_eq!(m.row_range(3), None);
}

#[test]
fn csr_length_mismatch_is_an_error_not_a_panic() {
    assert_eq!(
        CsrMatrix::<f64>::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0]),
        Err(CsrError::LengthMismatch),
    );
}

#[test]
fn csr_row_ptr_length_mismatch_is_an_error_not_a_panic() {
    assert_eq!(
        CsrMatrix::<f64>::new(3, 3, vec![0, 1], vec![], vec![]),
        Err(CsrError::RowPtrLengthMismatch),
    );
}

#[test]
fn csr_row_ptr_invalid_start_is_an_error_not_a_panic() {
    assert_eq!(
        CsrMatrix::<f64>::new(2, 2, vec![1, 1, 1], vec![], vec![]),
        Err(CsrError::RowPtrInvalid),
    );
}

#[test]
fn csr_row_ptr_not_monotone_is_an_error_not_a_panic() {
    assert_eq!(
        CsrMatrix::<f64>::new(3, 3, vec![0, 2, 1, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]),
        Err(CsrError::RowPtrInvalid),
    );
}

#[test]
fn csr_row_ptr_last_entry_mismatch_is_an_error_not_a_panic() {
    assert_eq!(
        CsrMatrix::<f64>::new(2, 2, vec![0, 1, 99], vec![0, 1], vec![1.0, 2.0]),
        Err(CsrError::RowPtrInvalid),
    );
}

#[test]
fn csr_col_index_out_of_bounds_is_an_error_not_a_panic() {
    assert_eq!(
        CsrMatrix::<f64>::new(2, 2, vec![0, 1, 2], vec![0, 5], vec![1.0, 2.0]),
        Err(CsrError::ColIndexOutOfBounds),
    );
}

#[test]
fn csr_partial_eq_holds_for_identical_data() {
    let a = CsrMatrix::new(2, 3, vec![0, 1, 2], vec![2, 0], vec![7.0_f64, 4.0]).unwrap();
    let b = CsrMatrix::new(2, 3, vec![0, 1, 2], vec![2, 0], vec![7.0_f64, 4.0]).unwrap();
    assert_eq!(a, b);
}

#[test]
fn csr_partial_eq_distinguishes_different_values() {
    let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
    let b = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 9.0]).unwrap();
    assert_ne!(a, b);
}

#[test]
fn csr_partial_eq_distinguishes_different_shapes() {
    let a = CsrMatrix::<f64>::new(2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
    let b = CsrMatrix::<f64>::new(3, 2, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    assert_ne!(a, b);
}

#[test]
fn csr_zero_row_matrix_is_valid() {
    let m = CsrMatrix::<f64>::new(0, 5, vec![0], vec![], vec![]).unwrap();
    assert_eq!(m.rows(), 0);
    assert_eq!(m.nnz(), 0);
    assert_eq!(m.row_range(0), None);
}

use rustebra::sparse::{CscError, CscMatrix};

#[test]
fn constructs_empty_csc_matrix() {
    let m = CscMatrix::<f64>::new(3, 4, vec![0, 0, 0, 0, 0], vec![], vec![]).unwrap();
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 4);
    assert_eq!(m.nnz(), 0);
}

#[test]
fn constructs_identity_in_csc() {
    let eye = CscMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 1.0, 1.0],
    )
    .unwrap();
    assert_eq!(eye.nnz(), 3);
    assert_eq!(eye.col_ptr(), &[0, 1, 2, 3]);
    assert_eq!(eye.row_indices(), &[0, 1, 2]);
    assert_eq!(eye.col_range(0), Some(0..1));
    assert_eq!(eye.col_range(1), Some(1..2));
    assert_eq!(eye.col_range(2), Some(2..3));
    assert_eq!(eye.col_range(3), None);
}

#[test]
fn col_range_out_of_bounds_returns_none() {
    let m = CscMatrix::<f64>::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
    assert_eq!(m.col_range(2), None);
    assert_eq!(m.col_range(100), None);
}

#[test]
fn col_ptr_length_mismatch_is_an_error() {
    let err = CscMatrix::<f64>::new(3, 3, vec![0, 1], vec![], vec![]);
    assert_eq!(err, Err(CscError::ColPtrLengthMismatch));
}

#[test]
fn col_ptr_not_starting_at_zero_is_an_error() {
    let err = CscMatrix::<f64>::new(2, 2, vec![1, 2, 3], vec![], vec![]);
    assert_eq!(err, Err(CscError::ColPtrInvalid));
}

#[test]
fn col_ptr_not_monotone_is_an_error() {
    let err = CscMatrix::<f64>::new(3, 3, vec![0, 2, 1, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
    assert_eq!(err, Err(CscError::ColPtrInvalid));
}

#[test]
fn col_ptr_last_entry_mismatch_is_an_error() {
    let err = CscMatrix::<f64>::new(2, 2, vec![0, 1, 5], vec![0, 1], vec![1.0, 2.0]);
    assert_eq!(err, Err(CscError::ColPtrInvalid));
}

#[test]
fn row_index_out_of_bounds_is_an_error() {
    let err = CscMatrix::<f64>::new(2, 2, vec![0, 1, 2], vec![0, 9], vec![1.0, 2.0]);
    assert_eq!(err, Err(CscError::RowIndexOutOfBounds));
}

#[test]
fn length_mismatch_is_an_error() {
    let err = CscMatrix::<f64>::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0]);
    assert_eq!(err, Err(CscError::LengthMismatch));
}

#[test]
fn partial_eq_compares_all_fields() {
    let a = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
    let b = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
    let c = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![9.0_f64, 2.0]).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn zero_col_matrix_is_valid() {
    let m = CscMatrix::<f64>::new(5, 0, vec![0], vec![], vec![]).unwrap();
    assert_eq!(m.rows(), 5);
    assert_eq!(m.cols(), 0);
    assert_eq!(m.nnz(), 0);
    assert_eq!(m.col_range(0), None);
}

#[test]
fn empty_column_in_middle_is_valid() {
    // Col 0: row 0 → 1.0; col 1: empty; col 2: row 1 → 2.0.
    let m = CscMatrix::new(2, 3, vec![0, 1, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
    assert_eq!(m.col_range(0), Some(0..1));
    assert_eq!(m.col_range(1), Some(1..1));
    assert_eq!(m.col_range(2), Some(1..2));
}

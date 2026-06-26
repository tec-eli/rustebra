use rustebra::sparse::{CooError, CooMatrix};

#[test]
fn coo_construction_and_accessors() {
    let m = CooMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![1.0_f64, 2.0, 3.0]).unwrap();

    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 3);
    assert_eq!(m.nnz(), 3);
    assert_eq!(m.row_indices(), &[0, 1, 2]);
    assert_eq!(m.col_indices(), &[0, 1, 2]);
    assert_eq!(m.values(), &[1.0, 2.0, 3.0]);
}

#[test]
fn coo_empty_matrix_is_valid() {
    let m = CooMatrix::<f64>::new(100, 200, vec![], vec![], vec![]).unwrap();
    assert_eq!(m.rows(), 100);
    assert_eq!(m.cols(), 200);
    assert_eq!(m.nnz(), 0);
}

#[test]
fn coo_length_mismatch_is_an_error_not_a_panic() {
    assert_eq!(
        CooMatrix::<f64>::new(2, 2, vec![0, 1], vec![0], vec![1.0, 2.0]),
        Err(CooError::LengthMismatch),
    );
    assert_eq!(
        CooMatrix::<f64>::new(2, 2, vec![0], vec![0], vec![1.0, 2.0]),
        Err(CooError::LengthMismatch),
    );
}

#[test]
fn coo_row_index_out_of_bounds_is_an_error_not_a_panic() {
    assert_eq!(
        CooMatrix::<f64>::new(2, 2, vec![2], vec![0], vec![1.0]),
        Err(CooError::RowIndexOutOfBounds),
    );
}

#[test]
fn coo_col_index_out_of_bounds_is_an_error_not_a_panic() {
    assert_eq!(
        CooMatrix::<f64>::new(2, 2, vec![0], vec![2], vec![1.0]),
        Err(CooError::ColIndexOutOfBounds),
    );
}

#[test]
fn coo_unsorted_and_duplicate_entries_are_accepted() {
    // Unsorted triplets — COO imposes no ordering requirement.
    let m = CooMatrix::new(3, 3, vec![2, 0, 1], vec![2, 0, 1], vec![9.0_f64, 1.0, 5.0]).unwrap();
    assert_eq!(m.nnz(), 3);

    // Duplicate (row, col) — semantically summed when operated on.
    let dup = CooMatrix::new(2, 2, vec![0, 0], vec![0, 0], vec![3.0_f64, 4.0]).unwrap();
    assert_eq!(dup.nnz(), 2);
}

#[test]
fn coo_partial_eq_holds_for_identical_triplets() {
    let a = CooMatrix::new(2, 3, vec![0, 1], vec![2, 0], vec![7.0_f64, 4.0]).unwrap();
    let b = CooMatrix::new(2, 3, vec![0, 1], vec![2, 0], vec![7.0_f64, 4.0]).unwrap();
    assert_eq!(a, b);
}

#[test]
fn coo_partial_eq_distinguishes_different_values() {
    let a = CooMatrix::new(2, 2, vec![0], vec![0], vec![1.0_f64]).unwrap();
    let b = CooMatrix::new(2, 2, vec![0], vec![0], vec![2.0_f64]).unwrap();
    assert_ne!(a, b);
}

#[test]
fn coo_partial_eq_distinguishes_different_shapes() {
    let a = CooMatrix::<f64>::new(2, 3, vec![], vec![], vec![]).unwrap();
    let b = CooMatrix::<f64>::new(3, 2, vec![], vec![], vec![]).unwrap();
    assert_ne!(a, b);
}

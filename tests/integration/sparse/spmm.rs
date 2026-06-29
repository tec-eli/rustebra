use rustebra::sparse::{CsrMatrix, DimensionMismatch, spmm_csr};

#[test]
fn spmm_identity_times_identity() {
    // I × I = I for 3×3.
    let eye = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 1.0, 1.0],
    )
    .unwrap();
    let eye2 = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 1.0, 1.0],
    )
    .unwrap();
    let c = spmm_csr(&eye, &eye2).unwrap();
    assert_eq!(c.row_ptr(), &[0, 1, 2, 3]);
    assert_eq!(c.col_indices(), &[0, 1, 2]);
    assert_eq!(c.values(), &[1.0, 1.0, 1.0]);
}

#[test]
fn spmm_general_2x2() {
    // [1 0]   [1 2]   [1 2]
    // [0 2] × [3 4] = [6 8]
    let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
    let b = CsrMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let c = spmm_csr(&a, &b).unwrap();
    assert_eq!(c.row_ptr(), &[0, 2, 4]);
    assert_eq!(c.col_indices(), &[0, 1, 0, 1]);
    assert_eq!(c.values(), &[1.0, 2.0, 6.0, 8.0]);
}

#[test]
fn spmm_dimension_mismatch_is_an_error() {
    let a = CsrMatrix::<f64>::new(2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
    let b = CsrMatrix::<f64>::new(2, 2, vec![0, 0, 0], vec![], vec![]).unwrap();
    assert_eq!(spmm_csr(&a, &b), Err(DimensionMismatch));
}

#[test]
fn spmm_empty_sparse_matrices() {
    let a = CsrMatrix::<f64>::new(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let b = CsrMatrix::<f64>::new(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let c = spmm_csr(&a, &b).unwrap();
    assert_eq!(c.nnz(), 0);
    assert_eq!(c.row_ptr(), &[0, 0, 0, 0]);
}

#[test]
fn spmm_non_square() {
    // 2×3 times 3×4 = 2×4.
    // a = [[1,0,2],[0,3,0]], b = identity padded.
    let a = CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
    // b (3×4): I in top-left 3×3, zero last column.
    let b = CsrMatrix::new(
        3,
        4,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 1.0, 1.0],
    )
    .unwrap();
    let c = spmm_csr(&a, &b).unwrap();
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 4);
    // Row 0: a[0,0]*b[0] + a[0,2]*b[2] = 1*e0 + 2*e2 → cols [0,2] vals [1,2]
    // Row 1: a[1,1]*b[1] = 3*e1 → col [1] val [3]
    assert_eq!(c.row_ptr(), &[0, 2, 3]);
    assert_eq!(c.col_indices(), &[0, 2, 1]);
    assert_eq!(c.values(), &[1.0, 2.0, 3.0]);
}

#[test]
fn spmm_result_column_indices_are_sorted() {
    // Ensure output columns are in ascending order within each row.
    // a = I; b has entries in reverse column order.
    let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]).unwrap();
    let b = CsrMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![1, 0, 1, 0],
        vec![2.0_f64, 1.0, 4.0, 3.0],
    )
    .unwrap();
    let c = spmm_csr(&a, &b).unwrap();
    assert_eq!(c.col_indices(), &[0, 1, 0, 1]);
    assert_eq!(c.values(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn spmm_zero_row_gives_zero_output_row() {
    // a has an all-zero row; that row of c must also be empty.
    let a = CsrMatrix::new(3, 2, vec![0, 0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]).unwrap();
    let b = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![5.0_f64, 6.0]).unwrap();
    let c = spmm_csr(&a, &b).unwrap();
    assert_eq!(c.row_ptr()[0], c.row_ptr()[1]); // row 0 is empty in c
}

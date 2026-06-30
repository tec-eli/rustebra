use rustebra::sparse::{CscMatrix, CsrMatrix, matmat_csc, matmat_csr};

#[test]
fn matmat_csr_identity_times_dense() {
    let eye = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]).unwrap();
    // x = [[1, 2], [3, 4]] (row-major 2×2)
    let x = [1.0_f64, 2.0, 3.0, 4.0];
    let y = matmat_csr(&eye, &x, 2).unwrap();
    assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn matmat_csr_diagonal_times_dense() {
    // diag(2, 3) × [[1,0],[0,4]] = [[2,0],[0,12]]
    let d = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0]).unwrap();
    let x = [1.0_f64, 0.0, 0.0, 4.0];
    let y = matmat_csr(&d, &x, 2).unwrap();
    assert_eq!(y, vec![2.0, 0.0, 0.0, 12.0]);
}

#[test]
fn matmat_csr_non_square_sparse_times_dense() {
    // 2×3 sparse times 3×2 dense = 2×2 dense.
    // sparse: [[1, 0, 2], [0, 3, 0]]
    let m = CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
    // x (row-major 3×2): [[1,0],[0,1],[1,0]]
    let x = [1.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0];
    let y = matmat_csr(&m, &x, 2).unwrap();
    // Row 0: 1*[1,0] + 2*[1,0] = [3, 0]
    // Row 1: 3*[0,1]            = [0, 3]
    assert_eq!(y, vec![3.0, 0.0, 0.0, 3.0]);
}

#[test]
fn matmat_csr_shape_mismatch_returns_error() {
    let m = CsrMatrix::new(2, 3, vec![0, 0, 0], vec![], vec![]).unwrap();
    // x only has 4 elements but should have 3 * x_cols = 6 for x_cols=2.
    let x = [0.0_f64; 4];
    let err = matmat_csr(&m, &x, 2);
    assert!(err.is_err());
}

#[test]
fn matmat_csr_zero_x_cols_returns_error() {
    let m = CsrMatrix::<f64>::new(2, 2, vec![0, 0, 0], vec![], vec![]).unwrap();
    let err = matmat_csr(&m, &[], 0);
    assert!(err.is_err());
}

#[test]
fn matmat_csc_identity_times_dense() {
    let eye = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]).unwrap();
    let x = [1.0_f64, 2.0, 3.0, 4.0];
    let y = matmat_csc(&eye, &x, 2).unwrap();
    assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn matmat_csc_diagonal_times_dense() {
    let d = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0]).unwrap();
    let x = [1.0_f64, 0.0, 0.0, 4.0];
    let y = matmat_csc(&d, &x, 2).unwrap();
    assert_eq!(y, vec![2.0, 0.0, 0.0, 12.0]);
}

#[test]
fn matmat_csr_and_csc_agree_on_same_matrix() {
    // [[1, 2], [3, 4]] stored in both formats.
    let csr = CsrMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let csc = CscMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0_f64, 3.0, 2.0, 4.0],
    )
    .unwrap();
    let x = [1.0_f64, 0.0, 0.0, 1.0]; // identity
    let y_csr = matmat_csr(&csr, &x, 2).unwrap();
    let y_csc = matmat_csc(&csc, &x, 2).unwrap();
    assert_eq!(y_csr, y_csc);
}

#[test]
fn matmat_csc_shape_mismatch_returns_error() {
    let m = CscMatrix::new(2, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let x = [0.0_f64; 4]; // should be 3 * x_cols = 6 for x_cols=2
    let err = matmat_csc(&m, &x, 2);
    assert!(err.is_err());
}

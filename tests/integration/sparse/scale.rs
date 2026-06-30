use rustebra::sparse::{CscMatrix, CsrMatrix, scale_csc, scale_csr};

// ── scale_csr ────────────────────────────────────────────────────────────────

#[test]
fn scale_csr_basic() {
    let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.0_f64, 6.0]).unwrap();
    let scaled = scale_csr(m, 0.5);
    assert_eq!(scaled.values(), &[2.0, 3.0]);
}

#[test]
fn scale_csr_preserves_structure() {
    let m = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 1, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 2.0, 3.0],
    )
    .unwrap();
    let scaled = scale_csr(m, 10.0);
    assert_eq!(scaled.rows(), 3);
    assert_eq!(scaled.cols(), 3);
    assert_eq!(scaled.nnz(), 3);
    assert_eq!(scaled.row_ptr(), &[0, 1, 1, 3]);
    assert_eq!(scaled.col_indices(), &[0, 1, 2]);
    assert_eq!(scaled.values(), &[10.0, 20.0, 30.0]);
}

#[test]
fn scale_csr_by_one_is_identity() {
    let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![1, 0], vec![5.0_f64, 8.0]).unwrap();
    let scaled = scale_csr(m, 1.0);
    assert_eq!(scaled.values(), &[5.0, 8.0]);
    assert_eq!(scaled.row_ptr(), &[0, 1, 2]);
    assert_eq!(scaled.col_indices(), &[1, 0]);
}

#[test]
fn scale_csr_empty_matrix() {
    let m = CsrMatrix::<f64>::new(3, 4, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let scaled = scale_csr(m, 99.0);
    assert_eq!(scaled.rows(), 3);
    assert_eq!(scaled.cols(), 4);
    assert_eq!(scaled.nnz(), 0);
    assert_eq!(scaled.row_ptr(), &[0, 0, 0, 0]);
}

#[test]
fn scale_csr_by_zero_yields_zero_values() {
    let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
    let scaled = scale_csr(m, 0.0);
    assert_eq!(scaled.nnz(), 0);
    assert_eq!(scaled.values(), &[]);
}

// ── scale_csc ────────────────────────────────────────────────────────────────

#[test]
fn scale_csc_basic() {
    let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.0_f64, 6.0]).unwrap();
    let scaled = scale_csc(m, 0.5);
    assert_eq!(scaled.values(), &[2.0, 3.0]);
}

#[test]
fn scale_csc_preserves_structure() {
    let m = CscMatrix::new(
        3,
        3,
        vec![0, 1, 1, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 2.0, 3.0],
    )
    .unwrap();
    let scaled = scale_csc(m, 10.0);
    assert_eq!(scaled.rows(), 3);
    assert_eq!(scaled.cols(), 3);
    assert_eq!(scaled.nnz(), 3);
    assert_eq!(scaled.col_ptr(), &[0, 1, 1, 3]);
    assert_eq!(scaled.row_indices(), &[0, 1, 2]);
    assert_eq!(scaled.values(), &[10.0, 20.0, 30.0]);
}

#[test]
fn scale_csc_by_one_is_identity() {
    let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![1, 0], vec![5.0_f64, 8.0]).unwrap();
    let scaled = scale_csc(m, 1.0);
    assert_eq!(scaled.values(), &[5.0, 8.0]);
    assert_eq!(scaled.col_ptr(), &[0, 1, 2]);
    assert_eq!(scaled.row_indices(), &[1, 0]);
}

#[test]
fn scale_csc_empty_matrix() {
    let m = CscMatrix::<f64>::new(4, 5, vec![0, 0, 0, 0, 0, 0], vec![], vec![]).unwrap();
    let scaled = scale_csc(m, 3.0);
    assert_eq!(scaled.rows(), 4);
    assert_eq!(scaled.cols(), 5);
    assert_eq!(scaled.nnz(), 0);
    assert_eq!(scaled.col_ptr(), &[0, 0, 0, 0, 0, 0]);
}

#[test]
fn scale_csc_by_zero_yields_zero_values() {
    let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
    let scaled = scale_csc(m, 0.0);
    assert_eq!(scaled.nnz(), 0);
    assert_eq!(scaled.values(), &[]);
}

#[test]
fn scale_csc_negative_scalar() {
    let m = CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![2.0_f64, -3.0]).unwrap();
    let scaled = scale_csc(m, -1.0);
    assert_eq!(scaled.values(), &[-2.0, 3.0]);
}

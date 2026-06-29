use rustebra::sparse::{CscMatrix, CsrMatrix, csc_to_csr, csr_to_csc};

#[test]
fn csr_to_csc_identity() {
    let csr = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 1.0, 1.0],
    )
    .unwrap();
    let csc = csr_to_csc(csr);
    assert_eq!(csc.col_ptr(), &[0, 1, 2, 3]);
    assert_eq!(csc.row_indices(), &[0, 1, 2]);
    assert_eq!(csc.values(), &[1.0, 1.0, 1.0]);
}

#[test]
fn csc_to_csr_identity() {
    let csc = CscMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 1.0, 1.0],
    )
    .unwrap();
    let csr = csc_to_csr(csc);
    assert_eq!(csr.row_ptr(), &[0, 1, 2, 3]);
    assert_eq!(csr.col_indices(), &[0, 1, 2]);
    assert_eq!(csr.values(), &[1.0, 1.0, 1.0]);
}

#[test]
fn csr_to_csc_general_matrix() {
    // [[1, 2], [3, 4]] in CSR.
    let csr = CsrMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let csc = csr_to_csc(csr);
    // Column 0: rows [0, 1] with values [1, 3].
    // Column 1: rows [0, 1] with values [2, 4].
    assert_eq!(csc.col_ptr(), &[0, 2, 4]);
    assert_eq!(csc.row_indices(), &[0, 1, 0, 1]);
    assert_eq!(csc.values(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn csc_to_csr_general_matrix() {
    // [[1, 2], [3, 4]] in CSC (col-major).
    let csc = CscMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0_f64, 3.0, 2.0, 4.0],
    )
    .unwrap();
    let csr = csc_to_csr(csc);
    assert_eq!(csr.row_ptr(), &[0, 2, 4]);
    assert_eq!(csr.col_indices(), &[0, 1, 0, 1]);
    assert_eq!(csr.values(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn csr_csc_roundtrip() {
    // Convert CSR → CSC → CSR and verify identical result.
    // Entries sorted by (row, col) so the roundtrip preserves order.
    let csr = CsrMatrix::new(
        3,
        4,
        vec![0, 2, 3, 5],
        vec![0, 3, 1, 0, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap();
    let expected_row_ptr = csr.row_ptr().to_vec();
    let expected_col = csr.col_indices().to_vec();
    let expected_val = csr.values().to_vec();
    let csc = csr_to_csc(csr);
    let back = csc_to_csr(csc);
    assert_eq!(back.row_ptr(), expected_row_ptr.as_slice());
    assert_eq!(back.col_indices(), expected_col.as_slice());
    assert_eq!(back.values(), expected_val.as_slice());
}

#[test]
fn csr_to_csc_empty_matrix() {
    let csr = CsrMatrix::<f64>::new(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let csc = csr_to_csc(csr);
    assert_eq!(csc.col_ptr(), &[0, 0, 0, 0]);
    assert_eq!(csc.nnz(), 0);
}

#[test]
fn csc_to_csr_empty_matrix() {
    let csc = CscMatrix::<f64>::new(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
    let csr = csc_to_csr(csc);
    assert_eq!(csr.row_ptr(), &[0, 0, 0, 0]);
    assert_eq!(csr.nnz(), 0);
}

#[test]
fn csr_to_csc_non_square_matrix() {
    // 2×3 matrix with entries at (0,1), (1,0), (1,2).
    let csr = CsrMatrix::new(
        2,
        3,
        vec![0, 1, 3],
        vec![1, 0, 2],
        vec![10.0_f64, 20.0, 30.0],
    )
    .unwrap();
    let csc = csr_to_csc(csr);
    assert_eq!(csc.rows(), 2);
    assert_eq!(csc.cols(), 3);
    // Col 0: row 1 → 20.0; col 1: row 0 → 10.0; col 2: row 1 → 30.0.
    assert_eq!(csc.col_ptr(), &[0, 1, 2, 3]);
    assert_eq!(csc.row_indices(), &[1, 0, 1]);
    assert_eq!(csc.values(), &[20.0, 10.0, 30.0]);
}

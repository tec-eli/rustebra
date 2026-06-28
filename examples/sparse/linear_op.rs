use rustebra::sparse::{CscMatrix, CsrMatrix, SparseLinearOp};

pub(crate) fn run() {
    println!("\n== SparseLinearOp ==");

    let eye_csr =
        CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]).expect("valid CSR");
    let y = eye_csr.apply(&[3.0, 5.0]).expect("dimensions match");
    println!("CsrMatrix::apply (identity): {y:?}");

    let eye_csc =
        CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]).expect("valid CSC");
    let y = eye_csc.apply(&[3.0, 5.0]).expect("dimensions match");
    println!("CscMatrix::apply (identity): {y:?}");
}

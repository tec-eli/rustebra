use rustebra::sparse::{CscMatrix, CsrMatrix, matvec_csc, matvec_csr};

pub(crate) fn run() {
    println!("\n== matvec ==");

    // [2  0]   [1]   [ 2]
    // [0  5] × [3] = [15]
    let m_csr = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 5.0])
        .expect("valid CSR");
    let y = matvec_csr(&m_csr, &[1.0, 3.0]).expect("dimensions match");
    println!("matvec_csr: [2 0; 0 5] × [1, 3] = {y:?}");

    let m_csc = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 5.0])
        .expect("valid CSC");
    let y = matvec_csc(&m_csc, &[1.0, 3.0]).expect("dimensions match");
    println!("matvec_csc: [2 0; 0 5] × [1, 3] = {y:?}");
}

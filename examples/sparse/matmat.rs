use rustebra::sparse::{CscMatrix, CsrMatrix, matmat_csc, matmat_csr};

pub(crate) fn run() {
    println!("\n== matmat ==");

    // [2  0]   [1  0]   [2  0]
    // [0  3] × [0  4] = [0 12]   (x is row-major)
    let x = [1.0_f64, 0.0, 0.0, 4.0];

    let m_csr =
        CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0]).expect("valid CSR");
    let y = matmat_csr(&m_csr, &x, 2).expect("dimensions match");
    println!("matmat_csr: [2 0; 0 3] × [1 0; 0 4] = {y:?}");

    let m_csc =
        CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0]).expect("valid CSC");
    let y = matmat_csc(&m_csc, &x, 2).expect("dimensions match");
    println!("matmat_csc: [2 0; 0 3] × [1 0; 0 4] = {y:?}");
}

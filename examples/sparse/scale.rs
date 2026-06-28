use rustebra::sparse::{CscMatrix, CsrMatrix, scale_csc, scale_csr};

pub(crate) fn run() {
    println!("\n== scale ==");

    let csr = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.0_f64, 6.0])
        .expect("valid CSR");
    let scaled = scale_csr(csr, 0.5);
    println!("scale_csr by 0.5: values = {:?}", scaled.values());

    let csc = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.0_f64, 6.0])
        .expect("valid CSC");
    let scaled_csc = scale_csc(csc, 0.5);
    println!("scale_csc by 0.5: values = {:?}", scaled_csc.values());
}

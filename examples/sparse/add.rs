use rustebra::sparse::{CscMatrix, CsrMatrix, add_csc, add_csr};

pub(crate) fn run() {
    println!("\n== add ==");

    // [2  0]   [1  3]   [3  3]
    // [0  5] + [0  4] = [0  9]
    let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 5.0])
        .expect("valid CSR");
    let b = CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 3.0, 4.0])
        .expect("valid CSR");
    let c = add_csr(&a, &b).expect("shapes match");
    println!("add_csr: col_indices={:?}  values={:?}", c.col_indices(), c.values());

    let a = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 5.0])
        .expect("valid CSC");
    let b = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 4.0])
        .expect("valid CSC");
    let c = add_csc(&a, &b).expect("shapes match");
    println!("add_csc: row_indices={:?}  values={:?}", c.row_indices(), c.values());
}

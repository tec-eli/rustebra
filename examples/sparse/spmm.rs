use rustebra::sparse::{CsrMatrix, spmm_csr};

pub(crate) fn run() {
    println!("\n== spmm ==");

    // [1  0]   [1  2]   [1  2]
    // [0  2] × [3  4] = [6  8]
    let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]).expect("valid CSR");
    let b = CsrMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .expect("valid CSR");
    let c = spmm_csr(&a, &b).expect("inner dimensions match");
    println!("spmm_csr: [1 0; 0 2] × [1 2; 3 4]:");
    println!(
        "  row_ptr={:?}  col_indices={:?}  values={:?}",
        c.row_ptr(),
        c.col_indices(),
        c.values()
    );
}

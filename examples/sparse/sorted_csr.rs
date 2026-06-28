use rustebra::sparse::{CsrMatrix, SortedCsrMatrix};

pub(crate) fn run() {
    println!("\n== SortedCsrMatrix ==");

    // Row 0 is stored as [col 2, col 0] — out of order.
    let m = CsrMatrix::new(
        2,
        3,
        vec![0, 2, 3],
        vec![2, 0, 1],
        vec![9.0_f64, 4.0, 5.0],
    )
    .expect("valid CSR");

    let sorted = SortedCsrMatrix::from_csr(m);
    println!("from_csr (row 0 had cols [2,0], now sorted to [0,2]):");
    println!("  col_indices = {:?}", sorted.col_indices());
    println!("  values      = {:?}", sorted.values());

    // into_inner recovers the underlying CsrMatrix.
    let inner = sorted.into_inner();
    println!("into_inner: nnz={}", inner.nnz());
}

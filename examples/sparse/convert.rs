use rustebra::sparse::{
    CooMatrix, CscMatrix, CsrMatrix, SortedCsrMatrix, coo_to_csr, csc_to_csr, csr_to_coo,
    csr_to_csc,
};

pub(crate) fn run() {
    println!("\n== convert ==");

    // coo_to_csr: duplicate (row, col) entries are summed; columns sorted per row.
    let coo = CooMatrix::new(2, 2, vec![0, 0, 1], vec![0, 0, 1], vec![3.0_f64, 4.0, 5.0])
        .expect("valid COO");
    let csr: SortedCsrMatrix<f64> = coo_to_csr(coo);
    println!("coo_to_csr (duplicate (0,0) summed: 3+4=7):");
    println!(
        "  row_ptr={:?}  col_indices={:?}  values={:?}",
        csr.row_ptr(),
        csr.col_indices(),
        csr.values()
    );

    // csr_to_coo: expands row_ptr into explicit row indices.
    let csr_eye = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 1.0, 1.0],
    )
    .expect("valid CSR");
    let coo_out = csr_to_coo(csr_eye);
    println!(
        "\ncsr_to_coo (3×3 identity): row_indices={:?}",
        coo_out.row_indices()
    );

    // csr_to_csc: transposes storage layout, sorted by (col, row).
    let csr_rect = CsrMatrix::new(
        2,
        3,
        vec![0, 2, 4],
        vec![0, 2, 1, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .expect("valid 2×3 CSR");
    let csc_out = csr_to_csc(csr_rect);
    println!("\ncsr_to_csc (2×3):");
    println!(
        "  col_ptr={:?}  row_indices={:?}  values={:?}",
        csc_out.col_ptr(),
        csc_out.row_indices(),
        csc_out.values()
    );

    // csc_to_csr: transposes back, sorted by (row, col).
    let csc_eye = CscMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 1.0, 1.0],
    )
    .expect("valid CSC");
    let csr_back = csc_to_csr(csc_eye);
    println!(
        "\ncsc_to_csr (3×3 identity): row_ptr={:?}",
        csr_back.row_ptr()
    );
}

use rustebra::sparse::CsrMatrix;

pub(crate) fn run() {
    println!("\n== CsrMatrix ==");

    // row_ptr has rows+1 entries; row_ptr[i]..row_ptr[i+1] spans row i.
    let eye = CsrMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 1.0, 1.0],
    )
    .expect("valid CSR arrays");

    println!("3×3 identity:");
    println!(
        "  rows={}, cols={}, nnz={}",
        eye.rows(),
        eye.cols(),
        eye.nnz()
    );
    println!("  row_ptr     = {:?}", eye.row_ptr());
    println!("  col_indices = {:?}", eye.col_indices());
    println!("  values      = {:?}", eye.values());
    println!("  row_range(1) = {:?}", eye.row_range(1));
}

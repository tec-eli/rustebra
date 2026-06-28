use rustebra::sparse::CscMatrix;

pub(crate) fn run() {
    println!("\n== CscMatrix ==");

    // col_ptr has cols+1 entries; col_ptr[j]..col_ptr[j+1] spans column j.
    let eye = CscMatrix::new(
        3,
        3,
        vec![0, 1, 2, 3],
        vec![0, 1, 2],
        vec![1.0_f64, 1.0, 1.0],
    )
    .expect("valid CSC arrays");

    println!("3×3 identity:");
    println!("  rows={}, cols={}, nnz={}", eye.rows(), eye.cols(), eye.nnz());
    println!("  col_ptr     = {:?}", eye.col_ptr());
    println!("  row_indices = {:?}", eye.row_indices());
    println!("  values      = {:?}", eye.values());
    println!("  col_range(2) = {:?}", eye.col_range(2));
}

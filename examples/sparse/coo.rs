use rustebra::sparse::CooMatrix;

pub(crate) fn run() {
    println!("\n== CooMatrix ==");

    let eye = CooMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![1.0_f64, 1.0, 1.0])
        .expect("valid COO triplets");

    println!("3×3 identity:");
    println!(
        "  rows={}, cols={}, nnz={}",
        eye.rows(),
        eye.cols(),
        eye.nnz()
    );
    println!("  row_indices = {:?}", eye.row_indices());
    println!("  col_indices = {:?}", eye.col_indices());
    println!("  values      = {:?}", eye.values());

    // Duplicate (row, col) entries are legal — they are summed on conversion.
    let dup = CooMatrix::new(2, 2, vec![0, 0, 1], vec![0, 0, 1], vec![3.0_f64, 4.0, 5.0])
        .expect("valid COO with duplicate at (0,0)");
    println!(
        "with duplicate at (0,0): nnz={} (summed on coo_to_csr)",
        dup.nnz()
    );
}

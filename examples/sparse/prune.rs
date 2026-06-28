use rustebra::sparse::{CsrMatrix, prune_csr};

pub(crate) fn run() {
    println!("\n== prune ==");

    // Entries within [-tolerance, tolerance] are dropped.
    let m = CsrMatrix::new(
        1,
        3,
        vec![0, 3],
        vec![0, 1, 2],
        vec![1e-15_f64, 2.0, -1e-15],
    )
    .expect("valid CSR");
    let pruned = prune_csr(m, 1e-10);
    println!("prune_csr (tolerance=1e-10, two near-zeros removed):");
    println!("  nnz={}", pruned.nnz());
    println!(
        "  col_indices={:?}  values={:?}",
        pruned.col_indices(),
        pruned.values()
    );
}

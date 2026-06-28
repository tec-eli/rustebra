use rustebra::algorithm::matrix::{cholesky, cholesky_decompose};
use rustebra::storage::StaticStorage;

pub(crate) fn run() {
    println!("\n== Cholesky decomposition ==");
    // [[4, 2], [2, 2]], symmetric positive-definite.
    let a = StaticStorage::new([4.0, 2.0, 2.0, 2.0]);

    let mut l = [0.0; 4];
    cholesky(&a, 2, 2, &mut l).unwrap();
    println!("l = {l:?}");

    let mut l_explicit = [0.0; 4];
    cholesky_decompose(&a, 2, 2, &mut l_explicit, 1e-9).unwrap();
    println!("l (explicit) = {l_explicit:?}");
}

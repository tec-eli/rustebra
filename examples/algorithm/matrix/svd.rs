use rustebra::algorithm::matrix::{svd, svd_qr_iteration};
use rustebra::storage::StaticStorage;

pub(crate) fn run() {
    println!("\n== Singular Value Decomposition (SVD) ==");
    // [[2, 0], [0, 1]].
    let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
    let mut scratch = [0.0; 5 * 2 * 2 + 2 + 2];

    let mut u = [0.0; 4];
    let mut sigma = [0.0; 2];
    let mut v = [0.0; 4];
    svd(&a, 2, 2, &mut u, &mut sigma, &mut v, &mut scratch).unwrap();
    println!("sigma = {sigma:?}, u = {u:?}, v = {v:?}");

    let mut sigma_explicit = [0.0; 2];
    svd_qr_iteration(&a, 2, 2, &mut u, &mut sigma_explicit, &mut v, &mut scratch).unwrap();
    println!("sigma (explicit QR iteration) = {sigma_explicit:?}");
}

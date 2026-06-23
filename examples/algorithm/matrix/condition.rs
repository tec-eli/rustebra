use rustebra::algorithm::matrix::{condition_number, condition_number_svd};
use rustebra::storage::StaticStorage;

pub(crate) fn run() {
    println!("\n== Condition number ==");
    // [[100, 0], [0, 1]]: ill-conditioned, kappa = 100.
    let a = StaticStorage::new([100.0, 0.0, 0.0, 1.0]);
    let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

    println!(
        "kappa(a) = {:?}",
        condition_number(&a, 2, 2, &mut scratch).unwrap()
    );
    println!(
        "kappa(a) via SVD (explicit) = {:?}",
        condition_number_svd(&a, 2, 2, &mut scratch).unwrap()
    );
}

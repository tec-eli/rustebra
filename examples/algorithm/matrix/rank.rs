use rustebra::algorithm::matrix::rank;
use rustebra::storage::StaticStorage;

pub(crate) fn run() {
    println!("\n== rank ==");
    // [[1, 2], [2, 4]]; row 1 is twice row 0, so this is rank-deficient.
    let a = StaticStorage::new([1.0, 2.0, 2.0, 4.0]);
    let mut scratch = [0.0; 4];

    println!("rank(a) = {:?}", rank(&a, 2, 2, &mut scratch).unwrap());
}

use rustebra::algorithm::matrix::{lu, lu_partial_pivot};
use rustebra::storage::StaticStorage;

pub(crate) fn run() {
    println!("\n== LU decomposition ==");
    // [[4, 3], [6, 3]].
    let a = StaticStorage::new([4.0, 3.0, 6.0, 3.0]);
    let mut l = [0.0; 4];
    let mut u = [0.0; 4];

    let swaps = lu(&a, 2, 2, &mut l, &mut u).unwrap();
    println!("l = {l:?}, u = {u:?} (row swaps: {swaps})");

    let mut l_explicit = [0.0; 4];
    let mut u_explicit = [0.0; 4];
    lu_partial_pivot(&a, 2, 2, &mut l_explicit, &mut u_explicit).unwrap();
    println!("l (explicit partial pivot) = {l_explicit:?}, u = {u_explicit:?}");
}

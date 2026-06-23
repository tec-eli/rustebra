use rustebra::algorithm::matrix::{determinant, determinant_cofactor, determinant_lu};
use rustebra::storage::StaticStorage;

pub(crate) fn run() {
    println!("\n== determinant ==");
    // [[1, 2, 3], [4, 5, 6], [7, 8, 10]].
    let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);

    println!("det(a) = {:?}", determinant(&a, 3, 3).unwrap());
    println!(
        "det(a) via cofactor expansion (explicit) = {:?}",
        determinant_cofactor(&a, 3, 3).unwrap()
    );

    let mut scratch = [0.0; 2 * 3 * 3];
    println!(
        "det(a) via LU decomposition (explicit) = {:?}",
        determinant_lu(&a, 3, 3, &mut scratch).unwrap()
    );
}

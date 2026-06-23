use rustebra::algorithm::matrix::{add, mul_matrix, mul_scalar, mul_vector, sub, transpose};
use rustebra::storage::StaticStorage;

pub(crate) fn run() {
    println!("== arithmetic ==");
    // Row-major 2x2 matrices: [[1, 2], [3, 4]] and [[5, 6], [7, 8]].
    let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
    let b = StaticStorage::new([5.0, 6.0, 7.0, 8.0]);

    let mut sum = [0.0; 4];
    add(&a, 2, 2, &b, 2, 2, &mut sum).unwrap();
    println!("a + b = {sum:?}");

    let mut diff = [0.0; 4];
    sub(&a, 2, 2, &b, 2, 2, &mut diff).unwrap();
    println!("a - b = {diff:?}");

    let mut scaled = [0.0; 4];
    mul_scalar(&a, 2, 2, 2.0, &mut scaled).unwrap();
    println!("a * 2 = {scaled:?}");

    let v = StaticStorage::new([1.0, 1.0]);
    let mut product_v = [0.0; 2];
    mul_vector(&a, 2, 2, &v, &mut product_v).unwrap();
    println!("a * v = {product_v:?}");

    let mut product_m = [0.0; 4];
    mul_matrix(&a, 2, 2, &b, 2, 2, &mut product_m).unwrap();
    println!("a * b = {product_m:?}");

    let mut transposed = [0.0; 4];
    transpose(&a, 2, 2, &mut transposed).unwrap();
    println!("a^T = {transposed:?}");
}

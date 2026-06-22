//! Construct a `StaticMatrix` and run each matrix operation on it.
//!
//! Run with: `cargo run --example static_matrix`

use rustebra::matrix::StaticMatrix;
use rustebra::vector::StaticVector;

fn main() {
    let a = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let b = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
    let v = StaticVector::new([1.0, 1.0]);

    println!("a = {a:?}");
    println!("b = {b:?}");
    println!("a + b = {:?}", a.add(&b));
    println!("a - b = {:?}", a.sub(&b));
    println!("a scaled by 2 = {:?}", a.mul_scalar(2.0));
    println!("a * v = {:?}", a.mul_vector(&v));
    println!("a * b = {:?}", a.mul_matrix(&b));
    println!("a^T = {:?}", a.transpose());
    println!("det(a) = {:?}", a.determinant());
}

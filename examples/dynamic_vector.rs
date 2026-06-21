//! Construct a `DynamicVector` and run each vector operation on it.
//!
//! Run with: `cargo run --example dynamic_vector --features alloc`

use rustebra::vector::DynamicVector;

fn main() {
    let a = DynamicVector::new(vec![1.0, 2.0, 3.0]);
    let b = DynamicVector::new(vec![4.0, 5.0, 6.0]);

    println!("a = {a:?}");
    println!("b = {b:?}");

    let sum = a.add(&b).expect("a and b have the same length");
    println!("a + b = {sum:?}");

    let diff = a.sub(&b).expect("a and b have the same length");
    println!("a - b = {diff:?}");

    println!("a scaled by 2 = {:?}", a.scale(2.0));

    let dot = a.dot(&b).expect("a and b have the same length");
    println!("a . b = {dot}");

    println!("||a|| = {:.4}", a.norm());
}

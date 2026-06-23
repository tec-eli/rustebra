//! Run each `Scalar` operation on `f64`.
//!
//! Run with: `cargo run --example scalar`

use rustebra::scalar::Scalar;

fn main() {
    println!("f64::zero() = {:?}", f64::zero());
    println!("f64::one() = {:?}", f64::one());
    println!("2.0 + 3.0 = {:?}", 2.0f64.add(3.0));
    println!("5.0 - 3.0 = {:?}", 5.0f64.sub(3.0));
    println!("2.0 * 3.0 = {:?}", 2.0f64.mul(3.0));
    println!("6.0 / 3.0 = {:?}", 6.0f64.div(3.0));
    println!("sqrt(2.0) = {:?}", Scalar::sqrt(2.0f64));
    println!(
        "sin(pi/2) = {:?}",
        Scalar::sin(core::f64::consts::FRAC_PI_2)
    );
    println!("cos(pi) = {:?}", Scalar::cos(core::f64::consts::PI));
}

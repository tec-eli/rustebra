//! Run each `Scalar` operation on `f32` and `f64`.
//!
//! Run with: `cargo run --example scalar`

mod f32;
mod f64;

fn main() {
    f32::run();
    f64::run();
}

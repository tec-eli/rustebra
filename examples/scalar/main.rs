//! Run each `Scalar` operation on `f32` and `f64`: basic arithmetic identities per type
//! (`f32`, `f64`), then `sqrt` across both types.
//!
//! Run with: `cargo run --example scalar`

mod f32;
mod f64;
mod sqrt;

fn main() {
    f32::run();
    f64::run();
    sqrt::run();
}

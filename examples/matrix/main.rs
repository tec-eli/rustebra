//! Construct `StaticMatrix`/`DynamicMatrix` and run each matrix operation on them.
//!
//! Run with: `cargo run --example matrix` (static only), or
//! `cargo run --example matrix --features alloc` (static and dynamic)

mod static_matrix;

#[cfg(feature = "alloc")]
mod dynamic_matrix;

fn main() {
    static_matrix::run();

    #[cfg(feature = "alloc")]
    dynamic_matrix::run();
}

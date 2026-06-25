//! Construct `StaticVector`/`DynamicVector` and run each vector operation on them.
//!
//! Run with: `cargo run --example vector` (static only), or
//! `cargo run --example vector --features alloc` (static and dynamic)

mod static_vector;

#[cfg(feature = "alloc")]
mod dynamic_vector;

fn main() {
    static_vector::run();

    #[cfg(feature = "alloc")]
    dynamic_vector::run();
}

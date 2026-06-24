//! Construct `StaticStorage`/`DynamicStorage` and inspect them through the `Storage` trait.
//!
//! Run with: `cargo run --example storage` (static only), or
//! `cargo run --example storage --features alloc` (static and dynamic)

mod r#static;

#[cfg(feature = "alloc")]
mod dynamic;

fn main() {
    r#static::run();

    #[cfg(feature = "alloc")]
    dynamic::run();
}

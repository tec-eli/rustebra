//! Tour `algorithm::vector` and `algorithm::matrix`'s functions directly, the layer
//! `StaticVector`/`DynamicVector` and `StaticMatrix`/`DynamicMatrix` are built on top of: each
//! function takes `Storage` operands and caller-provided output buffers instead of a
//! vector/matrix type — useful for callers who want explicit control over which algorithm
//! runs, even though every operation here is also reachable as an ergonomic method on
//! `StaticMatrix`/`DynamicMatrix`.
//!
//! Run with: `cargo run --example algorithm`

mod matrix;
mod vector;

fn main() {
    vector::run();
    matrix::run();
}

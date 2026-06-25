//! Tour `algorithm::vector` and `algorithm::matrix`'s functions directly, the layer
//! `StaticVector`/`DynamicVector` and `StaticMatrix`/`DynamicMatrix` are built on top of: each
//! function takes `Storage` operands and caller-provided output buffers instead of a
//! vector/matrix type, and several matrix operations (LU, QR, Cholesky, SVD, condition number)
//! aren't wrapped by either matrix type at all yet.
//!
//! Run with: `cargo run --example algorithm`

mod matrix;
mod vector;

fn main() {
    vector::run();
    matrix::run();
}

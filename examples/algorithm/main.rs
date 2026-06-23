//! Tour `algorithm::matrix`'s functions directly, the layer `StaticMatrix`/`DynamicMatrix`
//! are built on top of: each function takes `Storage` operands and caller-provided output
//! buffers instead of a matrix type, and several operations (LU, QR, Cholesky, SVD, condition
//! number) aren't wrapped by either matrix type at all yet.
//!
//! Run with: `cargo run --example algorithm`

mod matrix;

fn main() {
    matrix::run();
}

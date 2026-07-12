//! Tour `rustebra::krylov`'s subspace methods: Lanczos iteration, which builds an orthonormal
//! basis of a Krylov subspace and the symmetric tridiagonal matrix a symmetric operator
//! projects onto within it.
//!
//! Run with: `cargo run --example krylov`

mod lanczos;

fn main() {
    lanczos::run();
}

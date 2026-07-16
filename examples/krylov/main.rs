//! Tour `rustebra::krylov`'s subspace methods: Lanczos iteration, which builds an orthonormal
//! basis of a Krylov subspace and the symmetric tridiagonal matrix a symmetric operator
//! projects onto within it, and Arnoldi iteration, its non-symmetric counterpart producing an
//! upper Hessenberg projection.
//!
//! Run with: `cargo run --example krylov`

mod arnoldi;
mod lanczos;

fn main() {
    lanczos::run();
    arnoldi::run();
}

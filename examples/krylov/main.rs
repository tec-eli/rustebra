//! Tour `rustebra::krylov`'s subspace methods: Lanczos iteration, which builds an orthonormal
//! basis of a Krylov subspace and the symmetric tridiagonal matrix a symmetric operator
//! projects onto within it, Arnoldi iteration, its non-symmetric counterpart producing an
//! upper Hessenberg projection, and GMRES(m), the restarted linear solver built on top of it.
//!
//! Run with: `cargo run --example krylov` (add `--features alloc` for the GMRES section, since
//! it takes its operator as a `SparseLinearOp`).

mod arnoldi;
#[cfg(feature = "alloc")]
mod gmres;
mod lanczos;

fn main() {
    lanczos::run();
    arnoldi::run();
    #[cfg(feature = "alloc")]
    gmres::run();
}

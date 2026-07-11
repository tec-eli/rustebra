//! Shared helpers for property tests that diff this crate's algorithms against nalgebra.

mod svd;

/// Whether `a` and `b` differ by no more than `tol`.
pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

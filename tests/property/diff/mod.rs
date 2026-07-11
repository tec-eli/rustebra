//! Differential tests: compare this crate's decompositions against `nalgebra`'s reference
//! implementations on random inputs, to catch correctness bugs that unit tests on
//! hand-picked matrices might miss.

mod qr;

/// Returns `true` if `a` and `b` differ by no more than `tol`.
pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

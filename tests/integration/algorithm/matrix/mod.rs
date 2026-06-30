//! Black-box mathematical tests for `algorithm::matrix`'s decompositions and derived
//! quantities (LU, QR, Cholesky, SVD, condition number). Each test checks a property the
//! decomposition is mathematically required to satisfy (orthogonality, triangularity,
//! reconstruction, or agreement with an independently-computed quantity from this crate's
//! own public API) on inputs distinct from the ones already exercised by the unit tests
//! colocated with the implementation, rather than re-asserting specific algorithm output.

mod cholesky;
mod condition;
mod lu;
mod qr;
mod svd;

pub(crate) const TOL: f64 = 1e-9;
pub(crate) const TOL_ITERATIVE: f64 = 1e-6;

pub(crate) fn assert_all_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < tol, "expected {e}, got {a}");
    }
}

pub(crate) fn assert_lower_triangular(m: &[f64], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            assert_eq!(m[i * n + j], 0.0, "({i}, {j}) should be 0");
        }
    }
}

pub(crate) fn assert_unit_lower_triangular(m: &[f64], n: usize) {
    assert_lower_triangular(m, n);
    for i in 0..n {
        assert_eq!(m[i * n + i], 1.0, "({i}, {i}) should be 1");
    }
}

pub(crate) fn assert_upper_triangular(m: &[f64], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols.min(i) {
            assert_eq!(m[i * cols + j], 0.0, "({i}, {j}) should be 0");
        }
    }
}

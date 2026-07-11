//! Differential property test: `svd`'s singular values against nalgebra's independently
//! computed SVD, on matrices built with a known, well-separated spectrum so the fixed
//! 100-sweep QR eigendecomposition behind our algorithm has converged well past this test's
//! comparison tolerance.

use nalgebra::{Matrix4, Vector4};
use proptest::prelude::*;
use rustebra::algorithm::matrix::svd;
use rustebra::storage::StaticStorage;

use super::approx_eq;

/// Side length of every matrix this test generates.
const N: usize = 4;

/// Length of the flat scratch buffer `svd` needs for an `N x N` matrix.
const SCRATCH_LEN: usize = 5 * N * N + N + N;

/// Comparison tolerance for singular values, per the differential test requirement.
const TOL: f64 = 1e-9;

prop_compose! {
    /// Positive, strictly descending singular values with `sigma[i + 1] / sigma[i] <= 0.8`:
    /// a gap tight enough that 100 fixed QR-iteration sweeps converge well past [`TOL`]
    /// (`0.8 ^ 100 ~ 2e-10`), so both this crate's algorithm and nalgebra's should recover
    /// them (and thus each other) within tolerance.
    fn descending_spectrum()(
        top in 1.0..20.0f64,
        ratios in prop::array::uniform3(0.1..0.8f64),
    ) -> [f64; N] {
        let mut sigma = [top, 0.0, 0.0, 0.0];
        for i in 1..N {
            sigma[i] = sigma[i - 1] * ratios[i - 1];
        }
        sigma
    }
}

/// Orthogonal matrix obtained by QR-decomposing the given random entries; Householder QR
/// yields an orthogonal `Q` for any input, so no rejection is needed.
fn orthogonal_from(entries: &[f64; N * N]) -> Matrix4<f64> {
    Matrix4::from_row_slice(entries).qr().q()
}

/// `m` flattened row-major, the layout this crate's `Storage`-backed types expect.
fn row_major(m: &Matrix4<f64>) -> [f64; N * N] {
    let mut out = [0.0; N * N];
    for r in 0..N {
        for c in 0..N {
            out[r * N + c] = m[(r, c)];
        }
    }
    out
}

proptest! {
    /// `A = Q1 * diag(sigma) * Q2ᵗ` for random orthogonal `Q1`, `Q2` has singular values
    /// exactly `sigma` by construction. Checks that this crate's `svd` recovers them within
    /// [`TOL`] of nalgebra's independently computed singular values for the same matrix.
    #[test]
    fn agrees_with_nalgebra_svd(
        sigma in descending_spectrum(),
        q1_seed in prop::array::uniform16(-1.0..1.0f64),
        q2_seed in prop::array::uniform16(-1.0..1.0f64),
    ) {
        let q1 = orthogonal_from(&q1_seed);
        let q2 = orthogonal_from(&q2_seed);
        let d = Matrix4::from_diagonal(&Vector4::from_row_slice(&sigma));
        let a_matrix = q1 * d * q2.transpose();
        let a = row_major(&a_matrix);

        let mut u = [0.0; N * N];
        let mut out_sigma = [0.0; N];
        let mut v = [0.0; N * N];
        let mut scratch = [0.0; SCRATCH_LEN];
        let result = svd(
            &StaticStorage::new(a),
            N,
            N,
            &mut u,
            &mut out_sigma,
            &mut v,
            &mut scratch,
        );
        prop_assert!(result.is_ok(), "svd failed: {:?}", result);

        let oracle = nalgebra::SVD::new(a_matrix, false, false);

        for (i, (&actual, &expected)) in out_sigma.iter().zip(oracle.singular_values.iter()).enumerate() {
            prop_assert!(
                approx_eq(actual, expected, TOL),
                "singular value {} disagrees with nalgebra: {} vs {}",
                i,
                actual,
                expected,
            );
        }
    }
}

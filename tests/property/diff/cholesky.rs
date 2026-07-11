//! Differential oracle: for random symmetric positive-definite matrices, nalgebra's
//! `Cholesky` is an independent implementation to diff every entry of `L` against.

use proptest::prelude::*;
use rustebra::algorithm::matrix::{cholesky_decompose, mul_matrix, transpose};
use rustebra::storage::StaticStorage;

use super::approx_eq;

const N: usize = 4;
const TOL: f64 = 1e-9;

proptest! {
    /// Builds a guaranteed symmetric positive-definite matrix `a = mᵀm + n·I` from random
    /// entries `m`, runs this crate's [`cholesky_decompose`], and checks every entry of the
    /// resulting `L` against nalgebra's `Cholesky::l()` within [`TOL`].
    #[test]
    fn agrees_with_nalgebra_cholesky(
        entries in prop::collection::vec(-10.0..10.0f64, N * N),
    ) {
        let mut m = [0.0; N * N];
        m.copy_from_slice(&entries);

        // `mᵀm` is positive-semi-definite for any `m`; adding `n·I` pushes every eigenvalue
        // strictly positive, so `a` is genuinely positive-definite regardless of `m`.
        let mut m_t = [0.0; N * N];
        transpose(&StaticStorage::new(m), N, N, &mut m_t).unwrap();
        let mut a = [0.0; N * N];
        mul_matrix(&StaticStorage::new(m_t), N, N, &StaticStorage::new(m), N, N, &mut a).unwrap();
        for i in 0..N {
            a[i * N + i] += N as f64;
        }

        let mut l = [0.0; N * N];
        let result = cholesky_decompose(&StaticStorage::new(a), N, N, &mut l, TOL);
        prop_assert!(result.is_ok(), "expected a valid decomposition, got {:?}", result);

        let oracle = nalgebra::Cholesky::new(nalgebra::Matrix4::from_row_slice(&a))
            .expect("a is positive-definite by construction");
        let oracle_l = oracle.l();

        for r in 0..N {
            for c in 0..N {
                let expected = oracle_l[(r, c)];
                let actual = l[r * N + c];
                prop_assert!(
                    approx_eq(actual, expected, TOL),
                    "L[{},{}] = {} disagrees with nalgebra's {}",
                    r,
                    c,
                    actual,
                    expected,
                );
            }
        }
    }
}

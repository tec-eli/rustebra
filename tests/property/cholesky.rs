//! Property tests for [`cholesky`]: unlike `diff::cholesky`'s comparison against `nalgebra`,
//! these check the defining invariants directly (`L` lower triangular, `L * Lᵗ ≈ A`)
//! against positive-definite matrices built with a known, controllable condition number.

use proptest::prelude::*;
use rustebra::algorithm::matrix::{cholesky, mul_matrix, transpose};
use rustebra::storage::StaticStorage;

const N: usize = 4;
const TOL: f64 = 1e-9;

/// Whether the `n x n` matrix `l` (row-major) has every strictly-above-diagonal entry equal
/// to zero exactly, as `cholesky` produces it by construction (it never writes those
/// entries, rather than computing them and cancelling to zero, so no floating-point
/// tolerance is needed here unlike [`crate::qr::is_upper_triangular`]).
fn is_lower_triangular(l: &[f64], n: usize) -> bool {
    for row in 0..n {
        for col in (row + 1)..n {
            if l[row * n + col] != 0.0 {
                return false;
            }
        }
    }
    true
}

/// `a = mᵀm + n·I` from arbitrary `m`: `mᵀm` is positive-semi-definite for any `m`, and
/// adding `n·I` pushes every eigenvalue strictly positive, so `a` is always genuinely
/// positive-definite regardless of `m` (the same construction `diff::cholesky` uses).
fn positive_definite_from(entries: &[f64; N * N]) -> [f64; N * N] {
    let mut m = [0.0_f64; N * N];
    m.copy_from_slice(entries);

    let mut m_t = [0.0_f64; N * N];
    transpose(&StaticStorage::new(m), N, N, &mut m_t).unwrap();
    let mut a = [0.0_f64; N * N];
    mul_matrix(
        &StaticStorage::new(m_t),
        N,
        N,
        &StaticStorage::new(m),
        N,
        N,
        &mut a,
    )
    .unwrap();
    for i in 0..N {
        a[i * N + i] += N as f64;
    }
    a
}

proptest! {
    /// `cholesky` on a random positive-definite matrix produces a lower-triangular `L` whose
    /// `L * Lᵗ` reconstructs the input.
    #[test]
    fn cholesky_of_positive_definite_matrix_is_lower_triangular_and_reconstructs(
        entries in prop::collection::vec(-10.0..10.0f64, N * N),
    ) {
        let mut seed = [0.0_f64; N * N];
        seed.copy_from_slice(&entries);
        let a = positive_definite_from(&seed);

        let mut l = [0.0_f64; N * N];
        cholesky(&StaticStorage::new(a), N, N, &mut l)
            .expect("a is positive-definite by construction");

        prop_assert!(is_lower_triangular(&l, N), "L is not lower triangular: {:?}", l);

        let mut l_t = [0.0_f64; N * N];
        transpose(&StaticStorage::new(l), N, N, &mut l_t).unwrap();
        let mut ll_t = [0.0_f64; N * N];
        mul_matrix(&StaticStorage::new(l), N, N, &StaticStorage::new(l_t), N, N, &mut ll_t).unwrap();
        for (actual, expected) in ll_t.iter().zip(a.iter()) {
            prop_assert!((actual - expected).abs() < TOL);
        }
    }
}

#[test]
fn cholesky_of_ill_conditioned_matrix_still_reconstructs() {
    // Diagonal positive-definite matrix with condition number 1e8 (ratio of largest to
    // smallest diagonal entry): `L` is exactly `sqrt(diag)` here, so this isolates
    // ill-conditioning's effect on accuracy from any other source of error.
    #[rustfmt::skip]
    let a = StaticStorage::new([
        1e8, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]);

    let mut l = [0.0; 16];
    cholesky(&a, 4, 4, &mut l).expect("diagonal matrix with positive entries is positive-definite");

    // Looser tolerance than the well-conditioned property test above: squaring `l[0][0] ~
    // 1e4` back up to `1e8` amplifies its relative rounding error by the same factor.
    let mut l_t = [0.0; 16];
    transpose(&StaticStorage::new(l), 4, 4, &mut l_t).unwrap();
    let mut ll_t = [0.0; 16];
    let l_storage = StaticStorage::new(l);
    mul_matrix(&l_storage, 4, 4, &StaticStorage::new(l_t), 4, 4, &mut ll_t).unwrap();

    #[rustfmt::skip]
    let expected_a: [f64; 16] = [
        1e8, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    for (actual, expected) in ll_t.iter().zip(expected_a.iter()) {
        assert!((actual - expected).abs() < 1e-2);
    }
}

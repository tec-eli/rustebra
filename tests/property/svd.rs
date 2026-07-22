//! Property tests for [`svd`]: unlike `diff::svd`'s comparison against `nalgebra`, these
//! check the defining invariants directly (non-negative descending singular values,
//! orthonormal `U`/`V` columns, reconstruction) on both full-rank and rank-deficient inputs.

use proptest::prelude::*;
use rustebra::algorithm::matrix::{mul_matrix, qr_householder, svd_qr_iteration, transpose};
use rustebra::storage::StaticStorage;

const N: usize = 4;
const SCRATCH_LEN: usize = 5 * N * N + N + N;
const TOL: f64 = 1e-6;

/// Orthogonal matrix obtained by QR-decomposing arbitrary entries: `qr_householder` always
/// produces an orthogonal `Q`, whatever the input, so no rejection sampling is needed.
fn orthogonal_from(entries: [f64; N * N]) -> [f64; N * N] {
    let mut q = [0.0_f64; N * N];
    let mut r = [0.0_f64; N * N];
    let mut scratch = [0.0_f64; N];
    qr_householder(
        &StaticStorage::new(entries),
        N,
        N,
        &mut q,
        &mut r,
        &mut scratch,
    )
    .unwrap();
    q
}

fn mat_mul(a: &[f64; N * N], b: &[f64; N * N]) -> [f64; N * N] {
    let mut out = [0.0_f64; N * N];
    mul_matrix(
        &StaticStorage::new(*a),
        N,
        N,
        &StaticStorage::new(*b),
        N,
        N,
        &mut out,
    )
    .unwrap();
    out
}

fn mat_transpose(a: &[f64; N * N]) -> [f64; N * N] {
    let mut out = [0.0_f64; N * N];
    transpose(&StaticStorage::new(*a), N, N, &mut out).unwrap();
    out
}

/// `A = q1 * diag(sigma) * q2ᵗ`, which has exactly `sigma` as its singular values by
/// construction (`q1`, `q2` orthogonal).
fn matrix_with_spectrum(q1: [f64; N * N], q2: [f64; N * N], sigma: [f64; N]) -> [f64; N * N] {
    let mut diag = [0.0_f64; N * N];
    for i in 0..N {
        diag[i * N + i] = sigma[i];
    }
    mat_mul(&mat_mul(&q1, &diag), &mat_transpose(&q2))
}

/// Checks the invariants every `svd` output must satisfy regardless of rank: `sigma`
/// non-negative and descending, `V`'s columns orthonormal, and `U * diag(sigma) * Vᵗ`
/// reconstructs `a`. `U`'s columns are checked separately by [`assert_u_columns_valid`],
/// since a negligible singular value leaves its `U` column at zero by design rather than a
/// unit vector (see `svd_qr_iteration`'s docs).
fn assert_common_invariants(
    a: &[f64; N * N],
    u: &[f64; N * N],
    sigma: &[f64; N],
    v: &[f64; N * N],
) {
    for i in 0..N {
        assert!(sigma[i] >= 0.0, "sigma[{}] = {} is negative", i, sigma[i]);
        if i > 0 {
            assert!(
                sigma[i - 1] >= sigma[i],
                "sigma is not descending: {:?}",
                sigma
            );
        }
    }

    for i in 0..N {
        for j in 0..N {
            let dot: f64 = (0..N).map(|k| v[k * N + i] * v[k * N + j]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < TOL,
                "V columns {} and {} disagree: {:?}",
                i,
                j,
                v
            );
        }
    }

    let mut diag = [0.0_f64; N * N];
    for i in 0..N {
        diag[i * N + i] = sigma[i];
    }
    let reconstructed = mat_mul(&mat_mul(u, &diag), &mat_transpose(v));
    for (actual, expected) in reconstructed.iter().zip(a.iter()) {
        assert!(
            (actual - expected).abs() < TOL,
            "reconstruction mismatch: {:?} vs {:?}",
            reconstructed,
            a
        );
    }
}

/// `svd_qr_iteration` classifies singular value `i` as negligible by comparing it against
/// `tolerance * sigma_max` (see its docs), not `tolerance` alone; the caller below passes
/// [`TOL`] as `tolerance`, so this reproduces that same scaled comparison, with `sigma[0]`
/// standing in for `sigma_max` since `sigma` is already sorted descending.
fn negligibility_threshold(sigma: &[f64; N]) -> f64 {
    TOL * sigma[0]
}

/// For every singular value above [`negligibility_threshold`], its `U` column is unit
/// length and orthogonal to every other such column; for one at or below it, the column is
/// left at zero (the documented behavior when there's no well-defined direction left to
/// divide out). This only holds because the caller passes [`TOL`] itself as
/// `svd_qr_iteration`'s tolerance parameter, matching the threshold computed here — with the
/// default (much smaller) tolerance `svd` computes automatically, a singular value could
/// sit below this test's threshold yet still exceed `svd`'s own, in which case dividing
/// `a * v_i` by it amplifies residual rounding error into a non-tiny `U` column instead.
fn assert_u_columns_valid(u: &[f64; N * N], sigma: &[f64; N]) {
    let threshold = negligibility_threshold(sigma);
    let significant: Vec<usize> = (0..N).filter(|&i| sigma[i] > threshold).collect();
    for &i in &significant {
        let norm_sq: f64 = (0..N).map(|k| u[k * N + i] * u[k * N + i]).sum();
        assert!(
            (norm_sq - 1.0).abs() < TOL,
            "U column {} is not unit length: {}",
            i,
            norm_sq
        );
        for &j in &significant {
            if i == j {
                continue;
            }
            let dot: f64 = (0..N).map(|k| u[k * N + i] * u[k * N + j]).sum();
            assert!(
                dot.abs() < TOL,
                "U columns {} and {} are not orthogonal: {}",
                i,
                j,
                dot
            );
        }
    }
    for i in 0..N {
        if sigma[i] <= threshold {
            for k in 0..N {
                assert_eq!(u[k * N + i], 0.0, "U column {} should be left zero", i);
            }
        }
    }
}

proptest! {
    /// A full-rank matrix built from a strictly positive, descending spectrum: `svd`
    /// recovers `sigma`, `U` and `V` orthonormal, and reconstructs the input.
    #[test]
    fn svd_of_full_rank_matrix_satisfies_invariants(
        top in 1.0..20.0f64,
        ratios in prop::array::uniform3(0.1..0.9f64),
        q1_seed in prop::array::uniform16(-1.0..1.0f64),
        q2_seed in prop::array::uniform16(-1.0..1.0f64),
    ) {
        let mut sigma = [top, 0.0, 0.0, 0.0];
        for i in 1..N {
            sigma[i] = sigma[i - 1] * ratios[i - 1];
        }
        let q1 = orthogonal_from(q1_seed);
        let q2 = orthogonal_from(q2_seed);
        let a = matrix_with_spectrum(q1, q2, sigma);

        let mut u = [0.0_f64; N * N];
        let mut out_sigma = [0.0_f64; N];
        let mut v = [0.0_f64; N * N];
        let mut scratch = [0.0_f64; SCRATCH_LEN];
        let result =
            svd_qr_iteration(&StaticStorage::new(a), N, N, &mut u, &mut out_sigma, &mut v, &mut scratch, TOL);
        prop_assert!(result.is_ok(), "svd failed: {:?}", result);

        assert_common_invariants(&a, &u, &out_sigma, &v);
        assert_u_columns_valid(&u, &out_sigma);
    }

    /// A rank-2 matrix (two of its four singular values are exactly zero by construction):
    /// `svd` still recovers non-negative descending `sigma`, still reconstructs `a`, and
    /// leaves the `U` columns for the zero singular values at zero rather than producing
    /// nonsense directions.
    #[test]
    fn svd_of_rank_deficient_matrix_satisfies_invariants(
        top in 1.0..20.0f64,
        ratio in 0.1..0.9f64,
        q1_seed in prop::array::uniform16(-1.0..1.0f64),
        q2_seed in prop::array::uniform16(-1.0..1.0f64),
    ) {
        let sigma = [top, top * ratio, 0.0, 0.0];
        let q1 = orthogonal_from(q1_seed);
        let q2 = orthogonal_from(q2_seed);
        let a = matrix_with_spectrum(q1, q2, sigma);

        let mut u = [0.0_f64; N * N];
        let mut out_sigma = [0.0_f64; N];
        let mut v = [0.0_f64; N * N];
        let mut scratch = [0.0_f64; SCRATCH_LEN];
        let result =
            svd_qr_iteration(&StaticStorage::new(a), N, N, &mut u, &mut out_sigma, &mut v, &mut scratch, TOL);
        prop_assert!(result.is_ok(), "svd failed: {:?}", result);

        let threshold = negligibility_threshold(&out_sigma);
        prop_assert!(out_sigma[2] <= threshold, "expected sigma[2] negligible, got {}", out_sigma[2]);
        prop_assert!(out_sigma[3] <= threshold, "expected sigma[3] negligible, got {}", out_sigma[3]);

        assert_common_invariants(&a, &u, &out_sigma, &v);
        assert_u_columns_valid(&u, &out_sigma);
    }
}

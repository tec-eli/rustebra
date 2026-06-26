use super::{abs, n_as_scalar};
use crate::scalar::{FloatTolerance, Scalar};
use crate::storage::Storage;

/// Error returned by Cholesky decomposition.
///
/// Beyond the shape problems [`crate::algorithm::matrix::DimensionMismatch`] covers,
/// Cholesky has two failure modes of its own: the input might not be symmetric, or it
/// might not be positive-definite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CholeskyError {
    /// `a` is not square, or `a`/`out_l` doesn't have exactly `rows * cols` elements.
    DimensionMismatch,
    /// `a` is not symmetric: `a[i][j]` and `a[j][i]` differ by more than `tolerance`.
    NotSymmetric,
    /// `a` is not positive-definite: a value that should never be negative before its
    /// square root is taken went negative instead.
    NotPositiveDefinite,
}

/// Computes the Cholesky decomposition of the `rows x cols` matrix `a`: factors it as
/// `l * lᵗ`, where `l` is lower triangular with positive diagonal entries. This is the
/// general-user entry point: it delegates to [`cholesky_decompose`] with an
/// automatically-computed tolerance, so callers don't need to pick one themselves.
///
/// The default tolerance is `n * epsilon() * scale`, where `n` is `rows` and `scale` is the
/// largest-magnitude diagonal entry in `a` — the same quantities
/// [`cholesky_decompose`]'s positive-definiteness check itself compares against.
///
/// Only defined for square matrices, like [`crate::algorithm::matrix::determinant`]. `a`
/// must also be symmetric positive-definite for the decomposition to exist at all; see
/// [`cholesky_decompose`] for what happens when it isn't.
///
/// # Errors
///
/// Returns `Err(CholeskyError::DimensionMismatch)` if `a` is not square (`rows != cols`), or
/// if `a` or `out_l` doesn't have exactly `rows * cols` elements. Returns
/// `Err(CholeskyError::NotSymmetric)` if `a` is not symmetric within the auto-computed
/// tolerance. Returns `Err(CholeskyError::NotPositiveDefinite)` if `a` is not
/// positive-definite.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::cholesky;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 symmetric positive-definite matrix: [[4, 2], [2, 2]].
/// let a = StaticStorage::new([4.0, 2.0, 2.0, 2.0]);
/// let mut l = [0.0; 4];
/// cholesky(&a, 2, 2, &mut l).unwrap();
/// assert_eq!(l, [2.0, 0.0, 1.0, 1.0]);
/// ```
pub fn cholesky<S, T>(a: &S, rows: usize, cols: usize, out_l: &mut [T]) -> Result<(), CholeskyError>
where
    S: Storage<Item = T>,
    T: Scalar + FloatTolerance + PartialOrd,
{
    let n = rows;
    let mut scale = T::zero();
    for j in 0..n.min(cols) {
        let Some(&a_jj) = a.get(j * cols + j) else {
            return Err(CholeskyError::DimensionMismatch);
        };
        let a_jj_abs = abs(a_jj);
        if a_jj_abs > scale {
            scale = a_jj_abs;
        }
    }
    let tolerance = n_as_scalar::<T>(n).mul(T::epsilon()).mul(scale);

    cholesky_decompose(a, rows, cols, out_l, tolerance)
}

/// Computes the Cholesky decomposition of the `rows x cols` matrix `a` column by column:
/// for column `j`,
///
/// ```text
/// l[j][j] = sqrt(a[j][j] - sum_{k<j} l[j][k]^2)
/// l[i][j] = (a[i][j] - sum_{k<j} l[i][k] * l[j][k]) / l[j][j]   (for i > j)
/// ```
///
/// Unlike [`crate::algorithm::matrix::lu_partial_pivot`], this needs no pivoting: a
/// symmetric positive-definite matrix is guaranteed to produce a positive value under every
/// square root, so the diagonal entries of `l` never need reordering to stay nonzero.
///
/// If `a[j][j] - sum_{k<j} l[j][k]^2` is more than `tolerance` below zero for some `j`, `a`
/// is not positive-definite, and `Err(CholeskyError::NotPositiveDefinite)` is returned
/// instead of taking the square root of a negative number. A value within `tolerance` of
/// zero (including exactly zero) is treated as positive-*semi*-definite along that column
/// rather than strictly positive-definite (rounding noise on a matrix that is genuinely PSD
/// can otherwise compute as a tiny negative number): `l[j][j]` is left at `0` and the rest
/// of column `j` is left at `0` too, instead of dividing by that zero — the same "leave a
/// zero instead of erroring" choice [`crate::algorithm::matrix::lu_partial_pivot`] makes for
/// a zero pivot.
///
/// `tolerance` is a caller-chosen, absolute threshold (see [`cholesky`] for an
/// automatically-computed default).
///
/// # Errors
///
/// Returns `Err(CholeskyError::DimensionMismatch)` if `a` is not square (`rows != cols`), or
/// if `a` or `out_l` doesn't have exactly `rows * cols` elements, rather than panicking.
/// Returns `Err(CholeskyError::NotSymmetric)` if any off-diagonal pair `a[i][j]` and
/// `a[j][i]` differs by more than `tolerance` in absolute value.
/// Returns `Err(CholeskyError::NotPositiveDefinite)` if `a` is not positive-definite.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::cholesky_decompose;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 3x3 symmetric positive-definite matrix: [[4, 12, -16], [12, 37, -43],
/// // [-16, -43, 98]].
/// let a = StaticStorage::new([4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]);
/// let mut l = [0.0; 9];
/// cholesky_decompose(&a, 3, 3, &mut l, 1e-9).unwrap();
/// assert_eq!(l, [2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]);
/// ```
pub fn cholesky_decompose<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_l: &mut [T],
    tolerance: T,
) -> Result<(), CholeskyError>
where
    S: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    if rows != cols {
        return Err(CholeskyError::DimensionMismatch);
    }
    let n = rows;
    let len = n * n;
    if a.len() != len || out_l.len() != len {
        return Err(CholeskyError::DimensionMismatch);
    }

    let zero = T::zero();

    for i in 0..n {
        for j in (i + 1)..n {
            // Both indices are < n, so n*n == a.len() guarantees these are always Some.
            let Some(&a_ij) = a.get(i * n + j) else {
                return Err(CholeskyError::DimensionMismatch);
            };
            let Some(&a_ji) = a.get(j * n + i) else {
                return Err(CholeskyError::DimensionMismatch);
            };
            if abs(a_ij.sub(a_ji)) > tolerance {
                return Err(CholeskyError::NotSymmetric);
            }
        }
    }

    for slot in out_l.iter_mut() {
        *slot = zero;
    }

    for j in 0..n {
        let mut diag_sum = zero;
        for k in 0..j {
            let l_jk = out_l[j * n + k];
            diag_sum = diag_sum.add(l_jk.mul(l_jk));
        }
        // `j * n + j < n * n == a.len()`, so `get` below is always `Some`; handled
        // explicitly rather than panicking.
        let Some(&a_jj) = a.get(j * n + j) else {
            return Err(CholeskyError::DimensionMismatch);
        };
        let diag_sq = a_jj.sub(diag_sum);
        if diag_sq < zero.sub(tolerance) {
            return Err(CholeskyError::NotPositiveDefinite);
        }
        // `diag_sq.sqrt()` already returns `0` for any non-positive input (including the
        // tolerated negative sliver above), per `Scalar::sqrt`'s own contract — no separate
        // clamping is needed here.
        let l_jj = diag_sq.sqrt();
        out_l[j * n + j] = l_jj;
        if l_jj == zero {
            continue;
        }

        for i in (j + 1)..n {
            let mut sum = zero;
            for k in 0..j {
                sum = sum.add(out_l[i * n + k].mul(out_l[j * n + k]));
            }
            // `i * n + j < n * n == a.len()`, so `get` below is always `Some`; handled
            // explicitly
            let Some(&a_ij) = a.get(i * n + j) else {
                return Err(CholeskyError::DimensionMismatch);
            };
            out_l[i * n + j] = a_ij.sub(sum).div(l_jj);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CholeskyError, cholesky, cholesky_decompose};
    use crate::algorithm::matrix::{mul_matrix, transpose};
    use crate::storage::StaticStorage;

    #[test]
    fn cholesky_of_known_2x2_positive_definite_matrix_l_times_l_transpose_reconstructs_a() {
        // [[4, 2], [2, 2]]; l = [[2, 0], [1, 1]].
        let a = StaticStorage::new([4.0_f64, 2.0, 2.0, 2.0]);
        let mut l = [0.0; 4];

        assert_eq!(cholesky_decompose(&a, 2, 2, &mut l, 1e-9), Ok(()));
        assert_eq!(l, [2.0, 0.0, 1.0, 1.0]);

        let mut l_t = [0.0; 4];
        transpose(&StaticStorage::new(l), 2, 2, &mut l_t).unwrap();
        let mut l_l_t = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(l),
            2,
            2,
            &StaticStorage::new(l_t),
            2,
            2,
            &mut l_l_t,
        )
        .unwrap();
        for (actual, expected) in l_l_t.iter().zip([4.0, 2.0, 2.0, 2.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn cholesky_of_known_3x3_positive_definite_matrix_l_times_l_transpose_reconstructs_a() {
        // [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]; l = [[2, 0, 0], [6, 1, 0],
        // [-8, 5, 3]].
        let a = StaticStorage::new([4.0_f64, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]);
        let mut l = [0.0; 9];

        assert_eq!(cholesky_decompose(&a, 3, 3, &mut l, 1e-9), Ok(()));
        assert_eq!(l, [2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]);

        let mut l_t = [0.0; 9];
        transpose(&StaticStorage::new(l), 3, 3, &mut l_t).unwrap();
        let mut l_l_t = [0.0; 9];
        mul_matrix(
            &StaticStorage::new(l),
            3,
            3,
            &StaticStorage::new(l_t),
            3,
            3,
            &mut l_l_t,
        )
        .unwrap();
        let expected_a = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
        for (actual, expected) in l_l_t.iter().zip(expected_a) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn cholesky_decompose_of_non_symmetric_matrix_is_an_error() {
        // Lower triangle matches a PD matrix, but upper triangle differs: a[0][1] = 2 while
        // a[1][0] = 99. Without the symmetry check this would silently produce an L using
        // only the lower triangle.
        let a = StaticStorage::new([4.0_f64, 99.0, 2.0, 2.0]);
        let mut l = [0.0; 4];
        assert_eq!(
            cholesky_decompose(&a, 2, 2, &mut l, 1e-9),
            Err(CholeskyError::NotSymmetric)
        );
    }

    #[test]
    fn cholesky_decompose_nearly_symmetric_within_tolerance_is_accepted() {
        // a[0][1] = 2.0 + 1e-12; well within 1e-9 tolerance.
        let a = StaticStorage::new([4.0_f64, 2.0 + 1e-12, 2.0, 2.0]);
        let mut l = [0.0; 4];
        assert_eq!(cholesky_decompose(&a, 2, 2, &mut l, 1e-9), Ok(()));
    }

    #[test]
    fn cholesky_high_level_of_non_symmetric_matrix_is_an_error() {
        let a = StaticStorage::new([4.0_f64, 99.0, 2.0, 2.0]);
        let mut l = [0.0; 4];
        assert_eq!(cholesky(&a, 2, 2, &mut l), Err(CholeskyError::NotSymmetric));
    }

    #[test]
    fn cholesky_of_non_positive_definite_matrix_is_an_error() {
        // [[1, 2], [2, 1]]; l[1][1] would need sqrt(1 - 4) = sqrt(-3).
        let a = StaticStorage::new([1.0, 2.0, 2.0, 1.0]);
        let mut l = [0.0; 4];

        assert_eq!(
            cholesky_decompose(&a, 2, 2, &mut l, 1e-9),
            Err(CholeskyError::NotPositiveDefinite)
        );
    }

    #[test]
    fn cholesky_of_non_square_matrix_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut l = [0.0; 9];

        assert_eq!(
            cholesky_decompose(&a, 2, 3, &mut l, 1e-9),
            Err(CholeskyError::DimensionMismatch)
        );
    }

    #[test]
    fn cholesky_mismatched_output_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([4.0, 2.0, 2.0, 2.0]);
        let mut l = [0.0; 3];

        assert_eq!(
            cholesky_decompose(&a, 2, 2, &mut l, 1e-9),
            Err(CholeskyError::DimensionMismatch)
        );
    }

    #[test]
    fn cholesky_matches_cholesky_decompose() {
        let a = StaticStorage::new([4.0, 2.0, 2.0, 2.0]);

        let mut l_high_level = [0.0; 4];
        assert_eq!(cholesky(&a, 2, 2, &mut l_high_level), Ok(()));

        let mut l_explicit = [0.0; 4];
        assert_eq!(cholesky_decompose(&a, 2, 2, &mut l_explicit, 1e-9), Ok(()));

        assert_eq!(l_high_level, l_explicit);
    }

    #[test]
    fn cholesky_default_tolerance_absorbs_rounding_noise_on_a_genuinely_psd_matrix() {
        // [[3, 3], [3, 3]]; rank 1, eigenvalues 0 and 6, so this is genuinely
        // positive-*semi*-definite. Computing `diag_sq` for column 1 involves
        // `3 - (3 / sqrt(3))^2`, which is mathematically exactly `0` but lands at
        // `-4.440892098500626e-16` once `sqrt`/division rounding is involved — the same
        // kind of residual a positive-semi-definite (not strictly positive-definite) input
        // can produce. The default tolerance should absorb it instead of reporting
        // `NotPositiveDefinite`.
        let a = StaticStorage::new([3.0_f64, 3.0, 3.0, 3.0]);
        let mut l = [0.0; 4];

        assert_eq!(cholesky(&a, 2, 2, &mut l), Ok(()));
        assert_eq!(l[3], 0.0);
    }

    #[test]
    fn cholesky_explicit_zero_tolerance_rejects_the_same_rounding_noise() {
        // Same matrix as above, but with an explicit tolerance of exactly 0 (the old,
        // pre-ADR-0009 behavior) instead of the scale-aware default: the rounding residual
        // is no longer absorbed, so this genuinely PSD matrix is wrongly rejected.
        let a = StaticStorage::new([3.0_f64, 3.0, 3.0, 3.0]);
        let mut l = [0.0; 4];

        assert_eq!(
            cholesky_decompose(&a, 2, 2, &mut l, 0.0),
            Err(CholeskyError::NotPositiveDefinite)
        );
    }

    #[test]
    fn cholesky_rejects_upper_triangle_mismatch_across_larger_matrix() {
        // 3x3 matrix where the lower triangle is symmetric-compatible and PD, but the upper
        // triangle is deliberately changed. This catches the bug where the algorithm would
        // silently produce L using only the lower triangle if symmetry wasn't validated.
        // Lower triangle: [[4, ?, ?], [2, 2, ?], [-8, 5, 98]]
        // Upper triangle: [[4, 99, -16], [2, 2, 43], [-8, 5, 98]]
        //                      ^^       ^^     ^^-- changed from -43 to 43
        let a = StaticStorage::new([
            4.0_f64, 99.0, -16.0, // row 0: [0][1] deliberately ≠ [1][0]
            2.0, 2.0, 43.0, // row 1: [1][2] deliberately ≠ [2][1]
            -8.0, 5.0, 98.0, // row 2: diagonal and lower triangle intact
        ]);
        let mut l = [0.0; 9];
        assert_eq!(
            cholesky_decompose(&a, 3, 3, &mut l, 1e-9),
            Err(CholeskyError::NotSymmetric)
        );
    }
}

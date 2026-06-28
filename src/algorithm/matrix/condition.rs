use super::{DimensionMismatch, QR_ITERATIONS, n_as_scalar, svd_qr_iteration};
use crate::scalar::{FloatTolerance, Scalar};
use crate::storage::Storage;

/// Error returned by condition number computation.
///
/// Beyond the shape problems [`DimensionMismatch`] covers, computing a condition number has
/// a failure mode of its own: a singular matrix has no well-defined condition number (it
/// would require dividing by a zero smallest singular value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionNumberError {
    /// `a` is not square, or `a`/`scratch` doesn't have the expected number of elements.
    DimensionMismatch,
    /// `a` is singular: its smallest singular value is `0`.
    Singular,
}

impl From<DimensionMismatch> for ConditionNumberError {
    fn from(_: DimensionMismatch) -> Self {
        ConditionNumberError::DimensionMismatch
    }
}

/// Computes the condition number of the `rows x cols` matrix `a`: `kappa(a) = sigma_max /
/// sigma_min`, the ratio of its largest to smallest singular value. This is the general-user
/// entry point: it delegates to [`condition_number_svd`] with an automatically-computed
/// tolerance, so callers don't need to pick one themselves.
///
/// The default tolerance is `n * QR_ITERATIONS * epsilon()`, where `n` is `rows` — the same
/// default [`crate::algorithm::matrix::svd`] uses, for the same reason: this reuses
/// [`crate::algorithm::matrix::svd_qr_iteration`] internally, so the same accumulated
/// rounding error its own default accounts for applies here too (see
/// [`crate::algorithm::matrix::svd`]'s docs).
///
/// A large condition number means `a` is "ill-conditioned": small changes to `a` or to a
/// right-hand side `b` can produce disproportionately large changes in the solution to
/// `a * x = b`. The identity matrix has the smallest possible condition number, `1`
/// (every singular value equals every other).
///
/// Only defined for square matrices, like [`crate::algorithm::matrix::determinant`].
///
/// # Errors
///
/// Returns `Err(ConditionNumberError::DimensionMismatch)` if `a` is not square
/// (`rows != cols`), or if `a` or `scratch` doesn't have the expected number of elements.
/// Returns `Err(ConditionNumberError::Singular)` if `a`'s smallest singular value is
/// negligible relative to its largest.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::condition_number;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 identity matrix: [[1, 0], [0, 1]].
/// let a = StaticStorage::new([1.0_f64, 0.0, 0.0, 1.0]);
/// let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];
/// let kappa = condition_number(&a, 2, 2, &mut scratch).unwrap();
/// assert!((kappa - 1.0).abs() < 1e-9);
/// ```
pub fn condition_number<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    scratch: &mut [T],
) -> Result<T, ConditionNumberError>
where
    S: Storage<Item = T>,
    T: Scalar + FloatTolerance + PartialOrd,
{
    // `* QR_ITERATIONS`: this reaches the same fixed-iteration QR eigendecomposition
    // [`svd`]'s own default accounts for the same way, and for the same reason — see its
    // doc comment.
    let tolerance = n_as_scalar::<T>(rows * QR_ITERATIONS).mul(T::epsilon());
    condition_number_svd(a, rows, cols, scratch, tolerance)
}

/// Computes the condition number of the `rows x cols` matrix `a` via its singular value
/// decomposition (reusing [`crate::algorithm::matrix::svd_qr_iteration`], passing `tolerance`
/// through to it as well): `kappa(a) = sigma_max / sigma_min`, where `sigma_max` and
/// `sigma_min` are the largest and smallest entries of the decomposition's `sigma` output
/// (already sorted descending, so they're simply its first and last elements).
///
/// If `sigma_min` doesn't exceed `tolerance * sigma_max`, `a` is treated as singular — its
/// columns are negligibly far from linearly dependent, so there's no well-defined condition
/// number — and `Err(ConditionNumberError::Singular)` is returned instead of dividing by a
/// value that small. `tolerance` is a caller-chosen, relative threshold (see
/// [`condition_number`] for an automatically-computed default).
///
/// `scratch` is a single caller-provided buffer this function partitions internally into the
/// `u`, `sigma`, and `v` outputs the decomposition needs plus its own scratch requirement,
/// rather than exposing all of those as separate parameters when this function only needs
/// the two extreme singular values, not the decomposition itself; it must have exactly
/// `7 * rows * rows + 3 * rows` elements (`rows == cols`, checked below, so `rows * rows`
/// names the same single dimension [`crate::algorithm::matrix::svd`]'s formula calls
/// `cols * cols`).
///
/// # Errors
///
/// Returns `Err(ConditionNumberError::DimensionMismatch)` if `a` is not square
/// (`rows != cols`), if `a` doesn't have exactly `rows * cols` elements, or if `scratch`
/// doesn't have exactly `7 * rows * rows + 3 * rows` elements, rather than panicking.
/// Returns `Err(ConditionNumberError::Singular)` if `a`'s smallest singular value is
/// negligible relative to its largest.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::condition_number_svd;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 diagonal matrix: [[100, 0], [0, 1]]; ill-conditioned, kappa = 100.
/// let a = StaticStorage::new([100.0_f64, 0.0, 0.0, 1.0]);
/// let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];
/// let kappa = condition_number_svd(&a, 2, 2, &mut scratch, 1e-9).unwrap();
/// assert!((kappa - 100.0).abs() < 1e-6);
/// ```
pub fn condition_number_svd<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    scratch: &mut [T],
    tolerance: T,
) -> Result<T, ConditionNumberError>
where
    S: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    if rows != cols {
        return Err(ConditionNumberError::DimensionMismatch);
    }
    let n = rows;
    let nn = n * n;
    if a.len() != nn || scratch.len() != 7 * nn + 3 * n {
        return Err(ConditionNumberError::DimensionMismatch);
    }

    let (u_buf, rest) = scratch.split_at_mut(nn);
    let (v_buf, rest) = rest.split_at_mut(nn);
    let (sigma_buf, svd_scratch) = rest.split_at_mut(n);

    svd_qr_iteration(a, n, n, u_buf, sigma_buf, v_buf, svd_scratch, tolerance)?;

    // The decomposition sorts `sigma_buf` descending, so the largest and smallest singular
    // values are simply its first and last elements; `n >= 1` whenever `sigma_buf` is
    // non-empty, so both are present together or absent together.
    let (Some(&sigma_max), Some(&sigma_min)) = (sigma_buf.first(), sigma_buf.last()) else {
        return Err(ConditionNumberError::DimensionMismatch);
    };

    if sigma_min <= tolerance.mul(sigma_max) {
        return Err(ConditionNumberError::Singular);
    }

    Ok(sigma_max.div(sigma_min))
}

#[cfg(test)]
mod tests {
    use super::{ConditionNumberError, condition_number, condition_number_svd};
    use crate::storage::StaticStorage;

    #[test]
    fn condition_number_of_identity_matrix_is_one() {
        #[rustfmt::skip]
        let a = StaticStorage::new([
            1.0_f64, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
        let mut scratch = [0.0; 7 * 3 * 3 + 3 * 3];

        let kappa = condition_number_svd(&a, 3, 3, &mut scratch, 1e-9).unwrap();
        assert!((kappa - 1.0).abs() < 1e-9);
    }

    #[test]
    fn condition_number_of_ill_conditioned_diagonal_matrix_matches_known_ratio() {
        // [[100, 0], [0, 1]]; singular values are exactly 100 and 1, so kappa = 100.
        let a = StaticStorage::new([100.0_f64, 0.0, 0.0, 1.0]);
        let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

        let kappa = condition_number_svd(&a, 2, 2, &mut scratch, 1e-9).unwrap();
        assert!((kappa - 100.0).abs() < 1e-6);
    }

    #[test]
    fn condition_number_of_singular_matrix_is_an_error() {
        // [[1, 2], [2, 4]]; row 1 is twice row 0, so this is singular (rank 1).
        let a = StaticStorage::new([1.0_f64, 2.0, 2.0, 4.0]);
        let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

        assert_eq!(
            condition_number_svd(&a, 2, 2, &mut scratch, 1e-9),
            Err(ConditionNumberError::Singular)
        );
    }

    #[test]
    fn condition_number_of_non_square_matrix_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut scratch = [0.0; 7 * 3 * 3 + 3 * 3];

        assert_eq!(
            condition_number_svd(&a, 2, 3, &mut scratch, 1e-9),
            Err(ConditionNumberError::DimensionMismatch)
        );
    }

    #[test]
    fn condition_number_mismatched_scratch_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 0.0, 0.0, 1.0]);
        let mut scratch = [0.0; 4];

        assert_eq!(
            condition_number_svd(&a, 2, 2, &mut scratch, 1e-9),
            Err(ConditionNumberError::DimensionMismatch)
        );
    }

    #[test]
    fn condition_number_matches_condition_number_svd() {
        let a = StaticStorage::new([100.0_f64, 0.0, 0.0, 1.0]);

        let mut scratch_high_level = [0.0; 7 * 2 * 2 + 3 * 2];
        let kappa_high_level = condition_number(&a, 2, 2, &mut scratch_high_level).unwrap();

        let mut scratch_explicit = [0.0; 7 * 2 * 2 + 3 * 2];
        let kappa_explicit = condition_number_svd(&a, 2, 2, &mut scratch_explicit, 1e-9).unwrap();

        assert_eq!(kappa_high_level, kappa_explicit);
    }

    #[test]
    fn condition_number_default_tolerance_flags_an_extremely_ill_conditioned_matrix() {
        // [[1, 0], [0, 1e-20]]; singular values are exactly 1 and 1e-20 by construction,
        // far below what this algorithm's own fixed-iteration eigendecomposition of `aᵗ * a`
        // can actually resolve (computing `sigma_min` involves squaring it down to `1e-40`,
        // then recovering it via `sqrt` after 100 QR sweeps each contributing their own
        // rounding error) — the default tolerance accounts for that accumulated error (see
        // `condition_number`'s docs), so this is correctly reported as singular.
        let a = StaticStorage::new([1.0_f64, 0.0, 0.0, 1e-20]);
        let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

        assert_eq!(
            condition_number(&a, 2, 2, &mut scratch),
            Err(ConditionNumberError::Singular)
        );
    }

    #[test]
    fn condition_number_explicit_tolerance_too_strict_misses_the_same_case() {
        // Same matrix as above, but with an explicit tolerance far too strict (1e-25) for
        // what this algorithm can actually resolve: the singular value never reads back
        // as negligible, so a huge-but-finite kappa is reported instead of `Singular`.
        let a = StaticStorage::new([1.0_f64, 0.0, 0.0, 1e-20]);
        let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

        assert!(condition_number_svd(&a, 2, 2, &mut scratch, 1e-25).is_ok());
    }
}

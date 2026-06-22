use super::{DimensionMismatch, svd};
use crate::scalar::Scalar;
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
/// sigma_min`, the ratio of its largest to smallest singular value.
///
/// This is the high-level entry point: it always delegates to [`condition_number_svd`], the
/// only way this crate currently computes a condition number.
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
/// Returns `Err(ConditionNumberError::Singular)` if `a`'s smallest singular value is `0`.
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
    T: Scalar + PartialOrd,
{
    condition_number_svd(a, rows, cols, scratch)
}

/// Computes the condition number of the `rows x cols` matrix `a` via its singular value
/// decomposition (reusing [`crate::algorithm::matrix::svd`]): `kappa(a) = sigma_max /
/// sigma_min`, where `sigma_max` and `sigma_min` are the largest and smallest entries of
/// `svd`'s `sigma` output (already sorted descending, so they're simply its first and last
/// elements).
///
/// If `sigma_min` is `0`, `a` is singular — its columns are linearly dependent, so there's
/// no well-defined condition number — and `Err(ConditionNumberError::Singular)` is returned
/// instead of dividing by that zero.
///
/// `scratch` is a single caller-provided buffer this function partitions internally into the
/// `u`, `sigma`, and `v` outputs `svd` needs plus `svd`'s own scratch requirement, rather
/// than exposing all of those as separate parameters when this function only needs the two
/// extreme singular values, not the decomposition itself; it must have exactly
/// `7 * rows * rows + 3 * rows` elements (`rows == cols`, checked below, so `rows * rows`
/// names the same single dimension `svd`'s formula calls `cols * cols`).
///
/// # Errors
///
/// Returns `Err(ConditionNumberError::DimensionMismatch)` if `a` is not square
/// (`rows != cols`), if `a` doesn't have exactly `rows * cols` elements, or if `scratch`
/// doesn't have exactly `7 * rows * rows + 3 * rows` elements, rather than panicking.
/// Returns `Err(ConditionNumberError::Singular)` if `a`'s smallest singular value is `0`.
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
/// let kappa = condition_number_svd(&a, 2, 2, &mut scratch).unwrap();
/// assert!((kappa - 100.0).abs() < 1e-6);
/// ```
pub fn condition_number_svd<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    scratch: &mut [T],
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

    svd(a, n, n, u_buf, sigma_buf, v_buf, svd_scratch)?;

    // `svd` sorts `sigma_buf` descending, so the largest and smallest singular values are
    // simply its first and last elements; `n >= 1` whenever `sigma_buf` is non-empty, so
    // both are present together or absent together.
    let (Some(&sigma_max), Some(&sigma_min)) = (sigma_buf.first(), sigma_buf.last()) else {
        return Err(ConditionNumberError::DimensionMismatch);
    };

    if sigma_min == T::zero() {
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

        let kappa = condition_number_svd(&a, 3, 3, &mut scratch).unwrap();
        assert!((kappa - 1.0).abs() < 1e-9);
    }

    #[test]
    fn condition_number_of_ill_conditioned_diagonal_matrix_matches_known_ratio() {
        // [[100, 0], [0, 1]]; singular values are exactly 100 and 1, so kappa = 100.
        let a = StaticStorage::new([100.0_f64, 0.0, 0.0, 1.0]);
        let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

        let kappa = condition_number_svd(&a, 2, 2, &mut scratch).unwrap();
        assert!((kappa - 100.0).abs() < 1e-6);
    }

    #[test]
    fn condition_number_of_singular_matrix_is_an_error() {
        // [[1, 2], [2, 4]]; row 1 is twice row 0, so this is singular (rank 1).
        let a = StaticStorage::new([1.0_f64, 2.0, 2.0, 4.0]);
        let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

        assert_eq!(
            condition_number_svd(&a, 2, 2, &mut scratch),
            Err(ConditionNumberError::Singular)
        );
    }

    #[test]
    fn condition_number_of_non_square_matrix_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut scratch = [0.0; 7 * 3 * 3 + 3 * 3];

        assert_eq!(
            condition_number_svd(&a, 2, 3, &mut scratch),
            Err(ConditionNumberError::DimensionMismatch)
        );
    }

    #[test]
    fn condition_number_mismatched_scratch_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 0.0, 0.0, 1.0]);
        let mut scratch = [0.0; 4];

        assert_eq!(
            condition_number_svd(&a, 2, 2, &mut scratch),
            Err(ConditionNumberError::DimensionMismatch)
        );
    }

    #[test]
    fn condition_number_matches_condition_number_svd() {
        let a = StaticStorage::new([100.0_f64, 0.0, 0.0, 1.0]);

        let mut scratch_high_level = [0.0; 7 * 2 * 2 + 3 * 2];
        let kappa_high_level = condition_number(&a, 2, 2, &mut scratch_high_level).unwrap();

        let mut scratch_explicit = [0.0; 7 * 2 * 2 + 3 * 2];
        let kappa_explicit = condition_number_svd(&a, 2, 2, &mut scratch_explicit).unwrap();

        assert_eq!(kappa_high_level, kappa_explicit);
    }
}

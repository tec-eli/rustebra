use crate::scalar::Scalar;
use crate::storage::Storage;

/// Error returned by Cholesky decomposition.
///
/// Beyond the shape problems [`crate::algorithm::matrix::DimensionMismatch`] covers,
/// Cholesky has a failure mode of its own: the input might not be symmetric
/// positive-definite, discovered partway through the decomposition when a value that's
/// about to be square-rooted turns out negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CholeskyError {
    /// `a` is not square, or `a`/`out_l` doesn't have exactly `rows * cols` elements.
    DimensionMismatch,
    /// `a` is not positive-definite: a value that should never be negative before its
    /// square root is taken went negative instead.
    NotPositiveDefinite,
}

/// Computes the Cholesky decomposition of the `rows x cols` matrix `a`: factors it as
/// `l * lᵗ`, where `l` is lower triangular with positive diagonal entries.
///
/// This is the high-level entry point: it always delegates to [`cholesky_decompose`], the
/// only Cholesky algorithm this crate currently implements.
///
/// Only defined for square matrices, like [`crate::algorithm::matrix::determinant`]. `a`
/// must also be symmetric positive-definite for the decomposition to exist at all; see
/// [`cholesky_decompose`] for what happens when it isn't.
///
/// # Errors
///
/// Returns `Err(CholeskyError::DimensionMismatch)` if `a` is not square (`rows != cols`), or
/// if `a` or `out_l` doesn't have exactly `rows * cols` elements. Returns
/// `Err(CholeskyError::NotPositiveDefinite)` if `a` is not positive-definite.
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
    T: Scalar + PartialOrd,
{
    cholesky_decompose(a, rows, cols, out_l)
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
/// If `a[j][j] - sum_{k<j} l[j][k]^2` is negative for some `j`, `a` is not
/// positive-definite, and `Err(CholeskyError::NotPositiveDefinite)` is returned instead of
/// taking the square root of a negative number. A value of exactly zero is allowed (`a` is
/// positive *semi*-definite along that column rather than strictly positive-definite);
/// `l[j][j]` is left at `0` and the rest of column `j` is left at `0` too, instead of
/// dividing by that zero — the same "leave a zero instead of erroring" choice
/// [`crate::algorithm::matrix::lu_partial_pivot`] makes for a zero pivot.
///
/// # Errors
///
/// Returns `Err(CholeskyError::DimensionMismatch)` if `a` is not square (`rows != cols`), or
/// if `a` or `out_l` doesn't have exactly `rows * cols` elements, rather than panicking.
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
/// cholesky_decompose(&a, 3, 3, &mut l).unwrap();
/// assert_eq!(l, [2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]);
/// ```
pub fn cholesky_decompose<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_l: &mut [T],
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
        if diag_sq < zero {
            return Err(CholeskyError::NotPositiveDefinite);
        }
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

        assert_eq!(cholesky_decompose(&a, 2, 2, &mut l), Ok(()));
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

        assert_eq!(cholesky_decompose(&a, 3, 3, &mut l), Ok(()));
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
    fn cholesky_of_non_positive_definite_matrix_is_an_error() {
        // [[1, 2], [2, 1]]; l[1][1] would need sqrt(1 - 4) = sqrt(-3).
        let a = StaticStorage::new([1.0, 2.0, 2.0, 1.0]);
        let mut l = [0.0; 4];

        assert_eq!(
            cholesky_decompose(&a, 2, 2, &mut l),
            Err(CholeskyError::NotPositiveDefinite)
        );
    }

    #[test]
    fn cholesky_of_non_square_matrix_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut l = [0.0; 9];

        assert_eq!(
            cholesky_decompose(&a, 2, 3, &mut l),
            Err(CholeskyError::DimensionMismatch)
        );
    }

    #[test]
    fn cholesky_mismatched_output_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([4.0, 2.0, 2.0, 2.0]);
        let mut l = [0.0; 3];

        assert_eq!(
            cholesky_decompose(&a, 2, 2, &mut l),
            Err(CholeskyError::DimensionMismatch)
        );
    }

    #[test]
    fn cholesky_matches_cholesky_decompose() {
        let a = StaticStorage::new([4.0, 2.0, 2.0, 2.0]);

        let mut l_high_level = [0.0; 4];
        assert_eq!(cholesky(&a, 2, 2, &mut l_high_level), Ok(()));

        let mut l_explicit = [0.0; 4];
        assert_eq!(cholesky_decompose(&a, 2, 2, &mut l_explicit), Ok(()));

        assert_eq!(l_high_level, l_explicit);
    }
}

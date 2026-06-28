use super::{DimensionMismatch, abs, n_as_scalar};
use crate::scalar::{FloatTolerance, Scalar};
use crate::storage::Storage;

/// Computes the rank of the `rows x cols` matrix `a`: the number of linearly independent
/// rows. This is the general-user entry point: it delegates to [`rank_with_tolerance`] with an
/// automatically-computed tolerance, so callers don't need to pick one themselves.
///
/// The default tolerance is `n * epsilon() * scale`, where `n` is `max(rows, cols)` and
/// `scale` is the largest-magnitude entry in `a` — the same `n * machine-epsilon * matrix
/// scale` convention LAPACK/NumPy use, applied here since `a`'s own elimination has no
/// cheaper scale (such as a singular value) available for free.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` under the same conditions as [`rank_with_tolerance`].
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::rank;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[1, 2], [2, 4]]; row 1 is twice row 0, so rank is 1.
/// let a = StaticStorage::new([1.0, 2.0, 2.0, 4.0]);
/// let mut scratch = [0.0; 4];
/// assert_eq!(rank(&a, 2, 2, &mut scratch), Ok(1));
/// ```
pub fn rank<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    scratch: &mut [T],
) -> Result<usize, DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + FloatTolerance + PartialOrd,
{
    let mut scale = T::zero();
    for i in 0..a.len() {
        let Some(&x) = a.get(i) else {
            return Err(DimensionMismatch);
        };
        let x_abs = abs(x);
        if x_abs > scale {
            scale = x_abs;
        }
    }
    let n: T = n_as_scalar(rows.max(cols));
    let tolerance = n.mul(T::epsilon()).mul(scale);

    rank_with_tolerance(a, rows, cols, scratch, tolerance)
}

/// Computes the rank of the `rows x cols` matrix `a`: the number of linearly independent
/// rows, found by reducing `a` to row echelon form via Gaussian elimination (with partial
/// pivoting) and counting the rows that don't reduce to negligible (within `tolerance`)
/// entries everywhere. This is the mathematical-user entry point: the caller chooses
/// `tolerance` directly, rather than the automatically-computed default [`rank`] uses.
///
/// At each step, the largest-magnitude entry at or below the current pivot row in the
/// current column is swapped into pivot position — the same largest-magnitude pivoting
/// [`crate::algorithm::matrix::lu_partial_pivot`] uses, and for the same reason: bounding how
/// much any elimination step can amplify rounding error. A column is treated as having no
/// pivot (and elimination for it skipped) once even its largest-magnitude candidate's
/// absolute value doesn't exceed `tolerance` — seeing whether that ever happens at all is
/// exactly what makes this a genuinely approximate, tolerance-based judgment rather than an
/// algebraic fact about `a`.
///
/// Unlike [`crate::algorithm::matrix::determinant`], `a` doesn't need to be square.
///
/// `Storage` is read-only, so the elimination can't mutate `a` in place; `scratch` is a
/// caller-provided buffer that this function copies `a` into and reduces, leaving `a`
/// itself untouched.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a` or `scratch` doesn't have exactly `rows * cols`
/// elements, rather than panicking.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::rank_with_tolerance;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[1, 2], [2, 4]]; row 1 is twice row 0, so rank is 1.
/// let a = StaticStorage::new([1.0, 2.0, 2.0, 4.0]);
/// let mut scratch = [0.0; 4];
/// assert_eq!(rank_with_tolerance(&a, 2, 2, &mut scratch, 1e-9), Ok(1));
/// ```
pub fn rank_with_tolerance<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    scratch: &mut [T],
    tolerance: T,
) -> Result<usize, DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    let len = rows * cols;
    if a.len() != len || scratch.len() != len {
        return Err(DimensionMismatch);
    }
    for (i, slot) in scratch.iter_mut().enumerate() {
        // `i < len == a.len()`, so `get` below is always `Some`; handled explicitly rather
        // than panicking.
        let Some(&x) = a.get(i) else {
            return Err(DimensionMismatch);
        };
        *slot = x;
    }

    let mut pivot_row = 0;
    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }

        let mut best_row = pivot_row;
        let mut best_abs = abs(scratch[pivot_row * cols + col]);
        for r in (pivot_row + 1)..rows {
            let candidate_abs = abs(scratch[r * cols + col]);
            if candidate_abs > best_abs {
                best_abs = candidate_abs;
                best_row = r;
            }
        }
        if best_abs <= tolerance {
            continue;
        }
        if best_row != pivot_row {
            for c in 0..cols {
                scratch.swap(best_row * cols + c, pivot_row * cols + c);
            }
        }

        let pivot_value = scratch[pivot_row * cols + col];
        for r in (pivot_row + 1)..rows {
            let factor = scratch[r * cols + col].div(pivot_value);
            for c in col..cols {
                let term = factor.mul(scratch[pivot_row * cols + c]);
                scratch[r * cols + c] = scratch[r * cols + c].sub(term);
            }
        }
        pivot_row += 1;
    }

    let rank = (0..rows)
        .filter(|&r| (0..cols).any(|c| abs(scratch[r * cols + c]) > tolerance))
        .count();
    Ok(rank)
}

#[cfg(test)]
mod tests {
    use super::{DimensionMismatch, rank, rank_with_tolerance};
    use crate::storage::StaticStorage;

    #[test]
    fn rank_of_full_rank_matrix() {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 10]]; linearly independent rows, rank 3.
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let mut scratch = [0.0; 9];

        assert_eq!(rank(&a, 3, 3, &mut scratch), Ok(3));
    }

    #[test]
    fn rank_of_rank_deficient_matrix() {
        // [[1, 2, 3], [2, 4, 6], [0, 1, 1]]; row 1 is twice row 0, so rank is 2, not 3.
        let a = StaticStorage::new([1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 1.0, 1.0]);
        let mut scratch = [0.0; 9];

        assert_eq!(rank(&a, 3, 3, &mut scratch), Ok(2));
    }

    #[test]
    fn rank_of_non_square_matrix() {
        // [[1, 2, 3], [4, 5, 6]]; linearly independent rows, rank 2.
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut scratch = [0.0; 6];

        assert_eq!(rank(&a, 2, 3, &mut scratch), Ok(2));
    }

    #[test]
    fn rank_mismatched_scratch_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let mut scratch = [0.0; 3];

        assert_eq!(rank(&a, 2, 2, &mut scratch), Err(DimensionMismatch));
    }

    #[test]
    fn rank_with_explicit_tolerance_matches_the_default() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 1.0, 1.0]);
        let mut scratch = [0.0; 9];

        assert_eq!(rank_with_tolerance(&a, 3, 3, &mut scratch, 1e-9), Ok(2));
    }

    #[test]
    fn rank_default_tolerance_absorbs_rounding_noise_from_an_otherwise_dependent_row() {
        // Row 1 is meant to equal row 0 exactly (rank 1), but its last entry is computed as
        // `3.0 * 0.1 * 10.0` instead of written as a literal `3.0` — a classic
        // floating-point round-trip that lands one ULP away from `3.0`
        // (`3.0000000000000004`, not `3.0`). A truly exact-zero comparison after elimination
        // would see that residual as a "real" nonzero entry and report rank 2; the default,
        // scale-aware tolerance should recognize it as noise instead.
        let perturbed = 3.0 * 0.1 * 10.0;
        assert_ne!(
            perturbed, 3.0,
            "test setup requires a genuine rounding residual"
        );

        let a = StaticStorage::new([1.0, 2.0, 3.0, 1.0, 2.0, perturbed]);
        let mut scratch = [0.0; 6];

        assert_eq!(rank(&a, 2, 3, &mut scratch), Ok(1));
    }

    #[test]
    fn rank_explicit_tolerance_of_zero_is_not_robust_to_the_same_rounding_noise() {
        // Same matrix as above, but with an explicit tolerance of exactly 0 (the old,
        // pre-ADR-0009 behavior) instead of the scale-aware default: the rounding residual
        // is no longer absorbed, so the row is (wrongly) counted as independent.
        let perturbed = 3.0 * 0.1 * 10.0;
        let a = StaticStorage::new([1.0, 2.0, 3.0, 1.0, 2.0, perturbed]);
        let mut scratch = [0.0; 6];

        assert_eq!(rank_with_tolerance(&a, 2, 3, &mut scratch, 0.0), Ok(2));
    }

    #[test]
    fn rank_of_zero_by_zero_matrix_is_zero_not_a_panic() {
        let a: StaticStorage<f64, 0> = StaticStorage::new([]);
        let mut scratch: [f64; 0] = [];

        assert_eq!(rank(&a, 0, 0, &mut scratch), Ok(0));
    }

    #[test]
    fn rank_pivots_to_the_largest_magnitude_entry_even_when_neither_is_zero() {
        // [[1, 2], [2, 1]]; column 0 has both rows nonzero, but row 1's entry (2) is larger
        // in magnitude than row 0's (1) — true partial pivoting swaps it in.
        let a = StaticStorage::new([1.0, 2.0, 2.0, 1.0]);
        let mut scratch = [0.0; 4];

        // Rows aren't multiples of each other, so this is still full rank either way; the
        // point is that this doesn't panic or divide by an unnecessarily small pivot.
        assert_eq!(rank(&a, 2, 2, &mut scratch), Ok(2));
    }
}

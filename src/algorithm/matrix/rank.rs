use super::DimensionMismatch;
use crate::scalar::Scalar;
use crate::storage::Storage;

/// Computes the rank of the `rows x cols` matrix `a`: the number of linearly independent
/// rows, found by reducing `a` to row echelon form via Gaussian elimination (with partial
/// pivoting) and counting the rows that aren't entirely zero.
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
    T: Scalar + PartialEq,
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

    let zero = T::zero();
    let mut pivot_row = 0;
    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }
        let Some(found) = (pivot_row..rows).find(|&r| scratch[r * cols + col] != zero) else {
            continue;
        };
        if found != pivot_row {
            for c in 0..cols {
                scratch.swap(found * cols + c, pivot_row * cols + c);
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
        .filter(|&r| (0..cols).any(|c| scratch[r * cols + c] != zero))
        .count();
    Ok(rank)
}

#[cfg(test)]
mod tests {
    use super::{DimensionMismatch, rank};
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
}

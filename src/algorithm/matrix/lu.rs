use super::DimensionMismatch;
use crate::scalar::Scalar;
use crate::storage::Storage;

/// Computes the LU decomposition of the `rows x cols` matrix `a`: factors it as `l * u`,
/// where `l` is unit lower triangular (1s on the diagonal) and `u` is upper triangular, up to
/// a row permutation recorded as a swap count rather than materialized as its own matrix —
/// `l * u == p * a`, where `p` is the permutation built from applying that many row swaps, in
/// order, to the identity.
///
/// This is the high-level entry point: it always delegates to [`lu_partial_pivot`], which
/// documents the pivoting strategy. A non-pivoting variant isn't offered, since plain
/// Gaussian elimination fails outright on inputs with a zero pivot that pivoting would
/// otherwise route around (see [`lu_partial_pivot`]'s examples).
///
/// Only defined for square matrices, like [`crate::algorithm::matrix::determinant`].
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a` is not square (`rows != cols`), or if `a`,
/// `out_l`, or `out_u` doesn't have exactly `rows * cols` elements, rather than panicking,
/// per ADR 0004.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::lu;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[4, 3], [6, 3]].
/// let a = StaticStorage::new([4.0, 3.0, 6.0, 3.0]);
/// let mut l = [0.0; 4];
/// let mut u = [0.0; 4];
/// let swap_count = lu(&a, 2, 2, &mut l, &mut u).unwrap();
/// assert_eq!(swap_count, 0);
/// assert_eq!(l, [1.0, 0.0, 1.5, 1.0]);
/// assert_eq!(u, [4.0, 3.0, 0.0, -1.5]);
/// ```
pub fn lu<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_l: &mut [T],
    out_u: &mut [T],
) -> Result<usize, DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialEq,
{
    lu_partial_pivot(a, rows, cols, out_l, out_u)
}

/// Computes the LU decomposition of the `rows x cols` matrix `a` via Gaussian elimination
/// with partial pivoting: at each step `k`, before eliminating column `k`, the first row at
/// or below `k` with a nonzero entry in column `k` is swapped into position `k` (the same
/// "swap to the first nonzero" pivoting [`crate::algorithm::matrix::rank`] already uses) —
/// this is what lets the decomposition succeed on inputs plain (non-pivoting) Gaussian
/// elimination would fail on, such as a zero already sitting on the diagonal.
///
/// `out_l` and `out_u` start as the identity and a copy of `a`, respectively. Each swap
/// exchanges the full row of `out_u` (which still holds every column not yet eliminated), but
/// only columns `0..k` of `out_l` (the multipliers already computed for earlier columns) —
/// `out_l`'s diagonal and everything to its right is left untouched by the swap, which is
/// what keeps `out_l` unit lower triangular despite the row reordering. If no nonzero entry
/// exists in column `k` at or below row `k`, `a` is singular along that column; elimination
/// for `k` is skipped (there is no pivot to divide by) and decomposition continues — `out_u`
/// ends up with a zero on that diagonal entry instead of returning an error, since a zero
/// pivot is a property of `a`, not a malformed call.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a` is not square (`rows != cols`), or if `a`,
/// `out_l`, or `out_u` doesn't have exactly `rows * cols` elements, rather than panicking,
/// per ADR 0004.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::lu_partial_pivot;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[0, 1], [1, 1]]; a zero already sits on the diagonal, so this
/// // would fail without pivoting.
/// let a = StaticStorage::new([0.0, 1.0, 1.0, 1.0]);
/// let mut l = [0.0; 4];
/// let mut u = [0.0; 4];
/// let swap_count = lu_partial_pivot(&a, 2, 2, &mut l, &mut u).unwrap();
/// assert_eq!(swap_count, 1);
/// // l * u == p * a, where p swapped rows 0 and 1: [[1, 1], [0, 1]].
/// assert_eq!(l, [1.0, 0.0, 0.0, 1.0]);
/// assert_eq!(u, [1.0, 1.0, 0.0, 1.0]);
/// ```
pub fn lu_partial_pivot<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_l: &mut [T],
    out_u: &mut [T],
) -> Result<usize, DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialEq,
{
    if rows != cols {
        return Err(DimensionMismatch);
    }
    let n = rows;
    let len = n * n;
    if a.len() != len || out_l.len() != len || out_u.len() != len {
        return Err(DimensionMismatch);
    }

    for (i, slot) in out_u.iter_mut().enumerate() {
        // `i < len == a.len()`, so `get` below is always `Some`; handled explicitly rather
        // than panicking, per ADR 0004.
        let Some(&x) = a.get(i) else {
            return Err(DimensionMismatch);
        };
        *slot = x;
    }

    let zero = T::zero();
    let one = T::one();
    for slot in out_l.iter_mut() {
        *slot = zero;
    }
    for i in 0..n {
        out_l[i * n + i] = one;
    }

    let mut swap_count = 0;
    for k in 0..n {
        if let Some(p) = (k..n).find(|&r| out_u[r * n + k] != zero)
            && p != k
        {
            for c in 0..n {
                out_u.swap(k * n + c, p * n + c);
            }
            for c in 0..k {
                out_l.swap(k * n + c, p * n + c);
            }
            swap_count += 1;
        }

        let pivot = out_u[k * n + k];
        if pivot == zero {
            continue;
        }

        for i in (k + 1)..n {
            let factor = out_u[i * n + k].div(pivot);
            out_l[i * n + k] = factor;
            for c in k..n {
                let term = factor.mul(out_u[k * n + c]);
                out_u[i * n + c] = out_u[i * n + c].sub(term);
            }
        }
    }

    Ok(swap_count)
}

#[cfg(test)]
mod tests {
    use super::{DimensionMismatch, lu, lu_partial_pivot};
    use crate::algorithm::matrix::mul_matrix;
    use crate::storage::StaticStorage;

    #[test]
    fn lu_of_known_matrix_with_no_pivoting_needed() {
        // [[4, 3], [6, 3]]; the diagonal is already nonzero, so no row swap is needed.
        let a = StaticStorage::new([4.0, 3.0, 6.0, 3.0]);
        let mut l = [0.0; 4];
        let mut u = [0.0; 4];

        assert_eq!(lu(&a, 2, 2, &mut l, &mut u), Ok(0));
        assert_eq!(l, [1.0, 0.0, 1.5, 1.0]);
        assert_eq!(u, [4.0, 3.0, 0.0, -1.5]);

        // l * u == p * a; p is the identity here, since no swap occurred.
        let mut lu_product = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(l),
            2,
            2,
            &StaticStorage::new(u),
            2,
            2,
            &mut lu_product,
        )
        .unwrap();
        assert_eq!(lu_product, [4.0, 3.0, 6.0, 3.0]);
    }

    #[test]
    fn lu_pivots_when_a_zero_sits_on_the_diagonal() {
        // [[0, 1], [1, 1]]; plain Gaussian elimination would divide by the zero pivot at
        // (0, 0), so this only succeeds because of partial pivoting.
        let a = StaticStorage::new([0.0, 1.0, 1.0, 1.0]);
        let mut l = [0.0; 4];
        let mut u = [0.0; 4];

        assert_eq!(lu_partial_pivot(&a, 2, 2, &mut l, &mut u), Ok(1));
        assert_eq!(l, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(u, [1.0, 1.0, 0.0, 1.0]);

        // l * u == p * a, where p swapped rows 0 and 1 of a: [[1, 1], [0, 1]].
        let mut lu_product = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(l),
            2,
            2,
            &StaticStorage::new(u),
            2,
            2,
            &mut lu_product,
        )
        .unwrap();
        assert_eq!(lu_product, [1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn lu_of_singular_matrix_leaves_a_zero_pivot_instead_of_erroring() {
        // [[0, 0], [0, 5]]; column 0 is entirely zero, so there's no pivot to swap in for
        // column 0 at all (singular along that column), not just a zero on the diagonal.
        let a = StaticStorage::new([0.0, 0.0, 0.0, 5.0]);
        let mut l = [0.0; 4];
        let mut u = [0.0; 4];

        assert_eq!(lu(&a, 2, 2, &mut l, &mut u), Ok(0));
        assert_eq!(l, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(u, [0.0, 0.0, 0.0, 5.0]);

        let mut lu_product = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(l),
            2,
            2,
            &StaticStorage::new(u),
            2,
            2,
            &mut lu_product,
        )
        .unwrap();
        assert_eq!(lu_product, [0.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn lu_of_non_square_matrix_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut l = [0.0; 9];
        let mut u = [0.0; 9];

        assert_eq!(lu(&a, 2, 3, &mut l, &mut u), Err(DimensionMismatch));
    }

    #[test]
    fn lu_mismatched_output_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let mut l = [0.0; 3];
        let mut u = [0.0; 4];

        assert_eq!(lu(&a, 2, 2, &mut l, &mut u), Err(DimensionMismatch));
    }
}

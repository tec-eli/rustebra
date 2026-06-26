use super::{DimensionMismatch, abs};
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
/// // Row-major 2x2 matrix: [[6, 3], [4, 3]]; row 0 already holds the largest-magnitude
/// // entry (6) in column 0, so no row swap is needed.
/// let a = StaticStorage::new([6.0, 3.0, 4.0, 3.0]);
/// let mut l = [0.0; 4];
/// let mut u = [0.0; 4];
/// let swap_count = lu(&a, 2, 2, &mut l, &mut u).unwrap();
/// assert_eq!(swap_count, 0);
/// assert_eq!(l, [1.0, 0.0, 4.0 / 6.0, 1.0]);
/// assert_eq!(u, [6.0, 3.0, 0.0, 1.0]);
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
    T: Scalar + PartialOrd,
{
    lu_partial_pivot(a, rows, cols, out_l, out_u)
}

/// Computes the LU decomposition of the `rows x cols` matrix `a` via Gaussian elimination
/// with partial pivoting: at each step `k`, before eliminating column `k`, the
/// largest-magnitude entry at or below row `k` in column `k` is swapped into position `k` —
/// this is what lets the decomposition succeed on inputs plain (non-pivoting) Gaussian
/// elimination would fail on, such as a zero already sitting on the diagonal, and it's also
/// what keeps the multipliers computed in the elimination step below from blowing up: dividing
/// by the largest available candidate, rather than merely the first nonzero one, bounds how
/// much any single step can amplify rounding error.
///
/// `out_l` and `out_u` start as the identity and a copy of `a`, respectively. Each swap
/// exchanges the full row of `out_u` (which still holds every column not yet eliminated), but
/// only columns `0..k` of `out_l` (the multipliers already computed for earlier columns) —
/// `out_l`'s diagonal and everything to its right is left untouched by the swap, which is
/// what keeps `out_l` unit lower triangular despite the row reordering. If every entry in
/// column `k` at or below row `k` is exactly `0`, `a` is singular along that column;
/// elimination for `k` is skipped (there is no pivot to divide by) and decomposition
/// continues — `out_u` ends up with a zero on that diagonal entry instead of returning an
/// error, since a zero pivot is a property of `a`, not a malformed call. A nonzero pivot is
/// always used even if it's tiny, since the resulting error amplification is the honest
/// numerical consequence of an ill-conditioned `a`, not a defect this function should mask;
/// [`crate::algorithm::matrix::condition_number`] is the right tool for diagnosing that, rather
/// than adding a tolerance parameter here.
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
    T: Scalar + PartialOrd,
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
        let mut best_row = k;
        let mut best_abs = abs(out_u[k * n + k]);
        for r in (k + 1)..n {
            let candidate_abs = abs(out_u[r * n + k]);
            if candidate_abs > best_abs {
                best_abs = candidate_abs;
                best_row = r;
            }
        }
        if best_row != k {
            for c in 0..n {
                out_u.swap(k * n + c, best_row * n + c);
            }
            for c in 0..k {
                out_l.swap(k * n + c, best_row * n + c);
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
        // [[6, 3], [4, 3]]; row 0 already holds the largest-magnitude entry (6) in column 0,
        // so no row swap is needed.
        let a = StaticStorage::new([6.0, 3.0, 4.0, 3.0]);
        let mut l = [0.0; 4];
        let mut u = [0.0; 4];

        assert_eq!(lu(&a, 2, 2, &mut l, &mut u), Ok(0));
        assert_eq!(l, [1.0, 0.0, 4.0 / 6.0, 1.0]);
        assert_eq!(u, [6.0, 3.0, 0.0, 1.0]);

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
        assert_eq!(lu_product, [6.0, 3.0, 4.0, 3.0]);
    }

    #[test]
    fn lu_pivots_to_the_largest_magnitude_entry_even_when_neither_is_zero() {
        // [[4, 3], [6, 3]]; neither entry in column 0 is zero, so the old "first nonzero"
        // strategy would never have swapped here — but row 1's entry (6) is larger in
        // magnitude than row 0's (4), so true partial pivoting swaps it in anyway, to bound
        // the elimination's multiplier (and its rounding error) instead of needlessly using
        // the smaller available pivot.
        let a = StaticStorage::new([4.0, 3.0, 6.0, 3.0]);
        let mut l = [0.0; 4];
        let mut u = [0.0; 4];

        assert_eq!(lu(&a, 2, 2, &mut l, &mut u), Ok(1));
        assert_eq!(l, [1.0, 0.0, 4.0 / 6.0, 1.0]);
        assert_eq!(u, [6.0, 3.0, 0.0, 1.0]);

        // l * u == p * a, where p swapped rows 0 and 1 of a: [[6, 3], [4, 3]].
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
        assert_eq!(lu_product, [6.0, 3.0, 4.0, 3.0]);
    }

    #[test]
    fn lu_pivots_away_from_a_tiny_nonzero_entry_toward_a_much_larger_one() {
        // [[1e-300, 1], [1, 1]]; row 0's entry in column 0 is nonzero but minuscule. The old
        // "first nonzero" strategy would pivot on it anyway (it's not exactly zero) and
        // divide by it, catastrophically amplifying rounding error; true partial pivoting
        // swaps in row 1's entry (1) instead.
        let a = StaticStorage::new([1e-300_f64, 1.0, 1.0, 1.0]);
        let mut l = [0.0_f64; 4];
        let mut u = [0.0_f64; 4];

        assert_eq!(lu(&a, 2, 2, &mut l, &mut u), Ok(1));
        // The pivot used is 1, not 1e-300, so no multiplier anywhere near 1e300 appears.
        assert!(l.iter().all(|x: &f64| x.abs() < 1e10));
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

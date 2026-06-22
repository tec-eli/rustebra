use super::DimensionMismatch;
use crate::scalar::Scalar;
use crate::storage::Storage;

/// Computes the QR decomposition of the `rows x cols` matrix `a` via Householder
/// reflections: factors it as `q * r`, where `q` is a `rows x rows` orthogonal matrix
/// (`qᵗ * q` is the identity) and `r` is a `rows x cols` upper triangular matrix.
///
/// For each column `k`, the subvector `x = a[k.., k]` (the entries of column `k` from row
/// `k` down) is reflected onto a multiple of the first coordinate axis by the Householder
/// reflector `h = i - 2 * v * vᵗ / (vᵗ * v)`, where `v = x + sign(x₁) * ‖x‖ * e₁` (`e₁` the
/// first standard basis vector, `x₁` the first entry of `x`). Choosing the sign of the
/// `‖x‖` term to match `x₁` avoids subtracting two nearly equal numbers when `x₁` and `‖x‖`
/// are close in magnitude (catastrophic cancellation); `x₁ == 0` is treated as positive,
/// since there is no cancellation to avoid either way.
///
/// Applying `h` to `a` (restricted to rows/columns `k..`) zeroes out every entry below the
/// diagonal in column `k`, without disturbing the zeros already produced in earlier
/// columns. Repeating this for every column, `hₙ * ... * h₂ * h₁ * a = r`; since each `h` is
/// symmetric and its own inverse, `a = h₁ * h₂ * ... * hₙ * r`, so `q = h₁ * h₂ * ... * hₙ`.
/// `out_q` starts as the identity and is right-multiplied by each `h` as it's built — both
/// `out_r` and `out_q` are updated directly via the reflection formula applied to their
/// columns (for `out_r`) and rows (for `out_q`), rather than materializing every `h` as its
/// own `rows x rows` matrix and calling [`crate::algorithm::matrix::mul_matrix`] repeatedly.
///
/// `scratch` is a caller-provided buffer of length `rows` this function uses to hold `v` for
/// the column currently being eliminated; `Storage` exposes no way to allocate a scratch
/// buffer internally (the same constraint [`crate::algorithm::matrix::rank`]'s `scratch`
/// parameter works around).
///
/// Only defined for `rows >= cols`: `q` is always square (`rows x rows`), so `rows < cols`
/// would force `r` to have more nonzero rows than columns, which an upper triangular matrix
/// cannot represent.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `rows < cols`, if `a` or `out_r` doesn't have exactly
/// `rows * cols` elements, if `out_q` doesn't have exactly `rows * rows` elements, or if
/// `scratch` doesn't have exactly `rows` elements, rather than panicking.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::{mul_matrix, qr_householder};
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[3, 5], [4, 0]].
/// let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
/// let mut q = [0.0; 4];
/// let mut r = [0.0; 4];
/// let mut scratch = [0.0; 2];
/// qr_householder(&a, 2, 2, &mut q, &mut r, &mut scratch).unwrap();
///
/// // q * r reconstructs a (checked within tolerance: q's entries involve dividing by
/// // ‖x‖, an irrational square root for this a).
/// let mut qr = [0.0; 4];
/// mul_matrix(
///     &StaticStorage::new(q),
///     2,
///     2,
///     &StaticStorage::new(r),
///     2,
///     2,
///     &mut qr,
/// )
/// .unwrap();
/// for (actual, expected) in qr.iter().zip([3.0, 5.0, 4.0, 0.0]) {
///     assert!((actual - expected).abs() < 1e-9);
/// }
/// ```
pub fn qr_householder<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_q: &mut [T],
    out_r: &mut [T],
    scratch: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    if rows < cols {
        return Err(DimensionMismatch);
    }
    if a.len() != rows * cols
        || out_r.len() != rows * cols
        || out_q.len() != rows * rows
        || scratch.len() != rows
    {
        return Err(DimensionMismatch);
    }

    for (i, slot) in out_r.iter_mut().enumerate() {
        // `i < rows * cols == a.len()`, so `get` below is always `Some`;
        let Some(&x) = a.get(i) else {
            return Err(DimensionMismatch);
        };
        *slot = x;
    }

    let zero = T::zero();
    let one = T::one();
    let two = one.add(one);
    for slot in out_q.iter_mut() {
        *slot = zero;
    }
    for i in 0..rows {
        out_q[i * rows + i] = one;
    }

    for k in 0..cols {
        let mut norm_sq = zero;
        for i in k..rows {
            let xi = out_r[i * cols + k];
            norm_sq = norm_sq.add(xi.mul(xi));
        }
        let norm = norm_sq.sqrt();
        if norm == zero {
            // Column `k` is already zero from row `k` down; no reflection needed.
            continue;
        }

        let x1 = out_r[k * cols + k];
        let alpha = if x1 < zero { zero.sub(norm) } else { norm };

        scratch[k] = x1.add(alpha);
        for i in (k + 1)..rows {
            scratch[i] = out_r[i * cols + k];
        }

        let mut v_dot_v = zero;
        for &v_i in &scratch[k..rows] {
            v_dot_v = v_dot_v.add(v_i.mul(v_i));
        }
        if v_dot_v == zero {
            continue;
        }

        for c in k..cols {
            let mut dot = zero;
            for i in k..rows {
                dot = dot.add(scratch[i].mul(out_r[i * cols + c]));
            }
            let factor = two.mul(dot).div(v_dot_v);
            for i in k..rows {
                out_r[i * cols + c] = out_r[i * cols + c].sub(factor.mul(scratch[i]));
            }
        }

        for i in 0..rows {
            let mut dot = zero;
            for j in k..rows {
                dot = dot.add(scratch[j].mul(out_q[i * rows + j]));
            }
            let factor = two.mul(dot).div(v_dot_v);
            for j in k..rows {
                out_q[i * rows + j] = out_q[i * rows + j].sub(factor.mul(scratch[j]));
            }
        }
    }

    Ok(())
}

/// Computes the QR decomposition of the `rows x cols` matrix `a` via modified Gram-Schmidt
/// orthogonalization: factors it as `q * r`, where `q` is a `rows x cols` matrix with
/// orthonormal columns (`qᵗ * q` is the identity) and `r` is a `cols x cols` upper
/// triangular matrix.
///
/// For each column `j` of `a`, `v` starts as a copy of that column, then has its projection
/// onto every already-computed `qᵢ` (`i < j`) subtracted out in turn —
/// `v ← v − (qᵢ · v) · qᵢ` — before normalizing what's left to get `qⱼ = v / ‖v‖`. Projecting
/// against the running `v` rather than the original column (the "modified" part of modified
/// Gram-Schmidt) uses the most up-to-date, already-orthogonalized `v` for each projection,
/// which is what keeps rounding error from compounding as badly as plain (classical)
/// Gram-Schmidt does. The projection coefficients `qᵢ · v` become `r[i][j]`, and `‖v‖`
/// (computed after the last projection) becomes `r[j][j]`; `r[i][j]` for `i > j` is left at
/// its initial `0`, which is what makes `r` upper triangular.
///
/// `out_q` is `rows x cols` rather than `rows x rows` (unlike [`qr_householder`]'s `out_q`):
/// Gram-Schmidt only ever produces as many orthonormal columns as `a` has, with no further
/// columns to complete `q` into a square, fully orthogonal matrix.
///
/// `scratch` is a caller-provided buffer of length `rows` this function uses to hold `v` for
/// the column currently being orthogonalized; `Storage` exposes no way to allocate a scratch
/// buffer internally (the same constraint [`crate::algorithm::matrix::rank`]'s `scratch`
/// parameter works around).
///
/// If column `j` of `a` lies entirely in the span of the previous columns (or is zero to
/// begin with), `v` reduces to the zero vector and there is no direction left to normalize
/// into `qⱼ`; column `j` of `out_q` is left at `0` instead of dividing by a zero norm, the
/// same "leave a zero instead of erroring" choice
/// [`crate::algorithm::matrix::lu_partial_pivot`] makes for a zero pivot, since linear
/// dependence is a property of `a`, not a malformed call.
///
/// Only defined for `rows >= cols`, the same requirement [`qr_householder`] has: `q`'s
/// columns can only be made pairwise orthogonal if there are at least as many rows as
/// columns to spread them across.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `rows < cols`, if `a` doesn't have exactly
/// `rows * cols` elements, if `out_q` doesn't have exactly `rows * cols` elements, if `out_r`
/// doesn't have exactly `cols * cols` elements, or if `scratch` doesn't have exactly `rows`
/// elements, rather than panicking.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::{mul_matrix, qr_gram_schmidt};
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[3, 5], [4, 0]].
/// let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
/// let mut q = [0.0; 4];
/// let mut r = [0.0; 4];
/// let mut scratch = [0.0; 2];
/// qr_gram_schmidt(&a, 2, 2, &mut q, &mut r, &mut scratch).unwrap();
///
/// // q * r reconstructs a (checked within tolerance: q's entries involve dividing by
/// // ‖v‖, an irrational square root for this a).
/// let mut qr = [0.0; 4];
/// mul_matrix(
///     &StaticStorage::new(q),
///     2,
///     2,
///     &StaticStorage::new(r),
///     2,
///     2,
///     &mut qr,
/// )
/// .unwrap();
/// for (actual, expected) in qr.iter().zip([3.0, 5.0, 4.0, 0.0]) {
///     assert!((actual - expected).abs() < 1e-9);
/// }
/// ```
pub fn qr_gram_schmidt<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_q: &mut [T],
    out_r: &mut [T],
    scratch: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialEq,
{
    if rows < cols {
        return Err(DimensionMismatch);
    }
    if a.len() != rows * cols
        || out_q.len() != rows * cols
        || out_r.len() != cols * cols
        || scratch.len() != rows
    {
        return Err(DimensionMismatch);
    }

    let zero = T::zero();
    for slot in out_r.iter_mut() {
        *slot = zero;
    }

    for j in 0..cols {
        for (r, slot) in scratch.iter_mut().enumerate() {
            // `r < rows` and `j < cols`, so `r * cols + j < rows * cols == a.len()`; `get`
            // below is always `Some`, handled explicitly rather than panicking.
            let Some(&x) = a.get(r * cols + j) else {
                return Err(DimensionMismatch);
            };
            *slot = x;
        }

        for i in 0..j {
            let mut dot = zero;
            for r in 0..rows {
                dot = dot.add(out_q[r * cols + i].mul(scratch[r]));
            }
            out_r[i * cols + j] = dot;
            for r in 0..rows {
                scratch[r] = scratch[r].sub(dot.mul(out_q[r * cols + i]));
            }
        }

        let mut norm_sq = zero;
        for &v_r in scratch.iter() {
            norm_sq = norm_sq.add(v_r.mul(v_r));
        }
        let norm = norm_sq.sqrt();
        out_r[j * cols + j] = norm;

        if norm == zero {
            // Column `j` of `a` is linearly dependent on the earlier columns (or zero);
            // there's no direction left to normalize into `q`'s column `j`.
            for r in 0..rows {
                out_q[r * cols + j] = zero;
            }
            continue;
        }
        for r in 0..rows {
            out_q[r * cols + j] = scratch[r].div(norm);
        }
    }

    Ok(())
}

/// Computes the QR decomposition of the `rows x cols` matrix `a`: factors it as `q * r`,
/// where `q` is a `rows x rows` orthogonal matrix and `r` is a `rows x cols` upper
/// triangular matrix.
///
/// This is the high-level entry point: it always delegates to [`qr_householder`], since
/// Householder reflections are more numerically stable than [`qr_gram_schmidt`] for general
/// use (see [`qr_householder`]'s docs for why), and there's no observable property of `a`
/// this function currently uses to decide Gram-Schmidt would do better instead.
///
/// Note this returns the same shapes [`qr_householder`] does (`out_q` is `rows x rows`),
/// which differs from [`qr_gram_schmidt`]'s `rows x cols` `out_q` — a caller switching
/// between this function and [`qr_gram_schmidt`] directly needs differently-sized buffers.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` under the same conditions as [`qr_householder`].
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::qr;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[3, 5], [4, 0]].
/// let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
/// let mut q = [0.0; 4];
/// let mut r = [0.0; 4];
/// let mut scratch = [0.0; 2];
/// qr(&a, 2, 2, &mut q, &mut r, &mut scratch).unwrap();
/// assert_eq!(r[1 * 2 + 0], 0.0); // r is upper triangular.
/// ```
pub fn qr<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_q: &mut [T],
    out_r: &mut [T],
    scratch: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    qr_householder(a, rows, cols, out_q, out_r, scratch)
}

#[cfg(test)]
mod tests {
    use super::{DimensionMismatch, qr, qr_gram_schmidt, qr_householder};
    use crate::algorithm::matrix::{mul_matrix, transpose};
    use crate::storage::StaticStorage;

    #[test]
    fn qr_householder_of_square_matrix_q_is_orthogonal() {
        // [[3, 5], [4, 0]]; column 0 has norm 5, chosen so the reflection lands on nice
        // integers in r, even though q itself doesn't.
        let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_householder(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut q_t = [0.0; 4];
        transpose(&StaticStorage::new(q), 2, 2, &mut q_t).unwrap();
        let mut q_t_q = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(q_t),
            2,
            2,
            &StaticStorage::new(q),
            2,
            2,
            &mut q_t_q,
        )
        .unwrap();
        for (actual, expected) in q_t_q.iter().zip([1.0, 0.0, 0.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_householder_of_square_matrix_q_times_r_reconstructs_a() {
        // Same matrix as the orthogonality test above.
        let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_householder(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut qr = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(q),
            2,
            2,
            &StaticStorage::new(r),
            2,
            2,
            &mut qr,
        )
        .unwrap();
        for (actual, expected) in qr.iter().zip([3.0, 5.0, 4.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_householder_of_non_square_matrix_with_more_rows_than_columns() {
        // Row-major 3x2 matrix: [[12, 1], [6, 2], [-4, 3]]; column 0 has norm 14, chosen so
        // the first reflection lands on nice integers.
        let a = StaticStorage::new([12.0_f64, 1.0, 6.0, 2.0, -4.0, 3.0]);
        let mut q = [0.0; 9];
        let mut r = [0.0; 6];
        let mut scratch = [0.0; 3];

        assert_eq!(
            qr_householder(&a, 3, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut q_t = [0.0; 9];
        transpose(&StaticStorage::new(q), 3, 3, &mut q_t).unwrap();
        let mut q_t_q = [0.0; 9];
        mul_matrix(
            &StaticStorage::new(q_t),
            3,
            3,
            &StaticStorage::new(q),
            3,
            3,
            &mut q_t_q,
        )
        .unwrap();
        #[rustfmt::skip]
        let identity = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        for (actual, expected) in q_t_q.iter().zip(identity) {
            assert!((actual - expected).abs() < 1e-9);
        }

        let mut qr = [0.0; 6];
        mul_matrix(
            &StaticStorage::new(q),
            3,
            3,
            &StaticStorage::new(r),
            3,
            2,
            &mut qr,
        )
        .unwrap();
        let expected_a = [12.0, 1.0, 6.0, 2.0, -4.0, 3.0];
        for (actual, expected) in qr.iter().zip(expected_a) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_householder_of_matrix_with_more_columns_than_rows_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 6];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_householder(&a, 2, 3, &mut q, &mut r, &mut scratch),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn qr_householder_mismatched_scratch_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([3.0, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 3];

        assert_eq!(
            qr_householder(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn qr_gram_schmidt_of_square_matrix_q_is_orthogonal() {
        // Same matrix as the qr_householder orthogonality test above.
        let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_gram_schmidt(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut q_t = [0.0; 4];
        transpose(&StaticStorage::new(q), 2, 2, &mut q_t).unwrap();
        let mut q_t_q = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(q_t),
            2,
            2,
            &StaticStorage::new(q),
            2,
            2,
            &mut q_t_q,
        )
        .unwrap();
        for (actual, expected) in q_t_q.iter().zip([1.0, 0.0, 0.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_gram_schmidt_of_square_matrix_q_times_r_reconstructs_a() {
        // Same matrix as the orthogonality test above.
        let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_gram_schmidt(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut qr_product = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(q),
            2,
            2,
            &StaticStorage::new(r),
            2,
            2,
            &mut qr_product,
        )
        .unwrap();
        for (actual, expected) in qr_product.iter().zip([3.0, 5.0, 4.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_gram_schmidt_of_non_square_matrix_with_more_rows_than_columns() {
        // Same matrix as the qr_householder non-square test above.
        let a = StaticStorage::new([12.0_f64, 1.0, 6.0, 2.0, -4.0, 3.0]);
        let mut q = [0.0; 6];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 3];

        assert_eq!(
            qr_gram_schmidt(&a, 3, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut q_t = [0.0; 6];
        transpose(&StaticStorage::new(q), 3, 2, &mut q_t).unwrap();
        let mut q_t_q = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(q_t),
            2,
            3,
            &StaticStorage::new(q),
            3,
            2,
            &mut q_t_q,
        )
        .unwrap();
        for (actual, expected) in q_t_q.iter().zip([1.0, 0.0, 0.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }

        let mut qr_product = [0.0; 6];
        mul_matrix(
            &StaticStorage::new(q),
            3,
            2,
            &StaticStorage::new(r),
            2,
            2,
            &mut qr_product,
        )
        .unwrap();
        let expected_a = [12.0, 1.0, 6.0, 2.0, -4.0, 3.0];
        for (actual, expected) in qr_product.iter().zip(expected_a) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_gram_schmidt_of_matrix_with_more_columns_than_rows_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 9];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_gram_schmidt(&a, 2, 3, &mut q, &mut r, &mut scratch),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn qr_gram_schmidt_mismatched_scratch_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([3.0, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 3];

        assert_eq!(
            qr_gram_schmidt(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn qr_matches_qr_householder() {
        let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);

        let mut q_high_level = [0.0; 4];
        let mut r_high_level = [0.0; 4];
        let mut scratch_high_level = [0.0; 2];
        assert_eq!(
            qr(
                &a,
                2,
                2,
                &mut q_high_level,
                &mut r_high_level,
                &mut scratch_high_level
            ),
            Ok(())
        );

        let mut q_explicit = [0.0; 4];
        let mut r_explicit = [0.0; 4];
        let mut scratch_explicit = [0.0; 2];
        assert_eq!(
            qr_householder(
                &a,
                2,
                2,
                &mut q_explicit,
                &mut r_explicit,
                &mut scratch_explicit
            ),
            Ok(())
        );

        assert_eq!(q_high_level, q_explicit);
        assert_eq!(r_high_level, r_explicit);
    }
}

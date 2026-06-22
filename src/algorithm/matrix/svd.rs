use super::{DimensionMismatch, mul_matrix, mul_vector, qr_householder};
use crate::scalar::Scalar;
use crate::storage::Storage;

/// Number of QR iterations [`svd_qr_iteration`] runs to eigendecompose `aᵗ * a`. Fixed
/// rather than convergence-checked, so behavior is predictable in `no_std` contexts (the
/// same amount of work runs regardless of the input) — the same trade-off
/// [`crate::scalar::Scalar::sqrt`] makes for its own fixed-iteration Newton-Raphson.
///
/// Must be even: [`svd_qr_iteration`] accumulates eigenvectors by ping-ponging between
/// `out_v` and a scratch buffer, one swap per iteration, and relies on an even iteration
/// count to land the final result back in `out_v` itself rather than the scratch buffer.
const QR_ITERATIONS: usize = 100;

/// A read-only [`Storage`] view over a flat slice, so a caller-provided scratch buffer
/// (plain `&mut [T]`, not yet any [`Storage`] implementor) can be passed as the `a`/`b`
/// operand of other algorithms in this module that require one.
struct Slice<'a, T> {
    data: &'a [T],
}

impl<T> Storage for Slice<'_, T> {
    type Item = T;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<&Self::Item> {
        self.data.get(index)
    }
}

/// A read-only [`Storage`] view over column `col` of a flat, row-major, `stride`-wide
/// slice — used to extract a single eigenvector (one column of the accumulated `v` matrix)
/// as the `v` operand [`crate::algorithm::matrix::mul_vector`] expects.
struct StridedColumn<'a, T> {
    data: &'a [T],
    col: usize,
    stride: usize,
    len: usize,
}

impl<T> Storage for StridedColumn<'_, T> {
    type Item = T;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> Option<&Self::Item> {
        if index >= self.len {
            return None;
        }
        self.data.get(index * self.stride + self.col)
    }
}

/// Computes the singular value decomposition of the `rows x cols` matrix `a`: factors it as
/// `u * diag(sigma) * vᵗ`, where `u` is `rows x cols` with orthonormal columns, `sigma` is a
/// length-`cols` vector of non-negative singular values sorted descending, and `v` is a
/// `cols x cols` orthogonal matrix.
///
/// This is the high-level entry point: it always delegates to [`svd_qr_iteration`], the only
/// SVD algorithm this crate currently implements.
///
/// Unlike [`crate::algorithm::matrix::determinant`] or
/// [`crate::algorithm::matrix::lu_partial_pivot`], `a` doesn't need to be square — the
/// singular value decomposition exists for any matrix.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` under the same conditions as [`svd_qr_iteration`].
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::{mul_matrix, svd, transpose};
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[2, 0], [0, 1]].
/// let a = StaticStorage::new([2.0_f64, 0.0, 0.0, 1.0]);
/// let mut u = [0.0; 4];
/// let mut sigma = [0.0; 2];
/// let mut v = [0.0; 4];
/// let mut scratch = [0.0; 5 * 2 * 2 + 2 + 2];
/// svd(&a, 2, 2, &mut u, &mut sigma, &mut v, &mut scratch).unwrap();
///
/// // Singular values come out sorted descending and non-negative.
/// assert!(sigma[0] >= sigma[1]);
/// assert!(sigma[1] >= 0.0);
///
/// // u * diag(sigma) * vᵗ reconstructs a (within tolerance: the eigenvectors behind u and v
/// // come from a fixed-iteration QR algorithm, not exact arithmetic).
/// let diag = [sigma[0], 0.0, 0.0, sigma[1]];
/// let mut u_sigma = [0.0; 4];
/// mul_matrix(
///     &StaticStorage::new(u),
///     2,
///     2,
///     &StaticStorage::new(diag),
///     2,
///     2,
///     &mut u_sigma,
/// )
/// .unwrap();
/// let mut v_t = [0.0; 4];
/// transpose(&StaticStorage::new(v), 2, 2, &mut v_t).unwrap();
/// let mut reconstructed = [0.0; 4];
/// mul_matrix(
///     &StaticStorage::new(u_sigma),
///     2,
///     2,
///     &StaticStorage::new(v_t),
///     2,
///     2,
///     &mut reconstructed,
/// )
/// .unwrap();
/// for (actual, expected) in reconstructed.iter().zip([2.0, 0.0, 0.0, 1.0]) {
///     assert!((actual - expected).abs() < 1e-6);
/// }
/// ```
pub fn svd<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_u: &mut [T],
    out_sigma: &mut [T],
    out_v: &mut [T],
    scratch: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    svd_qr_iteration(a, rows, cols, out_u, out_sigma, out_v, scratch)
}

/// Computes the singular value decomposition of the `rows x cols` matrix `a` by
/// eigendecomposing the `cols x cols` symmetric positive-semi-definite matrix `aᵗ * a` via
/// unshifted QR iteration:
///
/// 1. Form `aᵗ * a`.
/// 2. Repeatedly factor the current matrix as `q * r` (via [`qr_householder`]) and replace it
///    with `r * q`, accumulating the product of every `q` into `out_v`. This converges to a
///    diagonal matrix whose diagonal holds the eigenvalues of `aᵗ * a`, with `out_v`'s
///    columns converging to the corresponding eigenvectors — for a symmetric input, this
///    always converges (no complex eigenvalue pairs to stall on), even with repeated
///    eigenvalues.
/// 3. Singular value `i` is `sqrt(eigenvalue i)` (an eigenvalue of `aᵗ * a` is never negative
///    for a real `a`, but a tiny negative numerical residue is possible; [`Scalar::sqrt`]
///    already returns `0` for non-positive input, so this needs no extra clamping).
/// 4. Sort the eigenvalues descending, permuting `out_v`'s columns (and thus `out_u`'s,
///    computed next) to match.
/// 5. For each non-zero singular value `i`, the left singular vector is
///    `out_u[:, i] = (a * out_v[:, i]) / sigma_i`. A zero singular value leaves `out_u[:, i]`
///    at `0` instead of dividing by it — `a * v_i` is itself the zero vector whenever
///    `sigma_i` is `0`, so there's no well-defined direction to divide out anyway; the same
///    "leave a zero instead of erroring" choice
///    [`crate::algorithm::matrix::lu_partial_pivot`] makes for a zero pivot.
///
/// Runs a fixed [`QR_ITERATIONS`] count rather than checking for convergence; see its docs
/// for why. Step 4's sort is a selection sort performed directly on the eigenvalues (the
/// diagonal of the working matrix) and `out_v`'s columns, rather than via a separate index
/// buffer, since there's no `usize`-typed scratch space to sort indices into — only `T`-typed
/// buffers are available.
///
/// `scratch` is a single caller-provided buffer this function partitions internally into
/// five `cols x cols` working matrices, a length-`cols` buffer for [`qr_householder`]'s own
/// internal scratch, and a length-`rows` buffer for `a * out_v[:, i]`; it must have exactly
/// `5 * cols * cols + cols + rows` elements. `Storage` exposes no way to allocate scratch
/// space internally (the same constraint [`crate::algorithm::matrix::rank`]'s `scratch`
/// parameter works around), and a single buffer keeps this function's signature from
/// growing one parameter per working matrix.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a` doesn't have exactly `rows * cols` elements, if
/// `out_u` doesn't have exactly `rows * cols` elements, if `out_sigma` doesn't have exactly
/// `cols` elements, if `out_v` doesn't have exactly `cols * cols` elements, or if `scratch`
/// doesn't have exactly `5 * cols * cols + cols + rows` elements, rather than panicking.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::svd_qr_iteration;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[2, 0], [0, 1]].
/// let a = StaticStorage::new([2.0_f64, 0.0, 0.0, 1.0]);
/// let mut u = [0.0; 4];
/// let mut sigma = [0.0; 2];
/// let mut v = [0.0; 4];
/// let mut scratch = [0.0; 5 * 2 * 2 + 2 + 2];
/// svd_qr_iteration(&a, 2, 2, &mut u, &mut sigma, &mut v, &mut scratch).unwrap();
/// assert!(sigma[0] >= sigma[1]);
/// ```
pub fn svd_qr_iteration<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_u: &mut [T],
    out_sigma: &mut [T],
    out_v: &mut [T],
    scratch: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    let m = rows;
    let n = cols;
    let nn = n * n;
    if a.len() != m * n
        || out_u.len() != m * n
        || out_sigma.len() != n
        || out_v.len() != nn
        || scratch.len() != 5 * nn + n + m
    {
        return Err(DimensionMismatch);
    }

    let (m_a, rest) = scratch.split_at_mut(nn);
    let (m_b, rest) = rest.split_at_mut(nn);
    let (q_buf, rest) = rest.split_at_mut(nn);
    let (r_buf, rest) = rest.split_at_mut(nn);
    let (v_buf, rest) = rest.split_at_mut(nn);
    let (householder_scratch, av_scratch) = rest.split_at_mut(n);

    let zero = T::zero();
    let one = T::one();

    // `m_a` <- aᵗ * a.
    for i in 0..n {
        for j in 0..n {
            let mut sum = zero;
            for k in 0..m {
                // `k * n + i` and `k * n + j` are both `< m * n == a.len()`, so both `get`
                // calls below are always `Some`; handled explicitly rather than panicking.
                let (Some(&a_ki), Some(&a_kj)) = (a.get(k * n + i), a.get(k * n + j)) else {
                    return Err(DimensionMismatch);
                };
                sum = sum.add(a_ki.mul(a_kj));
            }
            m_a[i * n + j] = sum;
        }
    }

    for slot in out_v.iter_mut() {
        *slot = zero;
    }
    for i in 0..n {
        out_v[i * n + i] = one;
    }

    let mut m_cur = m_a;
    let mut m_nxt = m_b;
    let mut v_cur = out_v;
    let mut v_nxt = v_buf;

    for _ in 0..QR_ITERATIONS {
        qr_householder(
            &Slice { data: m_cur },
            n,
            n,
            q_buf,
            r_buf,
            householder_scratch,
        )?;
        mul_matrix(
            &Slice { data: r_buf },
            n,
            n,
            &Slice { data: q_buf },
            n,
            n,
            m_nxt,
        )?;
        mul_matrix(
            &Slice { data: v_cur },
            n,
            n,
            &Slice { data: q_buf },
            n,
            n,
            v_nxt,
        )?;
        core::mem::swap(&mut m_cur, &mut m_nxt);
        core::mem::swap(&mut v_cur, &mut v_nxt);
    }

    // Sort the eigenvalues (the diagonal of `m_cur`) descending, permuting the matching
    // eigenvectors (`v_cur`'s columns) to keep each eigenvalue paired with its own
    // eigenvector.
    for i in 0..n {
        let mut max_idx = i;
        let mut max_val = m_cur[i * n + i];
        for j in (i + 1)..n {
            let val = m_cur[j * n + j];
            if val > max_val {
                max_val = val;
                max_idx = j;
            }
        }
        if max_idx != i {
            m_cur.swap(i * n + i, max_idx * n + max_idx);
            for r in 0..n {
                v_cur.swap(r * n + i, r * n + max_idx);
            }
        }
    }

    for i in 0..n {
        let lambda = m_cur[i * n + i];
        let sigma_i = lambda.sqrt();
        out_sigma[i] = sigma_i;

        if sigma_i == zero {
            for r in 0..m {
                out_u[r * n + i] = zero;
            }
            continue;
        }

        let v_i = StridedColumn {
            data: v_cur,
            col: i,
            stride: n,
            len: n,
        };
        mul_vector(a, m, n, &v_i, av_scratch)?;
        for r in 0..m {
            out_u[r * n + i] = av_scratch[r].div(sigma_i);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{svd, svd_qr_iteration};
    use crate::algorithm::matrix::DimensionMismatch;
    use crate::storage::StaticStorage;

    /// Computes `u * diag(sigma) * vᵗ` into `out`, entry by entry:
    /// `out[i][j] = sum_k u[i][k] * sigma[k] * v[j][k]` (`vᵗ[k][j] == v[j][k]`). Written
    /// directly rather than via [`super::super::mul_matrix`]/[`super::super::transpose`], so
    /// this test helper works for the runtime-determined `rows`/`cols` every test below
    /// uses without needing a `Vec` (this crate is `no_std`-first by default).
    fn reconstruct(u: &[f64], sigma: &[f64], v: &[f64], rows: usize, cols: usize, out: &mut [f64]) {
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0;
                for k in 0..cols {
                    sum += u[i * cols + k] * sigma[k] * v[j * cols + k];
                }
                out[i * cols + j] = sum;
            }
        }
    }

    #[test]
    fn svd_of_2x2_shear_matrix_reconstructs_a() {
        // [[1, 1], [0, 1]]; a non-diagonal, well-conditioned case (golden-ratio singular
        // values).
        let a = StaticStorage::new([1.0_f64, 1.0, 0.0, 1.0]);
        let mut u = [0.0; 4];
        let mut sigma = [0.0; 2];
        let mut v = [0.0; 4];
        let mut scratch = [0.0; 5 * 2 * 2 + 2 + 2];

        assert_eq!(
            svd_qr_iteration(&a, 2, 2, &mut u, &mut sigma, &mut v, &mut scratch),
            Ok(())
        );
        assert!(sigma[0] >= sigma[1]);
        assert!(sigma[1] >= 0.0);

        let mut reconstructed = [0.0; 4];
        reconstruct(&u, &sigma, &v, 2, 2, &mut reconstructed);
        for (actual, expected) in reconstructed.iter().zip([1.0, 1.0, 0.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn svd_of_non_square_3x2_matrix_reconstructs_a() {
        // [[1, 0], [0, 1], [1, 1]]; full column rank, more rows than columns.
        let a = StaticStorage::new([1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let mut u = [0.0; 6];
        let mut sigma = [0.0; 2];
        let mut v = [0.0; 4];
        let mut scratch = [0.0; 5 * 2 * 2 + 2 + 3];

        assert_eq!(
            svd_qr_iteration(&a, 3, 2, &mut u, &mut sigma, &mut v, &mut scratch),
            Ok(())
        );
        assert!(sigma[0] >= sigma[1]);
        assert!(sigma[1] >= 0.0);

        let mut reconstructed = [0.0; 6];
        reconstruct(&u, &sigma, &v, 3, 2, &mut reconstructed);
        let expected_a = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        for (actual, expected) in reconstructed.iter().zip(expected_a) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn svd_of_diagonal_matrix_sorts_singular_values_descending() {
        // Row-major 3x3 diagonal matrix: [[1, 0, 0], [0, 3, 0], [0, 0, 2]]; singular values
        // are the diagonal entries themselves, in some other order (1, 3, 2) than sorted.
        #[rustfmt::skip]
        let a = StaticStorage::new([
            1.0_f64, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 2.0,
        ]);
        let mut u = [0.0; 9];
        let mut sigma = [0.0; 3];
        let mut v = [0.0; 9];
        let mut scratch = [0.0; 5 * 3 * 3 + 3 + 3];

        assert_eq!(
            svd_qr_iteration(&a, 3, 3, &mut u, &mut sigma, &mut v, &mut scratch),
            Ok(())
        );

        for (actual, expected) in sigma.iter().zip([3.0, 2.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
        assert!(sigma[0] >= sigma[1] && sigma[1] >= sigma[2]);
        for &s in &sigma {
            assert!(s >= 0.0);
        }
    }

    #[test]
    fn svd_of_rank_deficient_matrix_singular_value_count_matches_rank() {
        // [[1, 2], [2, 4], [3, 6]]; every row is a multiple of [1, 2], so rank is 1.
        let a = StaticStorage::new([1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0]);
        let mut u = [0.0; 6];
        let mut sigma = [0.0; 2];
        let mut v = [0.0; 4];
        let mut scratch = [0.0; 5 * 2 * 2 + 2 + 3];

        assert_eq!(
            svd_qr_iteration(&a, 3, 2, &mut u, &mut sigma, &mut v, &mut scratch),
            Ok(())
        );

        let nonzero_count = sigma.iter().filter(|&&s| s > 1e-6).count();
        assert_eq!(nonzero_count, 1);
    }

    #[test]
    fn svd_mismatched_output_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 1.0, 0.0, 1.0]);
        let mut u = [0.0; 3];
        let mut sigma = [0.0; 2];
        let mut v = [0.0; 4];
        let mut scratch = [0.0; 5 * 2 * 2 + 2 + 2];

        assert_eq!(
            svd_qr_iteration(&a, 2, 2, &mut u, &mut sigma, &mut v, &mut scratch),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn svd_mismatched_scratch_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 1.0, 0.0, 1.0]);
        let mut u = [0.0; 4];
        let mut sigma = [0.0; 2];
        let mut v = [0.0; 4];
        let mut scratch = [0.0; 4];

        assert_eq!(
            svd_qr_iteration(&a, 2, 2, &mut u, &mut sigma, &mut v, &mut scratch),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn svd_matches_svd_qr_iteration() {
        let a = StaticStorage::new([1.0_f64, 1.0, 0.0, 1.0]);

        let mut u_high_level = [0.0; 4];
        let mut sigma_high_level = [0.0; 2];
        let mut v_high_level = [0.0; 4];
        let mut scratch_high_level = [0.0; 5 * 2 * 2 + 2 + 2];
        assert_eq!(
            svd(
                &a,
                2,
                2,
                &mut u_high_level,
                &mut sigma_high_level,
                &mut v_high_level,
                &mut scratch_high_level
            ),
            Ok(())
        );

        let mut u_explicit = [0.0; 4];
        let mut sigma_explicit = [0.0; 2];
        let mut v_explicit = [0.0; 4];
        let mut scratch_explicit = [0.0; 5 * 2 * 2 + 2 + 2];
        assert_eq!(
            svd_qr_iteration(
                &a,
                2,
                2,
                &mut u_explicit,
                &mut sigma_explicit,
                &mut v_explicit,
                &mut scratch_explicit
            ),
            Ok(())
        );

        assert_eq!(u_high_level, u_explicit);
        assert_eq!(sigma_high_level, sigma_explicit);
        assert_eq!(v_high_level, v_explicit);
    }
}

use super::ConvergenceError;
use super::hessenberg::HessenbergMatrix;
use super::power_iteration::{Slice, normalize};
use crate::algorithm::matrix::mul_vector;
use crate::algorithm::vector::{dot, norm};
use crate::scalar::Scalar;
use crate::storage::{Basis, Storage};

/// `y -= coefficient * x`, the orthogonalization step's only vector update.
fn subtract_scaled<T: Scalar>(y: &mut [T], coefficient: T, x: &[T]) {
    for (slot, &x_i) in y.iter_mut().zip(x.iter()) {
        *slot = slot.sub(coefficient.mul(x_i));
    }
}

/// Reduces the general, row-major `n x n` matrix `a` to upper Hessenberg form over a
/// `K`-dimensional Krylov subspace by Arnoldi iteration: fills `basis` with an orthonormal
/// basis `Q` of `span{v0, a*v0, ..., a^{K-1}*v0}` and returns the projection `H = Qᵗ * a * Q`,
/// which is upper Hessenberg.
///
/// Starting from `q_0 = v0 / ‖v0‖`, each step `j` computes `w = a * q_j`, orthogonalizes it
/// against every basis vector built so far by modified Gram-Schmidt (recording each
/// coefficient `h_{i,j} = q_iᵗ * w` as it subtracts `h_{i,j} * q_i`), and normalizes the
/// remainder into `q_{j+1}`, recording its length as `h_{j+1,j}`. Unlike [`super::lanczos`],
/// there is no three-term recurrence to exploit — a non-symmetric `a` has no symmetric
/// projection — so every step orthogonalizes against the *entire* basis built so far, not
/// just the two previous vectors; this is also why modified Gram-Schmidt is applied directly
/// rather than as a reorthogonalization pass over an already-orthogonal candidate.
///
/// # Orthogonalization
///
/// Modified Gram-Schmidt (subtracting each projection immediately, rather than computing all
/// projections against the original `w` and subtracting them at the end) is the standard
/// trade for Arnoldi: it is markedly more stable than classical Gram-Schmidt at the same
/// `O(K * n)` per-step cost, though still less stable than Householder Arnoldi, which
/// trades that extra stability for losing the explicit basis vectors the Krylov projection
/// needs. No reorthogonalization pass is added on top, unlike Lanczos's full
/// reorthogonalization: Lanczos's three-term recurrence is why rounding error erodes its
/// basis so quickly that a second pass earns its keep, and Arnoldi's every-vector
/// orthogonalization does not share that failure mode to the same degree.
///
/// # Breakdown is success, not failure
///
/// When `‖w‖ <= tol * ‖a * q_j‖` after orthogonalization, `q_0, ..., q_j` already span an
/// invariant subspace of `a`: there is no new direction to extend the basis with, but the
/// `j + 1` vectors and the leading `(j + 1) x (j + 1)` block of `H` already built are exact
/// and useful. This is therefore reported as `Ok((h, j + 1))`, not an error — unlike
/// [`super::lanczos`], which reports the analogous condition as
/// [`ConvergenceError::Breakdown`]. The difference is what the caller does next: a Lanczos
/// caller that requested `K` vectors and got fewer has nothing it can use without changing
/// its request, while GMRES built on top of Arnoldi can solve directly in the smaller
/// subspace `Ok` reports — the exact subspace is often exactly where the true solution
/// already lives. Callers that do need the full `K` vectors distinguish this case from a
/// complete run by checking `reached < K`.
///
/// `scratch` is a caller-provided buffer of length `n` holding the candidate vector `w` each
/// step; `Storage` exposes no way to allocate one internally (the same constraint
/// [`super::power_iteration`]'s `scratch` parameter works around).
///
/// # Errors
///
/// - [`ConvergenceError::DimensionMismatch`] if `a` doesn't have exactly `n * n` elements,
///   `v0` or `scratch` doesn't have exactly `n` elements, `basis` vectors aren't of length
///   `n`, or `K > n` (an `n`-dimensional space has no `K` orthonormal directions to find).
/// - [`ConvergenceError::ZeroVector`] if `v0` has zero norm (including `n == 0`).
/// - [`ConvergenceError::NonFinite`] if `v0` or an iterate goes `NaN` or infinite.
///
/// # Examples
///
/// ```
/// use rustebra::krylov::arnoldi;
/// use rustebra::storage::{Basis, StaticStorage};
///
/// // Already upper Hessenberg: [[2, 1, 0], [3, 2, 1], [0, 1, 2]]. Starting from e1, Arnoldi
/// // walks the matrix's own rows and reproduces it exactly.
/// let a = StaticStorage::new([2.0_f64, 1.0, 0.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0]);
/// let v0 = StaticStorage::new([1.0, 0.0, 0.0]);
/// let mut buffer = [0.0; 9];
/// let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
/// let mut scratch = [0.0; 3];
///
/// let (h, reached) = arnoldi(&a, 3, &v0, 1e-12, &mut basis, &mut scratch).unwrap();
///
/// assert_eq!(reached, 3);
/// for r in 0..3 {
///     for c in 0..3 {
///         let expected = if r <= c + 1 { [[2.0, 1.0, 0.0], [3.0, 2.0, 1.0], [0.0, 1.0, 2.0]][r][c] } else { 0.0 };
///         assert!((h.entry(r, c).unwrap() - expected).abs() < 1e-12);
///     }
/// }
/// ```
pub fn arnoldi<S, V, T, const K: usize>(
    a: &S,
    n: usize,
    v0: &V,
    tol: T,
    basis: &mut Basis<'_, T, K>,
    scratch: &mut [T],
) -> Result<(HessenbergMatrix<T, K>, usize), ConvergenceError>
where
    S: Storage<Item = T>,
    V: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    if a.len() != n * n || v0.len() != n || basis.vector_len() != n || scratch.len() != n || K > n {
        return Err(ConvergenceError::DimensionMismatch);
    }

    // q_0 = v0 / ‖v0‖, staged through `scratch` and validated before the `K == 0` fast path
    // below: the documented `ZeroVector`/`NonFinite` contract on `v0` holds even when `K == 0`
    // means no basis vector is ever written.
    for (i, slot) in scratch.iter_mut().enumerate() {
        let Some(&x) = v0.get(i) else {
            return Err(ConvergenceError::DimensionMismatch);
        };
        *slot = x;
    }
    normalize(scratch)?;

    let mut data = [[T::zero(); K]; K];
    if K == 0 {
        return Ok((HessenbergMatrix::new(data), 0));
    }

    // `K >= 1` here, so `vector_mut(0)` is always `Some`; handled explicitly rather than
    // panicking, like every other validated access below.
    {
        let Some(q_0) = basis.vector_mut(0) else {
            return Err(ConvergenceError::DimensionMismatch);
        };
        q_0.copy_from_slice(scratch);
    }

    for j in 0..K {
        // w = a * q_j, and its norm — the local scale the breakdown test is relative to.
        let norm_aq = {
            let Some(q_j) = basis.vector(j) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            if mul_vector(a, n, n, &Slice { data: q_j }, scratch).is_err() {
                return Err(ConvergenceError::DimensionMismatch);
            }
            norm(&Slice { data: &*scratch })
        };

        // Modified Gram-Schmidt: subtract the projection onto each earlier basis vector as
        // soon as it's computed, recording it as the Hessenberg entry h_{i,j}.
        for (i, row) in data.iter_mut().enumerate().take(j + 1) {
            let Some(q_i) = basis.vector(i) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            let Ok(h_ij) = dot(&Slice { data: q_i }, &Slice { data: &*scratch }) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            row[j] = h_ij;
            subtract_scaled(scratch, h_ij, q_i);
        }

        let h_next = norm(&Slice { data: &*scratch });
        // `x - x` is `0` for every finite `x` and `NaN` for `NaN` and ±infinity — the only
        // values unequal to themselves. A `NaN`/`Inf` anywhere in this step's arithmetic ends
        // up in `w`, hence in this norm.
        let probe = h_next.sub(h_next);
        #[allow(clippy::eq_op)] // Self-inequality is the point: it holds only for NaN.
        let non_finite = probe != probe;
        if non_finite {
            return Err(ConvergenceError::NonFinite);
        }

        // Breakdown: q_0..q_j already span an invariant subspace of `a`. This is a good
        // outcome, not a failure — return what was built rather than an error.
        if h_next <= tol.mul(norm_aq) {
            return Ok((HessenbergMatrix::new(data), j + 1));
        }

        // The last step only needs column K - 1 of H; h_{K, K-1} falls outside the K x K
        // projection, so it's computed above (for the breakdown test) but never stored.
        if j + 1 < K {
            let Some(q_next) = basis.vector_mut(j + 1) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            let inv = T::one().div(h_next);
            for (slot, &w_i) in q_next.iter_mut().zip(scratch.iter()) {
                *slot = w_i.mul(inv);
            }
            data[j + 1][j] = h_next;
        }
    }

    Ok((HessenbergMatrix::new(data), K))
}

#[cfg(test)]
mod tests {
    use super::arnoldi;
    use crate::krylov::ConvergenceError;
    use crate::storage::{Basis, StaticStorage};

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual}"
        );
    }

    /// `Qᵗ * A * Q` entry `(r, c)` from the basis, against the row-major `n x n` matrix `a`.
    fn projection_entry<const K: usize>(
        a: &[f64],
        n: usize,
        basis: &Basis<'_, f64, K>,
        r: usize,
        c: usize,
    ) -> f64 {
        let q_r = basis.vector(r).unwrap();
        let q_c = basis.vector(c).unwrap();
        let mut sum = 0.0;
        for row in 0..n {
            for col in 0..n {
                sum += q_r[row] * a[row * n + col] * q_c[col];
            }
        }
        sum
    }

    #[test]
    fn hessenberg_input_from_e1_reproduces_itself() {
        // [[2, 1, 0], [3, 2, 1], [0, 1, 2]] with v0 = e1: the Krylov walk visits e1, e2, e3 in
        // order, so H is read straight off the input matrix.
        let a = StaticStorage::new([2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0]);
        let v0 = StaticStorage::new([1.0, 0.0, 0.0]);
        let mut buffer = [0.0; 9];
        let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
        let mut scratch = [0.0; 3];

        let (h, reached) = arnoldi(&a, 3, &v0, 1e-12, &mut basis, &mut scratch).unwrap();

        assert_eq!(reached, 3);
        let expected = [[2.0, 1.0, 0.0], [3.0, 2.0, 1.0], [0.0, 1.0, 2.0]];
        for (r, expected_row) in expected.iter().enumerate() {
            for (c, &expected_val) in expected_row.iter().enumerate() {
                assert_close(h.entry(r, c).unwrap(), expected_val, 1e-12);
            }
        }
        // The basis is the identity's columns, up to rounding.
        for k in 0..3 {
            let q_k = basis.vector(k).unwrap();
            for (i, &entry) in q_k.iter().enumerate() {
                assert_close(entry, if i == k { 1.0 } else { 0.0 }, 1e-12);
            }
        }
    }

    #[test]
    fn basis_is_orthonormal_and_hessenberg_structure_holds() {
        // Non-symmetric on purpose: Arnoldi's projection is upper Hessenberg, not tridiagonal.
        let a = [4.0, 1.0, 2.0, 3.0, 3.0, 1.0, 5.0, 1.0, 5.0];
        let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
        let mut buffer = [0.0; 9];
        let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
        let mut scratch = [0.0; 3];

        let (h, reached) = arnoldi(
            &StaticStorage::new(a),
            3,
            &v0,
            1e-12,
            &mut basis,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(reached, 3);

        for r in 0..3 {
            for c in 0..3 {
                let q_r = basis.vector(r).unwrap();
                let q_c = basis.vector(c).unwrap();
                let inner: f64 = q_r.iter().zip(q_c.iter()).map(|(x, y)| x * y).sum();
                assert_close(inner, if r == c { 1.0 } else { 0.0 }, 1e-12);
            }
        }

        // Upper Hessenberg: zero strictly below the first subdiagonal.
        assert_close(h.entry(2, 0).unwrap(), 0.0, 1e-10);

        assert_close(
            projection_entry(&a, 3, &basis, 0, 0),
            h.entry(0, 0).unwrap(),
            1e-10,
        );
        assert_close(
            projection_entry(&a, 3, &basis, 1, 0),
            h.entry(1, 0).unwrap(),
            1e-10,
        );
        assert_close(
            projection_entry(&a, 3, &basis, 2, 1),
            h.entry(2, 1).unwrap(),
            1e-10,
        );
    }

    #[test]
    fn partial_basis_projects_onto_the_leading_block() {
        // K == 2 < n == 3: the projection onto the two-vector basis is the 2x2 leading block
        // of the full Hessenberg form.
        let a = [4.0, 1.0, 2.0, 3.0, 3.0, 1.0, 5.0, 1.0, 5.0];
        let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
        let mut buffer = [0.0; 6];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 3).unwrap();
        let mut scratch = [0.0; 3];

        let (h, reached) = arnoldi(
            &StaticStorage::new(a),
            3,
            &v0,
            1e-12,
            &mut basis,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(reached, 2);

        let q_0 = basis.vector(0).unwrap();
        let q_1 = basis.vector(1).unwrap();
        let inner: f64 = q_0.iter().zip(q_1.iter()).map(|(x, y)| x * y).sum();
        assert_close(inner, 0.0, 1e-12);

        assert_close(
            projection_entry(&a, 3, &basis, 0, 0),
            h.entry(0, 0).unwrap(),
            1e-10,
        );
        assert_close(
            projection_entry(&a, 3, &basis, 0, 1),
            h.entry(0, 1).unwrap(),
            1e-10,
        );
        assert_close(
            projection_entry(&a, 3, &basis, 1, 0),
            h.entry(1, 0).unwrap(),
            1e-10,
        );
        assert_close(
            projection_entry(&a, 3, &basis, 1, 1),
            h.entry(1, 1).unwrap(),
            1e-10,
        );
    }

    #[test]
    fn k_equals_one_yields_the_rayleigh_quotient() {
        // With a single basis vector there is no off-diagonal: h_00 is the Rayleigh quotient
        // of the normalized v0. v0 = [3, 4]/5, h_00 = v0ᵗ A v0 = (9*2 + 16*1)/25.
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([3.0, 4.0]);
        let mut buffer = [0.0; 2];
        let mut basis = Basis::<f64, 1>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let (h, reached) = arnoldi(&a, 2, &v0, 1e-12, &mut basis, &mut scratch).unwrap();

        assert_eq!(reached, 1);
        assert_close(h.entry(0, 0).unwrap(), 34.0 / 25.0, 1e-12);
        assert_close(basis.vector(0).unwrap()[0], 0.6, 1e-12);
        assert_close(basis.vector(0).unwrap()[1], 0.8, 1e-12);
    }

    #[test]
    fn k_zero_produces_an_empty_projection() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut buffer: [f64; 0] = [];
        let mut basis = Basis::<f64, 0>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let (h, reached) = arnoldi(&a, 2, &v0, 1e-12, &mut basis, &mut scratch).unwrap();

        assert_eq!(reached, 0);
        assert_eq!(h.entry(0, 0), None);
    }

    #[test]
    fn repeated_eigenvalue_breaks_down_as_success_not_error() {
        // The identity's Krylov subspace is one-dimensional for every v0 (A*v0 = v0), so a
        // second basis vector can't exist: h_10 == 0 exactly. Unlike Lanczos, this is `Ok`.
        let a = StaticStorage::new([1.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 1.0]);
        let mut buffer = [0.0; 4];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let (h, reached) = arnoldi(&a, 2, &v0, 1e-12, &mut basis, &mut scratch).unwrap();

        assert_eq!(reached, 1);
        assert_close(h.entry(0, 0).unwrap(), 1.0, 1e-12);
    }

    #[test]
    fn v0_inside_an_invariant_subspace_breaks_down_as_success_not_error() {
        // diag(2, 1) with v0 = e1: an exact eigenvector spans an invariant subspace, so the
        // basis stops at one vector even though the eigenvalues are distinct.
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut buffer = [0.0; 4];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let (h, reached) = arnoldi(&a, 2, &v0, 1e-12, &mut basis, &mut scratch).unwrap();

        assert_eq!(reached, 1);
        assert_close(h.entry(0, 0).unwrap(), 2.0, 1e-12);
    }

    #[test]
    fn rank_deficient_matrix_breaks_down_early() {
        // Rank-1 matrix a = v vᵗ with v = [1, 1, 1]: A*v0 is always a multiple of v, so the
        // Krylov subspace from v0 = v collapses to one dimension immediately.
        let a = StaticStorage::new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
        let mut buffer = [0.0; 9];
        let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
        let mut scratch = [0.0; 3];

        let (_, reached) = arnoldi(&a, 3, &v0, 1e-10, &mut basis, &mut scratch).unwrap();

        assert_eq!(reached, 1);
    }

    #[test]
    fn zero_initial_vector_is_an_error_not_a_panic() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([0.0, 0.0]);
        let mut buffer = [0.0; 4];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let result = arnoldi(&a, 2, &v0, 1e-12, &mut basis, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::ZeroVector));
    }

    #[test]
    fn zero_initial_vector_is_an_error_even_when_k_is_zero() {
        // K == 0 never writes a basis vector, but v0 is still validated: it must not
        // silently succeed with an empty projection just because there's nothing to write.
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([0.0, 0.0]);
        let mut buffer: [f64; 0] = [];
        let mut basis = Basis::<f64, 0>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let result = arnoldi(&a, 2, &v0, 1e-12, &mut basis, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::ZeroVector));
    }

    #[test]
    fn non_finite_initial_vector_is_a_distinct_error_from_zero_vector() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([f64::NAN, 0.0]);
        let mut buffer = [0.0; 4];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let result = arnoldi(&a, 2, &v0, 1e-12, &mut basis, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::NonFinite));
    }

    #[test]
    fn non_finite_matrix_entry_is_an_error_not_a_breakdown() {
        for poison in [f64::NAN, f64::INFINITY] {
            let a = StaticStorage::new([poison, 0.0, 0.0, 1.0]);
            let v0 = StaticStorage::new([1.0, 1.0]);
            let mut buffer = [0.0; 4];
            let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
            let mut scratch = [0.0; 2];

            let result = arnoldi(&a, 2, &v0, 1e-12, &mut basis, &mut scratch);

            assert_eq!(result, Err(ConvergenceError::NonFinite));
        }
    }

    #[test]
    fn mismatched_dimensions_are_an_error_not_a_panic() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);

        // `a` has 4 elements but 3 x 3 is claimed.
        let v0_3 = StaticStorage::new([1.0, 0.0, 0.0]);
        let mut buffer_3 = [0.0; 6];
        let mut basis_3 = Basis::<f64, 2>::new(&mut buffer_3, 3).unwrap();
        let mut scratch_3 = [0.0; 3];
        assert_eq!(
            arnoldi(&a, 3, &v0_3, 1e-12, &mut basis_3, &mut scratch_3),
            Err(ConvergenceError::DimensionMismatch)
        );

        // `v0` too short for n == 2.
        let v0_short = StaticStorage::new([1.0]);
        let mut buffer = [0.0; 4];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];
        assert_eq!(
            arnoldi(&a, 2, &v0_short, 1e-12, &mut basis, &mut scratch),
            Err(ConvergenceError::DimensionMismatch)
        );

        // Basis vectors of the wrong length.
        let mut buffer_wrong = [0.0; 2];
        let mut basis_wrong = Basis::<f64, 2>::new(&mut buffer_wrong, 1).unwrap();
        assert_eq!(
            arnoldi(&a, 2, &v0, 1e-12, &mut basis_wrong, &mut scratch),
            Err(ConvergenceError::DimensionMismatch)
        );

        // Scratch buffer too short.
        let mut scratch_short = [0.0; 1];
        assert_eq!(
            arnoldi(&a, 2, &v0, 1e-12, &mut basis, &mut scratch_short),
            Err(ConvergenceError::DimensionMismatch)
        );

        // K > n: a 2-dimensional space has no 3 orthonormal directions.
        let mut buffer_deep = [0.0; 6];
        let mut basis_deep = Basis::<f64, 3>::new(&mut buffer_deep, 2).unwrap();
        assert_eq!(
            arnoldi(&a, 2, &v0, 1e-12, &mut basis_deep, &mut scratch),
            Err(ConvergenceError::DimensionMismatch)
        );
    }
}

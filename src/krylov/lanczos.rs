use super::ConvergenceError;
use super::power_iteration::{Slice, normalize};
use super::tridiagonal::TridiagonalMatrix;
use crate::algorithm::matrix::mul_vector;
use crate::algorithm::vector::{dot, norm};
use crate::scalar::Scalar;
use crate::storage::{Basis, Storage};

/// `y -= coefficient * x`, the orthogonalization step's only vector update. `zip` stops at
/// the shorter operand, but every length is validated before this is called.
fn subtract_scaled<T: Scalar>(y: &mut [T], coefficient: T, x: &[T]) {
    for (slot, &x_i) in y.iter_mut().zip(x.iter()) {
        *slot = slot.sub(coefficient.mul(x_i));
    }
}

/// Tridiagonalizes the symmetric `n x n`, row-major matrix `a` over a `K`-dimensional Krylov
/// subspace by Lanczos iteration: fills `basis` with an orthonormal basis `Q` of
/// `span{v0, a*v0, ..., a^{K-1}*v0}` and returns the projection `T = Qᵗ * a * Q`, which is
/// symmetric tridiagonal.
///
/// Starting from `q_0 = v0 / ‖v0‖`, each step computes `w = a * q_j`, records the diagonal
/// entry `α_j = q_jᵗ * w`, orthogonalizes `w` against the basis built so far, and normalizes
/// the remainder into `q_{j+1}`, recording its length as the off-diagonal entry `β_j`. The
/// eigenvalues of `T` (the Ritz values) approximate eigenvalues of `a`, with the extreme ones
/// converging first — with `K == n` and no breakdown, `T` has exactly the spectrum of `a`.
/// `a` is *assumed* symmetric, never verified: for a non-symmetric input the projection isn't
/// tridiagonal, and the returned `T` silently misrepresents it.
///
/// # Orthogonality
///
/// The three-term recurrence guarantees orthogonality only in exact arithmetic; in floating
/// point, rounding error famously erodes it as Ritz values converge. Each step therefore
/// re-orthogonalizes `w` against *every* basis vector built so far (full reorthogonalization,
/// one modified-Gram-Schmidt pass), keeping `Q` orthonormal to working precision at `O(K * n)`
/// extra work per step — the right trade at the basis sizes a stack-allocated `K` implies,
/// where the selective alternative's bookkeeping outweighs the dot products it skips.
///
/// # Breakdown and `tol`
///
/// When `‖w‖ <= tol * ‖a * q_j‖` after orthogonalization, the Krylov subspace is (numerically)
/// invariant: there is no new direction to extend the basis with, and the call fails fast with
/// [`ConvergenceError::Breakdown`] rather than dividing by a vanishing norm and iterating on
/// noise. This happens when `v0` lies in (or near) an invariant subspace of dimension less
/// than `K` — a matrix with a repeated eigenvalue reaches at most one basis vector per
/// *distinct* eigenvalue, so e.g. the identity breaks down immediately for any `v0`. The
/// entries of `T` and `basis` computed before the failure are left in place but partial.
/// `tol` is required, with no auto-computed default, per the crate's Krylov tolerance
/// convention; `0` detects only exact breakdown.
///
/// `scratch` is a caller-provided buffer of length `n` holding the candidate vector `w` each
/// step; `Storage` exposes no way to allocate one internally (the same constraint
/// [`crate::krylov::power_iteration`]'s `scratch` parameter works around).
///
/// # Errors
///
/// - [`ConvergenceError::DimensionMismatch`] if `a` doesn't have exactly `n * n` elements,
///   `v0` or `scratch` doesn't have exactly `n` elements, `basis` vectors aren't of length
///   `n`, or `K > n` (an `n`-dimensional space has no `K` orthonormal directions to find).
/// - [`ConvergenceError::ZeroVector`] if `v0` has zero norm (including `n == 0`).
/// - [`ConvergenceError::NonFinite`] if `v0` or an iterate goes `NaN` or infinite.
/// - [`ConvergenceError::Breakdown`] if the basis can't be extended to `K` vectors, as
///   described above.
///
/// # Examples
///
/// ```
/// use rustebra::krylov::lanczos;
/// use rustebra::storage::{Basis, StaticStorage};
///
/// // Already tridiagonal: [[2, 1, 0], [1, 2, 1], [0, 1, 2]]. Starting from e1, Lanczos
/// // walks the matrix's own rows and reproduces it exactly.
/// let a = StaticStorage::new([2.0_f64, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0]);
/// let v0 = StaticStorage::new([1.0, 0.0, 0.0]);
/// let mut buffer = [0.0; 9];
/// let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
/// let mut scratch = [0.0; 3];
///
/// let t = lanczos(&a, 3, &v0, 1e-12, &mut basis, &mut scratch).unwrap();
///
/// for j in 0..3 {
///     assert!((t.diagonal()[j] - 2.0).abs() < 1e-12);
/// }
/// for j in 0..2 {
///     assert!((t.off_diagonal()[j] - 1.0).abs() < 1e-12);
/// }
/// ```
pub fn lanczos<S, V, T, const K: usize>(
    a: &S,
    n: usize,
    v0: &V,
    tol: T,
    basis: &mut Basis<'_, T, K>,
    scratch: &mut [T],
) -> Result<TridiagonalMatrix<T, K>, ConvergenceError>
where
    S: Storage<Item = T>,
    V: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    if a.len() != n * n || v0.len() != n || basis.vector_len() != n || scratch.len() != n || K > n {
        return Err(ConvergenceError::DimensionMismatch);
    }

    let mut diagonal = [T::zero(); K];
    let mut off_diagonal = [T::zero(); K];
    if K == 0 {
        return Ok(TridiagonalMatrix::new(diagonal, off_diagonal));
    }

    // q_0 = v0 / ‖v0‖. `K >= 1` here, so `vector_mut(0)` is always `Some`; handled explicitly
    // rather than panicking, like every other validated access below.
    {
        let Some(q_0) = basis.vector_mut(0) else {
            return Err(ConvergenceError::DimensionMismatch);
        };
        for (i, slot) in q_0.iter_mut().enumerate() {
            let Some(&x) = v0.get(i) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            *slot = x;
        }
        normalize(q_0)?;
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

        // α_j = q_jᵗ * w, then w -= α_j * q_j - the diagonal entry and the recurrence's
        // projection along the newest basis vector.
        {
            let Some(q_j) = basis.vector(j) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            let Ok(alpha_j) = dot(&Slice { data: q_j }, &Slice { data: &*scratch }) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            diagonal[j] = alpha_j;
            subtract_scaled(scratch, alpha_j, q_j);
        }

        // w -= β_{j-1} * q_{j-1}, completing the three-term recurrence.
        if j > 0 {
            let Some(q_prev) = basis.vector(j - 1) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            subtract_scaled(scratch, off_diagonal[j - 1], q_prev);
        }

        // Full reorthogonalization: one modified-Gram-Schmidt pass removing the rounding
        // residue the recurrence leaves along every earlier basis vector.
        for i in 0..=j {
            let Some(q_i) = basis.vector(i) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            let Ok(residue) = dot(&Slice { data: q_i }, &Slice { data: &*scratch }) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            subtract_scaled(scratch, residue, q_i);
        }

        let beta_j = norm(&Slice { data: &*scratch });
        // `x - x` is `0` for every finite `x` and `NaN` for `NaN` and ±infinity — the only
        // values unequal to themselves. A `NaN`/`Inf` anywhere in this step's arithmetic ends
        // up in `w`, hence in this norm.
        let probe = beta_j.sub(beta_j);
        #[allow(clippy::eq_op)] // Self-inequality is the point: it holds only for NaN.
        let non_finite = probe != probe;
        if non_finite {
            return Err(ConvergenceError::NonFinite);
        }

        // The last step only needs α_{K-1}; β_{K-1} is outside the K x K projection, so a
        // vanishing remainder there is expected, not a breakdown.
        if j + 1 < K {
            if beta_j <= tol.mul(norm_aq) {
                return Err(ConvergenceError::Breakdown);
            }
            let Some(q_next) = basis.vector_mut(j + 1) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            let inv = T::one().div(beta_j);
            for (slot, &w_i) in q_next.iter_mut().zip(scratch.iter()) {
                *slot = w_i.mul(inv);
            }
            off_diagonal[j] = beta_j;
        }
    }

    Ok(TridiagonalMatrix::new(diagonal, off_diagonal))
}

#[cfg(test)]
mod tests {
    use super::lanczos;
    use crate::krylov::ConvergenceError;
    use crate::storage::{Basis, StaticStorage};

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual}"
        );
    }

    /// `Qᵗ * A * Q` entry `(r, c)` from the basis, against the row-major `n x n` matrix `a`.
    fn projection_entry(a: &[f64], n: usize, basis: &Basis<'_, f64, 3>, r: usize, c: usize) -> f64 {
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
    fn tridiagonal_input_from_e1_reproduces_itself() {
        // [[2, 1, 0], [1, 2, 1], [0, 1, 2]] with v0 = e1: the Krylov walk visits e1, e2, e3
        // in order, so α and β are read straight off the input matrix.
        let a = StaticStorage::new([2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0]);
        let v0 = StaticStorage::new([1.0, 0.0, 0.0]);
        let mut buffer = [0.0; 9];
        let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
        let mut scratch = [0.0; 3];

        let t = lanczos(&a, 3, &v0, 1e-12, &mut basis, &mut scratch).unwrap();

        for j in 0..3 {
            assert_close(t.diagonal()[j], 2.0, 1e-12);
        }
        for j in 0..2 {
            assert_close(t.off_diagonal()[j], 1.0, 1e-12);
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
    fn basis_is_orthonormal_and_projection_matches_the_tridiagonal() {
        let a = [4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 5.0];
        let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
        let mut buffer = [0.0; 9];
        let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
        let mut scratch = [0.0; 3];

        let t = lanczos(
            &StaticStorage::new(a),
            3,
            &v0,
            1e-12,
            &mut basis,
            &mut scratch,
        )
        .unwrap();

        for r in 0..3 {
            for c in 0..3 {
                let q_r = basis.vector(r).unwrap();
                let q_c = basis.vector(c).unwrap();
                let inner: f64 = q_r.iter().zip(q_c.iter()).map(|(x, y)| x * y).sum();
                assert_close(inner, if r == c { 1.0 } else { 0.0 }, 1e-12);

                let expected = match (r, c) {
                    _ if r == c => t.diagonal()[r],
                    _ if r.abs_diff(c) == 1 => t.off_diagonal()[r.min(c)],
                    _ => 0.0,
                };
                assert_close(projection_entry(&a, 3, &basis, r, c), expected, 1e-12);
            }
        }
    }

    #[test]
    fn partial_basis_projects_onto_the_leading_block() {
        // K == 2 < n == 3: the projection of A onto the two-vector basis is the 2x2 leading
        // block of the tridiagonal form.
        let a = [4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 5.0];
        let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
        let mut buffer = [0.0; 6];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 3).unwrap();
        let mut scratch = [0.0; 3];

        let t = lanczos(
            &StaticStorage::new(a),
            3,
            &v0,
            1e-12,
            &mut basis,
            &mut scratch,
        )
        .unwrap();

        let q_0 = basis.vector(0).unwrap();
        let q_1 = basis.vector(1).unwrap();
        let inner: f64 = q_0.iter().zip(q_1.iter()).map(|(x, y)| x * y).sum();
        assert_close(inner, 0.0, 1e-12);

        let mut projected = [[0.0; 2]; 2];
        for (r, q_r) in [q_0, q_1].iter().enumerate() {
            for (c, q_c) in [q_0, q_1].iter().enumerate() {
                for row in 0..3 {
                    for col in 0..3 {
                        projected[r][c] += q_r[row] * a[row * 3 + col] * q_c[col];
                    }
                }
            }
        }
        assert_close(projected[0][0], t.diagonal()[0], 1e-12);
        assert_close(projected[1][1], t.diagonal()[1], 1e-12);
        assert_close(projected[0][1], t.off_diagonal()[0], 1e-12);
        assert_close(projected[1][0], t.off_diagonal()[0], 1e-12);
    }

    #[test]
    fn k_equals_one_yields_the_rayleigh_quotient() {
        // With a single basis vector there is no off-diagonal: α_0 is the Rayleigh quotient
        // of the normalized v0. v0 = [3, 4]/5, α_0 = v0ᵗ A v0 = (9*2 + 16*1)/25.
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([3.0, 4.0]);
        let mut buffer = [0.0; 2];
        let mut basis = Basis::<f64, 1>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let t = lanczos(&a, 2, &v0, 1e-12, &mut basis, &mut scratch).unwrap();

        assert_close(t.diagonal()[0], 34.0 / 25.0, 1e-12);
        assert!(t.off_diagonal().is_empty());
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

        let t = lanczos(&a, 2, &v0, 1e-12, &mut basis, &mut scratch).unwrap();

        assert!(t.diagonal().is_empty());
        assert!(t.off_diagonal().is_empty());
    }

    #[test]
    fn repeated_eigenvalue_breaks_down_instead_of_panicking() {
        // The identity's Krylov subspace is one-dimensional for every v0 (A*v0 = v0), so a
        // second basis vector can't exist: β_0 == 0 exactly.
        let a = StaticStorage::new([1.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 1.0]);
        let mut buffer = [0.0; 4];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let result = lanczos(&a, 2, &v0, 1e-12, &mut basis, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::Breakdown));
    }

    #[test]
    fn v0_inside_an_invariant_subspace_breaks_down_instead_of_panicking() {
        // diag(2, 1) with v0 = e1: an exact eigenvector spans an invariant subspace, so the
        // basis stops at one vector even though the eigenvalues are distinct.
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut buffer = [0.0; 4];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let result = lanczos(&a, 2, &v0, 1e-12, &mut basis, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::Breakdown));
    }

    #[test]
    fn zero_initial_vector_is_an_error_not_a_panic() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([0.0, 0.0]);
        let mut buffer = [0.0; 4];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let result = lanczos(&a, 2, &v0, 1e-12, &mut basis, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::ZeroVector));
    }

    #[test]
    fn non_finite_initial_vector_is_a_distinct_error_from_zero_vector() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([f64::NAN, 0.0]);
        let mut buffer = [0.0; 4];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let result = lanczos(&a, 2, &v0, 1e-12, &mut basis, &mut scratch);

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

            let result = lanczos(&a, 2, &v0, 1e-12, &mut basis, &mut scratch);

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
            lanczos(&a, 3, &v0_3, 1e-12, &mut basis_3, &mut scratch_3),
            Err(ConvergenceError::DimensionMismatch)
        );

        // `v0` too short for n == 2.
        let v0_short = StaticStorage::new([1.0]);
        let mut buffer = [0.0; 4];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];
        assert_eq!(
            lanczos(&a, 2, &v0_short, 1e-12, &mut basis, &mut scratch),
            Err(ConvergenceError::DimensionMismatch)
        );

        // Basis vectors of the wrong length.
        let mut buffer_wrong = [0.0; 2];
        let mut basis_wrong = Basis::<f64, 2>::new(&mut buffer_wrong, 1).unwrap();
        assert_eq!(
            lanczos(&a, 2, &v0, 1e-12, &mut basis_wrong, &mut scratch),
            Err(ConvergenceError::DimensionMismatch)
        );

        // Scratch buffer too short.
        let mut scratch_short = [0.0; 1];
        assert_eq!(
            lanczos(&a, 2, &v0, 1e-12, &mut basis, &mut scratch_short),
            Err(ConvergenceError::DimensionMismatch)
        );

        // K > n: a 2-dimensional space has no 3 orthonormal directions.
        let mut buffer_deep = [0.0; 6];
        let mut basis_deep = Basis::<f64, 3>::new(&mut buffer_deep, 2).unwrap();
        assert_eq!(
            lanczos(&a, 2, &v0, 1e-12, &mut basis_deep, &mut scratch),
            Err(ConvergenceError::DimensionMismatch)
        );
    }
}

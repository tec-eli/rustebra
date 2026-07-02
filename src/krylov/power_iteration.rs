use super::ConvergenceError;
use crate::algorithm::matrix::{abs, mul_vector};
use crate::algorithm::vector::{dot, norm};
use crate::scalar::Scalar;
use crate::storage::Storage;

/// A read-only [`Storage`] view over a flat slice, so the caller-provided iterate buffers
/// (plain `&mut [T]`, not any [`Storage`] implementor) can be passed as the vector operand of
/// [`crate::algorithm::matrix::mul_vector`] and [`crate::algorithm::vector::dot`].
pub(super) struct Slice<'a, T> {
    pub(super) data: &'a [T],
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

/// Computes the dominant (largest-magnitude) eigenvalue of the `n x n`, row-major matrix `a`
/// by power iteration, returning the eigenvalue and writing a corresponding unit-length
/// eigenvector into `out_eigenvector`.
///
/// Starting from `v0` (normalized first), each iteration multiplies the current estimate by
/// `a` and renormalizes: `v_{k+1} = a * v_k / ‖a * v_k‖`. The eigenvalue estimate is the
/// Rayleigh quotient `λ_k = v_kᵗ * (a * v_k)` (its denominator `v_kᵗ * v_k` is `1`, since
/// `v_k` is kept at unit length). Renormalizing on *every* iteration is what keeps repeated
/// multiplication from overflowing (`|λ| > 1`) or underflowing (`|λ| < 1`): the iterate's
/// magnitude is reset to `1` each round, so only its direction evolves.
///
/// # Convergence
///
/// The error in the eigenvector direction shrinks by a factor of roughly `|λ2 / λ1|` per
/// iteration, where `λ1` and `λ2` are the eigenvalues of largest and second-largest
/// magnitude. The method converges quickly when the dominant eigenvalue is well separated,
/// slowly when `|λ2|` is close to `|λ1|`, and not at all when `|λ2| == |λ1|` with `λ2 != λ1`
/// (e.g. `λ2 == -λ1`, or the complex-conjugate dominant pair of a non-symmetric matrix): the
/// iterate then oscillates forever and the call returns
/// [`ConvergenceError::MaxIterationsExceeded`]. A *repeated* dominant eigenvalue
/// (`λ2 == λ1`) is fine — every vector in the shared eigenspace is a valid answer, and the
/// iteration settles on one.
///
/// Iteration stops once both convergence measures fall within `tol`, relative to the
/// eigenvalue's magnitude:
///
/// - eigenvalue stabilization: `|λ_k - λ_{k-1}| <= tol * |λ_k|`, and
/// - eigenvector stabilization, measured by the residual of the eigenpair equation:
///   `‖a * v_k - λ_k * v_k‖ <= tol * |λ_k|`.
///
/// The first measure needs a previous estimate to compare against, so at least two iterations
/// always run and `max_iter < 2` can never converge. If `v0` has no component along the
/// dominant eigenvector, the iteration converges (in exact arithmetic) to the dominant
/// eigenpair of the invariant subspace `v0` does reach — rounding error usually leaks a
/// dominant component back in and rescues the full answer, but don't rely on it.
///
/// `scratch` is a caller-provided buffer of length `n` holding `a * v_k` each iteration;
/// `Storage` exposes no way to allocate one internally (the same constraint
/// [`crate::algorithm::matrix::qr_householder`]'s `scratch` parameter works around).
///
/// # Smallest eigenvalues: shift-and-invert (reference only)
///
/// Power iteration always targets the *largest*-magnitude eigenvalue. To find the smallest,
/// or the one nearest a shift `σ`, apply this same iteration to `(a - σ * I)⁻¹` instead of
/// `a` — i.e. replace the `a * v_k` product with solving `(a - σ * I) * w = v_k`, e.g. via
/// [`crate::algorithm::matrix::lu_partial_pivot`]. The eigenvalues of that inverse are
/// `1 / (λ - σ)`, so the eigenvalue of `a` nearest `σ` becomes the dominant one of the
/// iterated operator. No such variant is provided here.
///
/// # Errors
///
/// - [`ConvergenceError::DimensionMismatch`] if `a` doesn't have exactly `n * n` elements, or
///   any of `v0`, `out_eigenvector`, `scratch` doesn't have exactly `n` elements.
/// - [`ConvergenceError::ZeroVector`] if `v0` has zero norm (including `n == 0`, where every
///   vector is empty), or an iterate lands in the null space of `a` and maps to zero.
/// - [`ConvergenceError::MaxIterationsExceeded`] if the criteria above aren't met within
///   `max_iter` iterations.
///
/// # Examples
///
/// ```
/// use rustebra::krylov::power_iteration;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[2, 0], [0, 1]] — eigenvalues 2 and 1.
/// let a = StaticStorage::new([2.0_f64, 0.0, 0.0, 1.0]);
/// let v0 = StaticStorage::new([1.0, 1.0]);
/// let mut eigenvector = [0.0; 2];
/// let mut scratch = [0.0; 2];
/// let eigenvalue =
///     power_iteration(&a, 2, &v0, 200, 1e-12, &mut eigenvector, &mut scratch).unwrap();
///
/// assert!((eigenvalue - 2.0).abs() < 1e-9);
/// // The dominant eigenvector is ±[1, 0].
/// assert!((eigenvector[0].abs() - 1.0).abs() < 1e-9);
/// assert!(eigenvector[1].abs() < 1e-9);
/// ```
pub fn power_iteration<S, V, T>(
    a: &S,
    n: usize,
    v0: &V,
    max_iter: usize,
    tol: T,
    out_eigenvector: &mut [T],
    scratch: &mut [T],
) -> Result<T, ConvergenceError>
where
    S: Storage<Item = T>,
    V: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    if a.len() != n * n || v0.len() != n || out_eigenvector.len() != n || scratch.len() != n {
        return Err(ConvergenceError::DimensionMismatch);
    }

    for (i, slot) in out_eigenvector.iter_mut().enumerate() {
        // `i < n == v0.len()`, so `get` below is always `Some`; handled explicitly rather
        // than panicking.
        let Some(&x) = v0.get(i) else {
            return Err(ConvergenceError::DimensionMismatch);
        };
        *slot = x;
    }
    normalize(out_eigenvector)?;

    let mut prev_eigenvalue: Option<T> = None;
    for _ in 0..max_iter {
        // scratch = a * v_k. Every length was validated above, so this can't disagree;
        // handled explicitly rather than panicking.
        if mul_vector(
            a,
            n,
            n,
            &Slice {
                data: &*out_eigenvector,
            },
            scratch,
        )
        .is_err()
        {
            return Err(ConvergenceError::DimensionMismatch);
        }

        // Rayleigh quotient: `‖v_k‖ == 1`, so `λ_k = v_kᵗ * (a * v_k)` with no denominator.
        let Ok(eigenvalue) = dot(
            &Slice {
                data: &*out_eigenvector,
            },
            &Slice { data: &*scratch },
        ) else {
            return Err(ConvergenceError::DimensionMismatch);
        };

        // ‖a * v_k - λ_k * v_k‖, from the product already in `scratch` — the eigenvector
        // stabilization measure, at no extra matrix-vector cost.
        let mut residual_sq = T::zero();
        for (&w_i, &v_i) in scratch.iter().zip(out_eigenvector.iter()) {
            let r = w_i.sub(eigenvalue.mul(v_i));
            residual_sq = residual_sq.add(r.mul(r));
        }
        let residual = residual_sq.sqrt();

        // Advance to v_{k+1} before the convergence check, so the vector handed back on
        // success is the freshest, unit-length iterate.
        for (slot, &w_i) in out_eigenvector.iter_mut().zip(scratch.iter()) {
            *slot = w_i;
        }
        normalize(out_eigenvector)?;

        let scale = abs(eigenvalue);
        let eigenvalue_stable = match prev_eigenvalue {
            Some(prev) => abs(eigenvalue.sub(prev)) <= tol.mul(scale),
            None => false,
        };
        if eigenvalue_stable && residual <= tol.mul(scale) {
            return Ok(eigenvalue);
        }
        prev_eigenvalue = Some(eigenvalue);
    }
    Err(ConvergenceError::MaxIterationsExceeded)
}

/// Scales `v` in place to unit Euclidean length.
///
/// Errors with `ZeroVector` when `‖v‖` is not strictly positive — that covers the zero
/// vector, the empty (`n == 0`) vector, and a norm poisoned by non-finite input (`NaN`
/// compares false against everything, so it lands in the error arm rather than being
/// divided by).
pub(super) fn normalize<T: Scalar + PartialOrd>(v: &mut [T]) -> Result<(), ConvergenceError> {
    let length = norm(&Slice { data: &*v });
    if length > T::zero() {
        let inv = T::one().div(length);
        for slot in v.iter_mut() {
            *slot = slot.mul(inv);
        }
        Ok(())
    } else {
        Err(ConvergenceError::ZeroVector)
    }
}

#[cfg(test)]
mod tests {
    use super::power_iteration;
    use crate::krylov::ConvergenceError;
    use crate::storage::StaticStorage;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn symmetric_2x2_dominant_eigenpair() {
        // [[2, 1], [1, 2]]: eigenvalues 3 (eigenvector [1, 1]/√2) and 1.
        let a = StaticStorage::new([2.0, 1.0, 1.0, 2.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue =
            power_iteration(&a, 2, &v0, 500, 1e-12, &mut eigenvector, &mut scratch).unwrap();

        assert_close(eigenvalue, 3.0, 1e-9);
        let inv_sqrt2 = 1.0 / core::f64::consts::SQRT_2;
        assert_close(eigenvector[0].abs(), inv_sqrt2, 1e-6);
        assert_close(eigenvector[1].abs(), inv_sqrt2, 1e-6);
        // Both components share a sign: it really is ±[1, 1]/√2, not ±[1, -1]/√2.
        assert!(eigenvector[0] * eigenvector[1] > 0.0);
    }

    #[test]
    fn one_by_one_matrix() {
        let a = StaticStorage::new([-4.0]);
        let v0 = StaticStorage::new([2.0]);
        let mut eigenvector = [0.0; 1];
        let mut scratch = [0.0; 1];

        let eigenvalue =
            power_iteration(&a, 1, &v0, 10, 1e-12, &mut eigenvector, &mut scratch).unwrap();

        assert_close(eigenvalue, -4.0, 1e-12);
        assert_close(eigenvector[0].abs(), 1.0, 1e-12);
    }

    #[test]
    fn negative_dominant_eigenvalue() {
        // diag(-3, 1): dominant eigenvalue is -3, so the iterate's sign alternates each
        // step, but the Rayleigh quotient and residual still settle.
        let a = StaticStorage::new([-3.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 1.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue =
            power_iteration(&a, 2, &v0, 500, 1e-12, &mut eigenvector, &mut scratch).unwrap();

        assert_close(eigenvalue, -3.0, 1e-9);
        assert_close(eigenvector[0].abs(), 1.0, 1e-6);
        assert_close(eigenvector[1].abs(), 0.0, 1e-6);
    }

    #[test]
    fn repeated_dominant_eigenvalue_settles_in_the_eigenspace() {
        // The identity has eigenvalue 1 with multiplicity 2: any unit vector is a valid
        // eigenvector, and the iteration keeps the direction of v0.
        let a = StaticStorage::new([1.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([3.0, 4.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue =
            power_iteration(&a, 2, &v0, 10, 1e-12, &mut eigenvector, &mut scratch).unwrap();

        assert_close(eigenvalue, 1.0, 1e-12);
        assert_close(eigenvector[0], 0.6, 1e-12);
        assert_close(eigenvector[1], 0.8, 1e-12);
    }

    #[test]
    fn equal_magnitude_opposite_sign_spectrum_never_converges() {
        // [[0, 1], [1, 0]] has eigenvalues +1 and -1: |λ2/λ1| == 1, and a v0 that mixes
        // both eigenvectors oscillates forever between [1, 0] and [0, 1].
        let a = StaticStorage::new([0.0, 1.0, 1.0, 0.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        let result = power_iteration(&a, 2, &v0, 200, 1e-12, &mut eigenvector, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::MaxIterationsExceeded));
    }

    #[test]
    fn degenerate_spectrum_still_converges_from_an_exact_eigenvector() {
        // Same ±1 spectrum as above, but v0 = [1, 1] is already the λ == 1 eigenvector, so
        // there's nothing left to oscillate.
        let a = StaticStorage::new([0.0, 1.0, 1.0, 0.0]);
        let v0 = StaticStorage::new([1.0, 1.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue =
            power_iteration(&a, 2, &v0, 10, 1e-12, &mut eigenvector, &mut scratch).unwrap();

        assert_close(eigenvalue, 1.0, 1e-12);
    }

    #[test]
    fn v0_orthogonal_to_dominant_eigenvector_finds_the_reachable_subspace() {
        // diag(2, 1) with v0 = [0, 1]: v0 spans the λ == 1 invariant subspace exactly, and a
        // diagonal matrix produces no rounding leakage back into the first component.
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([0.0, 1.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue =
            power_iteration(&a, 2, &v0, 10, 1e-12, &mut eigenvector, &mut scratch).unwrap();

        assert_close(eigenvalue, 1.0, 1e-12);
        assert_close(eigenvector[0], 0.0, 1e-12);
        assert_close(eigenvector[1].abs(), 1.0, 1e-12);
    }

    #[test]
    fn returned_eigenvector_satisfies_the_eigenpair_equation() {
        // Non-symmetric matrix with well-separated real eigenvalues 4 and 1.
        let a = [3.0, 2.0, 1.0, 2.0];
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue = power_iteration(
            &StaticStorage::new(a),
            2,
            &v0,
            500,
            1e-12,
            &mut eigenvector,
            &mut scratch,
        )
        .unwrap();

        assert_close(eigenvalue, 4.0, 1e-9);
        for row in 0..2 {
            let av = a[row * 2] * eigenvector[0] + a[row * 2 + 1] * eigenvector[1];
            assert_close(av, eigenvalue * eigenvector[row], 1e-8);
        }
    }

    #[test]
    fn zero_initial_vector_is_an_error_not_a_panic() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([0.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        let result = power_iteration(&a, 2, &v0, 10, 1e-12, &mut eigenvector, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::ZeroVector));
    }

    #[test]
    fn iterate_mapped_into_the_null_space_is_an_error_not_a_panic() {
        // The zero matrix maps every v0 to zero: dominant eigenvalue 0 is a breakdown, not
        // a value this method can report.
        let a = StaticStorage::new([0.0, 0.0, 0.0, 0.0]);
        let v0 = StaticStorage::new([1.0, 1.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        let result = power_iteration(&a, 2, &v0, 10, 1e-12, &mut eigenvector, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::ZeroVector));
    }

    #[test]
    fn too_small_iteration_budget_is_an_error_not_a_panic() {
        // Convergence needs a previous eigenvalue estimate to compare against, so even an
        // already-exact eigenpair can't be accepted in under two iterations.
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        for max_iter in [0, 1] {
            let result =
                power_iteration(&a, 2, &v0, max_iter, 1e-12, &mut eigenvector, &mut scratch);
            assert_eq!(result, Err(ConvergenceError::MaxIterationsExceeded));
        }
    }

    #[test]
    fn mismatched_dimensions_are_an_error_not_a_panic() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);

        // `a` has 4 elements but 3 x 3 is claimed.
        let mut eigenvector3 = [0.0; 3];
        let mut scratch3 = [0.0; 3];
        let v0_3 = StaticStorage::new([1.0, 0.0, 0.0]);
        assert_eq!(
            power_iteration(&a, 3, &v0_3, 10, 1e-12, &mut eigenvector3, &mut scratch3),
            Err(ConvergenceError::DimensionMismatch)
        );

        // `v0` too short for n == 2.
        let v0_short = StaticStorage::new([1.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];
        assert_eq!(
            power_iteration(&a, 2, &v0_short, 10, 1e-12, &mut eigenvector, &mut scratch),
            Err(ConvergenceError::DimensionMismatch)
        );

        // Output buffer too short.
        let mut eigenvector_short = [0.0; 1];
        assert_eq!(
            power_iteration(&a, 2, &v0, 10, 1e-12, &mut eigenvector_short, &mut scratch),
            Err(ConvergenceError::DimensionMismatch)
        );

        // Scratch buffer too short.
        let mut scratch_short = [0.0; 1];
        assert_eq!(
            power_iteration(&a, 2, &v0, 10, 1e-12, &mut eigenvector, &mut scratch_short),
            Err(ConvergenceError::DimensionMismatch)
        );
    }
}

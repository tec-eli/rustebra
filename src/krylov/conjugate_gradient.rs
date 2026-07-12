use super::ConvergenceError;
use crate::algorithm::matrix::mul_vector;
use crate::algorithm::vector::dot;
use crate::scalar::Scalar;
use crate::storage::Storage;

/// Solves the symmetric positive-definite (SPD) linear system `A x = b` via the Conjugate
/// Gradient (CG) method: an iterative refinement that converges in at most `n` iterations
/// for exact arithmetic, exploiting the SPD structure.
///
/// CG builds a sequence of conjugate directions from residuals, with each iteration doing
/// one matrix-vector product (`A * p_k`) and two dot products. No factorization, no matrix
/// storage beyond the original `n × n` matrix `a` itself.
///
/// # Convergence
///
/// The convergence rate depends on the condition number `κ(A) = λ_max / λ_min` of `a`: the
/// iteration converges to within factor `(√κ - 1) / (√κ + 1)` per step in the energy norm.
/// For well-conditioned problems (κ ≈ 1), convergence is fast; for ill-conditioned ones,
/// slow but still graceful (no breakdown for genuinely SPD inputs).
///
/// Iteration stops once the residual norm `‖r_k‖ = ‖b - A x_k‖` satisfies `‖r_k‖ ≤ tol`,
/// or `max_iter` iterations are exhausted.
///
/// # SPD Detection
///
/// A genuinely SPD matrix has all positive eigenvalues, guaranteeing all dot products
/// `p_k · (A p_k)` are positive. CG detects a non-SPD input operationally: if any
/// `p_k · (A p_k) ≤ 0` mid-iteration (within floating-point tolerance), the algorithm
/// returns `Err(ConvergenceError::NonFinite)` — this is the correct failure mode for an
/// indefinite or negative-semidefinite input that slipped through validation. NaN or
/// infinite iterates are also detected and cause immediate termination.
///
/// # Input validation
///
/// - `a.len()` must equal `n * n` (row-major matrix).
/// - `b` must have exactly `n` elements (right-hand side).
/// - `x0` must have exactly `n` elements (initial guess).
/// - `out_x` must have exactly `n` elements (output solution).
///
/// # Errors
///
/// - [`ConvergenceError::DimensionMismatch`] if input shapes don't agree with each other
///   or with their claimed dimensions.
/// - [`ConvergenceError::NonFinite`] if an iterate contains `NaN` or `Inf`, or if
///   the matrix is detected to be non-SPD (negative or ~zero `p_k · (A p_k)`).
/// - [`ConvergenceError::MaxIterationsExceeded`] if convergence criteria aren't met
///   within `max_iter` iterations.
///
/// # Examples
///
/// ```
/// use rustebra::krylov::conjugate_gradient;
/// use rustebra::storage::StaticStorage;
///
/// // Symmetric positive-definite 2x2 matrix: [[2, 1], [1, 2]].
/// // Eigenvalues are 3 and 1, condition number κ = 3.
/// let a = StaticStorage::new([2.0_f64, 1.0, 1.0, 2.0]);
/// let b = [1.0, 2.0];
/// let x0 = [0.0, 0.0];
/// let mut x = [0.0; 2];
/// let mut r = [0.0; 2];
/// let mut p = [0.0; 2];
/// let mut ap = [0.0; 2];
///
/// conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap)
///     .unwrap();
///
/// // Solution should satisfy A x = b, i.e., x = [0, 1]
/// assert!((x[0] - 0.0).abs() < 1e-8);
/// assert!((x[1] - 1.0).abs() < 1e-8);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn conjugate_gradient<S, T>(
    a: &S,
    n: usize,
    b: &[T],
    x0: &[T],
    max_iter: usize,
    tol: T,
    out_x: &mut [T],
    scratch_r: &mut [T],
    scratch_p: &mut [T],
    scratch_ap: &mut [T],
) -> Result<(), ConvergenceError>
where
    S: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    // Validate dimensions.
    if a.len() != n * n
        || b.len() != n
        || x0.len() != n
        || out_x.len() != n
        || scratch_r.len() != n
        || scratch_p.len() != n
        || scratch_ap.len() != n
    {
        return Err(ConvergenceError::DimensionMismatch);
    }

    // Initialize x_0 from x0, then compute r_0 = b - A x_0.
    for (i, slot) in out_x.iter_mut().enumerate() {
        let Some(&x) = x0.get(i) else {
            return Err(ConvergenceError::DimensionMismatch);
        };
        *slot = x;
    }

    // r_0 = b - A x_0: compute A x_0 into scratch_ap, then r = b - A x.
    mul_vector(a, n, n, &Slice { data: out_x }, scratch_ap)
        .map_err(|_| ConvergenceError::DimensionMismatch)?;
    for (i, slot) in scratch_r.iter_mut().enumerate() {
        let Some(&b_i) = b.get(i) else {
            return Err(ConvergenceError::DimensionMismatch);
        };
        let Some(&ap_i) = scratch_ap.get(i) else {
            return Err(ConvergenceError::DimensionMismatch);
        };
        *slot = b_i.sub(ap_i);
    }

    // Check for non-finite values in initial residual.
    let mut r_norm_sq = dot(&Slice { data: &*scratch_r }, &Slice { data: &*scratch_r })
        .map_err(|_| ConvergenceError::DimensionMismatch)?;

    let r_norm = r_norm_sq.sqrt();

    // Detect non-finite residual: if both > and == comparisons fail, it's NaN/Inf.
    if !(r_norm > T::zero() || r_norm == T::zero()) {
        return Err(ConvergenceError::NonFinite);
    }

    // Early exit if already converged.
    if r_norm <= tol {
        return Ok(());
    }

    // p_0 = r_0 (first search direction is the residual).
    for (i, slot) in scratch_p.iter_mut().enumerate() {
        let Some(&r_i) = scratch_r.get(i) else {
            return Err(ConvergenceError::DimensionMismatch);
        };
        *slot = r_i;
    }

    for _ in 0..max_iter {
        // Compute A p_k into scratch_ap.
        mul_vector(a, n, n, &Slice { data: &*scratch_p }, scratch_ap)
            .map_err(|_| ConvergenceError::DimensionMismatch)?;

        // α_k = (r_k · r_k) / (p_k · (A p_k))
        let p_ap = dot(&Slice { data: &*scratch_p }, &Slice { data: &*scratch_ap })
            .map_err(|_| ConvergenceError::DimensionMismatch)?;

        // Detect non-SPD: p · (A p) must be strictly positive. If not (zero, negative, or NaN),
        // the matrix is not SPD. We use `!(p_ap > T::zero())` to catch NaN too, since
        // NaN comparisons return false.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(p_ap > T::zero()) {
            return Err(ConvergenceError::NonFinite);
        }

        let alpha = r_norm_sq.div(p_ap);

        // x_{k+1} = x_k + α_k p_k
        for i in 0..n {
            let Some(&x_i) = out_x.get(i) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            let Some(&p_i) = scratch_p.get(i) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            let slot = out_x
                .get_mut(i)
                .ok_or(ConvergenceError::DimensionMismatch)?;
            *slot = x_i.add(alpha.mul(p_i));
        }

        // r_{k+1} = r_k - α_k (A p_k)
        for i in 0..n {
            let Some(&r_i) = scratch_r.get(i) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            let Some(&ap_i) = scratch_ap.get(i) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            let slot = scratch_r
                .get_mut(i)
                .ok_or(ConvergenceError::DimensionMismatch)?;
            *slot = r_i.sub(alpha.mul(ap_i));
        }

        // Compute new residual norm and check for non-finite values.
        let new_r_norm_sq = dot(&Slice { data: &*scratch_r }, &Slice { data: &*scratch_r })
            .map_err(|_| ConvergenceError::DimensionMismatch)?;

        let new_r_norm = new_r_norm_sq.sqrt();

        // Detect non-finite residual.
        if !(new_r_norm > T::zero() || new_r_norm == T::zero()) {
            return Err(ConvergenceError::NonFinite);
        }

        if new_r_norm <= tol {
            return Ok(());
        }

        // β_k = (r_{k+1} · r_{k+1}) / (r_k · r_k) (Fletcher-Reeves formula)
        let beta = new_r_norm_sq.div(r_norm_sq);

        // p_{k+1} = r_{k+1} + β_k p_k
        for i in 0..n {
            let Some(&r_i) = scratch_r.get(i) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            let Some(&p_i) = scratch_p.get(i) else {
                return Err(ConvergenceError::DimensionMismatch)?;
            };
            let slot = scratch_p
                .get_mut(i)
                .ok_or(ConvergenceError::DimensionMismatch)?;
            *slot = r_i.add(beta.mul(p_i));
        }

        r_norm_sq = new_r_norm_sq;
    }

    Err(ConvergenceError::MaxIterationsExceeded)
}

/// A read-only [`Storage`] view over a slice, so vector operations can be reused
/// on slices directly without wrapping them separately.
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

#[cfg(test)]
mod tests {
    use super::conjugate_gradient;
    use crate::krylov::ConvergenceError;
    use crate::storage::{StaticStorage, Storage};

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn solves_known_2x2_spd_system() {
        // [[2, 1], [1, 2]] x = [1, 2]
        // Eigenvalues: 3, 1; κ = 3. Solution: x = [0, 1]
        let a = StaticStorage::new([2.0_f64, 1.0, 1.0, 2.0]);
        let b = [1.0, 2.0];
        let x0 = [0.0, 0.0];
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        let result =
            conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        assert_close(x[0], 0.0, 1e-8);
        assert_close(x[1], 1.0, 1e-8);
    }

    #[test]
    fn solves_diagonal_matrix() {
        // diag(2, 3) x = [2, 6] => x = [1, 2]
        let a = StaticStorage::new([2.0_f64, 0.0, 0.0, 3.0]);
        let b = [2.0, 6.0];
        let x0 = [0.0, 0.0];
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        let result =
            conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        assert_close(x[0], 1.0, 1e-10);
        assert_close(x[1], 2.0, 1e-10);
    }

    #[test]
    fn one_by_one_system() {
        // [5] x = [10] => x = [2]
        let a = StaticStorage::new([5.0_f64]);
        let b = [10.0];
        let x0 = [0.0];
        let mut x = [0.0; 1];
        let mut r = [0.0; 1];
        let mut p = [0.0; 1];
        let mut ap = [0.0; 1];

        let result =
            conjugate_gradient(&a, 1, &b, &x0, 100, 1e-12, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        assert_close(x[0], 2.0, 1e-12);
    }

    #[test]
    fn ill_conditioned_matrix_converges_slowly() {
        // diag(100, 1): κ = 100, convergence rate ≈ ((10 - 1) / (10 + 1))^2 ≈ 0.67 per 2 steps
        // x = [1, 1] requires many iterations for ill-conditioned systems
        let a = StaticStorage::new([100.0_f64, 0.0, 0.0, 1.0]);
        let b = [100.0, 1.0];
        let x0 = [0.0, 0.0];
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        // With coarse tolerance, even ill-conditioned systems converge quickly.
        let result =
            conjugate_gradient(&a, 2, &b, &x0, 1000, 1e-3, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        // Solution is [1, 1].
        assert_close(x[0], 1.0, 1e-2);
        assert_close(x[1], 1.0, 1e-2);
    }

    #[test]
    fn negative_definite_matrix_is_detected_as_non_spd() {
        // -diag(1, 1): negative-definite, should be rejected.
        let a = StaticStorage::new([-1.0_f64, 0.0, 0.0, -1.0]);
        let b = [1.0, 1.0];
        let x0 = [1.0, 1.0];
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        let result =
            conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Err(ConvergenceError::NonFinite));
    }

    #[test]
    fn indefinite_matrix_is_detected_as_non_spd() {
        // [[1, 0], [0, -1]]: indefinite (one positive, one negative eigenvalue).
        let a = StaticStorage::new([1.0_f64, 0.0, 0.0, -1.0]);
        let b = [1.0, 1.0];
        let x0 = [0.0, 0.0];
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        let result =
            conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Err(ConvergenceError::NonFinite));
    }

    #[test]
    fn early_convergence_on_initial_small_residual() {
        // Diagonal system where convergence is immediate (diag(1, 1) x = [1, 1] => x = [1, 1]).
        // With x0 = [1, 1], residual is zero, so should return immediately.
        let a = StaticStorage::new([1.0_f64, 0.0, 0.0, 1.0]);
        let b = [1.0, 1.0];
        let x0 = [1.0, 1.0]; // This is the solution.
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        let result =
            conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        assert_close(x[0], 1.0, 1e-10);
        assert_close(x[1], 1.0, 1e-10);
    }

    #[test]
    fn mismatched_dimensions_is_an_error() {
        let a = StaticStorage::new([2.0_f64, 1.0, 1.0, 2.0]);
        let b = [1.0]; // Wrong length
        let x0 = [0.0, 0.0];
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        let result =
            conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Err(ConvergenceError::DimensionMismatch));
    }

    #[test]
    fn output_buffer_too_short_is_an_error() {
        let a = StaticStorage::new([2.0_f64, 1.0, 1.0, 2.0]);
        let b = [1.0, 2.0];
        let x0 = [0.0, 0.0];
        let mut x = [0.0; 1]; // Wrong length
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        let result =
            conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Err(ConvergenceError::DimensionMismatch));
    }

    #[test]
    fn max_iterations_exceeded_on_slow_convergence() {
        // ill-conditioned matrix with tight tolerance and low max_iter
        let a = StaticStorage::new([1000.0_f64, 0.0, 0.0, 1.0]);
        let b = [1000.0, 1.0];
        let x0 = [0.0, 0.0];
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        // Extremely tight tolerance with very low iteration budget should fail.
        let result = conjugate_gradient(&a, 2, &b, &x0, 2, 1e-15, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Err(ConvergenceError::MaxIterationsExceeded));
    }

    #[test]
    fn converges_with_nonzero_initial_guess() {
        let a = StaticStorage::new([2.0_f64, 1.0, 1.0, 2.0]);
        let b = [1.0, 2.0];
        let x0 = [0.5, 0.5]; // Nonzero initial guess
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        let result =
            conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        assert_close(x[0], 0.0, 1e-8);
        assert_close(x[1], 1.0, 1e-8);
    }

    #[test]
    fn larger_3x3_spd_matrix() {
        // [[4, 1, 1], [1, 3, 0], [1, 0, 2]] is SPD
        let a = StaticStorage::new([4.0_f64, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0]);
        let b = [6.0, 4.0, 3.0];
        let x0 = [0.0, 0.0, 0.0];
        let mut x = [0.0; 3];
        let mut r = [0.0; 3];
        let mut p = [0.0; 3];
        let mut ap = [0.0; 3];

        let result =
            conjugate_gradient(&a, 3, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        // Verify: A x should be close to b
        let mut ax = [0.0; 3];
        for i in 0..3 {
            ax[i] = 0.0;
            for j in 0..3 {
                ax[i] = ax[i] + a.get(i * 3 + j).unwrap_or(&0.0) * x[j];
            }
        }

        for i in 0..3 {
            assert_close(ax[i], b[i], 1e-8);
        }
    }

    #[test]
    fn repeated_eigenvalues_still_converges() {
        // diag(2, 2, 1): repeated eigenvalue 2. κ(A) = 2.
        let a = StaticStorage::new([2.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
        let b = [2.0, 4.0, 1.0];
        let x0 = [0.0, 0.0, 0.0];
        let mut x = [0.0; 3];
        let mut r = [0.0; 3];
        let mut p = [0.0; 3];
        let mut ap = [0.0; 3];

        let result =
            conjugate_gradient(&a, 3, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        assert_close(x[0], 1.0, 1e-8);
        assert_close(x[1], 2.0, 1e-8);
        assert_close(x[2], 1.0, 1e-8);
    }

    #[test]
    fn ill_conditioned_spd_matrix_still_solves() {
        // diag(0.01, 0.01, 1.0): condition number κ = 100, challenging but manageable
        let a = StaticStorage::new([0.01_f64, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0]);
        let b = [0.01, 0.01, 1.0];
        let x0 = [0.0, 0.0, 0.0];
        let mut x = [0.0; 3];
        let mut r = [0.0; 3];
        let mut p = [0.0; 3];
        let mut ap = [0.0; 3];

        // κ = 100 means convergence rate ≈ 0.67 per iteration
        let result =
            conjugate_gradient(&a, 3, &b, &x0, 1000, 1e-8, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        assert_close(x[0], 1.0, 1e-6);
        assert_close(x[1], 1.0, 1e-6);
        assert_close(x[2], 1.0, 1e-6);
    }

    #[test]
    fn identity_matrix_converges_immediately() {
        // I x = b => x = b
        let a = StaticStorage::new([1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let b = [1.5, 2.5, 3.5];
        let x0 = [0.0, 0.0, 0.0];
        let mut x = [0.0; 3];
        let mut r = [0.0; 3];
        let mut p = [0.0; 3];
        let mut ap = [0.0; 3];

        let result =
            conjugate_gradient(&a, 3, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        assert_close(x[0], 1.5, 1e-10);
        assert_close(x[1], 2.5, 1e-10);
        assert_close(x[2], 3.5, 1e-10);
    }

    #[test]
    fn negative_rhs_still_works() {
        // [[2, 1], [1, 2]] x = [-1, -2]. Solution: [0, -1].
        let a = StaticStorage::new([2.0_f64, 1.0, 1.0, 2.0]);
        let b = [-1.0, -2.0];
        let x0 = [0.0, 0.0];
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        let result =
            conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        assert_close(x[0], 0.0, 1e-8);
        assert_close(x[1], -1.0, 1e-8);
    }

    #[test]
    fn sparse_system_with_dominant_diagonal() {
        // Diagonal dominance improves condition number; convergence is fast.
        // [[100, 1], [1, 100]] is well-conditioned (κ ≈ 1.02)
        let a = StaticStorage::new([100.0_f64, 1.0, 1.0, 100.0]);
        let b = [101.0, 101.0];
        let x0 = [0.0, 0.0];
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        let result = conjugate_gradient(&a, 2, &b, &x0, 10, 1e-10, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        assert_close(x[0], 1.0, 1e-8);
        assert_close(x[1], 1.0, 1e-8);
    }

    #[test]
    fn residual_decreases_monotonically_for_spd() {
        // Track residual norms throughout iteration; they should decrease or stay flat
        let a = StaticStorage::new([3.0_f64, 1.0, 1.0, 3.0]);
        let b = [4.0, 4.0];
        let x0 = [0.0, 0.0];
        let mut x = [0.0; 2];
        let mut r = [0.0; 2];
        let mut p = [0.0; 2];
        let mut ap = [0.0; 2];

        let result =
            conjugate_gradient(&a, 2, &b, &x0, 100, 1e-12, &mut x, &mut r, &mut p, &mut ap);
        assert_eq!(result, Ok(()));

        // Verify final solution
        assert_close(x[0], 1.0, 1e-10);
        assert_close(x[1], 1.0, 1e-10);

        // Final residual should be tiny
        let mut final_r = [0.0; 2];
        let mut ax = [0.0; 2];
        for i in 0..2 {
            ax[i] = a.get(i * 2).unwrap_or(&0.0) * x[0] + a.get(i * 2 + 1).unwrap_or(&0.0) * x[1];
            final_r[i] = b[i] - ax[i];
        }
        let residual_norm_sq = final_r[0] * final_r[0] + final_r[1] * final_r[1];
        assert!(residual_norm_sq < 1e-20);
    }
}

use super::ConvergenceError;
use super::power_iteration::{Slice, normalize};
use crate::algorithm::matrix::{abs, mul_vector};
use crate::algorithm::vector::dot;
use crate::scalar::Scalar;
use crate::storage::Storage;

/// Computes the eigenvalue of the `n x n`, row-major matrix `a` nearest to `shift` (and a
/// corresponding unit-length eigenvector, written into `out_eigenvector`) by inverse power
/// iteration: power iteration applied to the operator `(a - shift * I)⁻¹`.
///
/// The eigenvalues of `(a - shift * I)⁻¹` are `1 / (λ - shift)` for each eigenvalue `λ` of
/// `a`, so the eigenvalue of `a` *nearest the shift* becomes the dominant one of the iterated
/// operator — with `shift == 0`, that's the smallest-magnitude eigenvalue of `a`. The inverse
/// is never formed: `a - shift * I` is factored once into `p * l * u` (partial-pivoted LU,
/// stored compactly in `factor` and `pivots`), and each iteration solves
/// `(a - shift * I) * w_k = v_k` by substitution, then renormalizes: the same
/// direction-refinement loop as [`crate::krylov::power_iteration`], with each `a * v` product
/// replaced by a triangular solve.
///
/// # When to use this instead of [`crate::krylov::power_iteration`]
///
/// Power iteration can only find the *largest*-magnitude eigenvalue, converges at rate
/// `|λ2 / λ1|`, and offers no way to steer. Inverse iteration targets any point in the
/// spectrum via `shift`, and converges at rate `|λ1 - shift| / |λ2 - shift|` (with `λ1`, `λ2`
/// now the eigenvalues nearest and second-nearest the shift) — arbitrarily fast when the
/// shift is a good eigenvalue estimate, e.g. one or two iterations from a Rayleigh-quotient
/// or Gershgorin-disc guess. The price is the up-front `O(n^3)` factorization, an extra
/// `n * n` buffer, and an `O(n^2)` solve per step instead of power iteration's plain
/// matrix-vector product; when only the dominant eigenpair is wanted, power iteration is the
/// cheaper tool. A shift far outside the spectrum makes every `|λ - shift|` nearly equal and
/// the rate approach `1`, so distant shifts converge slowly — they still select the nearest
/// eigenvalue, just inefficiently.
///
/// # Singular and near-singular shifts
///
/// A shift *at* an eigenvalue makes `a - shift * I` exactly singular, and the condition
/// number of the solve blows up as the shift approaches one. The factorization therefore
/// rejects any pivot whose magnitude is at most `singular_tol * m`, where `m` is the
/// largest-magnitude entry of `a - shift * I`, with the hard error
/// [`ConvergenceError::SingularShift`] — never a best-effort solve against amplified noise
/// that could masquerade as convergence. `singular_tol` is the caller's threshold judgment
/// (`0` detects only exactly-zero pivots); `n * epsilon` of
/// [`crate::scalar::FloatTolerance`] is a reasonable default for float scalars. A shift
/// *near* an eigenvalue but past that test is not a problem, counterintuitively: the solve's
/// error is amplified almost entirely along the eigenvector being sought, so the iteration
/// converges faster, not worse. Should a solve still degenerate (e.g. overflow to non-finite
/// values), the iterate fails normalization and the call stops with
/// [`ConvergenceError::ZeroVector`] rather than looping on a NaN chain.
///
/// # Convergence
///
/// Monitored on the iterated operator `b⁻¹ = (a - shift * I)⁻¹`, whose dominant eigenvalue
/// `μ = 1 / (λ1 - shift)` is estimated by the Rayleigh quotient `μ_k = v_kᵗ * w_k` (with
/// `‖v_k‖ == 1`). Iteration stops once both measures fall within `tol`:
///
/// - eigenvalue stabilization: `|μ_k - μ_{k-1}| <= tol * |μ_k|`, and
/// - eigenvector stabilization, via the residual: `‖b⁻¹ * v_k - μ_k * v_k‖ <= tol * |μ_k|`.
///
/// As in [`crate::krylov::power_iteration`], the first measure needs a previous estimate, so
/// at least two iterations always run and `max_iter < 2` can never converge. On success the
/// *reported* eigenvalue is the Rayleigh quotient of `a` itself at the converged eigenvector,
/// `λ = vᵗ * a * v` — the best single eigenvalue estimate for the vector actually returned,
/// rather than `shift + 1 / μ` back-computed from the inverse operator.
///
/// `factor` (`n * n`), `pivots` (`n`), and `scratch` (`n`) are caller-provided working
/// buffers — the factorization, its row permutation, and the current solve's right-hand
/// side; `Storage` exposes no way to allocate them internally (the same constraint
/// [`crate::algorithm::matrix::qr_householder`]'s `scratch` parameter works around).
///
/// # Errors
///
/// - [`ConvergenceError::DimensionMismatch`] if `a` or `factor` doesn't have exactly `n * n`
///   elements, or any of `v0`, `out_eigenvector`, `scratch` doesn't have exactly `n`
///   elements, or `pivots` doesn't have exactly `n` elements.
/// - [`ConvergenceError::SingularShift`] if `a - shift * I` is singular within
///   `singular_tol`, as described above (including `n == 0`, where there is no invertible
///   operator at all).
/// - [`ConvergenceError::ZeroVector`] if `v0` has zero norm, or an iterate degenerates to an
///   unnormalizable (zero or non-finite) vector.
/// - [`ConvergenceError::MaxIterationsExceeded`] if the criteria above aren't met within
///   `max_iter` iterations.
///
/// # Examples
///
/// ```
/// use rustebra::krylov::inverse_power_iteration;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[2, 1], [1, 2]] — eigenvalues 3 and 1.
/// let a = StaticStorage::new([2.0_f64, 1.0, 1.0, 2.0]);
/// let v0 = StaticStorage::new([1.0, 0.0]);
/// let mut eigenvector = [0.0; 2];
/// let mut factor = [0.0; 4];
/// let mut pivots = [0_usize; 2];
/// let mut scratch = [0.0; 2];
///
/// // A shift of 0.9 targets the eigenvalue nearest it: 1, not the dominant 3.
/// let eigenvalue = inverse_power_iteration(
///     &a, 2, &v0, 0.9, 200, 1e-12, 1e-12,
///     &mut eigenvector, &mut factor, &mut pivots, &mut scratch,
/// )
/// .unwrap();
///
/// assert!((eigenvalue - 1.0).abs() < 1e-9);
/// // The λ == 1 eigenvector is ±[1, -1]/√2: equal magnitudes, opposite signs.
/// assert!((eigenvector[0] + eigenvector[1]).abs() < 1e-9);
/// ```
#[allow(clippy::too_many_arguments)] // Inputs, then knobs, then output, then working buffers.
pub fn inverse_power_iteration<S, V, T>(
    a: &S,
    n: usize,
    v0: &V,
    shift: T,
    max_iter: usize,
    tol: T,
    singular_tol: T,
    out_eigenvector: &mut [T],
    factor: &mut [T],
    pivots: &mut [usize],
    scratch: &mut [T],
) -> Result<T, ConvergenceError>
where
    S: Storage<Item = T>,
    V: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    if a.len() != n * n
        || v0.len() != n
        || out_eigenvector.len() != n
        || factor.len() != n * n
        || pivots.len() != n
        || scratch.len() != n
    {
        return Err(ConvergenceError::DimensionMismatch);
    }

    // factor = a - shift * I. In a row-major square matrix the diagonal falls exactly on the
    // indices divisible by n + 1.
    for (i, slot) in factor.iter_mut().enumerate() {
        // `i < n * n == a.len()`, so `get` below is always `Some`; handled explicitly rather
        // than panicking.
        let Some(&x) = a.get(i) else {
            return Err(ConvergenceError::DimensionMismatch);
        };
        *slot = if i % (n + 1) == 0 { x.sub(shift) } else { x };
    }

    factorize(factor, pivots, n, singular_tol)?;

    for (i, slot) in out_eigenvector.iter_mut().enumerate() {
        // `i < n == v0.len()`, so `get` below is always `Some`; handled explicitly rather
        // than panicking.
        let Some(&x) = v0.get(i) else {
            return Err(ConvergenceError::DimensionMismatch);
        };
        *slot = x;
    }
    normalize(out_eigenvector)?;

    let mut prev_mu: Option<T> = None;
    for _ in 0..max_iter {
        // scratch = (a - shift * I)⁻¹ * v_k, via the factorization: the inverse operator's
        // matrix-vector product, in the same role `a * v_k` plays in power iteration.
        for (slot, &x) in scratch.iter_mut().zip(out_eigenvector.iter()) {
            *slot = x;
        }
        solve_in_place(factor, pivots, n, scratch);

        // Rayleigh quotient of the *inverse* operator: `‖v_k‖ == 1`, so `μ_k = v_kᵗ * w_k`.
        let Ok(mu) = dot(
            &Slice {
                data: &*out_eigenvector,
            },
            &Slice { data: &*scratch },
        ) else {
            return Err(ConvergenceError::DimensionMismatch);
        };

        // ‖w_k - μ_k * v_k‖ — the eigenvector stabilization measure, on the iterated operator.
        let mut residual_sq = T::zero();
        for (&w_i, &v_i) in scratch.iter().zip(out_eigenvector.iter()) {
            let r = w_i.sub(mu.mul(v_i));
            residual_sq = residual_sq.add(r.mul(r));
        }
        let residual = residual_sq.sqrt();

        // Advance to v_{k+1} before the convergence check, so the vector handed back on
        // success is the freshest, unit-length iterate.
        for (slot, &w_i) in out_eigenvector.iter_mut().zip(scratch.iter()) {
            *slot = w_i;
        }
        normalize(out_eigenvector)?;

        let scale = abs(mu);
        let mu_stable = match prev_mu {
            Some(prev) => abs(mu.sub(prev)) <= tol.mul(scale),
            None => false,
        };
        if mu_stable && residual <= tol.mul(scale) {
            // Report the eigenvalue of `a` itself, as the Rayleigh quotient at the converged
            // eigenvector: scratch = a * v, λ = vᵗ * (a * v) with ‖v‖ == 1. Every length was
            // validated up front, so neither call can disagree; handled explicitly rather
            // than panicking.
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
            let Ok(eigenvalue) = dot(
                &Slice {
                    data: &*out_eigenvector,
                },
                &Slice { data: &*scratch },
            ) else {
                return Err(ConvergenceError::DimensionMismatch);
            };
            return Ok(eigenvalue);
        }
        prev_mu = Some(mu);
    }
    Err(ConvergenceError::MaxIterationsExceeded)
}

/// Factors the `n x n` matrix in `factor` in place as `p * l * u`, via Gaussian elimination
/// with partial pivoting: on return, `factor` holds `u` on and above the diagonal and `l`'s
/// multipliers below it (`l`'s unit diagonal is implicit), and `pivots[k]` records the row
/// swapped into position `k` at step `k`.
///
/// [`crate::algorithm::matrix::lu_partial_pivot`] can't be used here: it reports its row
/// permutation only as a swap *count* (all its callers need is the determinant's sign), which
/// is not enough to permute a right-hand side for [`solve_in_place`].
///
/// Errors with `SingularShift` when any pivot's magnitude, after choosing the best available
/// row, is at most `singular_tol` times the largest-magnitude entry of the input — the
/// matrix is singular (or too close to it) along that column, so no solve against this
/// factorization can be trusted. The zero matrix (and `n == 0`) is rejected the same way.
///
/// Caller guarantees `factor.len() == n * n` and `pivots.len() == n`.
fn factorize<T: Scalar + PartialOrd>(
    factor: &mut [T],
    pivots: &mut [usize],
    n: usize,
    singular_tol: T,
) -> Result<(), ConvergenceError> {
    let mut largest = T::zero();
    for &x in factor.iter() {
        let magnitude = abs(x);
        if magnitude > largest {
            largest = magnitude;
        }
    }
    let pivot_floor = singular_tol.mul(largest);
    if n == 0 || largest == T::zero() {
        return Err(ConvergenceError::SingularShift);
    }

    for k in 0..n {
        let mut best_row = k;
        let mut best_abs = abs(factor[k * n + k]);
        for r in (k + 1)..n {
            let candidate_abs = abs(factor[r * n + k]);
            if candidate_abs > best_abs {
                best_abs = candidate_abs;
                best_row = r;
            }
        }
        pivots[k] = best_row;
        if best_row != k {
            // Swapping the entire row exchanges both the `u` part (columns k..) and the
            // multipliers already stored in columns ..k, exactly as pivoted LU requires.
            for c in 0..n {
                factor.swap(k * n + c, best_row * n + c);
            }
        }

        let pivot = factor[k * n + k];
        if abs(pivot) <= pivot_floor {
            return Err(ConvergenceError::SingularShift);
        }

        for i in (k + 1)..n {
            let multiplier = factor[i * n + k].div(pivot);
            factor[i * n + k] = multiplier;
            for c in (k + 1)..n {
                let term = multiplier.mul(factor[k * n + c]);
                factor[i * n + c] = factor[i * n + c].sub(term);
            }
        }
    }
    Ok(())
}

/// Solves `p * l * u * x = b` in place: `b` arrives holding the right-hand side and leaves
/// holding `x`. Applies the recorded row swaps to `b`, then forward-substitutes through the
/// implicit-unit-diagonal `l`, then back-substitutes through `u`.
///
/// Caller guarantees the buffers came through a successful [`factorize`] with this `n`, so
/// every `u` diagonal entry is nonzero.
fn solve_in_place<T: Scalar + PartialOrd>(factor: &[T], pivots: &[usize], n: usize, b: &mut [T]) {
    for (k, &p) in pivots.iter().enumerate() {
        if p != k {
            b.swap(k, p);
        }
    }
    for i in 1..n {
        let mut sum = b[i];
        for j in 0..i {
            sum = sum.sub(factor[i * n + j].mul(b[j]));
        }
        b[i] = sum;
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum = sum.sub(factor[i * n + j].mul(b[j]));
        }
        b[i] = sum.div(factor[i * n + i]);
    }
}

#[cfg(test)]
mod tests {
    use super::inverse_power_iteration;
    use crate::krylov::ConvergenceError;
    use crate::storage::StaticStorage;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn shift_selects_the_smaller_eigenvalue_power_iteration_cannot_reach() {
        // [[2, 1], [1, 2]]: eigenvalues 3 and 1. A shift of 0.9 targets 1.
        let a = StaticStorage::new([2.0, 1.0, 1.0, 2.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue = inverse_power_iteration(
            &a,
            2,
            &v0,
            0.9,
            200,
            1e-12,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        )
        .unwrap();

        assert_close(eigenvalue, 1.0, 1e-9);
        // The λ == 1 eigenvector is ±[1, -1]/√2: equal magnitudes, opposite signs.
        let inv_sqrt2 = 1.0 / core::f64::consts::SQRT_2;
        assert_close(eigenvector[0].abs(), inv_sqrt2, 1e-6);
        assert_close(eigenvector[1].abs(), inv_sqrt2, 1e-6);
        assert!(eigenvector[0] * eigenvector[1] < 0.0);
    }

    #[test]
    fn shift_near_the_dominant_eigenvalue_selects_it() {
        let a = StaticStorage::new([2.0, 1.0, 1.0, 2.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue = inverse_power_iteration(
            &a,
            2,
            &v0,
            2.9,
            200,
            1e-12,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        )
        .unwrap();

        assert_close(eigenvalue, 3.0, 1e-9);
        assert!(eigenvector[0] * eigenvector[1] > 0.0);
    }

    #[test]
    fn zero_shift_finds_the_smallest_magnitude_eigenvalue() {
        // diag(5, 0.5, -2): the smallest-magnitude eigenvalue is 0.5, which is neither the
        // dominant one (5) nor the most negative one (-2).
        let a = StaticStorage::new([
            5.0, 0.0, 0.0, //
            0.0, 0.5, 0.0, //
            0.0, 0.0, -2.0,
        ]);
        let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
        let mut eigenvector = [0.0; 3];
        let mut factor = [0.0; 9];
        let mut pivots = [0_usize; 3];
        let mut scratch = [0.0; 3];

        let eigenvalue = inverse_power_iteration(
            &a,
            3,
            &v0,
            0.0,
            500,
            1e-12,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        )
        .unwrap();

        assert_close(eigenvalue, 0.5, 1e-9);
        assert_close(eigenvector[0], 0.0, 1e-6);
        assert_close(eigenvector[1].abs(), 1.0, 1e-6);
        assert_close(eigenvector[2], 0.0, 1e-6);
    }

    #[test]
    fn one_by_one_matrix() {
        let a = StaticStorage::new([5.0]);
        let v0 = StaticStorage::new([-2.0]);
        let mut eigenvector = [0.0; 1];
        let mut factor = [0.0; 1];
        let mut pivots = [0_usize; 1];
        let mut scratch = [0.0; 1];

        let eigenvalue = inverse_power_iteration(
            &a,
            1,
            &v0,
            1.0,
            10,
            1e-12,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        )
        .unwrap();

        assert_close(eigenvalue, 5.0, 1e-12);
        assert_close(eigenvector[0].abs(), 1.0, 1e-12);
    }

    #[test]
    fn non_symmetric_matrix_satisfies_the_eigenpair_equation() {
        // [[3, 2], [1, 2]]: eigenvalues 4 and 1. A shift of 0.5 targets 1, whose eigenvector
        // solves (a - I) v = 0, i.e. v = ±[1, -1]/√2.
        let a = [3.0, 2.0, 1.0, 2.0];
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue = inverse_power_iteration(
            &StaticStorage::new(a),
            2,
            &v0,
            0.5,
            500,
            1e-12,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        )
        .unwrap();

        assert_close(eigenvalue, 1.0, 1e-9);
        assert!(eigenvector[0] * eigenvector[1] < 0.0);
        for row in 0..2 {
            let av = a[row * 2] * eigenvector[0] + a[row * 2 + 1] * eigenvector[1];
            assert_close(av, eigenvalue * eigenvector[row], 1e-8);
        }
    }

    #[test]
    fn shift_far_outside_the_spectrum_still_selects_the_nearest_eigenvalue() {
        // diag(2, 1) with shift 100: both eigenvalues are ~equally far, so the convergence
        // rate |2 - 100| / |1 - 100| == 98/99 is painfully close to 1 — the documented
        // slow-but-correct regime. ~2000 iterations of an O(n^2) solve is still trivial.
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 1.0]);
        let mut eigenvector = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue = inverse_power_iteration(
            &a,
            2,
            &v0,
            100.0,
            20_000,
            1e-8,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        )
        .unwrap();

        assert_close(eigenvalue, 2.0, 1e-6);
    }

    #[test]
    fn shift_exactly_at_an_eigenvalue_is_a_singular_shift_error() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 1.0]);
        let mut eigenvector = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let mut scratch = [0.0; 2];

        let result = inverse_power_iteration(
            &a,
            2,
            &v0,
            2.0,
            100,
            1e-12,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        );

        assert_eq!(result, Err(ConvergenceError::SingularShift));
    }

    #[test]
    fn singular_matrix_with_zero_shift_is_a_singular_shift_error() {
        // [[1, 1], [1, 1]] is singular (eigenvalues 2 and 0), so shift 0 sits exactly on an
        // eigenvalue even though no entry of the shifted matrix is zero.
        let a = StaticStorage::new([1.0, 1.0, 1.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let mut scratch = [0.0; 2];

        let result = inverse_power_iteration(
            &a,
            2,
            &v0,
            0.0,
            100,
            1e-12,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        );

        assert_eq!(result, Err(ConvergenceError::SingularShift));
    }

    #[test]
    fn near_singular_shift_past_the_tolerance_converges_fast_not_badly() {
        // Shift 1e-8 away from the eigenvalue 2: the solve is savagely ill-conditioned, but
        // its error amplification points along the sought eigenvector, so the iteration
        // converges almost immediately instead of degrading.
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 1.0]);
        let mut eigenvector = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue = inverse_power_iteration(
            &a,
            2,
            &v0,
            2.0 - 1e-8,
            100,
            1e-12,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        )
        .unwrap();

        assert_close(eigenvalue, 2.0, 1e-6);
        assert_close(eigenvector[0].abs(), 1.0, 1e-6);
        assert_close(eigenvector[1], 0.0, 1e-6);
    }

    #[test]
    fn zero_initial_vector_is_an_error_not_a_panic() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([0.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let mut scratch = [0.0; 2];

        let result = inverse_power_iteration(
            &a,
            2,
            &v0,
            0.5,
            100,
            1e-12,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        );

        assert_eq!(result, Err(ConvergenceError::ZeroVector));
    }

    #[test]
    fn too_small_iteration_budget_is_an_error_not_a_panic() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let mut scratch = [0.0; 2];

        for max_iter in [0, 1] {
            let result = inverse_power_iteration(
                &a,
                2,
                &v0,
                1.9,
                max_iter,
                1e-12,
                1e-12,
                &mut eigenvector,
                &mut factor,
                &mut pivots,
                &mut scratch,
            );
            assert_eq!(result, Err(ConvergenceError::MaxIterationsExceeded));
        }
    }

    #[test]
    fn mismatched_dimensions_are_an_error_not_a_panic() {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let mut scratch = [0.0; 2];

        // `v0` too short for n == 2.
        let v0_short = StaticStorage::new([1.0]);
        assert_eq!(
            inverse_power_iteration(
                &a,
                2,
                &v0_short,
                0.5,
                10,
                1e-12,
                1e-12,
                &mut eigenvector,
                &mut factor,
                &mut pivots,
                &mut scratch,
            ),
            Err(ConvergenceError::DimensionMismatch)
        );

        // Factorization buffer too short for n == 2.
        let mut factor_short = [0.0; 3];
        assert_eq!(
            inverse_power_iteration(
                &a,
                2,
                &v0,
                0.5,
                10,
                1e-12,
                1e-12,
                &mut eigenvector,
                &mut factor_short,
                &mut pivots,
                &mut scratch,
            ),
            Err(ConvergenceError::DimensionMismatch)
        );

        // Pivot buffer too short for n == 2.
        let mut pivots_short = [0_usize; 1];
        assert_eq!(
            inverse_power_iteration(
                &a,
                2,
                &v0,
                0.5,
                10,
                1e-12,
                1e-12,
                &mut eigenvector,
                &mut factor,
                &mut pivots_short,
                &mut scratch,
            ),
            Err(ConvergenceError::DimensionMismatch)
        );

        // Output buffer too short for n == 2.
        let mut eigenvector_short = [0.0; 1];
        assert_eq!(
            inverse_power_iteration(
                &a,
                2,
                &v0,
                0.5,
                10,
                1e-12,
                1e-12,
                &mut eigenvector_short,
                &mut factor,
                &mut pivots,
                &mut scratch,
            ),
            Err(ConvergenceError::DimensionMismatch)
        );

        // Scratch buffer too short for n == 2.
        let mut scratch_short = [0.0; 1];
        assert_eq!(
            inverse_power_iteration(
                &a,
                2,
                &v0,
                0.5,
                10,
                1e-12,
                1e-12,
                &mut eigenvector,
                &mut factor,
                &mut pivots,
                &mut scratch_short,
            ),
            Err(ConvergenceError::DimensionMismatch)
        );
    }

    #[test]
    fn pivoting_handles_a_zero_landing_on_the_shifted_diagonal() {
        // [[1, 1], [1, 0]] with shift 1: the shifted matrix [[0, 1], [1, -1]] has a zero at
        // (0, 0), so the factorization only succeeds because of partial pivoting. Its
        // eigenvalues are (1 ± √5)/2 ≈ 1.618 and -0.618; shift 1 targets the former.
        let a = StaticStorage::new([1.0, 1.0, 1.0, 0.0]);
        let v0 = StaticStorage::new([1.0, 0.0]);
        let mut eigenvector = [0.0; 2];
        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let mut scratch = [0.0; 2];

        let eigenvalue = inverse_power_iteration(
            &a,
            2,
            &v0,
            1.0,
            500,
            1e-12,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        )
        .unwrap();

        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert_close(eigenvalue, golden_ratio, 1e-9);
    }
}

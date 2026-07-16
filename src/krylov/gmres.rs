use super::ConvergenceError;
use super::power_iteration::{Slice, normalize};
use crate::algorithm::vector::{dot, norm};
use crate::scalar::Scalar;
use crate::sparse::SparseLinearOp;
use crate::storage::Basis;

/// `y -= coefficient * x`, the orthogonalization step's only vector update.
fn subtract_scaled<T: Scalar>(y: &mut [T], coefficient: T, x: &[T]) {
    for (slot, &x_i) in y.iter_mut().zip(x.iter()) {
        *slot = slot.sub(coefficient.mul(x_i));
    }
}

/// Computes `r = b - a * x` into `scratch` and returns `‖r‖`, or `NonFinite` if that norm is
/// `NaN` or infinite.
fn residual_norm<T: Scalar + PartialOrd>(
    a: &impl SparseLinearOp<T>,
    b: &[T],
    x: &[T],
    scratch: &mut [T],
) -> Result<T, ConvergenceError> {
    a.apply(x, scratch)
        .map_err(|_| ConvergenceError::DimensionMismatch)?;
    for (slot, &b_i) in scratch.iter_mut().zip(b.iter()) {
        *slot = b_i.sub(*slot);
    }
    let r_norm = norm(&Slice { data: &*scratch });
    // `x - x` is `0` for every finite `x` and `NaN` for `NaN`/±infinity — the only values
    // unequal to themselves. `r_norm > T::zero()` alone would accept `+Infinity`.
    let probe = r_norm.sub(r_norm);
    #[allow(clippy::eq_op)]
    let non_finite = probe != probe;
    if non_finite {
        Err(ConvergenceError::NonFinite)
    } else {
        Ok(r_norm)
    }
}

/// Solves the general (possibly non-symmetric) linear system `A x = b` via restarted GMRES,
/// GMRES(`M`): Arnoldi iteration builds an `M`-dimensional Krylov basis from the current
/// residual, the resulting least-squares problem over that basis is solved via Givens
/// rotations, and the cycle restarts from the improved iterate until either the residual
/// meets `tol` or `max_restarts` cycles are exhausted.
///
/// `A` is supplied as a [`SparseLinearOp`] rather than a dense matrix: applying it never
/// allocates, so restart cycles reuse the same workspace (`out_x`, `basis`, `scratch`)
/// throughout.
///
/// # Algorithm
///
/// Each restart cycle:
///
/// 1. Computes the residual `r = b - A x` and its norm `β = ‖r‖`, returning `Ok` immediately
///    if `β <= tol`.
/// 2. Runs Arnoldi iteration from `q_0 = r / β`, building an orthonormal basis `Q` of up to
///    `M` vectors and the upper Hessenberg projection `H`, stopping early (before `M` steps)
///    on breakdown — an invariant subspace found before the basis filled up, the same
///    "success, not failure" case documented on [`super::arnoldi`].
/// 3. Solves `min_y ‖β e_1 - H y‖` (a `(reached + 1) x reached` least-squares problem) via
///    incremental Givens rotations, then updates `x <- x + Q y`.
///
/// # Convergence
///
/// GMRES's residual norm decreases monotonically within a cycle (each additional basis
/// vector can only improve the least-squares fit) and never increases across a restart,
/// because restarting recomputes the same residual the next cycle continues from. It is not
/// guaranteed to decrease *strictly* every cycle, though: a starting vector aligned with an
/// invariant subspace the operator doesn't expand (breakdown on the very first Arnoldi step)
/// leaves `x` unchanged, and the iteration stagnates. Restarting also discards the larger
/// Krylov subspace full (non-restarted) GMRES would have kept building, so GMRES(`M`) can
/// converge slower, or stagnate on problems full GMRES would resolve — the restart budget
/// `max_restarts` bounds the cost of that risk rather than eliminating it.
///
/// # Errors
///
/// - [`ConvergenceError::DimensionMismatch`] if `a.rows() != a.cols()`, `b`, `x0`, `out_x`,
///   or a `basis` vector doesn't have exactly `a.rows()` elements, or `M > a.rows()`.
/// - [`ConvergenceError::NonFinite`] if a residual or Arnoldi iterate goes `NaN` or infinite.
/// - [`ConvergenceError::Breakdown`] if the small least-squares system built from `H` has a
///   zero pivot that Arnoldi's own breakdown test didn't already catch (a coincidental exact
///   singularity in the projected system).
/// - [`ConvergenceError::MaxIterationsExceeded`] if the residual hasn't met `tol` after
///   `max_restarts` cycles.
///
/// # Examples
///
/// ```
/// use rustebra::krylov::gmres;
/// use rustebra::sparse::CsrMatrix;
/// use rustebra::storage::Basis;
///
/// // [[4, 1], [2, 3]] x = [1, 2]. Solution: x = [0.1, 0.6].
/// let a = CsrMatrix::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![4.0_f64, 1.0, 2.0, 3.0])
///     .unwrap();
/// let b = [1.0, 2.0];
/// let x0 = [0.0, 0.0];
/// let mut out_x = [0.0; 2];
/// let mut buffer = [0.0; 4];
/// let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
/// let mut scratch = [0.0; 2];
///
/// gmres(&a, &b, &x0, 10, 1e-10, &mut out_x, &mut basis, &mut scratch).unwrap();
///
/// assert!((out_x[0] - 0.1).abs() < 1e-8);
/// assert!((out_x[1] - 0.6).abs() < 1e-8);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn gmres<T, const M: usize>(
    a: &impl SparseLinearOp<T>,
    b: &[T],
    x0: &[T],
    max_restarts: usize,
    tol: T,
    out_x: &mut [T],
    basis: &mut Basis<'_, T, M>,
    scratch: &mut [T],
) -> Result<(), ConvergenceError>
where
    T: Scalar + PartialOrd,
{
    let n = a.rows();
    if a.cols() != n
        || b.len() != n
        || x0.len() != n
        || out_x.len() != n
        || basis.vector_len() != n
        || scratch.len() != n
        || M > n
    {
        return Err(ConvergenceError::DimensionMismatch);
    }

    out_x.copy_from_slice(x0);

    let mut beta = residual_norm(a, b, out_x, scratch)?;
    if beta <= tol {
        return Ok(());
    }

    for _ in 0..max_restarts {
        if M == 0 {
            // No basis vectors can be built; the residual can't be reduced this cycle.
            beta = residual_norm(a, b, out_x, scratch)?;
            if beta <= tol {
                return Ok(());
            }
            continue;
        }

        // q_0 = r / beta, where `r` (unnormalized) is already sitting in `scratch` from the
        // residual computation above.
        normalize(scratch)?;
        {
            let q_0 = basis
                .vector_mut(0)
                .ok_or(ConvergenceError::DimensionMismatch)?;
            q_0.copy_from_slice(scratch);
        }

        let mut h = [[T::zero(); M]; M];
        let mut h_sub = [T::zero(); M];
        let mut reached = 0usize;

        for j in 0..M {
            let norm_aq = {
                let q_j = basis.vector(j).ok_or(ConvergenceError::DimensionMismatch)?;
                a.apply(q_j, scratch)
                    .map_err(|_| ConvergenceError::DimensionMismatch)?;
                norm(&Slice { data: &*scratch })
            };

            for (i, row) in h.iter_mut().enumerate().take(j + 1) {
                let q_i = basis.vector(i).ok_or(ConvergenceError::DimensionMismatch)?;
                let h_ij = dot(&Slice { data: q_i }, &Slice { data: &*scratch })
                    .map_err(|_| ConvergenceError::DimensionMismatch)?;
                row[j] = h_ij;
                subtract_scaled(scratch, h_ij, q_i);
            }

            let h_next = norm(&Slice { data: &*scratch });
            // `x - x` is `0` for every finite `x` and `NaN` for `NaN`/±infinity — the only
            // values unequal to themselves.
            let probe = h_next.sub(h_next);
            #[allow(clippy::eq_op)]
            let non_finite = probe != probe;
            if non_finite {
                return Err(ConvergenceError::NonFinite);
            }

            reached = j + 1;
            // Breakdown: the Krylov subspace built so far is already invariant. Stop
            // extending the basis; the least-squares solve below uses what was built.
            if h_next <= tol.mul(norm_aq) {
                break;
            }
            h_sub[j] = h_next;

            if j + 1 < M {
                let q_next = basis
                    .vector_mut(j + 1)
                    .ok_or(ConvergenceError::DimensionMismatch)?;
                let inv = T::one().div(h_next);
                for (slot, &w_i) in q_next.iter_mut().zip(scratch.iter()) {
                    *slot = w_i.mul(inv);
                }
            }
        }

        // Reduce the (reached + 1) x reached Hessenberg block `[h; h_sub]` to upper
        // triangular form via incremental Givens rotations, tracking the same rotations'
        // effect on the right-hand side `g` (initialized to `beta * e_1`).
        let mut g = [T::zero(); M];
        g[0] = beta;
        let mut cs = [T::zero(); M];
        let mut sn = [T::zero(); M];

        for j in 0..reached {
            let mut col = [T::zero(); M];
            for (i, slot) in col.iter_mut().enumerate().take(j + 1) {
                *slot = h[i][j];
            }
            let sub = h_sub[j];

            for i in 0..j {
                let old_i = col[i];
                let old_i1 = col[i + 1];
                col[i] = cs[i].mul(old_i).add(sn[i].mul(old_i1));
                col[i + 1] = cs[i].mul(old_i1).sub(sn[i].mul(old_i));
            }

            let r = Scalar::sqrt(col[j].mul(col[j]).add(sub.mul(sub)));
            let (c, s) = if r == T::zero() {
                (T::one(), T::zero())
            } else {
                (col[j].div(r), sub.div(r))
            };
            cs[j] = c;
            sn[j] = s;
            col[j] = r;

            let g_j = g[j];
            let g_next = if j + 1 < M { g[j + 1] } else { T::zero() };
            g[j] = c.mul(g_j).add(s.mul(g_next));
            if j + 1 < M {
                g[j + 1] = c.mul(g_next).sub(s.mul(g_j));
            }

            for (i, &value) in col.iter().enumerate().take(j + 1) {
                h[i][j] = value;
            }
        }

        // Back substitution: R y = g, with R the (now upper triangular) leading block of `h`.
        let mut y = [T::zero(); M];
        for i in (0..reached).rev() {
            let mut sum = g[i];
            for (k, &y_k) in y.iter().enumerate().take(reached).skip(i + 1) {
                sum = sum.sub(h[i][k].mul(y_k));
            }
            let diag = h[i][i];
            if diag == T::zero() {
                return Err(ConvergenceError::Breakdown);
            }
            y[i] = sum.div(diag);
        }

        // x <- x + Q y
        for (k, &y_k) in y.iter().enumerate().take(reached) {
            let q_k = basis.vector(k).ok_or(ConvergenceError::DimensionMismatch)?;
            for (slot, &q_ki) in out_x.iter_mut().zip(q_k.iter()) {
                *slot = slot.add(y_k.mul(q_ki));
            }
        }

        beta = residual_norm(a, b, out_x, scratch)?;
        if beta <= tol {
            return Ok(());
        }
    }

    Err(ConvergenceError::MaxIterationsExceeded)
}

#[cfg(test)]
mod tests {
    use super::gmres;
    use crate::krylov::ConvergenceError;
    use crate::sparse::CsrMatrix;
    use crate::storage::Basis;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() < tol,
            "expected {expected}, got {actual}"
        );
    }

    fn csr_from_dense(a: &[f64], n: usize) -> CsrMatrix<f64> {
        let mut row_ptr = vec![0_u32];
        let mut col_indices = vec![];
        let mut values = vec![];
        for r in 0..n {
            for c in 0..n {
                let v = a[r * n + c];
                if v != 0.0 {
                    col_indices.push(c as u32);
                    values.push(v);
                }
            }
            row_ptr.push(col_indices.len() as u32);
        }
        CsrMatrix::new(n, n, row_ptr, col_indices, values).unwrap()
    }

    fn residual_norm(a: &[f64], n: usize, x: &[f64], b: &[f64]) -> f64 {
        let mut r_sq = 0.0;
        for row in 0..n {
            let mut ax = 0.0;
            for col in 0..n {
                ax += a[row * n + col] * x[col];
            }
            let r = b[row] - ax;
            r_sq += r * r;
        }
        r_sq.sqrt()
    }

    #[test]
    fn solves_a_small_nonsymmetric_system() {
        // [[4, 1], [2, 3]] x = [1, 2]. Solution: x = [0.1, 0.6].
        let a = [4.0, 1.0, 2.0, 3.0];
        let m = csr_from_dense(&a, 2);
        let b = [1.0, 2.0];
        let x0 = [0.0, 0.0];
        let mut out_x = [0.0; 2];
        let mut buffer = [0.0; 4];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        gmres(&m, &b, &x0, 10, 1e-10, &mut out_x, &mut basis, &mut scratch).unwrap();

        assert_close(out_x[0], 0.1, 1e-8);
        assert_close(out_x[1], 0.6, 1e-8);
    }

    #[test]
    fn solves_a_diagonal_system_in_one_restart() {
        // diag(2, 4, 5) x = [4, 8, 10] => x = [2, 2, 2]. Full-basis GMRES solves an
        // n-dimensional system exactly within one restart (n steps of Arnoldi).
        let a = [2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 5.0];
        let m = csr_from_dense(&a, 3);
        let b = [4.0, 8.0, 10.0];
        let x0 = [0.0, 0.0, 0.0];
        let mut out_x = [0.0; 3];
        let mut buffer = [0.0; 9];
        let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
        let mut scratch = [0.0; 3];

        gmres(&m, &b, &x0, 1, 1e-10, &mut out_x, &mut basis, &mut scratch).unwrap();

        assert_close(out_x[0], 2.0, 1e-8);
        assert_close(out_x[1], 2.0, 1e-8);
        assert_close(out_x[2], 2.0, 1e-8);
    }

    #[test]
    fn restarts_accumulate_progress_toward_the_solution() {
        // A well-conditioned 3x3 system, solved with a restart size (M = 1) too small to
        // reach the solution in a single cycle: correctness relies on restarts carrying the
        // residual forward, not on any single cycle's Krylov subspace being big enough.
        let a = [5.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 3.0];
        let m = csr_from_dense(&a, 3);
        let b = [6.0, 6.0, 4.0];
        let x0 = [0.0, 0.0, 0.0];
        let mut out_x = [0.0; 3];
        let mut buffer = [0.0; 3];
        let mut basis = Basis::<f64, 1>::new(&mut buffer, 3).unwrap();
        let mut scratch = [0.0; 3];

        gmres(
            &m,
            &b,
            &x0,
            500,
            1e-10,
            &mut out_x,
            &mut basis,
            &mut scratch,
        )
        .unwrap();

        assert!(residual_norm(&a, 3, &out_x, &b) < 1e-8);
    }

    #[test]
    fn m_equals_one_degenerates_toward_steepest_descent_like_behavior() {
        // A symmetric positive-definite system: with M == 1, every restart cycle can only
        // move along the current residual direction, the same search direction gradient
        // descent would take. It still converges, just gradually, given enough restarts.
        let a = [3.0, 1.0, 1.0, 2.0];
        let m = csr_from_dense(&a, 2);
        let b = [4.0, 3.0];
        let x0 = [0.0, 0.0];
        let mut out_x = [0.0; 2];
        let mut buffer = [0.0; 2];
        let mut basis = Basis::<f64, 1>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        gmres(
            &m,
            &b,
            &x0,
            200,
            1e-10,
            &mut out_x,
            &mut basis,
            &mut scratch,
        )
        .unwrap();

        assert!(residual_norm(&a, 2, &out_x, &b) < 1e-8);
    }

    #[test]
    fn zero_restart_budget_only_accepts_an_already_converged_guess() {
        let a = [2.0, 0.0, 0.0, 2.0];
        let m = csr_from_dense(&a, 2);
        let b = [2.0, 2.0];

        // x0 is already the solution: converges without spending any restart.
        let x0_exact = [1.0, 1.0];
        let mut out_x = [0.0; 2];
        let mut buffer = [0.0; 2];
        let mut basis = Basis::<f64, 1>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];
        gmres(
            &m,
            &b,
            &x0_exact,
            0,
            1e-10,
            &mut out_x,
            &mut basis,
            &mut scratch,
        )
        .unwrap();
        assert_close(out_x[0], 1.0, 1e-10);
        assert_close(out_x[1], 1.0, 1e-10);

        // x0 is not the solution and no restarts are budgeted: no progress can be made.
        let x0_wrong = [0.0, 0.0];
        let result = gmres(
            &m,
            &b,
            &x0_wrong,
            0,
            1e-10,
            &mut out_x,
            &mut basis,
            &mut scratch,
        );
        assert_eq!(result, Err(ConvergenceError::MaxIterationsExceeded));
    }

    #[test]
    fn stagnation_across_restarts_reports_exhaustion_not_a_hang() {
        // A 90-degree rotation: for *any* direction q, `A * q` is orthogonal to `q`. With
        // M == 1, GMRES minimizes `‖r - y * (A * q_0)‖` over the scalar `y`, and since
        // `A * q_0 ⟂ r` (r is a multiple of q_0), that minimum sits exactly at `y == 0` —
        // every restart cycle picks the step that changes nothing. A tight budget must fail
        // fast with `MaxIterationsExceeded`, not loop forever making zero progress.
        let a = [0.0, -1.0, 1.0, 0.0];
        let m = csr_from_dense(&a, 2);
        let b = [1.0, 0.0];
        let x0 = [0.0, 0.0];
        let mut out_x = [0.0; 2];
        let mut buffer = [0.0; 2];
        let mut basis = Basis::<f64, 1>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let result = gmres(&m, &b, &x0, 20, 1e-10, &mut out_x, &mut basis, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::MaxIterationsExceeded));
        // x never moved off the stagnation point.
        assert_close(out_x[0], 0.0, 1e-10);
        assert_close(out_x[1], 0.0, 1e-10);
    }

    #[test]
    fn m_zero_never_reduces_a_nonzero_residual() {
        let a = [2.0, 0.0, 0.0, 2.0];
        let m = csr_from_dense(&a, 2);
        let b = [2.0, 2.0];
        let x0 = [0.0, 0.0];
        let mut out_x = [0.0; 2];
        let mut buffer: [f64; 0] = [];
        let mut basis = Basis::<f64, 0>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        let result = gmres(&m, &b, &x0, 5, 1e-10, &mut out_x, &mut basis, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::MaxIterationsExceeded));
    }

    #[test]
    fn already_converged_initial_guess_returns_immediately() {
        let a = [2.0, 0.0, 0.0, 2.0];
        let m = csr_from_dense(&a, 2);
        let b = [2.0, 4.0];
        let x0 = [1.0, 2.0];
        let mut out_x = [0.0; 2];
        let mut buffer = [0.0; 2];
        let mut basis = Basis::<f64, 1>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        gmres(&m, &b, &x0, 10, 1e-10, &mut out_x, &mut basis, &mut scratch).unwrap();

        assert_close(out_x[0], 1.0, 1e-10);
        assert_close(out_x[1], 2.0, 1e-10);
    }

    #[test]
    fn mismatched_dimensions_are_an_error_not_a_panic() {
        let a = csr_from_dense(&[2.0, 0.0, 0.0, 1.0], 2);
        let b = [1.0, 0.0];
        let x0 = [0.0, 0.0];
        let mut out_x = [0.0; 2];
        let mut buffer = [0.0; 2];
        let mut basis = Basis::<f64, 1>::new(&mut buffer, 2).unwrap();
        let mut scratch = [0.0; 2];

        // `b` too short.
        assert_eq!(
            gmres(
                &a,
                &[1.0],
                &x0,
                10,
                1e-10,
                &mut out_x,
                &mut basis,
                &mut scratch
            ),
            Err(ConvergenceError::DimensionMismatch)
        );

        // `x0` too short.
        assert_eq!(
            gmres(
                &a,
                &b,
                &[1.0],
                10,
                1e-10,
                &mut out_x,
                &mut basis,
                &mut scratch
            ),
            Err(ConvergenceError::DimensionMismatch)
        );

        // `out_x` too short.
        let mut out_x_short = [0.0; 1];
        assert_eq!(
            gmres(
                &a,
                &b,
                &x0,
                10,
                1e-10,
                &mut out_x_short,
                &mut basis,
                &mut scratch
            ),
            Err(ConvergenceError::DimensionMismatch)
        );

        // `scratch` too short.
        let mut scratch_short = [0.0; 1];
        assert_eq!(
            gmres(
                &a,
                &b,
                &x0,
                10,
                1e-10,
                &mut out_x,
                &mut basis,
                &mut scratch_short
            ),
            Err(ConvergenceError::DimensionMismatch)
        );

        // `basis` vectors of the wrong length.
        let mut buffer_wrong = [0.0; 1];
        let mut basis_wrong = Basis::<f64, 1>::new(&mut buffer_wrong, 1).unwrap();
        assert_eq!(
            gmres(
                &a,
                &b,
                &x0,
                10,
                1e-10,
                &mut out_x,
                &mut basis_wrong,
                &mut scratch
            ),
            Err(ConvergenceError::DimensionMismatch)
        );

        // M > n: a 2-dimensional space has no 3 orthonormal directions.
        let mut buffer_deep = [0.0; 6];
        let mut basis_deep = Basis::<f64, 3>::new(&mut buffer_deep, 2).unwrap();
        assert_eq!(
            gmres(
                &a,
                &b,
                &x0,
                10,
                1e-10,
                &mut out_x,
                &mut basis_deep,
                &mut scratch
            ),
            Err(ConvergenceError::DimensionMismatch)
        );
    }

    #[test]
    fn non_square_operator_is_a_dimension_mismatch() {
        // 2x3, not square: GMRES requires a square operator.
        let a =
            CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 1, 2], vec![1.0_f64, 1.0, 1.0]).unwrap();
        let b = [1.0, 1.0];
        let x0 = [0.0, 0.0, 0.0];
        let mut out_x = [0.0; 3];
        let mut buffer = [0.0; 6];
        let mut basis = Basis::<f64, 2>::new(&mut buffer, 3).unwrap();
        let mut scratch = [0.0; 3];

        let result = gmres(&a, &b, &x0, 10, 1e-10, &mut out_x, &mut basis, &mut scratch);

        assert_eq!(result, Err(ConvergenceError::DimensionMismatch));
    }
}

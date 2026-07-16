//! Numerical stress tests for the Krylov iterations: ill-conditioned matrices, magnitude
//! extremes at the [-100, 100] generation boundary, and iteration-budget behavior near the
//! convergence threshold.

// The shared Krylov test harness lives with the property suite; only a subset of it is used
// per test target, hence the dead_code allowance.
#[path = "property/krylov/common.rs"]
#[allow(dead_code)]
mod common;

use rustebra::krylov::{ConvergenceError, arnoldi, inverse_power_iteration, lanczos, power_iteration};
use rustebra::storage::{Basis, StaticStorage};

use common::{
    ALGORITHM_TOL, ASSERTION_TOL, SINGULAR_TOL, approx_eq_eigenvector, fixed_eigenvector_3,
    fixed_similarity_3, max_iter_for,
};

#[test]
fn power_iteration_converges_at_the_gap_ratio_boundary_within_the_derived_budget() {
    // diag(100, 98): |λ2/λ1| = 0.98, the widest gap the property harness ever generates,
    // with entries at the magnitude boundary. The budget is derived from that exact rate.
    let a = StaticStorage::new([100.0, 0.0, 0.0, 98.0]);
    let v0 = StaticStorage::new([1.0, 1.0]);
    let mut eigenvector = [0.0; 2];
    let mut scratch = [0.0; 2];

    let eigenvalue = power_iteration(
        &a,
        2,
        &v0,
        max_iter_for(0.98, ALGORITHM_TOL),
        ALGORITHM_TOL,
        &mut eigenvector,
        &mut scratch,
    )
    .unwrap();

    assert!((eigenvalue - 100.0).abs() <= ASSERTION_TOL * 100.0);
    assert!(approx_eq_eigenvector(
        &eigenvector,
        &[1.0, 0.0],
        ASSERTION_TOL
    ));
}

#[test]
fn power_iteration_reports_exhaustion_when_the_budget_falls_short_of_the_gap() {
    // Same 0.98-ratio matrix, but a budget well below the ~1140 iterations the rate demands:
    // the residual criterion cannot be met yet, so exhaustion is the guaranteed outcome.
    let a = StaticStorage::new([100.0, 0.0, 0.0, 98.0]);
    let v0 = StaticStorage::new([1.0, 1.0]);
    let mut eigenvector = [0.0; 2];
    let mut scratch = [0.0; 2];

    let result = power_iteration(
        &a,
        2,
        &v0,
        500,
        ALGORITHM_TOL,
        &mut eigenvector,
        &mut scratch,
    );

    assert_eq!(result, Err(ConvergenceError::MaxIterationsExceeded));
}

#[test]
fn inverse_power_iteration_converges_at_the_rate_boundary_within_the_derived_budget() {
    // diag(1, 1.02, -3) with shift 0: convergence rate |1 - 0| / |1.02 - 0| ≈ 0.98, the same
    // boundary rate as the power-iteration case, approached through the shift instead of the
    // spectrum.
    let a = StaticStorage::new([
        1.0, 0.0, 0.0, //
        0.0, 1.02, 0.0, //
        0.0, 0.0, -3.0,
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
        max_iter_for(1.0 / 1.02, ALGORITHM_TOL),
        ALGORITHM_TOL,
        SINGULAR_TOL,
        &mut eigenvector,
        &mut factor,
        &mut pivots,
        &mut scratch,
    )
    .unwrap();

    assert!((eigenvalue - 1.0).abs() <= ASSERTION_TOL);
    assert!(approx_eq_eigenvector(
        &eigenvector,
        &[1.0, 0.0, 0.0],
        ASSERTION_TOL
    ));
}

#[test]
fn inverse_power_iteration_reports_exhaustion_when_the_budget_falls_short_of_the_rate() {
    let a = StaticStorage::new([
        1.0, 0.0, 0.0, //
        0.0, 1.02, 0.0, //
        0.0, 0.0, -3.0,
    ]);
    let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
    let mut eigenvector = [0.0; 3];
    let mut factor = [0.0; 9];
    let mut pivots = [0_usize; 3];
    let mut scratch = [0.0; 3];

    let result = inverse_power_iteration(
        &a,
        3,
        &v0,
        0.0,
        500,
        ALGORITHM_TOL,
        SINGULAR_TOL,
        &mut eigenvector,
        &mut factor,
        &mut pivots,
        &mut scratch,
    );

    assert_eq!(result, Err(ConvergenceError::MaxIterationsExceeded));
}

#[test]
fn power_iteration_finds_the_dominant_eigenpair_of_an_ill_conditioned_matrix() {
    // H diag(100, 1e-6, -50) Hᵀ: condition number 1e8, but the dominant eigenpair is well
    // separated (ratio 0.5), so conditioning must not disturb it.
    let a = fixed_similarity_3([100.0, 1e-6, -50.0]);
    let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
    let mut eigenvector = [0.0; 3];
    let mut scratch = [0.0; 3];

    let eigenvalue = power_iteration(
        &StaticStorage::new(a),
        3,
        &v0,
        max_iter_for(0.5, ALGORITHM_TOL),
        ALGORITHM_TOL,
        &mut eigenvector,
        &mut scratch,
    )
    .unwrap();

    assert!((eigenvalue - 100.0).abs() <= ASSERTION_TOL * 100.0);
    assert!(approx_eq_eigenvector(
        &eigenvector,
        &fixed_eigenvector_3(0),
        ASSERTION_TOL
    ));
}

#[test]
fn inverse_power_iteration_resolves_the_tiny_eigenvalue_of_an_ill_conditioned_matrix() {
    // Same 1e8-condition matrix, shift 0: the target is the eigenvalue 1e-6 that makes the
    // matrix ill-conditioned in the first place, and the solve runs against that near
    // singularity every iteration.
    let a = fixed_similarity_3([100.0, 1e-6, -50.0]);
    let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
    let mut eigenvector = [0.0; 3];
    let mut factor = [0.0; 9];
    let mut pivots = [0_usize; 3];
    let mut scratch = [0.0; 3];

    let eigenvalue = inverse_power_iteration(
        &StaticStorage::new(a),
        3,
        &v0,
        0.0,
        50,
        ALGORITHM_TOL,
        SINGULAR_TOL,
        &mut eigenvector,
        &mut factor,
        &mut pivots,
        &mut scratch,
    )
    .unwrap();

    // Absolute tolerance: relative to the spectral radius 100 the bound would be vacuous for
    // a 1e-6 target.
    assert!(
        (eigenvalue - 1e-6).abs() <= ASSERTION_TOL,
        "got {eigenvalue}"
    );
    assert!(approx_eq_eigenvector(
        &eigenvector,
        &fixed_eigenvector_3(1),
        ASSERTION_TOL
    ));
}

#[test]
fn power_iteration_handles_entries_at_the_positive_magnitude_boundary() {
    // [[100, -99], [-99, 100]]: every entry at or near the [-100, 100] boundary; eigenvalues
    // 199 (eigenvector ±[1, -1]/√2) and 1.
    let a = StaticStorage::new([100.0, -99.0, -99.0, 100.0]);
    let v0 = StaticStorage::new([1.0, 0.0]);
    let mut eigenvector = [0.0; 2];
    let mut scratch = [0.0; 2];

    let eigenvalue = power_iteration(
        &a,
        2,
        &v0,
        200,
        ALGORITHM_TOL,
        &mut eigenvector,
        &mut scratch,
    )
    .unwrap();

    assert!((eigenvalue - 199.0).abs() <= ASSERTION_TOL * 199.0);
    assert!(approx_eq_eigenvector(
        &eigenvector,
        &[1.0, -1.0],
        ASSERTION_TOL
    ));
}

#[test]
fn power_iteration_handles_entries_at_the_negative_magnitude_boundary() {
    // [[-100, 99], [99, -100]]: eigenvalues -199 (eigenvector ±[1, -1]/√2) and -1 — the
    // dominant eigenvalue is negative, so the iterate's sign alternates at full magnitude.
    let a = StaticStorage::new([-100.0, 99.0, 99.0, -100.0]);
    let v0 = StaticStorage::new([1.0, 0.0]);
    let mut eigenvector = [0.0; 2];
    let mut scratch = [0.0; 2];

    let eigenvalue = power_iteration(
        &a,
        2,
        &v0,
        200,
        ALGORITHM_TOL,
        &mut eigenvector,
        &mut scratch,
    )
    .unwrap();

    assert!((eigenvalue + 199.0).abs() <= ASSERTION_TOL * 199.0);
    assert!(approx_eq_eigenvector(
        &eigenvector,
        &[1.0, -1.0],
        ASSERTION_TOL
    ));
}

#[test]
fn inverse_power_iteration_handles_entries_at_the_magnitude_boundary() {
    // Same boundary matrix, shift 0: targets the eigenvalue 1 (eigenvector ±[1, 1]/√2) that
    // sits five orders of magnitude below the matrix entries' scale.
    let a = StaticStorage::new([100.0, -99.0, -99.0, 100.0]);
    let v0 = StaticStorage::new([1.0, 0.0]);
    let mut eigenvector = [0.0; 2];
    let mut factor = [0.0; 4];
    let mut pivots = [0_usize; 2];
    let mut scratch = [0.0; 2];

    let eigenvalue = inverse_power_iteration(
        &a,
        2,
        &v0,
        0.0,
        200,
        ALGORITHM_TOL,
        SINGULAR_TOL,
        &mut eigenvector,
        &mut factor,
        &mut pivots,
        &mut scratch,
    )
    .unwrap();

    assert!((eigenvalue - 1.0).abs() <= ASSERTION_TOL);
    assert!(approx_eq_eigenvector(
        &eigenvector,
        &[1.0, 1.0],
        ASSERTION_TOL
    ));
}

#[test]
fn lanczos_recovers_the_full_spectrum_of_an_ill_conditioned_matrix() {
    // Same 1e8-condition matrix as the power-iteration cases above: a full-dimension basis
    // (K == n) must still recover all three eigenvalues, including the tiny 1e-6 one that
    // makes the matrix ill-conditioned, not just the well-separated dominant one.
    let a = fixed_similarity_3([100.0, 1e-6, -50.0]);
    let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
    let mut buffer = [0.0; 9];
    let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
    let mut scratch = [0.0; 3];

    let t = lanczos(
        &StaticStorage::new(a),
        3,
        &v0,
        ALGORITHM_TOL,
        &mut basis,
        &mut scratch,
    )
    .unwrap();

    let mut dense = nalgebra::Matrix3::zeros();
    for j in 0..3 {
        dense[(j, j)] = t.diagonal()[j];
    }
    for (j, &beta) in t.off_diagonal().iter().enumerate() {
        dense[(j, j + 1)] = beta;
        dense[(j + 1, j)] = beta;
    }
    let mut actual: [f64; 3] = dense.symmetric_eigen().eigenvalues.into();
    actual.sort_by(|x, y| x.total_cmp(y));

    let mut expected: [f64; 3] = [100.0, 1e-6, -50.0];
    expected.sort_by(|x, y| x.total_cmp(y));

    // Absolute tolerance: relative to the spectral radius 100 the bound would be vacuous for
    // the 1e-6 eigenvalue.
    for (got, want) in actual.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() <= ASSERTION_TOL,
            "got {actual:?}, want {expected:?}"
        );
    }
}

#[test]
fn arnoldi_recovers_the_full_spectrum_of_an_ill_conditioned_non_symmetric_matrix() {
    // The companion matrix of the monic cubic with roots 100, 1e-6, -50 — the same 1e8-
    // condition spectrum as the Lanczos stress case above, but via a genuinely non-symmetric
    // matrix (companion matrices are never symmetric except in trivial cases) — the case
    // Arnoldi exists to handle and Lanczos cannot. Coefficients are derived from the roots
    // (rather than hand-expanded) to keep the -0.005/4999.9999.../50.000001 arithmetic exact.
    let roots = [100.0_f64, 1e-6, -50.0];
    let sum = roots[0] + roots[1] + roots[2];
    let pair_sum = roots[0] * roots[1] + roots[0] * roots[2] + roots[1] * roots[2];
    let product = roots[0] * roots[1] * roots[2];
    let a = StaticStorage::new([
        0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, //
        product, -pair_sum, sum,
    ]);
    let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
    let mut buffer = [0.0; 9];
    let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
    let mut scratch = [0.0; 3];

    let (h, reached) = arnoldi(&a, 3, &v0, ALGORITHM_TOL, &mut basis, &mut scratch).unwrap();
    assert_eq!(reached, 3);

    let mut dense = nalgebra::Matrix3::zeros();
    for r in 0..3 {
        for c in 0..3 {
            dense[(r, c)] = h.entry(r, c).unwrap();
        }
    }
    let mut actual: [f64; 3] = dense.complex_eigenvalues().map(|z| z.re).into();
    actual.sort_by(|x, y| x.total_cmp(y));

    let mut expected: [f64; 3] = [100.0, 1e-6, -50.0];
    expected.sort_by(|x, y| x.total_cmp(y));

    // Absolute tolerance: relative to the spectral radius 100 the bound would be vacuous for
    // the 1e-6 eigenvalue.
    for (got, want) in actual.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() <= ASSERTION_TOL,
            "got {actual:?}, want {expected:?}"
        );
    }
}

//! Curated, fixed-matrix edge cases for `power_iteration` and `inverse_power_iteration`:
//! degenerate spectra, singular shifts, non-finite inputs, dimension mismatches, and the
//! zero vector — cases the property harness deliberately never generates.

use rustebra::krylov::{ConvergenceError, inverse_power_iteration, power_iteration};
use rustebra::storage::StaticStorage;

use crate::common::{
    ALGORITHM_TOL, ASSERTION_TOL, SINGULAR_TOL, approx_eq_eigenvector, fixed_similarity_3,
    max_iter_for,
};

/// `‖A v - λ v‖` for a row-major `n x n` matrix — the eigenpair-equation residual, the only
/// valid check when the eigenspace holds more than one valid answer.
fn residual_norm(a: &[f64], n: usize, eigenvalue: f64, eigenvector: &[f64]) -> f64 {
    let mut sum = 0.0;
    for r in 0..n {
        let av: f64 = (0..n).map(|c| a[r * n + c] * eigenvector[c]).sum();
        let diff = av - eigenvalue * eigenvector[r];
        sum += diff * diff;
    }
    sum.sqrt()
}

#[test]
fn power_iteration_on_a_1x1_matrix() {
    let a = StaticStorage::new([-7.5]);
    let v0 = StaticStorage::new([3.0]);
    let mut eigenvector = [0.0; 1];
    let mut scratch = [0.0; 1];

    let eigenvalue = power_iteration(
        &a,
        1,
        &v0,
        10,
        ALGORITHM_TOL,
        &mut eigenvector,
        &mut scratch,
    )
    .unwrap();

    assert!((eigenvalue + 7.5).abs() <= ASSERTION_TOL * 7.5);
    assert!((eigenvector[0].abs() - 1.0).abs() <= ASSERTION_TOL);
}

#[test]
fn inverse_power_iteration_on_a_1x1_matrix() {
    let a = StaticStorage::new([3.0]);
    let v0 = StaticStorage::new([-2.0]);
    let mut eigenvector = [0.0; 1];
    let mut factor = [0.0; 1];
    let mut pivots = [0_usize; 1];
    let mut scratch = [0.0; 1];

    let eigenvalue = inverse_power_iteration(
        &a,
        1,
        &v0,
        0.0,
        10,
        ALGORITHM_TOL,
        SINGULAR_TOL,
        &mut eigenvector,
        &mut factor,
        &mut pivots,
        &mut scratch,
    )
    .unwrap();

    assert!((eigenvalue - 3.0).abs() <= ASSERTION_TOL * 3.0);
    assert!((eigenvector[0].abs() - 1.0).abs() <= ASSERTION_TOL);
}

#[test]
fn repeated_dominant_eigenvalue_converges_to_a_vector_in_the_eigenspace() {
    // H diag(5, 5, 2) Hᵀ: the dominant eigenvalue 5 has a two-dimensional eigenspace, so any
    // unit vector satisfying the eigenpair equation is a correct answer — the residual is
    // asserted, never one specific eigenvector.
    let a = fixed_similarity_3([5.0, 5.0, 2.0]);
    let v0 = StaticStorage::new([1.0, 0.4, -0.3]);
    let mut eigenvector = [0.0; 3];
    let mut scratch = [0.0; 3];

    let eigenvalue = power_iteration(
        &StaticStorage::new(a),
        3,
        &v0,
        500,
        ALGORITHM_TOL,
        &mut eigenvector,
        &mut scratch,
    )
    .unwrap();

    assert!(
        (eigenvalue - 5.0).abs() <= ASSERTION_TOL * 5.0,
        "expected the repeated dominant eigenvalue 5, got {eigenvalue}"
    );
    assert!(
        residual_norm(&a, 3, eigenvalue, &eigenvector) <= ASSERTION_TOL * 5.0,
        "returned vector does not satisfy A v = λ v"
    );
}

#[test]
fn dominant_complex_conjugate_pair_exhausts_the_iteration_budget() {
    // Rotation-and-scale in the xy-plane: eigenvalues ±2i (the dominant pair) and 1. The
    // iterate rotates forever, so MaxIterationsExceeded is the guaranteed outcome, not a
    // tolerated failure.
    let a = StaticStorage::new([
        0.0, -2.0, 0.0, //
        2.0, 0.0, 0.0, //
        0.0, 0.0, 1.0,
    ]);
    let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
    let mut eigenvector = [0.0; 3];
    let mut scratch = [0.0; 3];

    let result = power_iteration(
        &a,
        3,
        &v0,
        300,
        ALGORITHM_TOL,
        &mut eigenvector,
        &mut scratch,
    );

    assert_eq!(result, Err(ConvergenceError::MaxIterationsExceeded));
}

#[test]
fn shift_exactly_at_an_eigenvalue_is_a_singular_shift_error() {
    // [[2, 1], [1, 2]] has eigenvalues 3 and 1: shift 3 makes the shifted matrix singular
    // even though none of its entries becomes zero.
    let a = StaticStorage::new([2.0, 1.0, 1.0, 2.0]);
    let v0 = StaticStorage::new([1.0, 0.0]);
    let mut eigenvector = [0.0; 2];
    let mut factor = [0.0; 4];
    let mut pivots = [0_usize; 2];
    let mut scratch = [0.0; 2];

    let result = inverse_power_iteration(
        &a,
        2,
        &v0,
        3.0,
        100,
        ALGORITHM_TOL,
        SINGULAR_TOL,
        &mut eigenvector,
        &mut factor,
        &mut pivots,
        &mut scratch,
    );

    assert_eq!(result, Err(ConvergenceError::SingularShift));
}

#[test]
fn shift_far_from_the_spectrum_still_selects_the_nearest_eigenvalue() {
    // diag(2, 1) with shift -50: the distances 51 and 52 are nearly equal, so the rate 51/52
    // is close to 1 — the documented slow-but-correct regime, with the budget derived from
    // that rate.
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
        -50.0,
        max_iter_for(51.0 / 52.0, ALGORITHM_TOL),
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
        &[0.0, 1.0],
        ASSERTION_TOL
    ));
}

#[test]
fn non_finite_matrix_entries_are_an_error_not_a_panic_or_a_spurious_ok() {
    for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let a = StaticStorage::new([2.0, poison, 0.0, 1.0]);
        let v0 = StaticStorage::new([1.0, 1.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        let result = power_iteration(
            &a,
            2,
            &v0,
            100,
            ALGORITHM_TOL,
            &mut eigenvector,
            &mut scratch,
        );
        assert!(
            result.is_err(),
            "power_iteration accepted a {poison} entry: {result:?}"
        );

        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let result = inverse_power_iteration(
            &a,
            2,
            &v0,
            0.5,
            100,
            ALGORITHM_TOL,
            SINGULAR_TOL,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        );
        assert!(
            result.is_err(),
            "inverse_power_iteration accepted a {poison} entry: {result:?}"
        );
    }
}

#[test]
fn non_finite_initial_vector_is_a_non_finite_error() {
    // A non-finite v0 poisons the norm, which the normalization treats as non-finite.
    for poison in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
        let v0 = StaticStorage::new([poison, 1.0]);
        let mut eigenvector = [0.0; 2];
        let mut scratch = [0.0; 2];

        let result = power_iteration(
            &a,
            2,
            &v0,
            100,
            ALGORITHM_TOL,
            &mut eigenvector,
            &mut scratch,
        );
        assert_eq!(result, Err(ConvergenceError::NonFinite), "poison {poison}");

        let mut factor = [0.0; 4];
        let mut pivots = [0_usize; 2];
        let result = inverse_power_iteration(
            &a,
            2,
            &v0,
            0.5,
            100,
            ALGORITHM_TOL,
            SINGULAR_TOL,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        );
        assert_eq!(result, Err(ConvergenceError::NonFinite), "poison {poison}");
    }
}

#[test]
fn zero_initial_vector_is_a_zero_vector_error_for_both_iterations() {
    let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
    let v0 = StaticStorage::new([0.0, 0.0]);
    let mut eigenvector = [0.0; 2];
    let mut scratch = [0.0; 2];

    let result = power_iteration(
        &a,
        2,
        &v0,
        100,
        ALGORITHM_TOL,
        &mut eigenvector,
        &mut scratch,
    );
    assert_eq!(result, Err(ConvergenceError::ZeroVector));

    let mut factor = [0.0; 4];
    let mut pivots = [0_usize; 2];
    let result = inverse_power_iteration(
        &a,
        2,
        &v0,
        0.5,
        100,
        ALGORITHM_TOL,
        SINGULAR_TOL,
        &mut eigenvector,
        &mut factor,
        &mut pivots,
        &mut scratch,
    );
    assert_eq!(result, Err(ConvergenceError::ZeroVector));
}

#[test]
fn dimension_mismatches_are_an_error_for_both_iterations() {
    // 4 entries can't be a 3x3 matrix.
    let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
    let v0 = StaticStorage::new([1.0, 0.0, 0.0]);
    let mut eigenvector = [0.0; 3];
    let mut scratch = [0.0; 3];

    let result = power_iteration(
        &a,
        3,
        &v0,
        100,
        ALGORITHM_TOL,
        &mut eigenvector,
        &mut scratch,
    );
    assert_eq!(result, Err(ConvergenceError::DimensionMismatch));

    let mut factor = [0.0; 9];
    let mut pivots = [0_usize; 3];
    let result = inverse_power_iteration(
        &a,
        3,
        &v0,
        0.5,
        100,
        ALGORITHM_TOL,
        SINGULAR_TOL,
        &mut eigenvector,
        &mut factor,
        &mut pivots,
        &mut scratch,
    );
    assert_eq!(result, Err(ConvergenceError::DimensionMismatch));

    // v0 shorter than n.
    let v0_short = StaticStorage::new([1.0]);
    let mut eigenvector2 = [0.0; 2];
    let mut scratch2 = [0.0; 2];
    let result = power_iteration(
        &a,
        2,
        &v0_short,
        100,
        ALGORITHM_TOL,
        &mut eigenvector2,
        &mut scratch2,
    );
    assert_eq!(result, Err(ConvergenceError::DimensionMismatch));

    let mut factor2 = [0.0; 4];
    let mut pivots2 = [0_usize; 2];
    let result = inverse_power_iteration(
        &a,
        2,
        &v0_short,
        0.5,
        100,
        ALGORITHM_TOL,
        SINGULAR_TOL,
        &mut eigenvector2,
        &mut factor2,
        &mut pivots2,
        &mut scratch2,
    );
    assert_eq!(result, Err(ConvergenceError::DimensionMismatch));
}

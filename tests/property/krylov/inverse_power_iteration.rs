//! Known-spectrum property tests for `inverse_power_iteration`: the shift is placed a
//! controlled distance from a chosen target eigenvalue, so both the selected eigenpair and
//! the convergence rate are known and success within the derived budget is asserted.

use proptest::prelude::*;
use rustebra::krylov::inverse_power_iteration;
use rustebra::storage::StaticStorage;

use super::common::{
    ALGORITHM_TOL, ASSERTION_TOL, N, SINGULAR_TOL, algorithm_tol_f32, approx_eq_eigenvector,
    assertion_tol_f32, column, eigenbasis_coordinate, max_iter_for, nonsymmetric_with_spectrum,
    overlap, spectrum_with_gap, symmetric_with_spectrum,
};

/// Smallest allowed distance from the target eigenvalue to the rest of the spectrum, relative
/// to the spectral radius: keeps "nearest the shift" unambiguous and the solve nonsingular.
const MIN_SEPARATION: f64 = 0.05;

/// Places the shift a `delta_frac` fraction of `separation` away from the target eigenvalue,
/// returning the shift and the resulting convergence rate of the inverse operator (the
/// distance to the target over the worst-case distance to any other eigenvalue).
fn shift_near(lambda_t: f64, separation: f64, delta_frac: f64, below: bool) -> (f64, f64) {
    let delta = delta_frac * separation;
    let shift = if below {
        lambda_t - delta
    } else {
        lambda_t + delta
    };
    (shift, delta / (separation - delta))
}

/// Distance from `eigenvalues[target]` to the nearest other eigenvalue.
fn separation_of(eigenvalues: &[f64; N], target: usize) -> f64 {
    eigenvalues
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != target)
        .map(|(_, x)| (x - eigenvalues[target]).abs())
        .fold(f64::INFINITY, f64::min)
}

proptest! {
    /// `A = Q D Qᵀ`: a shift a known distance from a chosen eigenvalue must select exactly
    /// that eigenpair within the rate-derived budget.
    #[test]
    fn symmetric_known_spectrum_shift_selects_the_nearest_eigenpair(
        eigenvalues in spectrum_with_gap(),
        q_seed in prop::array::uniform16(-1.0..1.0f64),
        v0 in prop::array::uniform4(-1.0..1.0f64),
        target in 0usize..N,
        delta_frac in 0.05..0.45f64,
        below in any::<bool>(),
    ) {
        let separation = separation_of(&eigenvalues, target);
        prop_assume!(separation >= MIN_SEPARATION * eigenvalues[0].abs());
        let lambda_t = eigenvalues[target];
        let (shift, rate) = shift_near(lambda_t, separation, delta_frac, below);

        let (a, q) = symmetric_with_spectrum(&eigenvalues, &q_seed);
        let expected = column(&q, target);
        prop_assume!(overlap(&v0, &expected).abs() > 0.1);

        let mut eigenvector = [0.0; N];
        let mut factor = [0.0; N * N];
        let mut pivots = [0_usize; N];
        let mut scratch = [0.0; N];
        let result = inverse_power_iteration(
            &StaticStorage::new(a),
            N,
            &StaticStorage::new(v0),
            shift,
            max_iter_for(rate, ALGORITHM_TOL),
            ALGORITHM_TOL,
            SINGULAR_TOL,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        );

        prop_assert!(result.is_ok(), "expected convergence to {}, got {:?}", lambda_t, result);
        let eigenvalue = result.unwrap();
        prop_assert!(
            (eigenvalue - lambda_t).abs() <= ASSERTION_TOL * eigenvalues[0].abs(),
            "shift {} should select {}, got {}",
            shift,
            lambda_t,
            eigenvalue,
        );
        prop_assert!(
            approx_eq_eigenvector(&eigenvector, &expected, ASSERTION_TOL),
            "eigenvector {:?} is not aligned with the constructed one {:?}",
            eigenvector,
            expected,
        );
    }

    /// `A = P D P⁻¹` with a well-conditioned, non-orthogonal `P`: shift selection on a
    /// genuinely non-symmetric matrix with a known real spectrum.
    #[test]
    fn nonsymmetric_known_spectrum_shift_selects_the_nearest_eigenpair(
        eigenvalues in spectrum_with_gap(),
        q_seed in prop::array::uniform16(-1.0..1.0f64),
        upper in prop::array::uniform6(-0.3..0.3f64),
        v0 in prop::array::uniform4(-1.0..1.0f64),
        target in 0usize..N,
        delta_frac in 0.05..0.45f64,
        below in any::<bool>(),
    ) {
        let separation = separation_of(&eigenvalues, target);
        prop_assume!(separation >= MIN_SEPARATION * eigenvalues[0].abs());
        let lambda_t = eigenvalues[target];
        let (shift, rate) = shift_near(lambda_t, separation, delta_frac, below);

        let generated = nonsymmetric_with_spectrum(&eigenvalues, &q_seed, &upper);
        prop_assume!(generated.is_some());
        let (a, p, p_inv) = generated.unwrap();
        let expected = column(&p, target);
        prop_assume!(eigenbasis_coordinate(&p_inv, &v0, target).abs() > 0.1);

        let mut eigenvector = [0.0; N];
        let mut factor = [0.0; N * N];
        let mut pivots = [0_usize; N];
        let mut scratch = [0.0; N];
        let result = inverse_power_iteration(
            &StaticStorage::new(a),
            N,
            &StaticStorage::new(v0),
            shift,
            max_iter_for(rate, ALGORITHM_TOL),
            ALGORITHM_TOL,
            SINGULAR_TOL,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        );

        prop_assert!(result.is_ok(), "expected convergence to {}, got {:?}", lambda_t, result);
        let eigenvalue = result.unwrap();
        prop_assert!(
            (eigenvalue - lambda_t).abs() <= ASSERTION_TOL * eigenvalues[0].abs(),
            "shift {} should select {}, got {}",
            shift,
            lambda_t,
            eigenvalue,
        );
        prop_assert!(
            approx_eq_eigenvector(&eigenvector, &expected, ASSERTION_TOL),
            "eigenvector {:?} is not aligned with the constructed one {:?}",
            eigenvector,
            expected,
        );
    }

    /// f32 variant of the symmetric shift-selection test, with tolerances scaled by
    /// `sqrt(f32::EPSILON)` and `singular_tol = n * f32::EPSILON` (the documented default).
    #[test]
    fn symmetric_known_spectrum_shift_selects_the_nearest_eigenpair_f32(
        eigenvalues in spectrum_with_gap(),
        q_seed in prop::array::uniform16(-1.0..1.0f64),
        v0 in prop::array::uniform4(-1.0..1.0f64),
        target in 0usize..N,
        delta_frac in 0.05..0.45f64,
        below in any::<bool>(),
    ) {
        let separation = separation_of(&eigenvalues, target);
        prop_assume!(separation >= MIN_SEPARATION * eigenvalues[0].abs());
        let lambda_t = eigenvalues[target];
        let (shift, rate) = shift_near(lambda_t, separation, delta_frac, below);

        let (a, q) = symmetric_with_spectrum(&eigenvalues, &q_seed);
        let expected = column(&q, target);
        prop_assume!(overlap(&v0, &expected).abs() > 0.1);

        let a_f32 = a.map(|x| x as f32);
        let v0_f32 = v0.map(|x| x as f32);
        let algorithm_tol = algorithm_tol_f32();
        let mut eigenvector = [0.0_f32; N];
        let mut factor = [0.0_f32; N * N];
        let mut pivots = [0_usize; N];
        let mut scratch = [0.0_f32; N];
        let result = inverse_power_iteration(
            &StaticStorage::new(a_f32),
            N,
            &StaticStorage::new(v0_f32),
            shift as f32,
            max_iter_for(rate, algorithm_tol as f64),
            algorithm_tol,
            N as f32 * f32::EPSILON,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        );

        prop_assert!(result.is_ok(), "expected convergence to {}, got {:?}", lambda_t, result);
        let eigenvalue = result.unwrap() as f64;
        let assertion_tol = assertion_tol_f32() as f64;
        prop_assert!(
            (eigenvalue - lambda_t).abs() <= assertion_tol * eigenvalues[0].abs(),
            "shift {} should select {}, got {}",
            shift,
            lambda_t,
            eigenvalue,
        );
        let eigenvector_f64 = eigenvector.map(|x| x as f64);
        prop_assert!(
            approx_eq_eigenvector(&eigenvector_f64, &expected, assertion_tol),
            "eigenvector {:?} is not aligned with the constructed one {:?}",
            eigenvector_f64,
            expected,
        );
    }
}

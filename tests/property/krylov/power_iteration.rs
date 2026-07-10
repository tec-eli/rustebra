//! Known-spectrum property tests for `power_iteration`: matrices are constructed as
//! `Q D Qᵀ` / `P D P⁻¹` with a controlled spectral gap, so the exact dominant eigenpair is
//! known and convergence within the derived iteration budget is asserted, not just tolerated.

use proptest::prelude::*;
use rustebra::krylov::power_iteration;
use rustebra::storage::StaticStorage;

use super::common::{
    ALGORITHM_TOL, ASSERTION_TOL, N, algorithm_tol_f32, approx_eq_eigenvector, assertion_tol_f32,
    column, dominant_ratio, eigenbasis_coordinate, max_iter_for, nonsymmetric_with_spectrum,
    overlap, spectrum_with_gap, symmetric_with_spectrum,
};

proptest! {
    /// `A = Q D Qᵀ` with `|λ2/λ1| <= 0.98`: the dominant eigenpair is known exactly, so the
    /// call must converge within the gap-derived budget and reproduce it.
    #[test]
    fn symmetric_known_spectrum_recovers_the_dominant_eigenpair(
        eigenvalues in spectrum_with_gap(),
        q_seed in prop::array::uniform16(-1.0..1.0f64),
        v0 in prop::array::uniform4(-1.0..1.0f64),
    ) {
        let (a, q) = symmetric_with_spectrum(&eigenvalues, &q_seed);
        let expected = column(&q, 0);
        // v0 needs a real component along the dominant eigenvector to converge to it.
        prop_assume!(overlap(&v0, &expected).abs() > 0.1);

        let mut eigenvector = [0.0; N];
        let mut scratch = [0.0; N];
        let result = power_iteration(
            &StaticStorage::new(a),
            N,
            &StaticStorage::new(v0),
            max_iter_for(dominant_ratio(&eigenvalues), ALGORITHM_TOL),
            ALGORITHM_TOL,
            &mut eigenvector,
            &mut scratch,
        );

        prop_assert!(result.is_ok(), "expected convergence, got {:?}", result);
        let eigenvalue = result.unwrap();
        prop_assert!(
            (eigenvalue - eigenvalues[0]).abs() <= ASSERTION_TOL * eigenvalues[0].abs(),
            "eigenvalue {} is not the constructed dominant {}",
            eigenvalue,
            eigenvalues[0],
        );
        prop_assert!(
            approx_eq_eigenvector(&eigenvector, &expected, ASSERTION_TOL),
            "eigenvector {:?} is not aligned with the constructed one {:?}",
            eigenvector,
            expected,
        );
    }

    /// `A = P D P⁻¹` with a well-conditioned, non-orthogonal `P`: a genuinely non-symmetric
    /// matrix whose real spectrum is still known exactly.
    #[test]
    fn nonsymmetric_known_spectrum_recovers_the_dominant_eigenpair(
        eigenvalues in spectrum_with_gap(),
        q_seed in prop::array::uniform16(-1.0..1.0f64),
        upper in prop::array::uniform6(-0.3..0.3f64),
        v0 in prop::array::uniform4(-1.0..1.0f64),
    ) {
        let generated = nonsymmetric_with_spectrum(&eigenvalues, &q_seed, &upper);
        prop_assume!(generated.is_some());
        let (a, p, p_inv) = generated.unwrap();
        let expected = column(&p, 0);
        // In a non-orthogonal eigenbasis the reachable component is the eigenbasis
        // coordinate, not the plain overlap with the eigenvector.
        prop_assume!(eigenbasis_coordinate(&p_inv, &v0, 0).abs() > 0.1);

        let mut eigenvector = [0.0; N];
        let mut scratch = [0.0; N];
        let result = power_iteration(
            &StaticStorage::new(a),
            N,
            &StaticStorage::new(v0),
            max_iter_for(dominant_ratio(&eigenvalues), ALGORITHM_TOL),
            ALGORITHM_TOL,
            &mut eigenvector,
            &mut scratch,
        );

        prop_assert!(result.is_ok(), "expected convergence, got {:?}", result);
        let eigenvalue = result.unwrap();
        prop_assert!(
            (eigenvalue - eigenvalues[0]).abs() <= ASSERTION_TOL * eigenvalues[0].abs(),
            "eigenvalue {} is not the constructed dominant {}",
            eigenvalue,
            eigenvalues[0],
        );
        prop_assert!(
            approx_eq_eigenvector(&eigenvector, &expected, ASSERTION_TOL),
            "eigenvector {:?} is not aligned with the constructed one {:?}",
            eigenvector,
            expected,
        );
    }

    /// f32 variant of the symmetric known-spectrum test, with tolerances scaled by
    /// `sqrt(f32::EPSILON)` rather than reusing the f64 constants.
    #[test]
    fn symmetric_known_spectrum_recovers_the_dominant_eigenpair_f32(
        eigenvalues in spectrum_with_gap(),
        q_seed in prop::array::uniform16(-1.0..1.0f64),
        v0 in prop::array::uniform4(-1.0..1.0f64),
    ) {
        let (a, q) = symmetric_with_spectrum(&eigenvalues, &q_seed);
        let expected = column(&q, 0);
        prop_assume!(overlap(&v0, &expected).abs() > 0.1);

        let a_f32 = a.map(|x| x as f32);
        let v0_f32 = v0.map(|x| x as f32);
        let algorithm_tol = algorithm_tol_f32();
        let mut eigenvector = [0.0_f32; N];
        let mut scratch = [0.0_f32; N];
        let result = power_iteration(
            &StaticStorage::new(a_f32),
            N,
            &StaticStorage::new(v0_f32),
            max_iter_for(dominant_ratio(&eigenvalues), algorithm_tol as f64),
            algorithm_tol,
            &mut eigenvector,
            &mut scratch,
        );

        prop_assert!(result.is_ok(), "expected convergence, got {:?}", result);
        let eigenvalue = result.unwrap() as f64;
        let assertion_tol = assertion_tol_f32() as f64;
        prop_assert!(
            (eigenvalue - eigenvalues[0]).abs() <= assertion_tol * eigenvalues[0].abs(),
            "eigenvalue {} is not the constructed dominant {}",
            eigenvalue,
            eigenvalues[0],
        );
        let eigenvector_f64 = eigenvector.map(|x| x as f64);
        prop_assert!(
            approx_eq_eigenvector(&eigenvector_f64, &expected, assertion_tol),
            "eigenvector {:?} is not aligned with the constructed one {:?}",
            eigenvector_f64,
            expected,
        );
    }

    /// Differential oracle: for symmetric matrices, nalgebra's `SymmetricEigen` is an
    /// independent implementation to diff the dominant eigenpair against.
    #[test]
    fn agrees_with_nalgebra_symmetric_eigen(
        eigenvalues in spectrum_with_gap(),
        q_seed in prop::array::uniform16(-1.0..1.0f64),
        v0 in prop::array::uniform4(-1.0..1.0f64),
    ) {
        let (a, q) = symmetric_with_spectrum(&eigenvalues, &q_seed);
        prop_assume!(overlap(&v0, &column(&q, 0)).abs() > 0.1);

        let mut eigenvector = [0.0; N];
        let mut scratch = [0.0; N];
        let result = power_iteration(
            &StaticStorage::new(a),
            N,
            &StaticStorage::new(v0),
            max_iter_for(dominant_ratio(&eigenvalues), ALGORITHM_TOL),
            ALGORITHM_TOL,
            &mut eigenvector,
            &mut scratch,
        );
        prop_assert!(result.is_ok(), "expected convergence, got {:?}", result);
        let eigenvalue = result.unwrap();

        let oracle = nalgebra::SymmetricEigen::new(nalgebra::Matrix4::from_row_slice(&a));
        let dominant = (0..N)
            .max_by(|&i, &j| oracle.eigenvalues[i].abs().total_cmp(&oracle.eigenvalues[j].abs()))
            .unwrap();
        let oracle_eigenvalue = oracle.eigenvalues[dominant];
        let oracle_eigenvector = column(&oracle.eigenvectors, dominant);

        prop_assert!(
            (eigenvalue - oracle_eigenvalue).abs() <= ASSERTION_TOL * oracle_eigenvalue.abs(),
            "eigenvalue {} disagrees with nalgebra's {}",
            eigenvalue,
            oracle_eigenvalue,
        );
        prop_assert!(
            approx_eq_eigenvector(&eigenvector, &oracle_eigenvector, ASSERTION_TOL),
            "eigenvector {:?} disagrees with nalgebra's {:?}",
            eigenvector,
            oracle_eigenvector,
        );
    }
}

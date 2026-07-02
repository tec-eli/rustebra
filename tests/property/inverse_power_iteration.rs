use proptest::prelude::*;
use rustebra::krylov::{ConvergenceError, inverse_power_iteration};
use rustebra::storage::StaticStorage;

proptest! {
    /// Property test: any eigenpair `inverse_power_iteration` accepts satisfies its
    /// definition, regardless of the shift.
    ///
    /// Generates random 3x3 symmetric matrices (symmetrized so every eigenvalue is real) with
    /// random shifts and starting vectors, then verifies that whenever the call returns `Ok`:
    /// 1. The returned eigenvector has unit Euclidean length
    /// 2. `A v ≈ λ v` holds component-wise
    ///
    /// Shifts that land (numerically) on an eigenvalue, shifts far outside the spectrum
    /// (whose convergence rate approaches 1), and degenerate spectra may legitimately fail
    /// instead; those outcomes are accepted, but only as documented error variants, never a
    /// shape error or a panic.
    #[test]
    fn accepted_eigenpair_satisfies_a_v_equals_lambda_v(
        entries in prop::collection::vec(-100.0..100.0f64, 9),
        v0_entries in prop::collection::vec(-10.0..10.0f64, 3),
        shift in -150.0..150.0f64,
    ) {
        let mut a = [0.0; 9];
        for r in 0..3 {
            for c in 0..3 {
                a[r * 3 + c] = (entries[r * 3 + c] + entries[c * 3 + r]) / 2.0;
            }
        }

        let v0_norm: f64 = v0_entries.iter().map(|x| x * x).sum::<f64>().sqrt();
        prop_assume!(v0_norm > 1e-3);
        let mut v0 = [0.0; 3];
        v0.copy_from_slice(&v0_entries);

        let mut eigenvector = [0.0; 3];
        let mut factor = [0.0; 9];
        let mut pivots = [0_usize; 3];
        let mut scratch = [0.0; 3];
        let result = inverse_power_iteration(
            &StaticStorage::new(a),
            3,
            &StaticStorage::new(v0),
            shift,
            20_000,
            1e-10,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        );

        match result {
            Ok(eigenvalue) => {
                let norm: f64 = eigenvector.iter().map(|x| x * x).sum::<f64>().sqrt();
                prop_assert!(
                    (norm - 1.0).abs() < 1e-9,
                    "eigenvector is not unit length: ‖v‖ = {}",
                    norm
                );

                let scale = eigenvalue.abs().max(1.0);
                for r in 0..3 {
                    let av: f64 = (0..3).map(|c| a[r * 3 + c] * eigenvector[c]).sum();
                    prop_assert!(
                        (av - eigenvalue * eigenvector[r]).abs() < 1e-6 * scale,
                        "component {}: (A v)[{}] = {} but λ v[{}] = {}",
                        r,
                        r,
                        av,
                        r,
                        eigenvalue * eigenvector[r]
                    );
                }
            }
            Err(error) => {
                prop_assert!(
                    matches!(
                        error,
                        ConvergenceError::MaxIterationsExceeded
                            | ConvergenceError::ZeroVector
                            | ConvergenceError::SingularShift
                    ),
                    "unexpected error kind: {:?}",
                    error
                );
            }
        }
    }

    /// Property test: the shift selects the correct eigenvalue.
    ///
    /// Uses random *diagonal* matrices, whose eigenvalues are exactly the diagonal entries,
    /// and asserts the returned eigenvalue is the entry nearest the shift. Cases are assumed
    /// away when the shift nearly touches an eigenvalue (a documented singularity) or when
    /// the nearest entry isn't decisively nearest (a slow-convergence tie); with a 2x
    /// distance margin the convergence rate is at most 1/2, so success itself is asserted,
    /// not just checked when it happens.
    #[test]
    fn shift_selects_the_nearest_eigenvalue_of_a_diagonal_matrix(
        diagonal in prop::collection::vec(-50.0..50.0f64, 3),
        shift in -60.0..60.0f64,
    ) {
        let mut distances: Vec<f64> = diagonal.iter().map(|d| (d - shift).abs()).collect();
        distances.sort_by(|x, y| x.total_cmp(y));
        // Not so close to an eigenvalue that the solve is near-singular, and decisively
        // nearest so convergence is fast and the target unambiguous.
        prop_assume!(distances[0] > 1e-3);
        prop_assume!(distances[1] > 2.0 * distances[0]);

        let expected = diagonal
            .iter()
            .copied()
            .min_by(|x, y| (x - shift).abs().total_cmp(&(y - shift).abs()))
            .expect("diagonal has 3 entries");

        let a = [
            diagonal[0], 0.0, 0.0, //
            0.0, diagonal[1], 0.0, //
            0.0, 0.0, diagonal[2],
        ];
        let v0 = [1.0, 1.0, 1.0];

        let mut eigenvector = [0.0; 3];
        let mut factor = [0.0; 9];
        let mut pivots = [0_usize; 3];
        let mut scratch = [0.0; 3];
        let eigenvalue = inverse_power_iteration(
            &StaticStorage::new(a),
            3,
            &StaticStorage::new(v0),
            shift,
            10_000,
            1e-10,
            1e-12,
            &mut eigenvector,
            &mut factor,
            &mut pivots,
            &mut scratch,
        );

        prop_assert_eq!(eigenvalue.is_ok(), true, "expected convergence, got {:?}", eigenvalue);
        if let Ok(eigenvalue) = eigenvalue {
            prop_assert!(
                (eigenvalue - expected).abs() < 1e-6,
                "shift {} should select {}, got {}",
                shift,
                expected,
                eigenvalue
            );
        }
    }
}

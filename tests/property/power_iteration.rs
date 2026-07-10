use proptest::prelude::*;
use rustebra::krylov::{ConvergenceError, power_iteration};
use rustebra::storage::StaticStorage;

proptest! {
    /// Property test: any eigenpair `power_iteration` accepts satisfies its definition.
    ///
    /// Generates random 3x3 symmetric matrices (symmetrized so every eigenvalue is real —
    /// a complex-conjugate dominant pair can never converge) and random nonzero starting
    /// vectors, then verifies that whenever the call returns `Ok`:
    /// 1. The returned eigenvector has unit Euclidean length
    /// 2. `A v ≈ λ v` holds component-wise
    ///
    /// Degenerate spectra (`|λ1| == |λ2|` with `λ1 != λ2`) and near-zero matrices may
    /// legitimately fail to converge instead; those outcomes are accepted, but only as the
    /// two documented convergence errors, never a shape error or a panic.
    #[test]
    fn accepted_eigenpair_satisfies_a_v_equals_lambda_v(
        entries in prop::collection::vec(-100.0..100.0f64, 9),
        v0_entries in prop::collection::vec(-10.0..10.0f64, 3),
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
        let mut scratch = [0.0; 3];
        let result = power_iteration(
            &StaticStorage::new(a),
            3,
            &StaticStorage::new(v0),
            10_000,
            1e-10,
            &mut eigenvector,
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
                        ConvergenceError::MaxIterationsExceeded | ConvergenceError::ZeroVector
                    ),
                    "unexpected error kind: {:?}",
                    error
                );
            }
        }
    }
}

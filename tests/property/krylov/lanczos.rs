//! Known-spectrum property tests for `lanczos`: symmetric matrices are constructed as
//! `Q D Qᵀ`, and the returned basis and tridiagonal matrix are checked against the defining
//! identities — basis orthonormality and `Qᵀ A Q == T` — rather than hand-worked examples.

use proptest::prelude::*;
use rustebra::krylov::{ConvergenceError, lanczos};
use rustebra::storage::{Basis, StaticStorage};

use super::common::{ALGORITHM_TOL, ASSERTION_TOL, N, spectrum_with_gap, symmetric_with_spectrum};

/// `Qᵀ A Q` entry `(r, c)` from the basis vectors and the row-major `N x N` matrix `a`.
fn projection_entry(a: &[f64; N * N], basis: &Basis<'_, f64, N>, r: usize, c: usize) -> f64 {
    let q_r = basis.vector(r).unwrap();
    let q_c = basis.vector(c).unwrap();
    let mut sum = 0.0;
    for row in 0..N {
        for col in 0..N {
            sum += q_r[row] * a[row * N + col] * q_c[col];
        }
    }
    sum
}

proptest! {
    /// For `A = Q D Qᵀ` and a random starting vector, the basis must be orthonormal and its
    /// projection `Qᵀ A Q` must reproduce the returned tridiagonal matrix — including the
    /// zeros beyond the first off-diagonal, which is what makes it *tridiagonal*.
    #[test]
    fn basis_is_orthonormal_and_projects_a_onto_the_tridiagonal(
        eigenvalues in spectrum_with_gap(),
        q_seed in prop::array::uniform16(-1.0..1.0f64),
        v0 in prop::array::uniform4(-1.0..1.0f64),
    ) {
        let (a, _) = symmetric_with_spectrum(&eigenvalues, &q_seed);
        let norm_v0: f64 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
        prop_assume!(norm_v0 > 0.1);

        let mut buffer = [0.0; N * N];
        let mut basis = Basis::<f64, N>::new(&mut buffer, N).unwrap();
        let mut scratch = [0.0; N];
        let result = lanczos(
            &StaticStorage::new(a),
            N,
            &StaticStorage::new(v0),
            ALGORITHM_TOL,
            &mut basis,
            &mut scratch,
        );

        // A random v0 lying (numerically) inside a proper invariant subspace is legitimate
        // breakdown, not a defect; it is astronomically rare under this generator.
        prop_assume!(result != Err(ConvergenceError::Breakdown));
        let t = result.unwrap();

        // The assertion tolerance is relative to the matrix scale: eigenvalues reach
        // magnitude 50, so entries of A and T do too.
        let scale = eigenvalues.iter().fold(1.0f64, |acc, x| acc.max(x.abs()));
        for r in 0..N {
            for c in 0..N {
                let q_r = basis.vector(r).unwrap();
                let q_c = basis.vector(c).unwrap();
                let inner: f64 = q_r.iter().zip(q_c.iter()).map(|(x, y)| x * y).sum();
                let identity = if r == c { 1.0 } else { 0.0 };
                prop_assert!(
                    (inner - identity).abs() < ASSERTION_TOL,
                    "q_{r} · q_{c} = {inner}, expected {identity}",
                );

                let expected = match (r, c) {
                    _ if r == c => t.diagonal()[r],
                    _ if r.abs_diff(c) == 1 => t.off_diagonal()[r.min(c)],
                    _ => 0.0,
                };
                let actual = projection_entry(&a, &basis, r, c);
                prop_assert!(
                    (actual - expected).abs() < ASSERTION_TOL * scale,
                    "(Qᵀ A Q)[{r},{c}] = {actual}, expected {expected}",
                );
            }
        }
    }
}

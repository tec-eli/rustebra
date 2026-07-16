//! Known-spectrum property tests for `arnoldi`: non-symmetric matrices are constructed as
//! `P D P⁻¹`, and the returned basis and Hessenberg matrix are checked against the defining
//! identities — basis orthonormality, upper Hessenberg structure, and `Qᵗ A Q == H` — rather
//! than hand-worked examples.

use proptest::prelude::*;
use rustebra::krylov::arnoldi;
use rustebra::storage::{Basis, StaticStorage};

use super::common::{ALGORITHM_TOL, ASSERTION_TOL, N, nonsymmetric_with_spectrum, spectrum_with_gap};

/// `Qᵗ * A * Q` entry `(r, c)` from the basis, against the row-major `N x N` matrix `a`.
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
    /// For `A = P D P⁻¹` and a random starting vector, the basis must be orthonormal and its
    /// projection `Qᵗ A Q` must reproduce the returned Hessenberg matrix — including the
    /// zeros strictly below the first subdiagonal, which is what makes it *Hessenberg*.
    #[test]
    fn basis_is_orthonormal_and_projects_a_onto_the_hessenberg_matrix(
        eigenvalues in spectrum_with_gap(),
        q_seed in prop::array::uniform16(-1.0..1.0f64),
        upper in prop::array::uniform6(-1.0..1.0f64),
        v0 in prop::array::uniform4(-1.0..1.0f64),
    ) {
        let Some((a, _, _)) = nonsymmetric_with_spectrum(&eigenvalues, &q_seed, &upper) else {
            return Ok(());
        };
        let norm_v0: f64 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
        prop_assume!(norm_v0 > 0.1);

        let mut buffer = [0.0; N * N];
        let mut basis = Basis::<f64, N>::new(&mut buffer, N).unwrap();
        let mut scratch = [0.0; N];
        let (h, reached) = arnoldi(
            &StaticStorage::new(a),
            N,
            &StaticStorage::new(v0),
            ALGORITHM_TOL,
            &mut basis,
            &mut scratch,
        )
        .unwrap();

        // A random v0 lying (numerically) inside a proper invariant subspace is legitimate
        // early breakdown, not a defect; it is astronomically rare under this generator.
        prop_assume!(reached == N);

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

                let expected = h.entry(r, c).unwrap();
                if r > c + 1 {
                    prop_assert!(
                        expected.abs() < ASSERTION_TOL * scale,
                        "h[{r}][{c}] = {expected}, expected structurally zero (below the subdiagonal)",
                    );
                }
                let actual = projection_entry(&a, &basis, r, c);
                prop_assert!(
                    (actual - expected).abs() < ASSERTION_TOL * scale,
                    "(Qᵀ A Q)[{r},{c}] = {actual}, expected {expected}",
                );
            }
        }
    }
}

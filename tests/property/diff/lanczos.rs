//! Differential oracle: with a full-dimension basis (`K == n`) and no breakdown, the
//! tridiagonal matrix `lanczos` returns is orthogonally similar to the input, so nalgebra's
//! `SymmetricEigen` must recover the same spectrum from both.

use proptest::prelude::*;
use rustebra::krylov::{ConvergenceError, lanczos};
use rustebra::storage::{Basis, StaticStorage};

const N: usize = 4;
const ALGORITHM_TOL: f64 = 1e-10;
const TOL: f64 = 1e-8;

/// The returned `K x K` tridiagonal matrix as a dense nalgebra matrix.
fn dense_tridiagonal(t: &rustebra::krylov::TridiagonalMatrix<f64, N>) -> nalgebra::Matrix4<f64> {
    let mut dense = nalgebra::Matrix4::zeros();
    for j in 0..N {
        dense[(j, j)] = t.diagonal()[j];
    }
    for (j, &beta) in t.off_diagonal().iter().enumerate() {
        dense[(j, j + 1)] = beta;
        dense[(j + 1, j)] = beta;
    }
    dense
}

fn sorted_eigenvalues(m: &nalgebra::Matrix4<f64>) -> [f64; N] {
    let mut eigenvalues = [0.0; N];
    let symmetric_eigen = m.symmetric_eigen();
    eigenvalues.copy_from_slice(symmetric_eigen.eigenvalues.as_slice());
    eigenvalues.sort_by(|x, y| x.total_cmp(y));
    eigenvalues
}

proptest! {
    /// Builds a random symmetric matrix `a = (m + mᵀ) / 2`, tridiagonalizes it with a
    /// full-dimension basis, and diffs the sorted eigenvalues of the tridiagonal result
    /// against those of `a` itself, both computed by nalgebra's `SymmetricEigen`.
    #[test]
    fn full_basis_tridiagonal_has_the_spectrum_of_a(
        entries in prop::collection::vec(-10.0..10.0f64, N * N),
        v0 in prop::array::uniform4(-1.0..1.0f64),
    ) {
        let m = nalgebra::Matrix4::from_row_slice(&entries);
        let symmetric = (m + m.transpose()) * 0.5;
        let mut a = [0.0; N * N];
        for r in 0..N {
            for c in 0..N {
                a[r * N + c] = symmetric[(r, c)];
            }
        }
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

        // A v0 lying (numerically) inside a proper invariant subspace is legitimate
        // breakdown, not a defect; it is astronomically rare under this generator.
        prop_assume!(result != Err(ConvergenceError::Breakdown));
        let t = result.unwrap();

        let expected = sorted_eigenvalues(&symmetric);
        let actual = sorted_eigenvalues(&dense_tridiagonal(&t));
        for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            prop_assert!(
                (got - want).abs() < TOL,
                "eigenvalue {index}: tridiagonal gives {got}, input matrix gives {want}",
            );
        }
    }
}

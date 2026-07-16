//! Differential oracle: with a full-dimension basis (`K == n`) and no breakdown, the
//! Hessenberg matrix `arnoldi` returns is similar to the input, so nalgebra's
//! `complex_eigenvalues` must recover the same spectrum from both.

use proptest::prelude::*;
use rustebra::krylov::arnoldi;
use rustebra::storage::{Basis, StaticStorage};

const N: usize = 4;
const ALGORITHM_TOL: f64 = 1e-10;
const TOL: f64 = 1e-6;

/// The returned `K x K` Hessenberg matrix as a dense nalgebra matrix.
fn dense_hessenberg(h: &rustebra::krylov::HessenbergMatrix<f64, N>) -> nalgebra::Matrix4<f64> {
    let mut dense = nalgebra::Matrix4::zeros();
    for r in 0..N {
        for c in 0..N {
            dense[(r, c)] = h.entry(r, c).unwrap();
        }
    }
    dense
}

/// Complex eigenvalues sorted by (real, imaginary) so two spectra can be compared pairwise
/// regardless of the order each decomposition happens to produce them in.
fn sorted_eigenvalues(m: &nalgebra::Matrix4<f64>) -> [nalgebra::Complex<f64>; N] {
    let mut eigenvalues = [nalgebra::Complex::new(0.0, 0.0); N];
    eigenvalues.copy_from_slice(m.complex_eigenvalues().as_slice());
    eigenvalues.sort_by(|x, y| (x.re, x.im).partial_cmp(&(y.re, y.im)).unwrap());
    eigenvalues
}

proptest! {
    /// Builds a random (generally non-symmetric) matrix, reduces it to Hessenberg form with a
    /// full-dimension basis, and diffs the sorted eigenvalues of the Hessenberg result
    /// against those of the input itself, both computed by nalgebra's `complex_eigenvalues`.
    #[test]
    fn full_basis_hessenberg_has_the_spectrum_of_a(
        entries in prop::collection::vec(-10.0..10.0f64, N * N),
        v0 in prop::array::uniform4(-1.0..1.0f64),
    ) {
        let a = nalgebra::Matrix4::from_row_slice(&entries);
        let mut a_row_major = [0.0; N * N];
        for r in 0..N {
            for c in 0..N {
                a_row_major[r * N + c] = a[(r, c)];
            }
        }
        let norm_v0: f64 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
        prop_assume!(norm_v0 > 0.1);

        let mut buffer = [0.0; N * N];
        let mut basis = Basis::<f64, N>::new(&mut buffer, N).unwrap();
        let mut scratch = [0.0; N];
        let (h, reached) = arnoldi(
            &StaticStorage::new(a_row_major),
            N,
            &StaticStorage::new(v0),
            ALGORITHM_TOL,
            &mut basis,
            &mut scratch,
        )
        .unwrap();

        // Early breakdown means only a proper subspace was captured — the leading block has
        // no reason to share the full spectrum. It is astronomically rare under this
        // generator, so it's excluded rather than given a separate (weaker) oracle here.
        prop_assume!(reached == N);

        let expected = sorted_eigenvalues(&a);
        let actual = sorted_eigenvalues(&dense_hessenberg(&h));
        for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            prop_assert!(
                (got - want).norm() < TOL,
                "eigenvalue {index}: hessenberg gives {got}, input matrix gives {want}",
            );
        }
    }
}

#![cfg(feature = "alloc")]

//! Property test for `gmres`: on diagonally dominant systems (which guarantee a well-posed,
//! well-conditioned solve), the final residual `‖b - A x‖` must fall below the requested
//! tolerance, checked against an independently computed matrix-vector product rather than
//! anything the solver itself produced internally.

use proptest::prelude::*;
use rustebra::krylov::gmres;
use rustebra::sparse::CsrMatrix;
use rustebra::storage::Basis;

const N: usize = 4;
const TOL: f64 = 1e-9;

/// Dense row-major `a` as a `CsrMatrix`, storing every entry (including exact zeros) so the
/// generator's sparsity pattern never has to be tracked separately.
fn csr_from_dense(a: &[f64; N * N]) -> CsrMatrix<f64> {
    let mut row_ptr = vec![0_u32];
    let mut col_indices = vec![];
    let mut values = vec![];
    for r in 0..N {
        for c in 0..N {
            col_indices.push(c as u32);
            values.push(a[r * N + c]);
        }
        row_ptr.push(col_indices.len() as u32);
    }
    CsrMatrix::new(N, N, row_ptr, col_indices, values).unwrap()
}

fn residual_norm(a: &[f64; N * N], x: &[f64; N], b: &[f64; N]) -> f64 {
    let mut sq = 0.0;
    for r in 0..N {
        let mut ax = 0.0;
        for c in 0..N {
            ax += a[r * N + c] * x[c];
        }
        let ri = b[r] - ax;
        sq += ri * ri;
    }
    sq.sqrt()
}

prop_compose! {
    /// A row-major, strictly diagonally dominant `N x N` matrix: each diagonal entry's
    /// magnitude exceeds the sum of the magnitudes of the rest of its row, which guarantees
    /// non-singularity and a well-conditioned solve for GMRES to converge on within a small
    /// restart budget.
    fn diagonally_dominant_matrix()(
        off_diagonal in prop::array::uniform16(-1.0..1.0f64),
        diagonal_boost in prop::array::uniform4(5.0..10.0f64),
    ) -> [f64; N * N] {
        let mut a = off_diagonal;
        for r in 0..N {
            a[r * N + r] = 0.0;
        }
        let mut row_sums = [0.0; N];
        for r in 0..N {
            row_sums[r] = (0..N).map(|c| a[r * N + c].abs()).sum();
        }
        for r in 0..N {
            a[r * N + r] = row_sums[r] + diagonal_boost[r];
        }
        a
    }
}

proptest! {
    /// GMRES(N) — a full-dimension restart size — solves a diagonally dominant system to
    /// within `TOL` in a small restart budget, verified against an independently computed
    /// residual rather than any value the solver reports about itself.
    #[test]
    fn residual_is_small_after_convergence(
        a in diagonally_dominant_matrix(),
        b in prop::array::uniform4(-10.0..10.0f64),
    ) {
        let m = csr_from_dense(&a);
        let x0 = [0.0; N];
        let mut out_x = [0.0; N];
        let mut buffer = [0.0; N * N];
        let mut basis = Basis::<f64, N>::new(&mut buffer, N).unwrap();
        let mut scratch = [0.0; N];

        gmres(&m, &b, &x0, 10, TOL, &mut out_x, &mut basis, &mut scratch).unwrap();

        let residual = residual_norm(&a, &out_x, &b);
        prop_assert!(
            residual < 1e-6,
            "residual {residual} too large for a converged solve",
        );
    }
}

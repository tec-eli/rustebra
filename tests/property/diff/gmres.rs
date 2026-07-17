#![cfg(feature = "alloc")]

//! Differential property test: `gmres` against `nalgebra`'s direct LU solve on random
//! diagonally dominant systems, comparing the solution vector itself rather than only the
//! residual the solver reports about its own answer.

use super::approx_eq;
use nalgebra::{DMatrix, DVector};
use proptest::prelude::*;
use rustebra::krylov::gmres;
use rustebra::sparse::CsrMatrix;
use rustebra::storage::Basis;

const N: usize = 4;
const TOL: f64 = 1e-6;

/// Dense row-major `a` as a `CsrMatrix`, storing every entry (including exact zeros).
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

prop_compose! {
    /// A row-major, strictly diagonally dominant `N x N` matrix, guaranteeing a well-posed
    /// system nalgebra's direct solve and GMRES(`N`) both converge on.
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
    /// GMRES(N) — a full-dimension restart size, so no restart is ever needed for a
    /// diagonally dominant system — must land on the same solution nalgebra's direct LU solve
    /// produces for `A x = b`.
    #[test]
    fn solution_matches_nalgebra_direct_solve(
        a in diagonally_dominant_matrix(),
        b in prop::array::uniform4(-10.0..10.0f64),
    ) {
        let m = csr_from_dense(&a);
        let x0 = [0.0; N];
        let mut out_x = [0.0; N];
        let mut buffer = [0.0; N * N];
        let mut basis = Basis::<f64, N>::new(&mut buffer, N).unwrap();
        let mut scratch = [0.0; N];

        gmres(&m, &b, &x0, 10, 1e-10, &mut out_x, &mut basis, &mut scratch).unwrap();

        let a_na = DMatrix::from_row_slice(N, N, &a);
        let b_na = DVector::from_row_slice(&b);
        let x_na = a_na
            .lu()
            .solve(&b_na)
            .expect("diagonally dominant matrices are always invertible");

        for i in 0..N {
            prop_assert!(
                approx_eq(out_x[i], x_na[i], TOL),
                "x[{}]: ours={} vs nalgebra={}",
                i,
                out_x[i],
                x_na[i]
            );
        }
    }
}

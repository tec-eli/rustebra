//! Differential property test: `rustebra`'s LU decomposition against
//! `nalgebra`'s LU on random square matrices.
//!
//! Both libraries produce `L * U ≈ P * A` where `L` is unit lower triangular, `U` is upper
//! triangular, and `P` is a permutation matrix encoding row swaps. We verify that both
//! factorizations reconstruct a row-permuted version of the original matrix correctly.

use super::approx_eq;
use nalgebra::DMatrix;
use proptest::prelude::*;
use rustebra::algorithm::matrix::{lu, mul_matrix};
use rustebra::storage::StaticStorage;

const N: usize = 4;
const TOL: f64 = 1e-8;

proptest! {
    /// Property test: `lu` and `nalgebra`'s LU both reconstruct a (row-permuted) version of
    /// the input correctly.
    #[test]
    fn lu_reconstruction_matches_nalgebra_lu(
        entries in prop::collection::vec(-10.0..10.0f64, N * N),
    ) {
        let mut a = [0.0_f64; N * N];
        a.copy_from_slice(&entries);

        let mut l = [0.0_f64; N * N];
        let mut u = [0.0_f64; N * N];
        lu(&StaticStorage::new(a), N, N, &mut l, &mut u)
            .expect("N x N input with correctly-sized buffers never returns Err");

        // Get nalgebra's LU decomposition.
        let a_na = DMatrix::from_row_slice(N, N, &entries);
        let lu_na = a_na.clone().lu();
        let l_na = lu_na.l();
        let u_na = lu_na.u();

        // Compute both L*U products.
        let mut lu_product = [0.0_f64; N * N];
        mul_matrix(&StaticStorage::new(l), N, N, &StaticStorage::new(u), N, N, &mut lu_product)
            .expect("N x N matrices always multiply into N x N product");

        let lu_na_product = &l_na * &u_na;

        // Both L*U products should be identical (both reconstruct the same permuted input).
        for r in 0..N {
            for c in 0..N {
                let ours = lu_product[r * N + c];
                let theirs = lu_na_product[(r, c)];
                prop_assert!(
                    approx_eq(ours, theirs, TOL),
                    "L*U reconstruction[{},{}]: ours={} vs nalgebra={}",
                    r,
                    c,
                    ours,
                    theirs
                );
            }
        }

        // Compare L matrices entry-by-entry (L is unit lower triangular for both).
        for r in 0..N {
            for c in 0..N {
                let ours = l[r * N + c];
                let theirs = l_na[(r, c)];
                prop_assert!(
                    approx_eq(ours, theirs, TOL),
                    "L[{},{}]: ours={} vs nalgebra={}",
                    r,
                    c,
                    ours,
                    theirs
                );
            }
        }

        // Compare U matrices entry-by-entry (both upper triangular).
        for r in 0..N {
            for c in 0..N {
                let ours = u[r * N + c];
                let theirs = u_na[(r, c)];
                prop_assert!(
                    approx_eq(ours, theirs, TOL),
                    "U[{},{}]: ours={} vs nalgebra={}",
                    r,
                    c,
                    ours,
                    theirs
                );
            }
        }
    }
}

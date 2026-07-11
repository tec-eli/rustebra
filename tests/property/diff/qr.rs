//! Differential property test: `rustebra`'s Householder QR against `nalgebra`'s QR
//! decomposition, on random square matrices.
//!
//! For a square input both libraries produce a full, square `Q` and a same-shaped `R`, so
//! their outputs are directly comparable: this checks that both reconstruct the original
//! matrix (`Q * R ≈ A`) and that the `R` diagonal entries agree in magnitude.
//!
//! Signs are compared as magnitudes rather than asserted equal outright: `nalgebra`'s `r()`
//! always returns a non-negative diagonal (it stores the Householder scale's sign
//! separately, applying it to `q()` instead, and takes the modulus for `r()`), while
//! `qr_householder` keeps whichever sign its cancellation-avoiding reflection naturally
//! produces, which can be negative. Asserting raw sign equality would therefore fail on
//! valid input through no fault of either implementation; the magnitude is what's actually
//! convention-independent, and `nalgebra`'s non-negativity is asserted directly instead.

use super::approx_eq;
use nalgebra::DMatrix;
use proptest::prelude::*;
use rustebra::algorithm::matrix::{mul_matrix, qr_householder};
use rustebra::storage::StaticStorage;

const N: usize = 4;
const TOL: f64 = 1e-8;

proptest! {
    /// Property test: `qr_householder` and `nalgebra`'s QR agree on reconstruction and on
    /// the magnitude of every `R` diagonal entry.
    #[test]
    fn qr_matches_nalgebra_reconstruction_and_diagonal_magnitudes(
        entries in prop::collection::vec(-10.0..10.0f64, N * N),
    ) {
        let mut a = [0.0_f64; N * N];
        a.copy_from_slice(&entries);

        let mut q = [0.0_f64; N * N];
        let mut r = [0.0_f64; N * N];
        let mut scratch = [0.0_f64; N];
        qr_householder(&StaticStorage::new(a), N, N, &mut q, &mut r, &mut scratch)
            .expect("N x N input with correctly-sized buffers never returns Err");

        let a_na = DMatrix::from_row_slice(N, N, &entries);
        let qr_na = a_na.clone().qr();
        let q_na = qr_na.q();
        let r_na = qr_na.r();

        // Both factorizations reconstruct the original matrix.
        let mut qr_product = [0.0_f64; N * N];
        mul_matrix(&StaticStorage::new(q), N, N, &StaticStorage::new(r), N, N, &mut qr_product)
            .expect("N x N Q and R always multiply into an N x N product");
        for (actual, expected) in qr_product.iter().zip(entries.iter()) {
            prop_assert!(approx_eq(*actual, *expected, TOL));
        }

        let qr_na_product = &q_na * &r_na;
        for row in 0..N {
            for col in 0..N {
                prop_assert!(approx_eq(qr_na_product[(row, col)], a_na[(row, col)], TOL));
            }
        }

        // R diagonal magnitudes agree; nalgebra's diagonal is always non-negative by
        // construction (see the module doc comment for why raw signs aren't compared).
        for k in 0..N {
            let ours = r[k * N + k];
            let theirs = r_na[(k, k)];
            prop_assert!(theirs >= 0.0);
            prop_assert!(approx_eq(ours.abs(), theirs, TOL));
        }
    }
}

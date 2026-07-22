//! Property tests for [`qr`]: unlike the differential tests in `diff::qr`, these check the
//! defining invariants of a QR decomposition directly (orthogonality of `Q`, triangularity
//! of `R`, reconstruction) rather than comparing against `nalgebra`, so they also cover
//! shapes this crate's `qr` accepts that `nalgebra`'s QR wouldn't be a fair oracle for.
//!
//! `qr_householder` (and thus `qr`) is only defined for `rows >= cols`, so every shape
//! below respects that; a wide matrix is exercised instead as `cols x rows` (its transpose).

use proptest::prelude::*;
use rustebra::algorithm::matrix::{mul_matrix, qr, transpose};
use rustebra::storage::StaticStorage;

const TOL: f64 = 1e-8;

/// Whether the `n x n` matrix `q` (row-major) satisfies `qᵗq ≈ I` within `tol`.
fn is_orthogonal(q: &[f64], n: usize, tol: f64) -> bool {
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..n).map(|k| q[k * n + i] * q[k * n + j]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            if (dot - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Whether the `rows x cols` matrix `r` (row-major) has every strictly-below-diagonal entry
/// within `tol` of zero. `qr_householder` derives each such entry as the accumulated
/// difference of near-equal floating-point terms that are mathematically exactly zero, so
/// they land within a few `epsilon`-scale units of it rather than at it precisely.
fn is_upper_triangular(r: &[f64], rows: usize, cols: usize, tol: f64) -> bool {
    for row in 0..rows {
        for col in 0..cols.min(row) {
            if r[row * cols + col].abs() > tol {
                return false;
            }
        }
    }
    true
}

macro_rules! qr_property_test {
    ($name:ident, $rows:expr, $cols:expr) => {
        proptest! {
            /// `qr` on a random `rows x cols` matrix produces an orthogonal `Q`, an
            /// upper-triangular `R`, and `Q * R` reconstructs the input.
            #[test]
            fn $name(entries in prop::collection::vec(-10.0..10.0f64, $rows * $cols)) {
                const ROWS: usize = $rows;
                const COLS: usize = $cols;

                let mut a = [0.0_f64; ROWS * COLS];
                a.copy_from_slice(&entries);

                let mut q = [0.0_f64; ROWS * ROWS];
                let mut r = [0.0_f64; ROWS * COLS];
                let mut scratch = [0.0_f64; ROWS];
                qr(&StaticStorage::new(a), ROWS, COLS, &mut q, &mut r, &mut scratch)
                    .expect("ROWS x COLS input with correctly-sized buffers never returns Err");

                prop_assert!(is_orthogonal(&q, ROWS, TOL), "Q is not orthogonal: {:?}", q);
                prop_assert!(is_upper_triangular(&r, ROWS, COLS, TOL), "R is not upper triangular: {:?}", r);

                let mut qr_product = [0.0_f64; ROWS * COLS];
                mul_matrix(
                    &StaticStorage::new(q),
                    ROWS,
                    ROWS,
                    &StaticStorage::new(r),
                    ROWS,
                    COLS,
                    &mut qr_product,
                )
                .expect("ROWS x ROWS Q and ROWS x COLS R always multiply into a ROWS x COLS product");
                for (actual, expected) in qr_product.iter().zip(entries.iter()) {
                    prop_assert!((actual - expected).abs() < TOL);
                }
            }
        }
    };
}

qr_property_test!(qr_of_3x3_matrix_is_orthogonal_and_reconstructs, 3, 3);
qr_property_test!(qr_of_5x4_matrix_is_orthogonal_and_reconstructs, 5, 4);
qr_property_test!(qr_of_5x3_matrix_is_orthogonal_and_reconstructs, 5, 3);

proptest! {
    /// A wide `4x5` matrix is only decomposable through `qr` as its `5x4` transpose (`qr`
    /// requires `rows >= cols`); this checks that transposing back after decomposing still
    /// reconstructs the original wide matrix.
    #[test]
    fn qr_of_4x5_matrix_via_transpose_reconstructs(
        entries in prop::collection::vec(-10.0..10.0f64, 4 * 5),
    ) {
        let mut wide = [0.0_f64; 4 * 5];
        wide.copy_from_slice(&entries);

        let mut tall = [0.0_f64; 5 * 4];
        transpose(&StaticStorage::new(wide), 4, 5, &mut tall).unwrap();

        let mut q = [0.0_f64; 5 * 5];
        let mut r = [0.0_f64; 5 * 4];
        let mut scratch = [0.0_f64; 5];
        qr(&StaticStorage::new(tall), 5, 4, &mut q, &mut r, &mut scratch)
            .expect("5 x 4 input with correctly-sized buffers never returns Err");

        prop_assert!(is_orthogonal(&q, 5, TOL));
        prop_assert!(is_upper_triangular(&r, 5, 4, TOL));

        let mut qr_product = [0.0_f64; 5 * 4];
        mul_matrix(&StaticStorage::new(q), 5, 5, &StaticStorage::new(r), 5, 4, &mut qr_product)
            .unwrap();

        let mut reconstructed_wide = [0.0_f64; 4 * 5];
        transpose(&StaticStorage::new(qr_product), 5, 4, &mut reconstructed_wide).unwrap();
        for (actual, expected) in reconstructed_wide.iter().zip(entries.iter()) {
            prop_assert!((actual - expected).abs() < TOL);
        }
    }
}

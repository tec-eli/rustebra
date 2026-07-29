//! Edge-case tests for dimension extremes: degenerate `0xn` / `nx0` / `0x0` shapes, the
//! smallest nontrivial shape (`1x1`), and very rectangular shapes (`50x3`, `3x50`).
//!
//! The dense decompositions each carry their own shape constraint (LU/Cholesky require
//! square, QR requires `rows >= cols`), so a 0-dimension input is only a genuine
//! `DimensionMismatch` when it violates *that* operation's own constraint. When a
//! 0-dimension shape happens to satisfy an operation's constraint (a `0x0` "square" matrix,
//! or an `nx0` matrix for QR, which trivially satisfies `rows >= cols`), the decomposition is
//! mathematically well-defined and returns `Ok` — the same precedent already established and
//! tested for `determinant` (`0x0` is the empty product, `1`), `rank` (`0x0` is rank `0`), and
//! `CooMatrix`/`CsrMatrix` construction (a matrix with 0 rows or 0 columns is a valid, empty
//! sparse matrix). No operation below is ever expected to panic.
//!
//! For `StaticMatrix`, LU/Cholesky's square requirement is enforced by the type system: a
//! non-square `StaticMatrix<T, R, C>` simply has no `.lu()`/`.cholesky()` method to call, so
//! there's no runtime `Err` to observe there. The runtime `DimensionMismatch` these operations
//! report is instead exercised through `DynamicMatrix`, whose shape lives in its fields, not
//! its type.

use rustebra::algorithm::matrix::{
    CholeskyError, DimensionMismatch, cholesky_decompose, lu, mul_matrix, qr, svd,
};
use rustebra::matrix::StaticMatrix;
use rustebra::storage::{StaticStorage, Storage};
use rustebra::vector::StaticVector;

#[cfg(feature = "alloc")]
use rustebra::matrix::DynamicMatrix;
#[cfg(feature = "alloc")]
use rustebra::sparse::{CooError, CooMatrix, CscMatrix, CsrMatrix};

// ───────────────────────── 0-dimension: matrix construction ─────────────────────────

#[test]
fn static_matrix_constructs_at_every_zero_dimension_shape_not_a_panic() {
    let square = StaticMatrix::<f64, 0, 0>::new([]);
    assert_eq!(square.get(0), None);

    let zero_rows: StaticMatrix<f64, 0, 3> = StaticMatrix::new([]);
    assert_eq!(zero_rows.get(0), None);

    let zero_cols: StaticMatrix<f64, 3, 0> = StaticMatrix::new([[], [], []]);
    assert_eq!(zero_cols.get(0), None);
}

#[cfg(feature = "alloc")]
#[test]
fn dynamic_matrix_construction_accepts_every_zero_dimension_shape_not_a_panic() {
    let square = DynamicMatrix::<f64>::new(0, 0, vec![]).unwrap();
    assert_eq!((square.rows(), square.cols()), (0, 0));

    let zero_rows = DynamicMatrix::<f64>::new(0, 3, vec![]).unwrap();
    assert_eq!((zero_rows.rows(), zero_rows.cols()), (0, 3));

    let zero_cols = DynamicMatrix::<f64>::new(3, 0, vec![]).unwrap();
    assert_eq!((zero_cols.rows(), zero_cols.cols()), (3, 0));
}

#[cfg(feature = "alloc")]
#[test]
fn dynamic_matrix_construction_rejects_stray_data_at_a_zero_dimension_not_a_panic() {
    // 0 rows means 0 elements are expected; one stray element is still a length mismatch,
    // not silently ignored.
    assert_eq!(
        DynamicMatrix::<f64>::new(0, 3, vec![1.0]),
        Err(DimensionMismatch)
    );
}

// ───────────────────────── 0-dimension: LU (square only) ─────────────────────────

#[test]
fn lu_of_a_zero_by_zero_matrix_is_ok_with_no_swaps_and_empty_factors() {
    let m = StaticMatrix::<f64, 0, 0>::new([]);
    let (l, u, swap_count) = m.lu();
    assert_eq!(swap_count, 0);
    assert_eq!(l, StaticMatrix::new([]));
    assert_eq!(u, StaticMatrix::new([]));
}

#[cfg(feature = "alloc")]
#[test]
fn lu_of_a_non_square_zero_dimension_matrix_is_an_error_not_a_panic() {
    let zero_rows = DynamicMatrix::<f64>::new(0, 3, vec![]).unwrap();
    assert_eq!(zero_rows.lu(), Err(DimensionMismatch));

    let zero_cols = DynamicMatrix::<f64>::new(3, 0, vec![]).unwrap();
    assert_eq!(zero_cols.lu(), Err(DimensionMismatch));
}

// ───────────────────────── 0-dimension: QR (rows >= cols) ─────────────────────────

#[test]
fn qr_of_a_zero_by_zero_matrix_is_ok_trivially() {
    let m = StaticMatrix::<f64, 0, 0>::new([]);
    let (q, r) = m.qr().unwrap();
    assert_eq!(q, StaticMatrix::new([]));
    assert_eq!(r, StaticMatrix::new([]));
}

#[test]
fn qr_of_a_matrix_with_zero_columns_is_ok_with_q_as_the_identity() {
    // `rows >= cols` holds trivially once `cols == 0`, so this is a well-defined (if
    // degenerate) decomposition: no column is ever reflected, so `q` stays the `r x r`
    // identity, and `r` has no columns to hold any data.
    let m = StaticMatrix::<f64, 3, 0>::new([[], [], []]);
    let (q, r) = m.qr().unwrap();
    assert_eq!(
        q,
        StaticMatrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    );
    assert_eq!(r, StaticMatrix::new([[], [], []]));
}

#[test]
fn qr_of_a_matrix_with_zero_rows_is_an_error_since_it_would_need_more_columns_than_rows() {
    let m = StaticMatrix::<f64, 0, 3>::new([]);
    assert_eq!(m.qr(), Err(DimensionMismatch));
}

// ───────────────────────── 0-dimension: Cholesky (square only) ─────────────────────────

#[test]
fn cholesky_of_a_zero_by_zero_matrix_is_ok_trivially() {
    let m = StaticMatrix::<f64, 0, 0>::new([]);
    assert_eq!(m.cholesky(), Ok(StaticMatrix::new([])));
}

#[cfg(feature = "alloc")]
#[test]
fn cholesky_of_a_non_square_zero_dimension_matrix_is_an_error_not_a_panic() {
    let zero_rows = DynamicMatrix::<f64>::new(0, 3, vec![]).unwrap();
    assert_eq!(zero_rows.cholesky(), Err(CholeskyError::DimensionMismatch));

    let zero_cols = DynamicMatrix::<f64>::new(3, 0, vec![]).unwrap();
    assert_eq!(zero_cols.cholesky(), Err(CholeskyError::DimensionMismatch));
}

// ───────────────────────── 0-dimension: SVD (any shape) ─────────────────────────

#[test]
fn svd_of_a_zero_by_zero_matrix_is_ok_with_empty_outputs() {
    let m = StaticMatrix::<f64, 0, 0>::new([]);
    let mut scratch: [f64; 0] = [];
    let (u, sigma, v) = m.svd(&mut scratch).unwrap();
    assert_eq!(u, StaticMatrix::new([]));
    assert_eq!(sigma, StaticVector::new([]));
    assert_eq!(v, StaticMatrix::new([]));
}

#[test]
fn svd_of_a_matrix_with_zero_columns_is_ok_with_empty_outputs() {
    let m = StaticMatrix::<f64, 3, 0>::new([[], [], []]);
    let mut scratch = [0.0; 3]; // 5*0*0 + 0 + 3
    let (u, sigma, v) = m.svd(&mut scratch).unwrap();
    assert_eq!(u, StaticMatrix::new([[], [], []]));
    assert_eq!(sigma, StaticVector::new([]));
    assert_eq!(v, StaticMatrix::new([]));
}

#[test]
fn svd_of_a_matrix_with_zero_rows_is_ok_with_every_singular_value_zero() {
    // A 0-row matrix has no data at all, so `aᵗ * a` is genuinely the zero matrix: every
    // singular value is 0, not a shape error.
    let m = StaticMatrix::<f64, 0, 3>::new([]);
    let mut scratch = [0.0; 5 * 3 * 3 + 3]; // + 0 rows
    let (u, sigma, v) = m.svd(&mut scratch).unwrap();
    assert_eq!(u, StaticMatrix::new([]));
    assert_eq!(sigma, StaticVector::new([0.0, 0.0, 0.0]));
    assert_eq!(
        v,
        StaticMatrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    );
}

// ───────────────────────── 0-dimension: sparse construction ─────────────────────────

#[cfg(feature = "alloc")]
#[test]
fn coo_matrix_construction_accepts_every_zero_dimension_shape_not_a_panic() {
    let square = CooMatrix::<f64>::new(0, 0, vec![], vec![], vec![]).unwrap();
    assert_eq!((square.rows(), square.cols(), square.nnz()), (0, 0, 0));

    let zero_rows = CooMatrix::<f64>::new(0, 4, vec![], vec![], vec![]).unwrap();
    assert_eq!((zero_rows.rows(), zero_rows.cols()), (0, 4));

    let zero_cols = CooMatrix::<f64>::new(4, 0, vec![], vec![], vec![]).unwrap();
    assert_eq!((zero_cols.rows(), zero_cols.cols()), (4, 0));
}

#[cfg(feature = "alloc")]
#[test]
fn coo_matrix_construction_rejects_an_index_at_a_zero_dimension_not_a_panic() {
    // 0 columns means there is no valid column index at all; index 0 is already out of
    // bounds.
    let err = CooMatrix::<f64>::new(4, 0, vec![0], vec![0], vec![1.0]);
    assert_eq!(err, Err(CooError::ColIndexOutOfBounds));
}

#[cfg(feature = "alloc")]
#[test]
fn csr_matrix_construction_accepts_every_zero_dimension_shape_not_a_panic() {
    let square = CsrMatrix::<f64>::new(0, 0, vec![0], vec![], vec![]).unwrap();
    assert_eq!((square.rows(), square.cols(), square.nnz()), (0, 0, 0));

    let zero_rows = CsrMatrix::<f64>::new(0, 4, vec![0], vec![], vec![]).unwrap();
    assert_eq!((zero_rows.rows(), zero_rows.cols()), (0, 4));

    // 0 columns still needs one `row_ptr` entry per row plus a sentinel; every row is
    // forced empty, since no column index could ever be in bounds.
    let zero_cols = CsrMatrix::<f64>::new(4, 0, vec![0, 0, 0, 0, 0], vec![], vec![]).unwrap();
    assert_eq!((zero_cols.rows(), zero_cols.cols()), (4, 0));
}

#[cfg(feature = "alloc")]
#[test]
fn csc_matrix_construction_accepts_every_zero_dimension_shape_not_a_panic() {
    let square = CscMatrix::<f64>::new(0, 0, vec![0], vec![], vec![]).unwrap();
    assert_eq!((square.rows(), square.cols(), square.nnz()), (0, 0, 0));

    let zero_cols = CscMatrix::<f64>::new(4, 0, vec![0], vec![], vec![]).unwrap();
    assert_eq!((zero_cols.rows(), zero_cols.cols()), (4, 0));

    // 0 rows still needs one `col_ptr` entry per column plus a sentinel; every column is
    // forced empty, since no row index could ever be in bounds.
    let zero_rows = CscMatrix::<f64>::new(0, 4, vec![0, 0, 0, 0, 0], vec![], vec![]).unwrap();
    assert_eq!((zero_rows.rows(), zero_rows.cols()), (0, 4));
}

// ───────────────────────── 1x1 sanity checks ─────────────────────────

#[test]
fn lu_of_a_1x1_matrix_recovers_the_scalar_with_no_swaps() {
    let m = StaticMatrix::<f64, 1, 1>::new([[7.0]]);
    let (l, u, swap_count) = m.lu();
    assert_eq!(swap_count, 0);
    assert_eq!(l, StaticMatrix::new([[1.0]]));
    assert_eq!(u, StaticMatrix::new([[7.0]]));
}

#[test]
fn qr_of_a_1x1_matrix_reconstructs_the_scalar() {
    let m = StaticMatrix::<f64, 1, 1>::new([[7.0_f64]]);
    let (q, r) = m.qr().unwrap();

    let reconstructed = q.mul_matrix(&r);
    assert!((reconstructed.get(0).unwrap() - 7.0).abs() < 1e-9);
    // `q` is a 1x1 orthogonal matrix, so its only entry is +-1.
    assert!((q.get(0).unwrap().abs() - 1.0).abs() < 1e-9);
}

#[test]
fn cholesky_of_a_1x1_positive_matrix_is_its_square_root() {
    let m = StaticMatrix::<f64, 1, 1>::new([[4.0]]);
    assert_eq!(m.cholesky(), Ok(StaticMatrix::new([[2.0]])));
}

#[test]
fn svd_of_a_1x1_matrix_is_the_absolute_value_of_the_scalar() {
    let m = StaticMatrix::<f64, 1, 1>::new([[-7.0_f64]]);
    let mut scratch = [0.0; 7]; // 5*C*C + C + R with C = R = 1
    let (_, sigma, _) = m.svd(&mut scratch).unwrap();
    assert!((sigma.get(0).unwrap() - 7.0).abs() < 1e-9);
}

// ───────────────────────── very rectangular: 50x3 and 3x50 ─────────────────────────

const TALL_ROWS: usize = 50;
const WIDE_COLS: usize = 50;
const SHORT: usize = 3;

/// A deterministic, non-degenerate flat data set of `LEN` entries — distinct integer-valued
/// entries (not all equal, not all zero) — enough to exercise the decompositions below
/// without needing anything beyond integer arithmetic to construct.
fn rectangular_pattern<const LEN: usize>() -> [f64; LEN] {
    core::array::from_fn(|i| ((i * 37 + 5) % 97) as f64 + 1.0)
}

/// Computes `u * diag(sigma) * vᵗ` into `out`, entry by entry — used to check SVD
/// reconstruction for shapes too large to hardcode expected values for.
fn reconstruct(u: &[f64], sigma: &[f64], v: &[f64], rows: usize, cols: usize, out: &mut [f64]) {
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..cols {
                sum += u[i * cols + k] * sigma[k] * v[j * cols + k];
            }
            out[i * cols + j] = sum;
        }
    }
}

#[cfg(feature = "alloc")]
#[test]
fn matrix_construction_accepts_both_very_rectangular_shapes() {
    let tall = DynamicMatrix::new(
        TALL_ROWS,
        SHORT,
        rectangular_pattern::<{ TALL_ROWS * SHORT }>().to_vec(),
    )
    .unwrap();
    assert_eq!((tall.rows(), tall.cols()), (TALL_ROWS, SHORT));

    let wide = DynamicMatrix::new(
        SHORT,
        WIDE_COLS,
        rectangular_pattern::<{ SHORT * WIDE_COLS }>().to_vec(),
    )
    .unwrap();
    assert_eq!((wide.rows(), wide.cols()), (SHORT, WIDE_COLS));
}

#[test]
fn lu_and_cholesky_reject_both_very_rectangular_shapes_since_neither_is_square() {
    let tall = StaticStorage::new(rectangular_pattern::<{ TALL_ROWS * SHORT }>());
    let mut l = [0.0; TALL_ROWS * SHORT];
    let mut u = [0.0; TALL_ROWS * SHORT];
    assert_eq!(
        lu(&tall, TALL_ROWS, SHORT, &mut l, &mut u),
        Err(DimensionMismatch)
    );
    assert_eq!(
        cholesky_decompose(&tall, TALL_ROWS, SHORT, &mut l, 1e-9),
        Err(CholeskyError::DimensionMismatch)
    );

    let wide = StaticStorage::new(rectangular_pattern::<{ SHORT * WIDE_COLS }>());
    let mut l2 = [0.0; SHORT * WIDE_COLS];
    let mut u2 = [0.0; SHORT * WIDE_COLS];
    assert_eq!(
        lu(&wide, SHORT, WIDE_COLS, &mut l2, &mut u2),
        Err(DimensionMismatch)
    );
    assert_eq!(
        cholesky_decompose(&wide, SHORT, WIDE_COLS, &mut l2, 1e-9),
        Err(CholeskyError::DimensionMismatch)
    );
}

#[test]
fn qr_of_the_tall_rectangular_matrix_reconstructs_the_original() {
    let a = StaticStorage::new(rectangular_pattern::<{ TALL_ROWS * SHORT }>());
    let mut q = [0.0; TALL_ROWS * TALL_ROWS];
    let mut r = [0.0; TALL_ROWS * SHORT];
    let mut scratch = [0.0; TALL_ROWS];

    assert_eq!(
        qr(&a, TALL_ROWS, SHORT, &mut q, &mut r, &mut scratch),
        Ok(())
    );

    let mut reconstructed = [0.0; TALL_ROWS * SHORT];
    mul_matrix(
        &StaticStorage::new(q),
        TALL_ROWS,
        TALL_ROWS,
        &StaticStorage::new(r),
        TALL_ROWS,
        SHORT,
        &mut reconstructed,
    )
    .unwrap();
    for (actual, expected) in reconstructed
        .iter()
        .zip(rectangular_pattern::<{ TALL_ROWS * SHORT }>())
    {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn qr_of_the_wide_rectangular_matrix_is_an_error_since_it_has_more_columns_than_rows() {
    let a = StaticStorage::new(rectangular_pattern::<{ SHORT * WIDE_COLS }>());
    let mut q = [0.0; SHORT * SHORT];
    let mut r = [0.0; SHORT * WIDE_COLS];
    let mut scratch = [0.0; SHORT];

    assert_eq!(
        qr(&a, SHORT, WIDE_COLS, &mut q, &mut r, &mut scratch),
        Err(DimensionMismatch)
    );
}

#[test]
fn svd_of_the_tall_rectangular_matrix_reconstructs_the_original() {
    let a = StaticStorage::new(rectangular_pattern::<{ TALL_ROWS * SHORT }>());
    let mut u = [0.0; TALL_ROWS * SHORT];
    let mut sigma = [0.0; SHORT];
    let mut v = [0.0; SHORT * SHORT];
    let mut scratch = [0.0; 5 * SHORT * SHORT + SHORT + TALL_ROWS];

    assert_eq!(
        svd(
            &a,
            TALL_ROWS,
            SHORT,
            &mut u,
            &mut sigma,
            &mut v,
            &mut scratch
        ),
        Ok(())
    );
    for w in sigma.windows(2) {
        assert!(w[0] >= w[1]);
    }
    for &s in &sigma {
        assert!(s >= 0.0);
    }

    let mut reconstructed = [0.0; TALL_ROWS * SHORT];
    reconstruct(&u, &sigma, &v, TALL_ROWS, SHORT, &mut reconstructed);
    for (actual, expected) in reconstructed
        .iter()
        .zip(rectangular_pattern::<{ TALL_ROWS * SHORT }>())
    {
        assert!((actual - expected).abs() < 1e-4);
    }
}

#[test]
fn svd_of_the_wide_rectangular_matrix_reconstructs_the_original() {
    let a = StaticStorage::new(rectangular_pattern::<{ SHORT * WIDE_COLS }>());
    let mut u = [0.0; SHORT * WIDE_COLS];
    let mut sigma = [0.0; WIDE_COLS];
    let mut v = [0.0; WIDE_COLS * WIDE_COLS];
    let mut scratch = [0.0; 5 * WIDE_COLS * WIDE_COLS + WIDE_COLS + SHORT];

    assert_eq!(
        svd(
            &a,
            SHORT,
            WIDE_COLS,
            &mut u,
            &mut sigma,
            &mut v,
            &mut scratch
        ),
        Ok(())
    );
    for w in sigma.windows(2) {
        assert!(w[0] >= w[1]);
    }
    for &s in &sigma {
        assert!(s >= 0.0);
    }

    let mut reconstructed = [0.0; SHORT * WIDE_COLS];
    reconstruct(&u, &sigma, &v, SHORT, WIDE_COLS, &mut reconstructed);
    for (actual, expected) in reconstructed
        .iter()
        .zip(rectangular_pattern::<{ SHORT * WIDE_COLS }>())
    {
        assert!((actual - expected).abs() < 1e-4);
    }
}

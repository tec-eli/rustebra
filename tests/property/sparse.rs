#![cfg(feature = "alloc")]

//! Property tests for sparse arithmetic (`add_csr`, `spmm_csr`), checked against dense
//! equivalents computed directly from the same randomly generated entries.

use proptest::prelude::*;
use rustebra::sparse::{CsrMatrix, add_csr, spmm_csr};

/// Whether `a` and `b` differ by no more than `tol` in absolute value.
fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Generates `(rows, cols, dense)`, where `dense` is a row-major `rows * cols` grid whose
/// cells are either exactly zero or a random value in `-20.0..20.0`.
fn dense_matrix(max_dim: usize) -> impl Strategy<Value = (usize, usize, Vec<f64>)> {
    (1usize..=max_dim, 1usize..=max_dim).prop_flat_map(|(rows, cols)| {
        prop::collection::vec(prop_oneof![Just(0.0f64), -20.0..20.0f64], rows * cols)
            .prop_map(move |dense| (rows, cols, dense))
    })
}

/// Builds a `CsrMatrix` from a row-major dense grid, storing only the non-zero cells.
fn csr_from_dense(rows: usize, cols: usize, dense: &[f64]) -> CsrMatrix<f64> {
    let mut row_ptr = vec![0u32; rows + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = dense[r * cols + c];
            if v != 0.0 {
                col_indices.push(c as u32);
                values.push(v);
            }
        }
        row_ptr[r + 1] = col_indices.len() as u32;
    }
    CsrMatrix::new(rows, cols, row_ptr, col_indices, values).expect("built from a valid dense grid")
}

/// Expands a `CsrMatrix` back into a row-major dense grid for comparison.
fn dense_from_csr(m: &CsrMatrix<f64>) -> Vec<f64> {
    let mut dense = vec![0.0f64; m.rows() * m.cols()];
    for r in 0..m.rows() {
        for k in m.row_range(r).expect("row < rows") {
            let c = m.col_indices()[k] as usize;
            dense[r * m.cols() + c] = m.values()[k];
        }
    }
    dense
}

proptest! {
    /// `add_csr` matches element-wise dense addition of the same two matrices.
    #[test]
    fn add_csr_matches_dense_addition(
        (rows, cols, a_dense) in dense_matrix(8),
        b_entries in prop::collection::vec(prop_oneof![Just(0.0f64), -20.0..20.0f64], 0..64),
    ) {
        let b_dense: Vec<f64> = (0..rows * cols)
            .map(|i| b_entries.get(i).copied().unwrap_or(0.0))
            .collect();

        let a = csr_from_dense(rows, cols, &a_dense);
        let b = csr_from_dense(rows, cols, &b_dense);
        let c = add_csr(&a, &b).expect("same shape by construction");

        let expected: Vec<f64> = a_dense.iter().zip(b_dense.iter()).map(|(x, y)| x + y).collect();
        let actual = dense_from_csr(&c);
        for (act, exp) in actual.iter().zip(expected.iter()) {
            prop_assert!(approx_eq(*act, *exp, 1e-9));
        }

        // No spurious zeros: every stored value must be non-zero.
        for &v in c.values() {
            prop_assert_ne!(v, 0.0);
        }
    }

    /// `spmm_csr` matches a dense matrix multiply of the same two matrices.
    #[test]
    fn spmm_csr_matches_dense_matmul(
        m in 1usize..=6,
        k in 1usize..=6,
        n in 1usize..=6,
        a_entries in prop::collection::vec(prop_oneof![Just(0.0f64), -10.0..10.0f64], 0..64),
        b_entries in prop::collection::vec(prop_oneof![Just(0.0f64), -10.0..10.0f64], 0..64),
    ) {
        let a_dense: Vec<f64> = (0..m * k).map(|i| a_entries.get(i).copied().unwrap_or(0.0)).collect();
        let b_dense: Vec<f64> = (0..k * n).map(|i| b_entries.get(i).copied().unwrap_or(0.0)).collect();

        let a = csr_from_dense(m, k, &a_dense);
        let b = csr_from_dense(k, n, &b_dense);
        let c = spmm_csr(&a, &b).expect("a.cols() == b.rows() by construction");

        let mut expected = vec![0.0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                let av = a_dense[i * k + p];
                if av == 0.0 {
                    continue;
                }
                for j in 0..n {
                    expected[i * n + j] += av * b_dense[p * n + j];
                }
            }
        }

        let actual = dense_from_csr(&c);
        for (act, exp) in actual.iter().zip(expected.iter()) {
            prop_assert!(approx_eq(*act, *exp, 1e-6));
        }

        // No spurious zeros: every stored value must be non-zero.
        for &v in c.values() {
            prop_assert_ne!(v, 0.0);
        }
    }

    /// Adding a matrix to its negation always cancels every entry, regardless of sparsity
    /// pattern: this exercises `add_csr`'s zero-filtering on many random shapes/patterns.
    #[test]
    fn add_csr_cancels_to_empty((rows, cols, dense) in dense_matrix(8)) {
        let a = csr_from_dense(rows, cols, &dense);
        let negated: Vec<f64> = dense.iter().map(|v| -v).collect();
        let b = csr_from_dense(rows, cols, &negated);

        let c = add_csr(&a, &b).expect("same shape by construction");
        prop_assert_eq!(c.nnz(), 0);
    }

    /// `spmm_csr` filters exact cancellation in the accumulated product, not just in inputs:
    /// a 1x2 times 2x1 product where the two contributing terms are `v*w` and `(-v)*w`
    /// always cancels to exactly zero and must not be stored.
    #[test]
    fn spmm_csr_cancels_accumulated_product(v in -20.0..20.0f64, w in -20.0..20.0f64) {
        prop_assume!(v != 0.0 && w != 0.0);

        let a = CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![v, -v]).unwrap();
        let b = CsrMatrix::new(2, 1, vec![0, 1, 2], vec![0, 0], vec![w, w]).unwrap();

        let c = spmm_csr(&a, &b).expect("a.cols() == b.rows() by construction");
        prop_assert_eq!(c.nnz(), 0);
    }
}

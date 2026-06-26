use alloc::vec;
use alloc::vec::Vec;

use crate::scalar::Scalar;

use super::CsrMatrix;

/// Removes stored entries from `m` whose absolute value does not exceed `tolerance`,
/// returning a new `CsrMatrix<T>` with a reduced sparsity pattern and unchanged shape.
///
/// An entry `v` is **kept** when `v > tolerance || v < -tolerance` (strictly outside
/// `[-tolerance, tolerance]`). Setting `tolerance` to `T::zero()` removes only entries
/// that are exactly `0`.
///
/// This is useful after repeated sparse additions or Krylov-solver updates, where
/// accumulated fill-in or exact cancellations leave stored zeros that inflate `nnz`
/// without contributing to the matrix value.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, prune_csr};
///
/// // Row 0: three entries — two negligible, one significant.
/// let m = CsrMatrix::new(
///     1, 3,
///     vec![0, 3],
///     vec![0, 1, 2],
///     vec![1e-15_f64, 2.0, -1e-15],
/// )
/// .unwrap();
///
/// let pruned = prune_csr(m, 1e-10);
/// assert_eq!(pruned.nnz(), 1);
/// assert_eq!(pruned.col_indices(), &[1]);
/// assert_eq!(pruned.values(),      &[2.0]);
/// ```
pub fn prune_csr<T: Scalar + PartialOrd>(m: CsrMatrix<T>, tolerance: T) -> CsrMatrix<T> {
    let (rows, cols, old_row_ptr, old_col, old_val) = m.into_raw_parts();
    let neg_tol = T::zero().sub(tolerance);

    let mut new_row_ptr = vec![0usize; rows + 1];
    let mut new_col: Vec<usize> = Vec::new();
    let mut new_val: Vec<T> = Vec::new();

    for r in 0..rows {
        for k in old_row_ptr[r]..old_row_ptr[r + 1] {
            let v = old_val[k];
            if v > tolerance || v < neg_tol {
                new_col.push(old_col[k]);
                new_val.push(v);
            }
        }
        new_row_ptr[r + 1] = new_col.len();
    }

    CsrMatrix::new_raw(rows, cols, new_row_ptr, new_col, new_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CsrMatrix;

    #[test]
    fn keeps_entries_outside_tolerance_band() {
        let m = CsrMatrix::new(
            1,
            3,
            vec![0, 3],
            vec![0, 1, 2],
            vec![1e-15_f64, 2.0, -1e-15],
        )
        .unwrap();
        let p = prune_csr(m, 1e-10);
        assert_eq!(p.nnz(), 1);
        assert_eq!(p.col_indices(), &[1]);
        assert_eq!(p.values(), &[2.0]);
    }

    #[test]
    fn zero_tolerance_removes_exact_zeros() {
        let m = CsrMatrix::new(1, 3, vec![0, 3], vec![0, 1, 2], vec![0.0_f64, 5.0, 0.0]).unwrap();
        let p = prune_csr(m, 0.0);
        assert_eq!(p.nnz(), 1);
        assert_eq!(p.col_indices(), &[1]);
    }

    #[test]
    fn keeps_negative_values_outside_band() {
        let m = CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![-3.0_f64, 0.5]).unwrap();
        let p = prune_csr(m, 1.0);
        assert_eq!(p.nnz(), 1);
        assert_eq!(p.col_indices(), &[0]);
        assert_eq!(p.values(), &[-3.0]);
    }

    #[test]
    fn preserves_row_structure_with_multi_row_matrix() {
        // Row 0 keeps one entry; row 1 is fully pruned; row 2 keeps both.
        let m = CsrMatrix::new(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![0, 1, 0, 1, 0, 1],
            vec![1e-15_f64, 5.0, 1e-14, 1e-13, 7.0, 8.0],
        )
        .unwrap();
        let p = prune_csr(m, 1e-10);
        assert_eq!(p.row_ptr(), &[0, 1, 1, 3]);
        assert_eq!(p.col_indices(), &[1, 0, 1]);
        assert_eq!(p.values(), &[5.0, 7.0, 8.0]);
    }

    #[test]
    fn all_entries_below_tolerance_gives_empty_matrix() {
        let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1e-20_f64, 1e-20]).unwrap();
        let p = prune_csr(m, 1e-10);
        assert_eq!(p.nnz(), 0);
        assert_eq!(p.row_ptr(), &[0, 0, 0]);
    }

    #[test]
    fn empty_matrix_unchanged() {
        let m = CsrMatrix::<f64>::new(3, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
        let p = prune_csr(m, 1e-10);
        assert_eq!(p.nnz(), 0);
        assert_eq!(p.rows(), 3);
    }
}

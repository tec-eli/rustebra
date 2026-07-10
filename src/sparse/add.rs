use alloc::vec;
use alloc::vec::Vec;

use crate::scalar::Scalar;

use super::{CscMatrix, CsrMatrix, SortedCscMatrix, SortedCsrMatrix};

/// Error returned by sparse arithmetic functions when operands have incompatible shapes,
/// or when the vector length does not match the matrix column count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionMismatch {
    /// Operands have incompatible shapes, or a vector length does not match the expected
    /// matrix dimension.
    Shape,
    /// An arithmetic overflow occurred while computing a size (e.g. total non-zero count
    /// exceeding `u32::MAX`).
    DimensionOverflow,
}

/// Adds two CSR sparse matrices element-wise, merging and deduplicating entries per row.
///
/// For each row, entries from `a` and `b` are merged; entries at the same column position
/// have their values summed. The output always has sorted, deduplicated column indices
/// within each row (same invariant as matrices produced by [`crate::sparse::coo_to_csr`]),
/// which is why the result is a [`SortedCsrMatrix`] rather than a plain `CsrMatrix`.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch::Shape)` when `a` and `b` don't have the same shape.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, add_csr};
///
/// // [2  0]   [1  3]   [3  3]
/// // [0  5] + [0  4] = [0  9]
/// let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 5.0]).unwrap();
/// let b = CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 3.0, 4.0]).unwrap();
/// let c = add_csr(&a, &b).unwrap();
/// assert_eq!(c.row_ptr(),     &[0, 2, 3]);
/// assert_eq!(c.col_indices(), &[0, 1, 1]);
/// assert_eq!(c.values(),      &[3.0, 3.0, 9.0]);
/// ```
pub fn add_csr<T: Scalar>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
) -> Result<SortedCsrMatrix<T>, DimensionMismatch> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(DimensionMismatch::Shape);
    }
    let rows = a.rows();
    let cols = a.cols();

    let mut out_row_ptr = vec![0u32; rows + 1];
    let mut out_col: Vec<u32> = Vec::new();
    let mut out_val: Vec<T> = Vec::new();
    let mut pairs: Vec<(u32, T)> = Vec::new();

    for r in 0..rows {
        pairs.clear();
        let a_range = a.row_ptr()[r] as usize..a.row_ptr()[r + 1] as usize;
        for k in a_range {
            pairs.push((a.col_indices()[k], a.values()[k]));
        }
        let b_range = b.row_ptr()[r] as usize..b.row_ptr()[r + 1] as usize;
        for k in b_range {
            pairs.push((b.col_indices()[k], b.values()[k]));
        }
        pairs.sort_by_key(|&(c, _)| c);
        let mut k = 0;
        while k < pairs.len() {
            let (col, mut sum) = pairs[k];
            let mut j = k + 1;
            while j < pairs.len() && pairs[j].0 == col {
                sum = sum.add(pairs[j].1);
                j += 1;
            }
            if sum != T::zero() {
                out_col.push(col);
                out_val.push(sum);
            }
            k = j;
        }
        out_row_ptr[r + 1] =
            u32::try_from(out_col.len()).map_err(|_| DimensionMismatch::DimensionOverflow)?;
    }

    Ok(SortedCsrMatrix::from_sorted_unchecked(CsrMatrix::new_raw(
        rows,
        cols,
        out_row_ptr,
        out_col,
        out_val,
    )))
}

/// Adds two CSC sparse matrices element-wise, merging and deduplicating entries per column.
///
/// For each column, entries from `a` and `b` are merged; entries at the same row position
/// have their values summed. The output always has sorted, deduplicated row indices within
/// each column, which is why the result is a [`SortedCscMatrix`] rather than a plain
/// `CscMatrix`.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch::Shape)` when `a` and `b` don't have the same shape.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CscMatrix, add_csc};
///
/// // Column 0: a has row 0 → 2.0; b has row 0 → 1.0; sum = 3.0.
/// // Column 1: a has row 1 → 5.0; b has row 1 → 4.0; sum = 9.0.
/// let a = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 5.0]).unwrap();
/// let b = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 4.0]).unwrap();
/// let c = add_csc(&a, &b).unwrap();
/// assert_eq!(c.col_ptr(),     &[0, 1, 2]);
/// assert_eq!(c.row_indices(), &[0, 1]);
/// assert_eq!(c.values(),      &[3.0, 9.0]);
/// ```
pub fn add_csc<T: Scalar>(
    a: &CscMatrix<T>,
    b: &CscMatrix<T>,
) -> Result<SortedCscMatrix<T>, DimensionMismatch> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(DimensionMismatch::Shape);
    }
    let rows = a.rows();
    let cols = a.cols();

    let mut out_col_ptr = vec![0u32; cols + 1];
    let mut out_row: Vec<u32> = Vec::new();
    let mut out_val: Vec<T> = Vec::new();
    let mut pairs: Vec<(u32, T)> = Vec::new();

    for c in 0..cols {
        pairs.clear();
        let a_range = a.col_ptr()[c] as usize..a.col_ptr()[c + 1] as usize;
        for k in a_range {
            pairs.push((a.row_indices()[k], a.values()[k]));
        }
        let b_range = b.col_ptr()[c] as usize..b.col_ptr()[c + 1] as usize;
        for k in b_range {
            pairs.push((b.row_indices()[k], b.values()[k]));
        }
        pairs.sort_by_key(|&(r, _)| r);
        let mut k = 0;
        while k < pairs.len() {
            let (row, mut sum) = pairs[k];
            let mut j = k + 1;
            while j < pairs.len() && pairs[j].0 == row {
                sum = sum.add(pairs[j].1);
                j += 1;
            }
            if sum != T::zero() {
                out_row.push(row);
                out_val.push(sum);
            }
            k = j;
        }
        out_col_ptr[c + 1] =
            u32::try_from(out_row.len()).map_err(|_| DimensionMismatch::DimensionOverflow)?;
    }

    Ok(SortedCscMatrix::from_sorted_unchecked(CscMatrix::new_raw(
        rows,
        cols,
        out_col_ptr,
        out_row,
        out_val,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s1_add_csr_cancellation_zeros() {
        let a = CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![3.0_f64]).unwrap();
        let b = CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![-3.0_f64]).unwrap();
        let c = add_csr(&a, &b).unwrap();
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn s1_add_csc_cancellation_zeros() {
        let a = CscMatrix::new(2, 1, vec![0, 1], vec![0], vec![5.0_f64]).unwrap();
        let b = CscMatrix::new(2, 1, vec![0, 1], vec![0], vec![-5.0_f64]).unwrap();
        let c = add_csc(&a, &b).unwrap();
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn add_csr_result_has_sorted_col_indices_per_row() {
        // Row 0: a contributes col 2, b contributes cols 0 and 1 -> merged/sorted [0, 1, 2].
        let a = CsrMatrix::new(1, 3, vec![0, 1], vec![2], vec![9.0_f64]).unwrap();
        let b = CsrMatrix::new(1, 3, vec![0, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
        let c = add_csr(&a, &b).unwrap();
        assert_eq!(c.col_indices(), &[0, 1, 2]);
        for w in c.col_indices().windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn add_csc_result_has_sorted_row_indices_per_column() {
        // Column 0: a contributes row 2, b contributes rows 0 and 1 -> merged/sorted [0, 1, 2].
        let a = CscMatrix::new(3, 1, vec![0, 1], vec![2], vec![9.0_f64]).unwrap();
        let b = CscMatrix::new(3, 1, vec![0, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
        let c = add_csc(&a, &b).unwrap();
        assert_eq!(c.row_indices(), &[0, 1, 2]);
        for w in c.row_indices().windows(2) {
            assert!(w[0] < w[1]);
        }
    }
}

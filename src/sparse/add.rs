use alloc::vec;
use alloc::vec::Vec;

use crate::scalar::Scalar;

use super::{CscMatrix, CsrMatrix};

/// Error returned by sparse arithmetic functions when operands have incompatible shapes,
/// or when the vector length does not match the matrix column count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionMismatch;

/// Adds two CSR sparse matrices element-wise, merging and deduplicating entries per row.
///
/// For each row, entries from `a` and `b` are merged; entries at the same column position
/// have their values summed. The output has sorted, deduplicated column indices within each
/// row (same invariant as matrices produced by [`crate::sparse::coo_to_csr`]).
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` when `a` and `b` don't have the same shape.
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
) -> Result<CsrMatrix<T>, DimensionMismatch> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(DimensionMismatch);
    }
    let rows = a.rows();
    let cols = a.cols();

    let mut out_row_ptr = vec![0usize; rows + 1];
    let mut out_col: Vec<usize> = Vec::new();
    let mut out_val: Vec<T> = Vec::new();

    for r in 0..rows {
        let mut pairs: Vec<(usize, T)> = Vec::new();
        let a_range = a.row_ptr()[r]..a.row_ptr()[r + 1];
        for k in a_range {
            pairs.push((a.col_indices()[k], a.values()[k]));
        }
        let b_range = b.row_ptr()[r]..b.row_ptr()[r + 1];
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
            out_col.push(col);
            out_val.push(sum);
            k = j;
        }
        out_row_ptr[r + 1] = out_col.len();
    }

    Ok(CsrMatrix::new_raw(
        rows,
        cols,
        out_row_ptr,
        out_col,
        out_val,
    ))
}

/// Adds two CSC sparse matrices element-wise, merging and deduplicating entries per column.
///
/// For each column, entries from `a` and `b` are merged; entries at the same row position
/// have their values summed. The output has sorted, deduplicated row indices within each
/// column.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` when `a` and `b` don't have the same shape.
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
) -> Result<CscMatrix<T>, DimensionMismatch> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(DimensionMismatch);
    }
    let rows = a.rows();
    let cols = a.cols();

    let mut out_col_ptr = vec![0usize; cols + 1];
    let mut out_row: Vec<usize> = Vec::new();
    let mut out_val: Vec<T> = Vec::new();

    for c in 0..cols {
        let mut pairs: Vec<(usize, T)> = Vec::new();
        let a_range = a.col_ptr()[c]..a.col_ptr()[c + 1];
        for k in a_range {
            pairs.push((a.row_indices()[k], a.values()[k]));
        }
        let b_range = b.col_ptr()[c]..b.col_ptr()[c + 1];
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
            out_row.push(row);
            out_val.push(sum);
            k = j;
        }
        out_col_ptr[c + 1] = out_row.len();
    }

    Ok(CscMatrix::new_raw(
        rows,
        cols,
        out_col_ptr,
        out_row,
        out_val,
    ))
}

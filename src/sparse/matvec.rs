use alloc::vec;
use alloc::vec::Vec;

use crate::scalar::Scalar;

use super::add::DimensionMismatch;
use super::{CscMatrix, CsrMatrix};

/// Multiplies a CSR sparse matrix by a dense column vector, returning the result as a
/// `Vec<T>` of length `m.rows()`.
///
/// The CSR format allows efficient row-by-row traversal: for each row `r`, the stored
/// entries span `row_ptr[r]..row_ptr[r + 1]`, and each contributes `value * x[col]` to
/// `out[r]`.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` when `x.len() != m.cols()`.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, matvec_csr};
///
/// // [ 2  0 ]   [ 1 ]   [  2 ]
/// // [ 0  5 ] × [ 3 ] = [ 15 ]
/// let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 5.0]).unwrap();
/// let y = matvec_csr(&m, &[1.0, 3.0]).unwrap();
/// assert_eq!(y, vec![2.0, 15.0]);
/// ```
pub fn matvec_csr<T: Scalar>(m: &CsrMatrix<T>, x: &[T]) -> Result<Vec<T>, DimensionMismatch> {
    if x.len() != m.cols() {
        return Err(DimensionMismatch);
    }
    let mut out = vec![T::zero(); m.rows()];
    let row_ptr = m.row_ptr();
    let col_idx = m.col_indices();
    let vals = m.values();
    for r in 0..m.rows() {
        for k in row_ptr[r]..row_ptr[r + 1] {
            let prev = out[r];
            out[r] = prev.add(vals[k].mul(x[col_idx[k]]));
        }
    }
    Ok(out)
}

/// Multiplies a CSC sparse matrix by a dense column vector, returning the result as a
/// `Vec<T>` of length `m.rows()`.
///
/// CSC's column-oriented layout naturally accumulates output rows: for each column `c`,
/// every stored entry `(r, v)` contributes `v * x[c]` to `out[r]`.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` when `x.len() != m.cols()`.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CscMatrix, matvec_csc};
///
/// // [ 1  0 ]   [ 2 ]   [  2 ]
/// // [ 0  3 ] × [ 4 ] = [ 12 ]
/// let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 3.0]).unwrap();
/// let y = matvec_csc(&m, &[2.0, 4.0]).unwrap();
/// assert_eq!(y, vec![2.0, 12.0]);
/// ```
pub fn matvec_csc<T: Scalar>(m: &CscMatrix<T>, x: &[T]) -> Result<Vec<T>, DimensionMismatch> {
    if x.len() != m.cols() {
        return Err(DimensionMismatch);
    }
    let mut out = vec![T::zero(); m.rows()];
    let col_ptr = m.col_ptr();
    let row_idx = m.row_indices();
    let vals = m.values();
    for c in 0..m.cols() {
        for k in col_ptr[c]..col_ptr[c + 1] {
            let r = row_idx[k];
            let prev = out[r];
            out[r] = prev.add(vals[k].mul(x[c]));
        }
    }
    Ok(out)
}

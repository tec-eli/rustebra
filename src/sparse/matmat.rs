use alloc::vec;
use alloc::vec::Vec;

use crate::scalar::Scalar;

use super::add::DimensionMismatch;
use super::{CscMatrix, CsrMatrix};

/// Multiplies a CSR sparse matrix by a dense matrix, returning the result as a flat
/// row-major `Vec<T>` of shape `(m.rows(), x_cols)`.
///
/// `x` is a row-major dense matrix with `m.cols()` rows and `x_cols` columns, so
/// `x.len()` must equal `m.cols() * x_cols`.  The output is also row-major: element
/// `(i, j)` is at index `i * x_cols + j`.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` when `x_cols == 0` or `x.len() != m.cols() * x_cols`.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, matmat_csr};
///
/// // [2  0]   [1  0]   [2  0]
/// // [0  3] × [0  4] = [0 12]
/// let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0]).unwrap();
/// let x = [1.0_f64, 0.0, 0.0, 4.0]; // row-major 2×2
/// let y = matmat_csr(&m, &x, 2).unwrap();
/// assert_eq!(y, vec![2.0, 0.0, 0.0, 12.0]);
/// ```
pub fn matmat_csr<T: Scalar>(
    m: &CsrMatrix<T>,
    x: &[T],
    x_cols: usize,
) -> Result<Vec<T>, DimensionMismatch> {
    let expected = m.cols().checked_mul(x_cols).ok_or(DimensionMismatch)?;
    if x_cols == 0 || x.len() != expected {
        return Err(DimensionMismatch);
    }
    let out_len = m.rows().checked_mul(x_cols).ok_or(DimensionMismatch)?;
    let mut out = vec![T::zero(); out_len];
    let row_ptr = m.row_ptr();
    let col_idx = m.col_indices();
    let vals = m.values();
    for r in 0..m.rows() {
        for k in row_ptr[r]..row_ptr[r + 1] {
            let c = col_idx[k];
            let v = vals[k];
            for j in 0..x_cols {
                let prev = out[r * x_cols + j];
                out[r * x_cols + j] = prev.add(v.mul(x[c * x_cols + j]));
            }
        }
    }
    Ok(out)
}

/// Multiplies a CSC sparse matrix by a dense matrix, returning the result as a flat
/// row-major `Vec<T>` of shape `(m.rows(), x_cols)`.
///
/// `x` is a row-major dense matrix with `m.cols()` rows and `x_cols` columns.
/// For CSC the natural traversal is column-by-column: column `c` of the sparse matrix
/// contributes `v * x[c, j]` to row `r` of the output for every `(r, v)` stored in
/// that column.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` when `x_cols == 0` or `x.len() != m.cols() * x_cols`.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CscMatrix, matmat_csc};
///
/// // [2  0]   [1  0]   [2  0]
/// // [0  3] × [0  4] = [0 12]
/// let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2.0_f64, 3.0]).unwrap();
/// let x = [1.0_f64, 0.0, 0.0, 4.0]; // row-major 2×2
/// let y = matmat_csc(&m, &x, 2).unwrap();
/// assert_eq!(y, vec![2.0, 0.0, 0.0, 12.0]);
/// ```
pub fn matmat_csc<T: Scalar>(
    m: &CscMatrix<T>,
    x: &[T],
    x_cols: usize,
) -> Result<Vec<T>, DimensionMismatch> {
    let expected = m.cols().checked_mul(x_cols).ok_or(DimensionMismatch)?;
    if x_cols == 0 || x.len() != expected {
        return Err(DimensionMismatch);
    }
    let out_len = m.rows().checked_mul(x_cols).ok_or(DimensionMismatch)?;
    let mut out = vec![T::zero(); out_len];
    let col_ptr = m.col_ptr();
    let row_idx = m.row_indices();
    let vals = m.values();
    for c in 0..m.cols() {
        for k in col_ptr[c]..col_ptr[c + 1] {
            let r = row_idx[k];
            let v = vals[k];
            for j in 0..x_cols {
                let prev = out[r * x_cols + j];
                out[r * x_cols + j] = prev.add(v.mul(x[c * x_cols + j]));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::{matmat_csc, matmat_csr, CscMatrix, CsrMatrix};

    /// On 32-bit targets `65536 * 65536 == 0 (mod 2^32)`.  Without `checked_mul` a
    /// zero-length slice would bypass the length check and be silently accepted.
    #[cfg(target_pointer_width = "32")]
    #[test]
    fn matmat_csr_rejects_overflowing_dimensions() {
        let m = CsrMatrix::<f64>::new(1, 65536, vec![0, 0], vec![], vec![]).unwrap();
        let x: Vec<f64> = vec![];
        assert!(matmat_csr(&m, &x, 65536).is_err());
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn matmat_csc_rejects_overflowing_dimensions() {
        let m = CscMatrix::<f64>::new(65536, 65536, vec![0usize; 65537], vec![], vec![]).unwrap();
        let x: Vec<f64> = vec![];
        assert!(matmat_csc(&m, &x, 65536).is_err());
    }
}

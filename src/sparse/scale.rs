use alloc::vec;
use alloc::vec::Vec;

use crate::scalar::Scalar;

use super::{CscMatrix, CsrMatrix};

/// Multiplies every stored value of a CSR sparse matrix by `scalar`, returning a new matrix
/// with the same sparsity pattern and shape.
///
/// The `row_ptr` and `col_indices` arrays are preserved unchanged; only `values` changes.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, scale_csr};
///
/// let m = CsrMatrix::new(
///     2, 2,
///     vec![0, 1, 2],
///     vec![0, 1],
///     vec![4.0_f64, 6.0],
/// )
/// .unwrap();
///
/// let scaled = scale_csr(m, 0.5);
/// assert_eq!(scaled.values(),      &[2.0, 3.0]);
/// assert_eq!(scaled.row_ptr(),     &[0, 1, 2]);
/// assert_eq!(scaled.col_indices(), &[0, 1]);
/// ```
pub fn scale_csr<T: Scalar>(m: CsrMatrix<T>, scalar: T) -> CsrMatrix<T> {
    if scalar == T::zero() {
        let (rows, cols, _, _, _) = m.into_raw_parts();
        return CsrMatrix::new_raw(rows, cols, vec![0; rows + 1], Vec::new(), Vec::new());
    }
    let (rows, cols, row_ptr, col_indices, values) = m.into_raw_parts();
    let scaled: Vec<T> = values.into_iter().map(|v| v.mul(scalar)).collect();
    CsrMatrix::new_raw(rows, cols, row_ptr, col_indices, scaled)
}

/// Multiplies every stored value of a CSC sparse matrix by `scalar`, returning a new matrix
/// with the same sparsity pattern and shape.
///
/// The `col_ptr` and `row_indices` arrays are preserved unchanged; only `values` changes.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CscMatrix, scale_csc};
///
/// let m = CscMatrix::new(
///     2, 2,
///     vec![0, 1, 2],
///     vec![0, 1],
///     vec![4.0_f64, 6.0],
/// )
/// .unwrap();
///
/// let scaled = scale_csc(m, 0.5);
/// assert_eq!(scaled.values(),      &[2.0, 3.0]);
/// assert_eq!(scaled.col_ptr(),     &[0, 1, 2]);
/// assert_eq!(scaled.row_indices(), &[0, 1]);
/// ```
pub fn scale_csc<T: Scalar>(m: CscMatrix<T>, scalar: T) -> CscMatrix<T> {
    if scalar == T::zero() {
        let (rows, cols, _, _, _) = m.into_raw_parts();
        return CscMatrix::new_raw(rows, cols, vec![0; cols + 1], Vec::new(), Vec::new());
    }
    let (rows, cols, col_ptr, row_indices, values) = m.into_raw_parts();
    let scaled: Vec<T> = values.into_iter().map(|v| v.mul(scalar)).collect();
    CscMatrix::new_raw(rows, cols, col_ptr, row_indices, scaled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_csr_scales_values_and_preserves_structure() {
        let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.0_f64, 6.0]).unwrap();
        let scaled = scale_csr(m, 0.5);
        assert_eq!(scaled.values(), &[2.0, 3.0]);
        assert_eq!(scaled.row_ptr(), &[0, 1, 2]);
        assert_eq!(scaled.col_indices(), &[0, 1]);
    }

    #[test]
    fn scale_csc_scales_values_and_preserves_structure() {
        let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.0_f64, 6.0]).unwrap();
        let scaled = scale_csc(m, 0.5);
        assert_eq!(scaled.values(), &[2.0, 3.0]);
        assert_eq!(scaled.col_ptr(), &[0, 1, 2]);
        assert_eq!(scaled.row_indices(), &[0, 1]);
    }

    #[test]
    fn scale_csc_by_zero() {
        let m = CscMatrix::new(
            3,
            3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![1.0_f64, 2.0, 3.0],
        )
        .unwrap();
        let scaled = scale_csc(m, 0.0);
        assert_eq!(scaled.nnz(), 0);
        assert_eq!(scaled.values(), &[]);
    }

    #[test]
    fn scale_csc_negative_scalar() {
        let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![4.0_f64, 6.0]).unwrap();
        let scaled = scale_csc(m, -2.0);
        assert_eq!(scaled.values(), &[-8.0, -12.0]);
    }

    #[test]
    fn s3_scale_csr_by_zero() {
        let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
        let scaled = scale_csr(m, 0.0);
        assert_eq!(scaled.nnz(), 0);
    }

    #[test]
    fn s3_scale_csc_by_zero() {
        let m = CscMatrix::new(
            3,
            3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![1.0_f64, 2.0, 3.0],
        )
        .unwrap();
        let scaled = scale_csc(m, 0.0);
        assert_eq!(scaled.nnz(), 0);
    }
}

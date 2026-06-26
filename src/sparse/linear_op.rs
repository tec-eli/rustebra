use alloc::vec;
use alloc::vec::Vec;

use crate::scalar::Scalar;

use super::add::DimensionMismatch;
use super::{CscMatrix, CsrMatrix};

/// An abstract sparse linear operator that can be applied to a dense vector.
///
/// Implementing this trait for a storage format lets iterative Krylov solvers (CG, GMRES,
/// Lanczos, etc.) be written once, generically, rather than once per format.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, SparseLinearOp};
///
/// let eye = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]).unwrap();
/// let y = eye.apply(&[3.0, 5.0]).unwrap();
/// assert_eq!(y, vec![3.0, 5.0]);
/// ```
pub trait SparseLinearOp<T: Scalar> {
    /// Number of rows (length of the output vector from [`apply`](SparseLinearOp::apply)).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::{CsrMatrix, SparseLinearOp};
    ///
    /// let m = CsrMatrix::<f64>::new(3, 5, vec![0; 4], vec![], vec![]).unwrap();
    /// assert_eq!(m.rows(), 3);
    /// ```
    fn rows(&self) -> usize;

    /// Number of columns (required length of the input vector to [`apply`](SparseLinearOp::apply)).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::{CsrMatrix, SparseLinearOp};
    ///
    /// let m = CsrMatrix::<f64>::new(3, 5, vec![0; 4], vec![], vec![]).unwrap();
    /// assert_eq!(m.cols(), 5);
    /// ```
    fn cols(&self) -> usize;

    /// Multiplies the operator by the dense column vector `x`, returning a new `Vec<T>`
    /// of length `self.rows()`.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` when `x.len() != self.cols()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::{CscMatrix, SparseLinearOp};
    ///
    /// // [ 1  0 ]   [ 2 ]   [ 2 ]
    /// // [ 0  3 ] × [ 4 ] = [ 12 ]
    /// let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 3.0]).unwrap();
    /// let y = m.apply(&[2.0, 4.0]).unwrap();
    /// assert_eq!(y, vec![2.0, 12.0]);
    /// ```
    fn apply(&self, x: &[T]) -> Result<Vec<T>, DimensionMismatch>;
}

impl<T: Scalar> SparseLinearOp<T> for CsrMatrix<T> {
    fn rows(&self) -> usize {
        CsrMatrix::rows(self)
    }

    fn cols(&self) -> usize {
        CsrMatrix::cols(self)
    }

    fn apply(&self, x: &[T]) -> Result<Vec<T>, DimensionMismatch> {
        if x.len() != CsrMatrix::cols(self) {
            return Err(DimensionMismatch);
        }
        let mut out = vec![T::zero(); CsrMatrix::rows(self)];
        let row_ptr = self.row_ptr();
        let col_idx = self.col_indices();
        let vals = self.values();
        for r in 0..CsrMatrix::rows(self) {
            for k in row_ptr[r]..row_ptr[r + 1] {
                let prev = out[r];
                out[r] = prev.add(vals[k].mul(x[col_idx[k]]));
            }
        }
        Ok(out)
    }
}

impl<T: Scalar> SparseLinearOp<T> for CscMatrix<T> {
    fn rows(&self) -> usize {
        CscMatrix::rows(self)
    }

    fn cols(&self) -> usize {
        CscMatrix::cols(self)
    }

    fn apply(&self, x: &[T]) -> Result<Vec<T>, DimensionMismatch> {
        if x.len() != CscMatrix::cols(self) {
            return Err(DimensionMismatch);
        }
        let mut out = vec![T::zero(); CscMatrix::rows(self)];
        let col_ptr = self.col_ptr();
        let row_idx = self.row_indices();
        let vals = self.values();
        for c in 0..CscMatrix::cols(self) {
            for k in col_ptr[c]..col_ptr[c + 1] {
                let r = row_idx[k];
                let prev = out[r];
                out[r] = prev.add(vals[k].mul(x[c]));
            }
        }
        Ok(out)
    }
}

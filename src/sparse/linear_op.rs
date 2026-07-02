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

    /// Multiplies the operator by the dense column vector `x`, writing the result into the
    /// caller-supplied buffer `out` instead of allocating.
    ///
    /// `out` is overwritten entirely; its previous contents are ignored.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` when `x.len() != self.cols()` or
    /// `out.len() != self.rows()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::{CscMatrix, SparseLinearOp};
    ///
    /// // [ 1  0 ]   [ 2 ]   [ 2 ]
    /// // [ 0  3 ] × [ 4 ] = [ 12 ]
    /// let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 3.0]).unwrap();
    /// let mut y = [0.0; 2];
    /// m.apply_into(&[2.0, 4.0], &mut y).unwrap();
    /// assert_eq!(y, [2.0, 12.0]);
    /// ```
    fn apply_into(&self, x: &[T], out: &mut [T]) -> Result<(), DimensionMismatch>;
}

impl<T: Scalar> SparseLinearOp<T> for CsrMatrix<T> {
    fn rows(&self) -> usize {
        CsrMatrix::rows(self)
    }

    fn cols(&self) -> usize {
        CsrMatrix::cols(self)
    }

    fn apply(&self, x: &[T]) -> Result<Vec<T>, DimensionMismatch> {
        let mut out = vec![T::zero(); CsrMatrix::rows(self)];
        self.apply_into(x, &mut out)?;
        Ok(out)
    }

    fn apply_into(&self, x: &[T], out: &mut [T]) -> Result<(), DimensionMismatch> {
        if x.len() != CsrMatrix::cols(self) || out.len() != CsrMatrix::rows(self) {
            return Err(DimensionMismatch);
        }
        out.fill(T::zero());
        let row_ptr = self.row_ptr();
        let col_idx = self.col_indices();
        let vals = self.values();
        for r in 0..CsrMatrix::rows(self) {
            for k in row_ptr[r]..row_ptr[r + 1] {
                let prev = out[r];
                out[r] = prev.add(vals[k].mul(x[col_idx[k]]));
            }
        }
        Ok(())
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
        let mut out = vec![T::zero(); CscMatrix::rows(self)];
        self.apply_into(x, &mut out)?;
        Ok(out)
    }

    fn apply_into(&self, x: &[T], out: &mut [T]) -> Result<(), DimensionMismatch> {
        if x.len() != CscMatrix::cols(self) || out.len() != CscMatrix::rows(self) {
            return Err(DimensionMismatch);
        }
        out.fill(T::zero());
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_into_matches_apply_for_csr() {
        // [ 1  2  0 ]   [ 1 ]   [ 5 ]
        // [ 0  0  3 ] × [ 2 ] = [ 9 ]
        let m = CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 1, 2], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let x = [1.0, 2.0, 3.0];
        let expected = m.apply(&x).unwrap();
        let mut out = vec![7.0; 2];
        m.apply_into(&x, &mut out).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn apply_into_matches_apply_for_csc() {
        // [ 1  0 ]   [ 2 ]   [ 2 ]
        // [ 4  3 ] × [ 4 ] = [ 20 ]
        let m = CscMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 4.0, 3.0]).unwrap();
        let x = [2.0, 4.0];
        let expected = m.apply(&x).unwrap();
        let mut out = vec![7.0; 2];
        m.apply_into(&x, &mut out).unwrap();
        assert_eq!(out, expected);
    }
}

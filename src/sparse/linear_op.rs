use crate::scalar::Scalar;

use super::add::DimensionMismatch;
use super::{CscMatrix, CsrMatrix};

/// An abstract sparse linear operator that can be applied to a dense vector.
///
/// Implementing this trait for a storage format lets iterative Krylov solvers (CG, GMRES,
/// Lanczos, etc.) be written once, generically, rather than once per format. The output is
/// written into a caller-supplied buffer, so applying the operator never allocates — solver
/// loops can reuse the same workspace across iterations.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, SparseLinearOp};
///
/// let eye = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]).unwrap();
/// let mut y = [0.0; 2];
/// eye.apply(&[3.0, 5.0], &mut y).unwrap();
/// assert_eq!(y, [3.0, 5.0]);
/// ```
pub trait SparseLinearOp<T: Scalar> {
    /// Number of rows (required length of the output buffer for [`apply`](SparseLinearOp::apply)).
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

    /// Multiplies the operator by the dense column vector `x`, writing the result into the
    /// caller-supplied buffer `out`.
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
    /// m.apply(&[2.0, 4.0], &mut y).unwrap();
    /// assert_eq!(y, [2.0, 12.0]);
    /// ```
    fn apply(&self, x: &[T], out: &mut [T]) -> Result<(), DimensionMismatch>;
}

impl<T: Scalar> SparseLinearOp<T> for CsrMatrix<T> {
    fn rows(&self) -> usize {
        CsrMatrix::rows(self)
    }

    fn cols(&self) -> usize {
        CsrMatrix::cols(self)
    }

    fn apply(&self, x: &[T], out: &mut [T]) -> Result<(), DimensionMismatch> {
        if x.len() != CsrMatrix::cols(self) || out.len() != CsrMatrix::rows(self) {
            return Err(DimensionMismatch);
        }
        out.fill(T::zero());
        let row_ptr = self.row_ptr();
        let col_idx = self.col_indices();
        let vals = self.values();
        for r in 0..CsrMatrix::rows(self) {
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                let prev = out[r];
                out[r] = prev.add(vals[k].mul(x[col_idx[k] as usize]));
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

    fn apply(&self, x: &[T], out: &mut [T]) -> Result<(), DimensionMismatch> {
        if x.len() != CscMatrix::cols(self) || out.len() != CscMatrix::rows(self) {
            return Err(DimensionMismatch);
        }
        out.fill(T::zero());
        let col_ptr = self.col_ptr();
        let row_idx = self.row_indices();
        let vals = self.values();
        for c in 0..CscMatrix::cols(self) {
            for k in col_ptr[c] as usize..col_ptr[c + 1] as usize {
                let r = row_idx[k] as usize;
                let prev = out[r];
                out[r] = prev.add(vals[k].mul(x[c]));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn apply_overwrites_out_for_csr() {
        // [ 1  2  0 ]   [ 1 ]   [ 5 ]
        // [ 0  0  3 ] × [ 2 ] = [ 9 ]
        let m =
            CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 1, 2], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let x = [1.0, 2.0, 3.0];
        let mut out = [7.0; 2];
        m.apply(&x, &mut out).unwrap();
        assert_eq!(out, [5.0, 9.0]);
    }

    #[test]
    fn apply_overwrites_out_for_csc() {
        // [ 1  0 ]   [ 2 ]   [ 2 ]
        // [ 4  3 ] × [ 4 ] = [ 20 ]
        let m =
            CscMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 4.0, 3.0]).unwrap();
        let x = [2.0, 4.0];
        let mut out = [7.0; 2];
        m.apply(&x, &mut out).unwrap();
        assert_eq!(out, [2.0, 20.0]);
    }

    #[test]
    fn apply_rejects_wrong_input_length_for_csr() {
        let m =
            CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 1, 2], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let mut out = [0.0; 2];
        assert_eq!(m.apply(&[1.0, 2.0], &mut out), Err(DimensionMismatch));
    }

    #[test]
    fn apply_rejects_wrong_output_length_for_csc() {
        let m =
            CscMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 4.0, 3.0]).unwrap();
        let mut out = [0.0; 3];
        assert_eq!(m.apply(&[2.0, 4.0], &mut out), Err(DimensionMismatch));
    }
}

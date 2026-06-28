use alloc::vec::Vec;
use core::fmt;
use core::ops::Deref;

use crate::scalar::Scalar;

use super::CsrMatrix;
use super::add::DimensionMismatch;
use super::linear_op::SparseLinearOp;

/// A CSR sparse matrix that guarantees column indices within each row are in ascending order.
///
/// This invariant enables O(log(nnz/rows)) binary-search access to a specific `(row, col)`
/// entry and satisfies the sorted-index precondition required by sparse triangular solvers
/// and certain preconditioned iterative methods.
///
/// `SortedCsrMatrix<T>` implements [`Deref<Target = CsrMatrix<T>>`], so every read-only
/// accessor (`rows`, `cols`, `nnz`, `row_ptr`, `col_indices`, `values`, `row_range`) is
/// available directly without naming the inner type.
///
/// Construct via [`SortedCsrMatrix::from_csr`], or obtain one from [`coo_to_csr`] or
/// [`spmm_csr`], both of which produce sorted output as a side-effect.
///
/// [`coo_to_csr`]: crate::sparse::coo_to_csr
/// [`spmm_csr`]: crate::sparse::spmm_csr
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, SortedCsrMatrix};
///
/// // Row 0 has columns [2, 0] — out of order.
/// let m = CsrMatrix::new(
///     2, 3,
///     vec![0, 2, 3],
///     vec![2, 0, 1],
///     vec![9.0_f64, 4.0, 5.0],
/// )
/// .unwrap();
///
/// let sorted = SortedCsrMatrix::from_csr(m);
/// // Row 0 is now sorted: columns [0, 2], values [4.0, 9.0].
/// assert_eq!(sorted.col_indices(), &[0, 2, 1]);
/// assert_eq!(sorted.values(),      &[4.0, 9.0, 5.0]);
/// ```
pub struct SortedCsrMatrix<T>(CsrMatrix<T>);

impl<T: Scalar> SortedCsrMatrix<T> {
    /// Sorts column indices (and corresponding values) within each row of `m`, returning
    /// a `SortedCsrMatrix<T>` with the ascending-index invariant established.
    ///
    /// If a row already has sorted or singleton entries, no allocation is performed for
    /// that row. The sort is O(nnz log(nnz/rows)) overall.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::{CsrMatrix, SortedCsrMatrix};
    ///
    /// let m = CsrMatrix::new(
    ///     1, 3,
    ///     vec![0, 3],
    ///     vec![2, 0, 1],
    ///     vec![30.0_f64, 10.0, 20.0],
    /// )
    /// .unwrap();
    ///
    /// let s = SortedCsrMatrix::from_csr(m);
    /// assert_eq!(s.col_indices(), &[0, 1, 2]);
    /// assert_eq!(s.values(),      &[10.0, 20.0, 30.0]);
    /// ```
    pub fn from_csr(m: CsrMatrix<T>) -> Self {
        let (rows, cols, row_ptr, mut col_indices, mut values) = m.into_raw_parts();
        for r in 0..rows {
            let start = row_ptr[r];
            let end = row_ptr[r + 1];
            if end <= start + 1 {
                continue;
            }
            let mut pairs: Vec<(usize, T)> =
                (start..end).map(|k| (col_indices[k], values[k])).collect();
            pairs.sort_unstable_by_key(|&(c, _)| c);
            for (i, (c, v)) in pairs.into_iter().enumerate() {
                col_indices[start + i] = c;
                values[start + i] = v;
            }
        }
        Self(CsrMatrix::new_raw(rows, cols, row_ptr, col_indices, values))
    }

    /// Wraps `m` without sorting, trusting that column indices within every row are already
    /// in ascending order.
    ///
    /// Used internally by [`coo_to_csr`] and [`spmm_csr`], which produce sorted output as
    /// a side-effect of their algorithms.
    ///
    /// [`coo_to_csr`]: crate::sparse::coo_to_csr
    /// [`spmm_csr`]: crate::sparse::spmm_csr
    pub(super) fn from_sorted_unchecked(m: CsrMatrix<T>) -> Self {
        Self(m)
    }

    /// Unwraps the inner `CsrMatrix<T>`, consuming `self`.
    ///
    /// Useful when passing a `SortedCsrMatrix<T>` to a function that takes `CsrMatrix<T>`
    /// by value (e.g., [`csr_to_coo`] or [`csr_to_csc`]).
    ///
    /// [`csr_to_coo`]: crate::sparse::csr_to_coo
    /// [`csr_to_csc`]: crate::sparse::csr_to_csc
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::{CooMatrix, CsrMatrix, SortedCsrMatrix, coo_to_csr, csr_to_coo};
    ///
    /// let coo = CooMatrix::new(2, 2, vec![0, 1], vec![1, 0], vec![3.0_f64, 7.0]).unwrap();
    /// let sorted: SortedCsrMatrix<f64> = coo_to_csr(coo);
    /// let coo2 = csr_to_coo(sorted.into_inner());
    /// assert_eq!(coo2.nnz(), 2);
    /// ```
    pub fn into_inner(self) -> CsrMatrix<T> {
        self.0
    }
}

impl<T> Deref for SortedCsrMatrix<T> {
    type Target = CsrMatrix<T>;

    fn deref(&self) -> &CsrMatrix<T> {
        &self.0
    }
}

impl<T: PartialEq> PartialEq for SortedCsrMatrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: PartialEq> PartialEq<CsrMatrix<T>> for SortedCsrMatrix<T> {
    fn eq(&self, other: &CsrMatrix<T>) -> bool {
        &self.0 == other
    }
}

impl<T: PartialEq> PartialEq<SortedCsrMatrix<T>> for CsrMatrix<T> {
    fn eq(&self, other: &SortedCsrMatrix<T>) -> bool {
        self == &other.0
    }
}

impl<T: fmt::Debug> fmt::Debug for SortedCsrMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T> From<SortedCsrMatrix<T>> for CsrMatrix<T> {
    fn from(m: SortedCsrMatrix<T>) -> Self {
        m.0
    }
}

impl<T: Scalar> SparseLinearOp<T> for SortedCsrMatrix<T> {
    fn rows(&self) -> usize {
        self.0.rows()
    }

    fn cols(&self) -> usize {
        self.0.cols()
    }

    fn apply(&self, x: &[T]) -> Result<Vec<T>, DimensionMismatch> {
        SparseLinearOp::apply(&self.0, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CsrMatrix;

    #[test]
    fn already_sorted_row_is_unchanged() {
        let m = CsrMatrix::new(1, 3, vec![0, 3], vec![0, 1, 2], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let s = SortedCsrMatrix::from_csr(m);
        assert_eq!(s.col_indices(), &[0, 1, 2]);
        assert_eq!(s.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn unsorted_row_is_sorted() {
        let m =
            CsrMatrix::new(1, 3, vec![0, 3], vec![2, 0, 1], vec![30.0_f64, 10.0, 20.0]).unwrap();
        let s = SortedCsrMatrix::from_csr(m);
        assert_eq!(s.col_indices(), &[0, 1, 2]);
        assert_eq!(s.values(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn multi_row_sort() {
        // Row 0: [col 2, col 0] → sorted [col 0, col 2]
        // Row 1: [col 1] → unchanged
        let m =
            CsrMatrix::new(2, 3, vec![0, 2, 3], vec![2, 0, 1], vec![9.0_f64, 4.0, 5.0]).unwrap();
        let s = SortedCsrMatrix::from_csr(m);
        assert_eq!(s.col_indices(), &[0, 2, 1]);
        assert_eq!(s.values(), &[4.0, 9.0, 5.0]);
    }

    #[test]
    fn empty_matrix_is_valid() {
        let m = CsrMatrix::<f64>::new(3, 4, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
        let s = SortedCsrMatrix::from_csr(m);
        assert_eq!(s.nnz(), 0);
    }

    #[test]
    fn deref_gives_csr_accessors() {
        let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
        let s = SortedCsrMatrix::from_csr(m);
        assert_eq!(s.rows(), 2);
        assert_eq!(s.cols(), 2);
        assert_eq!(s.nnz(), 2);
        assert_eq!(s.row_range(0), Some(0..1));
    }

    #[test]
    fn into_inner_recovers_csr() {
        let m = CsrMatrix::new(1, 2, vec![0, 2], vec![1, 0], vec![3.0_f64, 7.0]).unwrap();
        let s = SortedCsrMatrix::from_csr(m);
        let inner = s.into_inner();
        assert_eq!(inner.col_indices(), &[0, 1]);
        assert_eq!(inner.values(), &[7.0, 3.0]);
    }
}

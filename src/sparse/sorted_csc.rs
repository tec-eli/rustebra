use alloc::vec::Vec;
use core::fmt;
use core::ops::Deref;

use crate::scalar::Scalar;

use super::CscMatrix;
use super::add::DimensionMismatch;
use super::linear_op::SparseLinearOp;

/// A CSC sparse matrix that guarantees row indices within each column are in ascending order.
///
/// This invariant enables O(log(nnz/cols)) binary-search access to a specific `(row, col)`
/// entry and satisfies the sorted-index precondition required by sparse triangular solvers
/// and certain preconditioned iterative methods.
///
/// `SortedCscMatrix<T>` implements [`Deref<Target = CscMatrix<T>>`], so every read-only
/// accessor (`rows`, `cols`, `nnz`, `col_ptr`, `row_indices`, `values`, `col_range`) is
/// available directly without naming the inner type.
///
/// Construct via [`SortedCscMatrix::from_csc`], or obtain one from [`add_csc`], which
/// produces sorted output as a side-effect.
///
/// [`add_csc`]: crate::sparse::add_csc
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CscMatrix, SortedCscMatrix};
///
/// // Column 0 has rows [2, 0] — out of order.
/// let m = CscMatrix::new(
///     3, 2,
///     vec![0, 2, 3],
///     vec![2, 0, 1],
///     vec![9.0_f64, 4.0, 5.0],
/// )
/// .unwrap();
///
/// let sorted = SortedCscMatrix::from_csc(m);
/// // Column 0 is now sorted: rows [0, 2], values [4.0, 9.0].
/// assert_eq!(sorted.row_indices(), &[0, 2, 1]);
/// assert_eq!(sorted.values(),      &[4.0, 9.0, 5.0]);
/// ```
pub struct SortedCscMatrix<T>(CscMatrix<T>);

impl<T: Scalar> SortedCscMatrix<T> {
    /// Sorts row indices (and corresponding values) within each column of `m`, returning
    /// a `SortedCscMatrix<T>` with the ascending-index invariant established.
    ///
    /// If a column already has sorted or singleton entries, no allocation is performed for
    /// that column. The sort is O(nnz log(nnz/cols)) overall.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::{CscMatrix, SortedCscMatrix};
    ///
    /// let m = CscMatrix::new(
    ///     3, 1,
    ///     vec![0, 3],
    ///     vec![2, 0, 1],
    ///     vec![30.0_f64, 10.0, 20.0],
    /// )
    /// .unwrap();
    ///
    /// let s = SortedCscMatrix::from_csc(m);
    /// assert_eq!(s.row_indices(), &[0, 1, 2]);
    /// assert_eq!(s.values(),      &[10.0, 20.0, 30.0]);
    /// ```
    pub fn from_csc(m: CscMatrix<T>) -> Self {
        let (rows, cols, col_ptr, mut row_indices, mut values) = m.into_raw_parts();
        for c in 0..cols {
            let start = col_ptr[c] as usize;
            let end = col_ptr[c + 1] as usize;
            if end <= start + 1 {
                continue;
            }
            let mut pairs: Vec<(u32, T)> =
                (start..end).map(|k| (row_indices[k], values[k])).collect();
            pairs.sort_unstable_by_key(|&(r, _)| r);
            for (i, (r, v)) in pairs.into_iter().enumerate() {
                row_indices[start + i] = r;
                values[start + i] = v;
            }
        }
        Self(CscMatrix::new_raw(rows, cols, col_ptr, row_indices, values))
    }

    /// Wraps `m` without sorting, trusting that row indices within every column are already
    /// in ascending order.
    ///
    /// Used internally by [`add_csc`], which produces sorted output as a side-effect of its
    /// merge algorithm.
    ///
    /// [`add_csc`]: crate::sparse::add_csc
    pub(super) fn from_sorted_unchecked(m: CscMatrix<T>) -> Self {
        Self(m)
    }

    /// Unwraps the inner `CscMatrix<T>`, consuming `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::{CscMatrix, SortedCscMatrix};
    ///
    /// let m = CscMatrix::new(2, 1, vec![0, 2], vec![1, 0], vec![3.0_f64, 7.0]).unwrap();
    /// let s = SortedCscMatrix::from_csc(m);
    /// let inner = s.into_inner();
    /// assert_eq!(inner.row_indices(), &[0, 1]);
    /// ```
    pub fn into_inner(self) -> CscMatrix<T> {
        self.0
    }
}

impl<T> Deref for SortedCscMatrix<T> {
    type Target = CscMatrix<T>;

    fn deref(&self) -> &CscMatrix<T> {
        &self.0
    }
}

impl<T: PartialEq> PartialEq for SortedCscMatrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: PartialEq> PartialEq<CscMatrix<T>> for SortedCscMatrix<T> {
    fn eq(&self, other: &CscMatrix<T>) -> bool {
        &self.0 == other
    }
}

impl<T: PartialEq> PartialEq<SortedCscMatrix<T>> for CscMatrix<T> {
    fn eq(&self, other: &SortedCscMatrix<T>) -> bool {
        self == &other.0
    }
}

impl<T: fmt::Debug> fmt::Debug for SortedCscMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T> From<SortedCscMatrix<T>> for CscMatrix<T> {
    fn from(m: SortedCscMatrix<T>) -> Self {
        m.0
    }
}

impl<T: Scalar> SparseLinearOp<T> for SortedCscMatrix<T> {
    fn rows(&self) -> usize {
        self.0.rows()
    }

    fn cols(&self) -> usize {
        self.0.cols()
    }

    fn apply(&self, x: &[T], out: &mut [T]) -> Result<(), DimensionMismatch> {
        SparseLinearOp::apply(&self.0, x, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    #[test]
    fn already_sorted_column_is_unchanged() {
        let m = CscMatrix::new(3, 1, vec![0, 3], vec![0, 1, 2], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let s = SortedCscMatrix::from_csc(m);
        assert_eq!(s.row_indices(), &[0, 1, 2]);
        assert_eq!(s.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn unsorted_column_is_sorted() {
        let m =
            CscMatrix::new(3, 1, vec![0, 3], vec![2, 0, 1], vec![30.0_f64, 10.0, 20.0]).unwrap();
        let s = SortedCscMatrix::from_csc(m);
        assert_eq!(s.row_indices(), &[0, 1, 2]);
        assert_eq!(s.values(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn multi_column_sort() {
        // Column 0: [row 2, row 0] → sorted [row 0, row 2]
        // Column 1: [row 1] → unchanged
        let m =
            CscMatrix::new(3, 2, vec![0, 2, 3], vec![2, 0, 1], vec![9.0_f64, 4.0, 5.0]).unwrap();
        let s = SortedCscMatrix::from_csc(m);
        assert_eq!(s.row_indices(), &[0, 2, 1]);
        assert_eq!(s.values(), &[4.0, 9.0, 5.0]);
    }

    #[test]
    fn empty_matrix_is_valid() {
        let m = CscMatrix::<f64>::new(4, 3, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
        let s = SortedCscMatrix::from_csc(m);
        assert_eq!(s.nnz(), 0);
    }

    #[test]
    fn deref_gives_csc_accessors() {
        let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
        let s = SortedCscMatrix::from_csc(m);
        assert_eq!(s.rows(), 2);
        assert_eq!(s.cols(), 2);
        assert_eq!(s.nnz(), 2);
        assert_eq!(s.col_range(0), Some(0..1));
    }

    #[test]
    fn into_inner_recovers_csc() {
        let m = CscMatrix::new(2, 1, vec![0, 2], vec![1, 0], vec![3.0_f64, 7.0]).unwrap();
        let s = SortedCscMatrix::from_csc(m);
        let inner = s.into_inner();
        assert_eq!(inner.row_indices(), &[0, 1]);
        assert_eq!(inner.values(), &[7.0, 3.0]);
    }
}

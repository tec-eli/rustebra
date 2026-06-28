use alloc::vec::Vec;
use core::fmt;
use core::ops::Range;

/// Error returned by [`CscMatrix::new`] when the supplied arrays are invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CscError {
    /// `row_indices` and `values` have different lengths.
    LengthMismatch,
    /// `col_ptr` must have exactly `cols + 1` entries.
    ColPtrLengthMismatch,
    /// `col_ptr[0]` must be `0`, every entry must be ≤ the next, and the last entry
    /// must equal `nnz`.
    ColPtrInvalid,
    /// A row index is >= the declared row count.
    RowIndexOutOfBounds,
}

/// A sparse matrix in compressed sparse column (CSC) format. Requires the `alloc` feature.
///
/// CSC stores non-zeros using three parallel arrays:
///
/// * `col_ptr` — length `cols + 1`; `col_ptr[j]..col_ptr[j + 1]` is the slice range
///   in `row_indices` / `values` that belongs to column `j`.
/// * `row_indices` — row index of each stored entry.
/// * `values` — value of each stored entry.
///
/// Within a column the row indices need not be sorted, but every row index must be
/// in-bounds (`< rows`) and `col_ptr` must be non-decreasing.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::CscMatrix;
///
/// // 3×3 identity matrix in CSC format.
/// let eye = CscMatrix::new(
///     3, 3,
///     vec![0, 1, 2, 3],           // col_ptr: one entry per column + sentinel
///     vec![0, 1, 2],              // row_indices
///     vec![1.0_f64, 1.0, 1.0],   // values
/// )
/// .unwrap();
///
/// assert_eq!(eye.rows(), 3);
/// assert_eq!(eye.cols(), 3);
/// assert_eq!(eye.nnz(), 3);
/// assert_eq!(eye.col_range(1), Some(1..2));
/// assert_eq!(eye.row_indices()[1], 1);
/// assert_eq!(eye.values()[1], 1.0);
/// ```
pub struct CscMatrix<T> {
    rows: usize,
    cols: usize,
    col_ptr: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<T>,
}

impl<T> CscMatrix<T> {
    pub(super) fn new_raw(
        rows: usize,
        cols: usize,
        col_ptr: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        Self {
            rows,
            cols,
            col_ptr,
            row_indices,
            values,
        }
    }

    pub(super) fn into_raw_parts(self) -> (usize, usize, Vec<usize>, Vec<usize>, Vec<T>) {
        (
            self.rows,
            self.cols,
            self.col_ptr,
            self.row_indices,
            self.values,
        )
    }

    /// Creates a new `CscMatrix` from its column-pointer, row-index, and value arrays.
    ///
    /// # Errors
    ///
    /// - [`CscError::LengthMismatch`] — `row_indices.len() != values.len()`.
    /// - [`CscError::ColPtrLengthMismatch`] — `col_ptr.len() != cols + 1`.
    /// - [`CscError::ColPtrInvalid`] — `col_ptr[0] != 0`, `col_ptr` is not non-decreasing,
    ///   or `col_ptr[cols] != row_indices.len()`.
    /// - [`CscError::RowIndexOutOfBounds`] — any row index `>= rows`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::{CscMatrix, CscError};
    ///
    /// // col_ptr must have cols + 1 = 4 entries for a 3-column matrix.
    /// let err = CscMatrix::<f64>::new(3, 3, vec![0, 1], vec![], vec![]);
    /// assert_eq!(err, Err(CscError::ColPtrLengthMismatch));
    /// ```
    pub fn new(
        rows: usize,
        cols: usize,
        col_ptr: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, CscError> {
        if row_indices.len() != values.len() {
            return Err(CscError::LengthMismatch);
        }
        if col_ptr.len() != cols + 1 {
            return Err(CscError::ColPtrLengthMismatch);
        }
        let nnz = row_indices.len();
        if col_ptr[0] != 0 || col_ptr[cols] != nnz {
            return Err(CscError::ColPtrInvalid);
        }
        for i in 1..=cols {
            if col_ptr[i] < col_ptr[i - 1] {
                return Err(CscError::ColPtrInvalid);
            }
        }
        for &r in &row_indices {
            if r >= rows {
                return Err(CscError::RowIndexOutOfBounds);
            }
        }
        Ok(Self {
            rows,
            cols,
            col_ptr,
            row_indices,
            values,
        })
    }

    /// Returns the number of rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CscMatrix;
    ///
    /// let m = CscMatrix::<f64>::new(4, 5, vec![0; 6], vec![], vec![]).unwrap();
    /// assert_eq!(m.rows(), 4);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CscMatrix;
    ///
    /// let m = CscMatrix::<f64>::new(4, 5, vec![0; 6], vec![], vec![]).unwrap();
    /// assert_eq!(m.cols(), 5);
    /// ```
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of stored entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CscMatrix;
    ///
    /// let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
    /// assert_eq!(m.nnz(), 2);
    /// ```
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Returns the column-pointer array.  Entry `j` is the start index into `row_indices`
    /// and `values` for column `j`; entry `cols` is `nnz`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CscMatrix;
    ///
    /// let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
    /// assert_eq!(m.col_ptr(), &[0, 1, 2]);
    /// ```
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    /// Returns the row-index array.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CscMatrix;
    ///
    /// let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
    /// assert_eq!(m.row_indices(), &[0, 1]);
    /// ```
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Returns the value array.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CscMatrix;
    ///
    /// let m = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
    /// assert_eq!(m.values(), &[3.0, 7.0]);
    /// ```
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns the index range into `row_indices` and `values` that belongs to `col`.
    /// Returns `None` if `col >= self.cols()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CscMatrix;
    ///
    /// let m = CscMatrix::new(3, 3, vec![0, 0, 1, 3], vec![0, 1, 2], vec![1.0_f64, 2.0, 3.0])
    ///     .unwrap();
    /// assert_eq!(m.col_range(0), Some(0..0)); // col 0 is empty
    /// assert_eq!(m.col_range(1), Some(0..1));
    /// assert_eq!(m.col_range(2), Some(1..3));
    /// assert_eq!(m.col_range(3), None);       // out of bounds
    /// ```
    pub fn col_range(&self, col: usize) -> Option<Range<usize>> {
        if col >= self.cols {
            None
        } else {
            Some(self.col_ptr[col]..self.col_ptr[col + 1])
        }
    }
}

impl<T: PartialEq> PartialEq for CscMatrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self.col_ptr == other.col_ptr
            && self.row_indices == other.row_indices
            && self.values == other.values
    }
}

impl<T: fmt::Debug> fmt::Debug for CscMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CscMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("nnz", &self.values.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{CscError, CscMatrix};

    #[test]
    fn constructs_empty_matrix() {
        let m = CscMatrix::<f64>::new(3, 4, vec![0, 0, 0, 0, 0], vec![], vec![]).unwrap();
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m.nnz(), 0);
        assert_eq!(m.col_ptr(), &[0, 0, 0, 0, 0]);
    }

    #[test]
    fn constructs_identity_matrix() {
        let m = CscMatrix::new(
            3,
            3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![1.0_f64, 1.0, 1.0],
        )
        .unwrap();
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.nnz(), 3);
        assert_eq!(m.row_indices(), &[0, 1, 2]);
        assert_eq!(m.values(), &[1.0, 1.0, 1.0]);
        assert_eq!(m.col_range(0), Some(0..1));
        assert_eq!(m.col_range(1), Some(1..2));
        assert_eq!(m.col_range(2), Some(2..3));
        assert_eq!(m.col_range(3), None);
    }

    #[test]
    fn constructs_matrix_with_empty_columns() {
        // Col 0 is empty, col 1 has one entry, col 2 has two entries.
        let m = CscMatrix::new(
            3,
            3,
            vec![0, 0, 1, 3],
            vec![0, 1, 2],
            vec![1.0_f64, 2.0, 3.0],
        )
        .unwrap();
        assert_eq!(m.col_range(0), Some(0..0));
        assert_eq!(m.col_range(1), Some(0..1));
        assert_eq!(m.col_range(2), Some(1..3));
    }

    #[test]
    fn length_mismatch_is_an_error_not_a_panic() {
        let err = CscMatrix::<f64>::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0]);
        assert_eq!(err, Err(CscError::LengthMismatch));
    }

    #[test]
    fn col_ptr_length_mismatch_is_an_error_not_a_panic() {
        let err = CscMatrix::<f64>::new(3, 3, vec![0, 1], vec![], vec![]);
        assert_eq!(err, Err(CscError::ColPtrLengthMismatch));
    }

    #[test]
    fn col_ptr_not_starting_at_zero_is_an_error_not_a_panic() {
        let err = CscMatrix::<f64>::new(2, 2, vec![1, 1, 1], vec![], vec![]);
        assert_eq!(err, Err(CscError::ColPtrInvalid));
    }

    #[test]
    fn col_ptr_not_monotone_is_an_error_not_a_panic() {
        let err = CscMatrix::<f64>::new(3, 3, vec![0, 2, 1, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        assert_eq!(err, Err(CscError::ColPtrInvalid));
    }

    #[test]
    fn col_ptr_last_entry_mismatch_is_an_error_not_a_panic() {
        let err = CscMatrix::<f64>::new(2, 2, vec![0, 1, 5], vec![0, 1], vec![1.0, 2.0]);
        assert_eq!(err, Err(CscError::ColPtrInvalid));
    }

    #[test]
    fn row_index_out_of_bounds_is_an_error_not_a_panic() {
        let err = CscMatrix::<f64>::new(2, 2, vec![0, 1, 2], vec![0, 3], vec![1.0, 2.0]);
        assert_eq!(err, Err(CscError::RowIndexOutOfBounds));
    }

    #[test]
    fn partial_eq_compares_all_fields() {
        let a = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
        let b = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
        let c = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 9.0]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn zero_col_matrix_is_valid() {
        let m = CscMatrix::<f64>::new(5, 0, vec![0], vec![], vec![]).unwrap();
        assert_eq!(m.cols(), 0);
        assert_eq!(m.nnz(), 0);
        assert_eq!(m.col_range(0), None);
    }
}

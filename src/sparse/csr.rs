use alloc::vec::Vec;
use core::fmt;
use core::ops::Range;

/// Error returned by [`CsrMatrix::new`] when the supplied arrays are invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsrError {
    /// `col_indices` and `values` have different lengths.
    LengthMismatch,
    /// `row_ptr` must have exactly `rows + 1` entries.
    RowPtrLengthMismatch,
    /// `row_ptr[0]` must be `0`, every entry must be ≤ the next, and the last entry
    /// must equal `nnz`.
    RowPtrInvalid,
    /// A column index is >= the declared column count.
    ColIndexOutOfBounds,
}

/// A sparse matrix in compressed sparse row (CSR) format. Requires the `alloc` feature.
///
/// CSR stores non-zeros using three parallel arrays:
///
/// * `row_ptr` — length `rows + 1`; `row_ptr[i]..row_ptr[i + 1]` is the slice range
///   in `col_indices` / `values` that belongs to row `i`.
/// * `col_indices` — column index of each stored entry.
/// * `values` — value of each stored entry.
///
/// Within a row the column indices need not be sorted, but every column index must be
/// in-bounds (`< cols`) and `row_ptr` must be non-decreasing.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::CsrMatrix;
///
/// // 3×3 identity matrix in CSR format.
/// let eye = CsrMatrix::new(
///     3, 3,
///     vec![0, 1, 2, 3],           // row_ptr: one entry per row + sentinel
///     vec![0, 1, 2],              // col_indices
///     vec![1.0_f64, 1.0, 1.0],   // values
/// )
/// .unwrap();
///
/// assert_eq!(eye.rows(), 3);
/// assert_eq!(eye.cols(), 3);
/// assert_eq!(eye.nnz(), 3);
/// assert_eq!(eye.row_range(1), Some(1..2));
/// assert_eq!(eye.col_indices()[1], 1);
/// assert_eq!(eye.values()[1], 1.0);
/// ```
pub struct CsrMatrix<T> {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
}

impl<T> CsrMatrix<T> {
    pub(super) fn new_raw(
        rows: usize,
        cols: usize,
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        Self {
            rows,
            cols,
            row_ptr,
            col_indices,
            values,
        }
    }

    pub(super) fn into_raw_parts(self) -> (usize, usize, Vec<usize>, Vec<usize>, Vec<T>) {
        (
            self.rows,
            self.cols,
            self.row_ptr,
            self.col_indices,
            self.values,
        )
    }

    /// Creates a new `CsrMatrix` from its row-pointer, column-index, and value arrays.
    ///
    /// # Errors
    ///
    /// - [`CsrError::LengthMismatch`] — `col_indices.len() != values.len()`.
    /// - [`CsrError::RowPtrLengthMismatch`] — `row_ptr.len() != rows + 1`.
    /// - [`CsrError::RowPtrInvalid`] — `row_ptr[0] != 0`, `row_ptr` is not non-decreasing,
    ///   or `row_ptr[rows] != col_indices.len()`.
    /// - [`CsrError::ColIndexOutOfBounds`] — any column index `>= cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::{CsrMatrix, CsrError};
    ///
    /// // row_ptr must have rows + 1 = 4 entries for a 3-row matrix.
    /// let err = CsrMatrix::<f64>::new(3, 3, vec![0, 1], vec![], vec![]);
    /// assert_eq!(err, Err(CsrError::RowPtrLengthMismatch));
    /// ```
    pub fn new(
        rows: usize,
        cols: usize,
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, CsrError> {
        if col_indices.len() != values.len() {
            return Err(CsrError::LengthMismatch);
        }
        if row_ptr.len() != rows + 1 {
            return Err(CsrError::RowPtrLengthMismatch);
        }
        let nnz = col_indices.len();
        if row_ptr[0] != 0 || row_ptr[rows] != nnz {
            return Err(CsrError::RowPtrInvalid);
        }
        for i in 1..=rows {
            if row_ptr[i] < row_ptr[i - 1] {
                return Err(CsrError::RowPtrInvalid);
            }
        }
        for &c in &col_indices {
            if c >= cols {
                return Err(CsrError::ColIndexOutOfBounds);
            }
        }
        Ok(Self {
            rows,
            cols,
            row_ptr,
            col_indices,
            values,
        })
    }

    /// Returns the number of rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CsrMatrix;
    ///
    /// let m = CsrMatrix::<f64>::new(4, 5, vec![0; 5], vec![], vec![]).unwrap();
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
    /// use rustebra::sparse::CsrMatrix;
    ///
    /// let m = CsrMatrix::<f64>::new(4, 5, vec![0; 5], vec![], vec![]).unwrap();
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
    /// use rustebra::sparse::CsrMatrix;
    ///
    /// let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
    /// assert_eq!(m.nnz(), 2);
    /// ```
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Returns the row-pointer array.  Entry `i` is the start index into `col_indices`
    /// and `values` for row `i`; entry `rows` is `nnz`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CsrMatrix;
    ///
    /// let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
    /// assert_eq!(m.row_ptr(), &[0, 1, 2]);
    /// ```
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Returns the column-index array.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CsrMatrix;
    ///
    /// let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
    /// assert_eq!(m.col_indices(), &[0, 1]);
    /// ```
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Returns the value array.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CsrMatrix;
    ///
    /// let m = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
    /// assert_eq!(m.values(), &[3.0, 7.0]);
    /// ```
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns the index range into `col_indices` and `values` that belongs to `row`.
    /// Returns `None` if `row >= self.rows()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CsrMatrix;
    ///
    /// let m = CsrMatrix::new(3, 3, vec![0, 0, 1, 3], vec![0, 1, 2], vec![1.0_f64, 2.0, 3.0])
    ///     .unwrap();
    /// assert_eq!(m.row_range(0), Some(0..0)); // row 0 is empty
    /// assert_eq!(m.row_range(1), Some(0..1));
    /// assert_eq!(m.row_range(2), Some(1..3));
    /// assert_eq!(m.row_range(3), None);       // out of bounds
    /// ```
    pub fn row_range(&self, row: usize) -> Option<Range<usize>> {
        if row >= self.rows {
            None
        } else {
            Some(self.row_ptr[row]..self.row_ptr[row + 1])
        }
    }
}

impl<T: PartialEq> PartialEq for CsrMatrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self.row_ptr == other.row_ptr
            && self.col_indices == other.col_indices
            && self.values == other.values
    }
}

impl<T: fmt::Debug> fmt::Debug for CsrMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CsrMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("nnz", &self.values.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{CsrError, CsrMatrix};

    #[test]
    fn constructs_empty_matrix() {
        let m = CsrMatrix::<f64>::new(3, 4, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m.nnz(), 0);
        assert_eq!(m.row_ptr(), &[0, 0, 0, 0]);
    }

    #[test]
    fn constructs_identity_matrix() {
        let m = CsrMatrix::new(
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
        assert_eq!(m.col_indices(), &[0, 1, 2]);
        assert_eq!(m.values(), &[1.0, 1.0, 1.0]);
        assert_eq!(m.row_range(0), Some(0..1));
        assert_eq!(m.row_range(1), Some(1..2));
        assert_eq!(m.row_range(2), Some(2..3));
        assert_eq!(m.row_range(3), None);
    }

    #[test]
    fn constructs_matrix_with_empty_rows() {
        // Row 0 is empty, row 1 has one entry, row 2 has two entries.
        let m = CsrMatrix::new(
            3,
            3,
            vec![0, 0, 1, 3],
            vec![0, 1, 2],
            vec![1.0_f64, 2.0, 3.0],
        )
        .unwrap();
        assert_eq!(m.row_range(0), Some(0..0));
        assert_eq!(m.row_range(1), Some(0..1));
        assert_eq!(m.row_range(2), Some(1..3));
    }

    #[test]
    fn length_mismatch_is_an_error_not_a_panic() {
        let err = CsrMatrix::<f64>::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0]);
        assert_eq!(err, Err(CsrError::LengthMismatch));
    }

    #[test]
    fn row_ptr_length_mismatch_is_an_error_not_a_panic() {
        let err = CsrMatrix::<f64>::new(3, 3, vec![0, 1], vec![], vec![]);
        assert_eq!(err, Err(CsrError::RowPtrLengthMismatch));
    }

    #[test]
    fn row_ptr_not_starting_at_zero_is_an_error_not_a_panic() {
        let err = CsrMatrix::<f64>::new(2, 2, vec![1, 1, 1], vec![], vec![]);
        assert_eq!(err, Err(CsrError::RowPtrInvalid));
    }

    #[test]
    fn row_ptr_not_monotone_is_an_error_not_a_panic() {
        let err = CsrMatrix::<f64>::new(3, 3, vec![0, 2, 1, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
        assert_eq!(err, Err(CsrError::RowPtrInvalid));
    }

    #[test]
    fn row_ptr_last_entry_mismatch_is_an_error_not_a_panic() {
        let err = CsrMatrix::<f64>::new(2, 2, vec![0, 1, 5], vec![0, 1], vec![1.0, 2.0]);
        assert_eq!(err, Err(CsrError::RowPtrInvalid));
    }

    #[test]
    fn col_index_out_of_bounds_is_an_error_not_a_panic() {
        let err = CsrMatrix::<f64>::new(2, 2, vec![0, 1, 2], vec![0, 3], vec![1.0, 2.0]);
        assert_eq!(err, Err(CsrError::ColIndexOutOfBounds));
    }

    #[test]
    fn partial_eq_compares_all_fields() {
        let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
        let b = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
        let c = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 9.0]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn zero_row_matrix_is_valid() {
        let m = CsrMatrix::<f64>::new(0, 5, vec![0], vec![], vec![]).unwrap();
        assert_eq!(m.rows(), 0);
        assert_eq!(m.nnz(), 0);
        assert_eq!(m.row_range(0), None);
    }
}

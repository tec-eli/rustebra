use alloc::vec::Vec;
use core::fmt;

/// Error returned by [`CooMatrix::new`] when the supplied triplets are invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CooError {
    /// The three component slices (`row_indices`, `col_indices`, `values`) don't all have
    /// the same length.
    LengthMismatch,
    /// A row index is >= the declared row count.
    RowIndexOutOfBounds,
    /// A column index is >= the declared column count.
    ColIndexOutOfBounds,
}

/// A sparse matrix in coordinate (COO) format. Requires the `alloc` feature.
///
/// COO format stores every non-zero as an explicit `(row, col, value)` triplet. The
/// triplets need not be sorted and the same `(row, col)` position may appear more than
/// once; duplicate entries are treated as logically summed when the matrix is used in
/// an operation (e.g. when converting to CSR or multiplying by a dense vector).
///
/// # Examples
///
/// ```
/// use rustebra::sparse::CooMatrix;
///
/// // 3×3 identity matrix stored as a COO sparse matrix.
/// let eye = CooMatrix::new(
///     3, 3,
///     vec![0, 1, 2],
///     vec![0, 1, 2],
///     vec![1.0_f64, 1.0, 1.0],
/// ).unwrap();
///
/// assert_eq!(eye.rows(), 3);
/// assert_eq!(eye.cols(), 3);
/// assert_eq!(eye.nnz(), 3);
/// assert_eq!(eye.row_indices(), &[0, 1, 2]);
/// assert_eq!(eye.col_indices(), &[0, 1, 2]);
/// assert_eq!(eye.values(), &[1.0, 1.0, 1.0]);
/// ```
pub struct CooMatrix<T> {
    rows: usize,
    cols: usize,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
}

impl<T> CooMatrix<T> {
    pub(super) fn new_raw(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        Self {
            rows,
            cols,
            row_indices,
            col_indices,
            values,
        }
    }

    pub(super) fn into_raw_parts(self) -> (usize, usize, Vec<usize>, Vec<usize>, Vec<T>) {
        (
            self.rows,
            self.cols,
            self.row_indices,
            self.col_indices,
            self.values,
        )
    }

    /// Creates a new `CooMatrix` from its row-indices, column-indices, and values.
    ///
    /// All three `Vec`s must have the same length. Every row index must be `< rows` and
    /// every column index must be `< cols`; otherwise `Err(CooError)` is returned.
    ///
    /// # Errors
    ///
    /// - `Err(CooError::LengthMismatch)` if the three vectors don't all have the same length.
    /// - `Err(CooError::RowIndexOutOfBounds)` if any row index >= `rows`.
    /// - `Err(CooError::ColIndexOutOfBounds)` if any column index >= `cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::{CooMatrix, CooError};
    ///
    /// // Column index 5 is out of bounds for a 3×3 matrix.
    /// let err = CooMatrix::<f64>::new(3, 3, vec![0], vec![5], vec![1.0]);
    /// assert_eq!(err, Err(CooError::ColIndexOutOfBounds));
    /// ```
    pub fn new(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, CooError> {
        if row_indices.len() != col_indices.len() || row_indices.len() != values.len() {
            return Err(CooError::LengthMismatch);
        }
        for &r in &row_indices {
            if r >= rows {
                return Err(CooError::RowIndexOutOfBounds);
            }
        }
        for &c in &col_indices {
            if c >= cols {
                return Err(CooError::ColIndexOutOfBounds);
            }
        }
        Ok(Self {
            rows,
            cols,
            row_indices,
            col_indices,
            values,
        })
    }

    /// Returns the number of rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CooMatrix;
    ///
    /// let m = CooMatrix::<f64>::new(4, 5, vec![], vec![], vec![]).unwrap();
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
    /// use rustebra::sparse::CooMatrix;
    ///
    /// let m = CooMatrix::<f64>::new(4, 5, vec![], vec![], vec![]).unwrap();
    /// assert_eq!(m.cols(), 5);
    /// ```
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of stored entries (including any explicit zeros or duplicate
    /// `(row, col)` positions).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CooMatrix;
    ///
    /// let m = CooMatrix::new(2, 2, vec![0, 1], vec![0, 1], vec![3.0_f64, 7.0]).unwrap();
    /// assert_eq!(m.nnz(), 2);
    /// ```
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Returns a slice of the row indices of all stored entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CooMatrix;
    ///
    /// let m = CooMatrix::new(2, 2, vec![0, 1], vec![1, 0], vec![5.0_f64, 9.0]).unwrap();
    /// assert_eq!(m.row_indices(), &[0, 1]);
    /// ```
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Returns a slice of the column indices of all stored entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CooMatrix;
    ///
    /// let m = CooMatrix::new(2, 2, vec![0, 1], vec![1, 0], vec![5.0_f64, 9.0]).unwrap();
    /// assert_eq!(m.col_indices(), &[1, 0]);
    /// ```
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Returns a slice of the values of all stored entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::sparse::CooMatrix;
    ///
    /// let m = CooMatrix::new(2, 2, vec![0, 1], vec![1, 0], vec![5.0_f64, 9.0]).unwrap();
    /// assert_eq!(m.values(), &[5.0, 9.0]);
    /// ```
    pub fn values(&self) -> &[T] {
        &self.values
    }
}

impl<T: PartialEq> PartialEq for CooMatrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self.row_indices == other.row_indices
            && self.col_indices == other.col_indices
            && self.values == other.values
    }
}

impl<T: fmt::Debug> fmt::Debug for CooMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CooMatrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("nnz", &self.values.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{CooError, CooMatrix};

    #[test]
    fn constructs_empty_matrix() {
        let m = CooMatrix::<f64>::new(3, 4, vec![], vec![], vec![]).unwrap();
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m.nnz(), 0);
    }

    #[test]
    fn constructs_triplet_matrix() {
        let m = CooMatrix::new(3, 3, vec![0, 1, 2], vec![0, 1, 2], vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.nnz(), 3);
        assert_eq!(m.row_indices(), &[0, 1, 2]);
        assert_eq!(m.col_indices(), &[0, 1, 2]);
        assert_eq!(m.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn length_mismatch_row_col_is_an_error_not_a_panic() {
        let err = CooMatrix::<f64>::new(2, 2, vec![0, 1], vec![0], vec![1.0, 2.0]);
        assert_eq!(err, Err(CooError::LengthMismatch));
    }

    #[test]
    fn length_mismatch_col_values_is_an_error_not_a_panic() {
        let err = CooMatrix::<f64>::new(2, 2, vec![0], vec![0], vec![1.0, 2.0]);
        assert_eq!(err, Err(CooError::LengthMismatch));
    }

    #[test]
    fn row_index_out_of_bounds_is_an_error_not_a_panic() {
        let err = CooMatrix::<f64>::new(2, 2, vec![2], vec![0], vec![1.0]);
        assert_eq!(err, Err(CooError::RowIndexOutOfBounds));
    }

    #[test]
    fn col_index_out_of_bounds_is_an_error_not_a_panic() {
        let err = CooMatrix::<f64>::new(2, 2, vec![0], vec![2], vec![1.0]);
        assert_eq!(err, Err(CooError::ColIndexOutOfBounds));
    }

    #[test]
    fn duplicate_positions_are_accepted() {
        let m = CooMatrix::new(2, 2, vec![0, 0], vec![0, 0], vec![1.0_f64, 2.0]).unwrap();
        assert_eq!(m.nnz(), 2);
    }

    #[test]
    fn partial_eq_compares_shape_indices_and_values() {
        let a = CooMatrix::new(2, 2, vec![0, 1], vec![1, 0], vec![5.0_f64, 9.0]).unwrap();
        let b = CooMatrix::new(2, 2, vec![0, 1], vec![1, 0], vec![5.0_f64, 9.0]).unwrap();
        let c = CooMatrix::new(2, 2, vec![0, 1], vec![1, 0], vec![5.0_f64, 8.0]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn zero_row_count_with_no_entries_is_valid() {
        let m = CooMatrix::<f64>::new(0, 5, vec![], vec![], vec![]).unwrap();
        assert_eq!(m.rows(), 0);
        assert_eq!(m.nnz(), 0);
    }
}

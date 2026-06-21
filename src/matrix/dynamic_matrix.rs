use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::algorithm::matrix::{self as algorithm, DimensionMismatch};
use crate::scalar::Scalar;
use crate::storage::{DynamicStorage, Storage};
use crate::vector::DynamicVector;

/// A heap-allocated matrix of runtime-determined shape, stored row-major. Requires the
/// `alloc` feature.
///
/// This is the public API layer (ADR 0006, layer 4) for matrices backed by dynamic storage:
/// it wires [`DynamicStorage`] together with the generic functions in
/// [`crate::algorithm::matrix`] into a concrete, ergonomic type, so callers don't need to
/// work with `Storage`/`Scalar` generics or row/column index arithmetic directly.
///
/// Unlike [`crate::matrix::StaticMatrix`], a `DynamicMatrix`'s shape lives in its fields, not
/// in its type, so operations between two of them can genuinely fail with
/// [`DimensionMismatch`] at runtime, rather than that case being statically unreachable.
///
/// # Examples
///
/// ```
/// use rustebra::matrix::DynamicMatrix;
///
/// let a = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let b = DynamicMatrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
/// let sum = DynamicMatrix::new(2, 2, vec![6.0, 8.0, 10.0, 12.0]).unwrap();
/// assert_eq!(a.add(&b), Ok(sum));
/// ```
pub struct DynamicMatrix<T> {
    storage: DynamicStorage<T>,
    rows: usize,
    cols: usize,
}

impl<T: Scalar> DynamicMatrix<T> {
    /// Creates a new `DynamicMatrix` from a flat, row-major `Vec` of elements.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` if `data.len() != rows * cols`, rather than
    /// panicking, per ADR 0004.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// ```
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, DimensionMismatch> {
        if data.len() != rows * cols {
            return Err(DimensionMismatch);
        }
        Ok(Self::from_parts(DynamicStorage::new(data), rows, cols))
    }

    fn from_parts(storage: DynamicStorage<T>, rows: usize, cols: usize) -> Self {
        Self {
            storage,
            rows,
            cols,
        }
    }

    /// Returns the number of rows in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// let m = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// assert_eq!(m.rows(), 2);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// let m = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// assert_eq!(m.cols(), 3);
    /// ```
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Computes the element-wise sum of `self` and `other`.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` if `self` and `other` don't have the same shape,
    /// rather than panicking, per ADR 0004.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// let a = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let b = DynamicMatrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    /// let sum = DynamicMatrix::new(2, 2, vec![6.0, 8.0, 10.0, 12.0]).unwrap();
    /// assert_eq!(a.add(&b), Ok(sum));
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self, DimensionMismatch> {
        let mut data = vec![T::zero(); self.rows * self.cols];
        algorithm::add(
            &self.storage,
            self.rows,
            self.cols,
            &other.storage,
            other.rows,
            other.cols,
            &mut data,
        )?;
        Ok(Self::from_parts(
            DynamicStorage::new(data),
            self.rows,
            self.cols,
        ))
    }

    /// Computes the element-wise difference of `self` and `other`.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` if `self` and `other` don't have the same shape,
    /// rather than panicking, per ADR 0004.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// let a = DynamicMatrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    /// let b = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let diff = DynamicMatrix::new(2, 2, vec![4.0, 4.0, 4.0, 4.0]).unwrap();
    /// assert_eq!(a.sub(&b), Ok(diff));
    /// ```
    pub fn sub(&self, other: &Self) -> Result<Self, DimensionMismatch> {
        let mut data = vec![T::zero(); self.rows * self.cols];
        algorithm::sub(
            &self.storage,
            self.rows,
            self.cols,
            &other.storage,
            other.rows,
            other.cols,
            &mut data,
        )?;
        Ok(Self::from_parts(
            DynamicStorage::new(data),
            self.rows,
            self.cols,
        ))
    }

    /// Computes the element-wise product of `self` and `factor`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let scaled = DynamicMatrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]).unwrap();
    /// assert_eq!(m.mul_scalar(2.0), scaled);
    /// ```
    pub fn mul_scalar(&self, factor: T) -> Self {
        let mut data = vec![T::zero(); self.rows * self.cols];
        // `data` is constructed with exactly `self.rows * self.cols` elements, so this can
        // never disagree in length.
        match algorithm::mul_scalar(&self.storage, self.rows, self.cols, factor, &mut data) {
            Ok(()) | Err(DimensionMismatch) => {}
        }
        Self::from_parts(DynamicStorage::new(data), self.rows, self.cols)
    }

    /// Computes the matrix-vector product `self * v`.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` if `v`'s length doesn't match `self.cols()` (the
    /// "inner dimension" matrix-vector multiplication requires), rather than panicking, per
    /// ADR 0004.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    /// use rustebra::vector::DynamicVector;
    ///
    /// let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let v = DynamicVector::new(vec![1.0, 1.0]);
    /// assert_eq!(m.mul_vector(&v), Ok(DynamicVector::new(vec![3.0, 7.0])));
    /// ```
    pub fn mul_vector(&self, v: &DynamicVector<T>) -> Result<DynamicVector<T>, DimensionMismatch> {
        let mut out = vec![T::zero(); self.rows];
        algorithm::mul_vector(&self.storage, self.rows, self.cols, v, &mut out)?;
        Ok(DynamicVector::new(out))
    }

    /// Computes the matrix-matrix product `self * other`.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` if `self.cols() != other.rows()` (the "inner
    /// dimension" matrix multiplication requires), rather than panicking, per ADR 0004.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// let a = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let b = DynamicMatrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    /// let product = DynamicMatrix::new(2, 2, vec![19.0, 22.0, 43.0, 50.0]).unwrap();
    /// assert_eq!(a.mul_matrix(&b), Ok(product));
    /// ```
    pub fn mul_matrix(&self, other: &Self) -> Result<Self, DimensionMismatch> {
        let mut data = vec![T::zero(); self.rows * other.cols];
        algorithm::mul_matrix(
            &self.storage,
            self.rows,
            self.cols,
            &other.storage,
            other.rows,
            other.cols,
            &mut data,
        )?;
        Ok(Self::from_parts(
            DynamicStorage::new(data),
            self.rows,
            other.cols,
        ))
    }

    /// Computes the transpose of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// let m = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let transposed = DynamicMatrix::new(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    /// assert_eq!(m.transpose(), transposed);
    /// ```
    pub fn transpose(&self) -> Self {
        let mut data = vec![T::zero(); self.rows * self.cols];
        // `data` is constructed with exactly `self.rows * self.cols` elements, so this can
        // never disagree in length.
        match algorithm::transpose(&self.storage, self.rows, self.cols, &mut data) {
            Ok(()) | Err(DimensionMismatch) => {}
        }
        Self::from_parts(DynamicStorage::new(data), self.cols, self.rows)
    }
}

impl<T> PartialEq for DynamicMatrix<T>
where
    T: Scalar + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && (0..self.storage.len()).all(|i| self.storage.get(i) == other.storage.get(i))
    }
}

impl<T> fmt::Debug for DynamicMatrix<T>
where
    T: Scalar + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..self.storage.len()).filter_map(|i| self.storage.get(i)))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::DynamicMatrix;
    use crate::algorithm::matrix::DimensionMismatch;
    use crate::vector::DynamicVector;

    #[test]
    fn constructs_from_flat_row_major_data() {
        let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            m,
            DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()
        );
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
    }

    #[test]
    fn constructs_mismatched_data_length_is_an_error_not_a_panic() {
        assert_eq!(
            DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0]),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn add_is_wired_to_the_algorithm_layer() {
        let a = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = DynamicMatrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        assert_eq!(
            a.add(&b),
            Ok(DynamicMatrix::new(2, 2, vec![6.0, 8.0, 10.0, 12.0]).unwrap())
        );
    }

    #[test]
    fn add_mismatched_shape_is_an_error_not_a_panic() {
        let a = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(a.add(&b), Err(DimensionMismatch));
    }

    #[test]
    fn sub_is_wired_to_the_algorithm_layer() {
        let a = DynamicMatrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let b = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        assert_eq!(
            a.sub(&b),
            Ok(DynamicMatrix::new(2, 2, vec![4.0, 4.0, 4.0, 4.0]).unwrap())
        );
    }

    #[test]
    fn sub_mismatched_shape_is_an_error_not_a_panic() {
        let a = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = DynamicMatrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(a.sub(&b), Err(DimensionMismatch));
    }

    #[test]
    fn mul_scalar_is_wired_to_the_algorithm_layer() {
        let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        assert_eq!(
            m.mul_scalar(2.0),
            DynamicMatrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]).unwrap()
        );
    }

    #[test]
    fn mul_vector_is_wired_to_the_algorithm_layer() {
        let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let v = DynamicVector::new(vec![1.0, 1.0]);

        assert_eq!(m.mul_vector(&v), Ok(DynamicVector::new(vec![3.0, 7.0])));
    }

    #[test]
    fn mul_vector_mismatched_inner_dimension_is_an_error_not_a_panic() {
        let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let v = DynamicVector::new(vec![1.0, 1.0, 1.0]);

        assert_eq!(m.mul_vector(&v), Err(DimensionMismatch));
    }

    #[test]
    fn mul_matrix_is_wired_to_the_algorithm_layer() {
        let a = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = DynamicMatrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        assert_eq!(
            a.mul_matrix(&b),
            Ok(DynamicMatrix::new(2, 2, vec![19.0, 22.0, 43.0, 50.0]).unwrap())
        );
    }

    #[test]
    fn mul_matrix_mismatched_inner_dimension_is_an_error_not_a_panic() {
        let a = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        assert_eq!(a.mul_matrix(&b), Err(DimensionMismatch));
    }

    #[test]
    fn transpose_is_wired_to_the_algorithm_layer() {
        let m = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(
            m.transpose(),
            DynamicMatrix::new(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap()
        );
    }
}

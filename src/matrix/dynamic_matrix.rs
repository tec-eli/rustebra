use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::algorithm::matrix::{
    self as algorithm, CholeskyError, ConditionNumberError, DeterminantError, DimensionMismatch,
};
use crate::scalar::{FloatTolerance, Scalar};
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

    /// Computes the determinant of `self`.
    ///
    /// # Errors
    ///
    /// Returns `Err(DeterminantError::DimensionMismatch)` if `self.rows() != self.cols()`,
    /// rather than panicking. Returns `Err(DeterminantError::MatrixTooLargeWithoutAlloc)` if
    /// the `alloc` feature is disabled and the matrix has more than 4 rows; in that case use
    /// [`crate::algorithm::matrix::determinant_lu`] with a caller-provided scratch buffer
    /// instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// assert_eq!(m.determinant(), Ok(-2.0));
    /// ```
    pub fn determinant(&self) -> Result<T, DeterminantError>
    where
        T: PartialOrd,
    {
        algorithm::determinant(&self.storage, self.rows, self.cols)
    }

    /// Computes the rank of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// // Row 1 is twice row 0, so rank is 1.
    /// let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
    /// assert_eq!(m.rank(), 1);
    /// ```
    pub fn rank(&self) -> usize
    where
        T: FloatTolerance + PartialOrd,
    {
        let mut scratch = vec![T::zero(); self.rows * self.cols];
        // `scratch` is constructed with exactly `self.rows * self.cols` elements, matching
        // `self.storage`, so this can never disagree in length.
        algorithm::rank(&self.storage, self.rows, self.cols, &mut scratch).unwrap_or(0)
    }

    /// Computes the LU decomposition of `self`: factors it as `l * u`, where `l` is unit
    /// lower triangular and `u` is upper triangular, up to a row permutation recorded as a
    /// swap count (see [`crate::algorithm::matrix::lu`]) rather than materialized as its own
    /// matrix.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` if `self.rows() != self.cols()`, rather than
    /// panicking, per ADR 0004.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// // Row 0 already holds the largest-magnitude entry (6) in column 0, so no swap is
    /// // needed.
    /// let m = DynamicMatrix::new(2, 2, vec![6.0, 3.0, 4.0, 3.0]).unwrap();
    /// let (l, u, swap_count) = m.lu().unwrap();
    /// assert_eq!(swap_count, 0);
    /// assert_eq!(l, DynamicMatrix::new(2, 2, vec![1.0, 0.0, 4.0 / 6.0, 1.0]).unwrap());
    /// assert_eq!(u, DynamicMatrix::new(2, 2, vec![6.0, 3.0, 0.0, 1.0]).unwrap());
    /// ```
    pub fn lu(&self) -> Result<(Self, Self, usize), DimensionMismatch>
    where
        T: PartialOrd,
    {
        let mut l = vec![T::zero(); self.rows * self.cols];
        let mut u = vec![T::zero(); self.rows * self.cols];
        let swap_count = algorithm::lu(&self.storage, self.rows, self.cols, &mut l, &mut u)?;
        Ok((
            Self::from_parts(DynamicStorage::new(l), self.rows, self.cols),
            Self::from_parts(DynamicStorage::new(u), self.rows, self.cols),
            swap_count,
        ))
    }

    /// Computes the QR decomposition of `self`: factors it as `q * r`, where `q` is a `self.
    /// rows() x self.rows()` orthogonal matrix and `r` is a `self.rows() x self.cols()` upper
    /// triangular matrix.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` if `self.rows() < self.cols()`, rather than panicking,
    /// per ADR 0004.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// let m = DynamicMatrix::new(2, 2, vec![3.0_f64, 5.0, 4.0, 0.0]).unwrap();
    /// let (q, r) = m.qr().unwrap();
    /// assert_eq!((q.rows(), q.cols()), (2, 2));
    /// assert_eq!((r.rows(), r.cols()), (2, 2));
    /// ```
    pub fn qr(&self) -> Result<(Self, Self), DimensionMismatch>
    where
        T: PartialOrd,
    {
        let mut q = vec![T::zero(); self.rows * self.rows];
        let mut r = vec![T::zero(); self.rows * self.cols];
        let mut scratch = vec![T::zero(); self.rows];
        algorithm::qr(
            &self.storage,
            self.rows,
            self.cols,
            &mut q,
            &mut r,
            &mut scratch,
        )?;
        Ok((
            Self::from_parts(DynamicStorage::new(q), self.rows, self.rows),
            Self::from_parts(DynamicStorage::new(r), self.rows, self.cols),
        ))
    }

    /// Computes the Cholesky decomposition of `self`: factors it as `l * lᵗ`, where `l` is
    /// lower triangular with positive diagonal entries.
    ///
    /// `self` must be symmetric positive-definite — see
    /// [`crate::algorithm::matrix::CholeskyError`].
    ///
    /// # Errors
    ///
    /// Returns `Err(CholeskyError::DimensionMismatch)` if `self.rows() != self.cols()`.
    /// Returns `Err(CholeskyError::NotPositiveDefinite)` if `self` is not positive-definite.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// // Symmetric positive-definite: [[4, 2], [2, 2]].
    /// let m = DynamicMatrix::new(2, 2, vec![4.0, 2.0, 2.0, 2.0]).unwrap();
    /// let l = m.cholesky().unwrap();
    /// assert_eq!(l, DynamicMatrix::new(2, 2, vec![2.0, 0.0, 1.0, 1.0]).unwrap());
    /// ```
    pub fn cholesky(&self) -> Result<Self, CholeskyError>
    where
        T: FloatTolerance + PartialOrd,
    {
        let mut l = vec![T::zero(); self.rows * self.cols];
        algorithm::cholesky(&self.storage, self.rows, self.cols, &mut l)?;
        Ok(Self::from_parts(
            DynamicStorage::new(l),
            self.rows,
            self.cols,
        ))
    }

    /// Computes the singular value decomposition of `self`: factors it as `u * diag(sigma) *
    /// vᵗ`, where `u` is `self.rows() x self.cols()` with orthonormal columns, `sigma` is a
    /// length-`self.cols()` vector of non-negative singular values sorted descending, and `v`
    /// is a `self.cols() x self.cols()` orthogonal matrix.
    ///
    /// Unlike [`Self::lu`] or [`Self::cholesky`], `self` doesn't need to be square — the
    /// singular value decomposition exists for any matrix.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` under the conditions documented at
    /// [`crate::algorithm::matrix::svd`]; unreachable here, since the scratch buffer this
    /// allocates always has the length that function expects.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    /// use rustebra::storage::Storage;
    ///
    /// let m = DynamicMatrix::new(2, 2, vec![2.0_f64, 0.0, 0.0, 1.0]).unwrap();
    /// let (u, sigma, v) = m.svd().unwrap();
    /// assert!(sigma.get(0) >= sigma.get(1));
    /// assert_eq!((u.rows(), u.cols()), (2, 2));
    /// assert_eq!((v.rows(), v.cols()), (2, 2));
    /// ```
    pub fn svd(&self) -> Result<(Self, DynamicVector<T>, Self), DimensionMismatch>
    where
        T: FloatTolerance + PartialOrd,
    {
        let mut u = vec![T::zero(); self.rows * self.cols];
        let mut sigma = vec![T::zero(); self.cols];
        let mut v = vec![T::zero(); self.cols * self.cols];
        let mut scratch = vec![T::zero(); 5 * self.cols * self.cols + self.cols + self.rows];
        algorithm::svd(
            &self.storage,
            self.rows,
            self.cols,
            &mut u,
            &mut sigma,
            &mut v,
            &mut scratch,
        )?;
        Ok((
            Self::from_parts(DynamicStorage::new(u), self.rows, self.cols),
            DynamicVector::new(sigma),
            Self::from_parts(DynamicStorage::new(v), self.cols, self.cols),
        ))
    }

    /// Computes the condition number of `self`: `kappa(self) = sigma_max / sigma_min`, the
    /// ratio of its largest to smallest singular value.
    ///
    /// Only defined for square matrices, like [`Self::determinant`].
    ///
    /// # Errors
    ///
    /// Returns `Err(ConditionNumberError::DimensionMismatch)` if `self.rows() !=
    /// self.cols()`. Returns `Err(ConditionNumberError::Singular)` if `self`'s smallest
    /// singular value is negligible relative to its largest.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::DynamicMatrix;
    ///
    /// let m = DynamicMatrix::new(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap();
    /// let kappa = m.condition_number().unwrap();
    /// assert!((kappa - 1.0).abs() < 1e-9);
    /// ```
    pub fn condition_number(&self) -> Result<T, ConditionNumberError>
    where
        T: FloatTolerance + PartialOrd,
    {
        let n = self.rows;
        let mut scratch = vec![T::zero(); 7 * n * n + 3 * n];
        algorithm::condition_number(&self.storage, self.rows, self.cols, &mut scratch)
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
    use crate::algorithm::matrix::{DeterminantError, DimensionMismatch};
    use crate::storage::Storage;
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

    #[test]
    fn determinant_is_wired_to_the_algorithm_layer() {
        let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        assert_eq!(m.determinant(), Ok(-2.0));
    }

    #[test]
    fn determinant_of_non_square_matrix_is_an_error_not_a_panic() {
        let m = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(m.determinant(), Err(DeterminantError::DimensionMismatch));
    }

    #[test]
    fn rank_is_wired_to_the_algorithm_layer() {
        let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();

        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn lu_is_wired_to_the_algorithm_layer() {
        let m = DynamicMatrix::new(2, 2, vec![6.0, 3.0, 4.0, 3.0]).unwrap();

        let (l, u, swap_count) = m.lu().unwrap();
        assert_eq!(swap_count, 0);
        assert_eq!(
            l,
            DynamicMatrix::new(2, 2, vec![1.0, 0.0, 4.0 / 6.0, 1.0]).unwrap()
        );
        assert_eq!(
            u,
            DynamicMatrix::new(2, 2, vec![6.0, 3.0, 0.0, 1.0]).unwrap()
        );
    }

    #[test]
    fn lu_of_non_square_matrix_is_an_error_not_a_panic() {
        let m = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(m.lu(), Err(DimensionMismatch));
    }

    #[test]
    fn qr_is_wired_to_the_algorithm_layer() {
        let m = DynamicMatrix::new(2, 2, vec![3.0_f64, 5.0, 4.0, 0.0]).unwrap();

        let (q, r) = m.qr().unwrap();
        let reconstructed = q.mul_matrix(&r).unwrap();
        for (actual, expected) in (0..4)
            .map(|i| *reconstructed.storage.get(i).unwrap())
            .zip([3.0, 5.0, 4.0, 0.0])
        {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_of_matrix_with_more_columns_than_rows_is_an_error_not_a_panic() {
        let m = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(m.qr(), Err(DimensionMismatch));
    }

    #[test]
    fn cholesky_is_wired_to_the_algorithm_layer() {
        let m = DynamicMatrix::new(2, 2, vec![4.0, 2.0, 2.0, 2.0]).unwrap();

        assert_eq!(
            m.cholesky(),
            Ok(DynamicMatrix::new(2, 2, vec![2.0, 0.0, 1.0, 1.0]).unwrap())
        );
    }

    #[test]
    fn cholesky_of_non_positive_definite_matrix_is_an_error_not_a_panic() {
        // [[1, 2], [2, 1]]; not positive-definite (its second leading principal minor,
        // 1*1 - 2*2 = -3, is negative).
        let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 2.0, 1.0]).unwrap();

        assert_eq!(
            m.cholesky(),
            Err(crate::algorithm::matrix::CholeskyError::NotPositiveDefinite)
        );
    }

    #[test]
    fn cholesky_of_non_square_matrix_is_an_error_not_a_panic() {
        let m = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(
            m.cholesky(),
            Err(crate::algorithm::matrix::CholeskyError::DimensionMismatch)
        );
    }

    #[test]
    fn svd_is_wired_to_the_algorithm_layer() {
        // [[1, 1], [0, 1]]; a non-diagonal, well-conditioned case (golden-ratio singular
        // values).
        let m = DynamicMatrix::new(2, 2, vec![1.0_f64, 1.0, 0.0, 1.0]).unwrap();

        let (u, sigma, v) = m.svd().unwrap();
        assert!(sigma.get(0) >= sigma.get(1));
        assert!(*sigma.get(1).unwrap() >= 0.0);
        assert_eq!((u.rows(), u.cols()), (2, 2));
        assert_eq!((v.rows(), v.cols()), (2, 2));
    }

    #[test]
    fn condition_number_is_wired_to_the_algorithm_layer() {
        let m = DynamicMatrix::new(2, 2, vec![100.0_f64, 0.0, 0.0, 1.0]).unwrap();

        let kappa = m.condition_number().unwrap();
        assert!((kappa - 100.0).abs() < 1e-6);
    }

    #[test]
    fn condition_number_of_singular_matrix_is_an_error() {
        // [[1, 2], [2, 4]]; row 1 is twice row 0, so this is singular (rank 1).
        let m = DynamicMatrix::new(2, 2, vec![1.0_f64, 2.0, 2.0, 4.0]).unwrap();

        assert_eq!(
            m.condition_number(),
            Err(crate::algorithm::matrix::ConditionNumberError::Singular)
        );
    }

    #[test]
    fn condition_number_of_non_square_matrix_is_an_error_not_a_panic() {
        let m = DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert_eq!(
            m.condition_number(),
            Err(crate::algorithm::matrix::ConditionNumberError::DimensionMismatch)
        );
    }
}

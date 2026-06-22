use crate::algorithm::matrix::{self as algorithm, DimensionMismatch};
use crate::scalar::Scalar;
use crate::storage::Storage;
use crate::vector::StaticVector;

/// A stack-allocated `R x C` matrix, stored row-major.
///
/// This is the public API layer (ADR 0006, layer 4) for matrices backed by static storage:
/// `R` rows of `C` elements each, laid out as `[[T; C]; R]` (the const-generic, stack-based
/// representation ADR 0001 calls for), wired to the generic functions in
/// [`crate::algorithm::matrix`] so callers don't need to work with `Storage`/`Scalar`
/// generics or row/column index arithmetic directly.
///
/// Unlike [`crate::vector::StaticVector`] (which wraps the existing 1D [`StaticStorage`]),
/// `R * C` can't be expressed as a single const-generic array length on stable Rust, so
/// `StaticMatrix` implements [`Storage`] directly over its own `[[T; C]; R]` field instead of
/// wrapping a separate flat storage type.
///
/// [`StaticStorage`]: crate::storage::StaticStorage
///
/// # Examples
///
/// ```
/// use rustebra::matrix::StaticMatrix;
///
/// let a = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// let b = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
/// assert_eq!(a.add(&b), StaticMatrix::new([[6.0, 8.0], [10.0, 12.0]]));
/// ```
#[derive(Debug, PartialEq)]
pub struct StaticMatrix<T, const R: usize, const C: usize> {
    data: [[T; C]; R],
}

impl<T, const R: usize, const C: usize> Storage for StaticMatrix<T, R, C> {
    type Item = T;

    fn len(&self) -> usize {
        R * C
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.data.as_flattened().get(index)
    }
}

impl<T: Scalar, const R: usize, const C: usize> StaticMatrix<T, R, C> {
    /// Creates a new `StaticMatrix` from a fixed-size array of rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// ```
    pub fn new(data: [[T; C]; R]) -> Self {
        Self { data }
    }

    /// Computes the element-wise sum of `self` and `other`.
    ///
    /// `self` and `other` are both `StaticMatrix<T, R, C>`, so they're guaranteed by the
    /// type system to have the same shape; the dimension mismatch
    /// [`crate::algorithm::matrix::add`] can report is unreachable here.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// let a = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let b = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
    /// assert_eq!(a.add(&b), StaticMatrix::new([[6.0, 8.0], [10.0, 12.0]]));
    /// ```
    pub fn add(&self, other: &Self) -> Self {
        let mut data = [[T::zero(); C]; R];
        match algorithm::add(self, R, C, other, R, C, data.as_flattened_mut()) {
            Ok(()) | Err(DimensionMismatch) => {}
        }
        Self::new(data)
    }

    /// Computes the element-wise difference of `self` and `other`.
    ///
    /// `self` and `other` are both `StaticMatrix<T, R, C>`, so they're guaranteed by the
    /// type system to have the same shape; the dimension mismatch
    /// [`crate::algorithm::matrix::sub`] can report is unreachable here.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// let a = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
    /// let b = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(a.sub(&b), StaticMatrix::new([[4.0, 4.0], [4.0, 4.0]]));
    /// ```
    pub fn sub(&self, other: &Self) -> Self {
        let mut data = [[T::zero(); C]; R];
        match algorithm::sub(self, R, C, other, R, C, data.as_flattened_mut()) {
            Ok(()) | Err(DimensionMismatch) => {}
        }
        Self::new(data)
    }

    /// Computes the element-wise product of `self` and `factor`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.mul_scalar(2.0), StaticMatrix::new([[2.0, 4.0], [6.0, 8.0]]));
    /// ```
    pub fn mul_scalar(&self, factor: T) -> Self {
        let mut data = [[T::zero(); C]; R];
        match algorithm::mul_scalar(self, R, C, factor, data.as_flattened_mut()) {
            Ok(()) | Err(DimensionMismatch) => {}
        }
        Self::new(data)
    }

    /// Computes the matrix-vector product `self * v`.
    ///
    /// `v` is a `StaticVector<T, C>`, so it's guaranteed by the type system to have exactly
    /// `C` elements, matching `self`'s column count; the dimension mismatch
    /// [`crate::algorithm::matrix::mul_vector`] can report is unreachable here.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    /// use rustebra::vector::StaticVector;
    ///
    /// let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let v = StaticVector::new([1.0, 1.0]);
    /// assert_eq!(m.mul_vector(&v), StaticVector::new([3.0, 7.0]));
    /// ```
    pub fn mul_vector(&self, v: &StaticVector<T, C>) -> StaticVector<T, R> {
        let mut out = [T::zero(); R];
        match algorithm::mul_vector(self, R, C, v, &mut out) {
            Ok(()) | Err(DimensionMismatch) => {}
        }
        StaticVector::new(out)
    }

    /// Computes the matrix-matrix product `self * other`.
    ///
    /// `other` is a `StaticMatrix<T, C, C2>`, so its row count is guaranteed by the type
    /// system to equal `C`, `self`'s column count — the "inner dimension" matrix
    /// multiplication requires; the dimension mismatch
    /// [`crate::algorithm::matrix::mul_matrix`] can report is unreachable here.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// let a = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let b = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
    /// assert_eq!(a.mul_matrix(&b), StaticMatrix::new([[19.0, 22.0], [43.0, 50.0]]));
    /// ```
    pub fn mul_matrix<const C2: usize>(
        &self,
        other: &StaticMatrix<T, C, C2>,
    ) -> StaticMatrix<T, R, C2> {
        let mut data = [[T::zero(); C2]; R];
        match algorithm::mul_matrix(self, R, C, other, C, C2, data.as_flattened_mut()) {
            Ok(()) | Err(DimensionMismatch) => {}
        }
        StaticMatrix::new(data)
    }

    /// Computes the transpose of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// let m = StaticMatrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// assert_eq!(m.transpose(), StaticMatrix::new([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]));
    /// ```
    pub fn transpose(&self) -> StaticMatrix<T, C, R> {
        let mut data = [[T::zero(); R]; C];
        match algorithm::transpose(self, R, C, data.as_flattened_mut()) {
            Ok(()) | Err(DimensionMismatch) => {}
        }
        StaticMatrix::new(data)
    }

    /// Computes the rank of `self`.
    ///
    /// `self` and the scratch buffer used internally both have exactly `R * C` elements, so
    /// the dimension mismatch [`crate::algorithm::matrix::rank`] can report is unreachable
    /// here.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// // Row 1 is twice row 0, so rank is 1.
    /// let m = StaticMatrix::new([[1.0, 2.0], [2.0, 4.0]]);
    /// assert_eq!(m.rank(), 1);
    /// ```
    pub fn rank(&self) -> usize
    where
        T: PartialEq,
    {
        let mut scratch = [[T::zero(); C]; R];
        algorithm::rank(self, R, C, scratch.as_flattened_mut()).unwrap_or(0)
    }
}

impl<T: Scalar, const N: usize> StaticMatrix<T, N, N> {
    /// Computes the determinant of `self`.
    ///
    /// `self` is a `StaticMatrix<T, N, N>`, so it's guaranteed by the type system to be
    /// square; the dimension mismatch [`crate::algorithm::matrix::determinant`] can report
    /// is unreachable here.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.determinant(), -2.0);
    /// ```
    pub fn determinant(&self) -> T
    where
        T: PartialEq,
    {
        algorithm::determinant(self, N, N).unwrap_or(T::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::StaticMatrix;
    use crate::vector::StaticVector;

    #[test]
    fn constructs_from_rows() {
        let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m, StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]));
    }

    #[test]
    fn add_is_wired_to_the_algorithm_layer() {
        let a = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);

        assert_eq!(a.add(&b), StaticMatrix::new([[6.0, 8.0], [10.0, 12.0]]));
    }

    #[test]
    fn sub_is_wired_to_the_algorithm_layer() {
        let a = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
        let b = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(a.sub(&b), StaticMatrix::new([[4.0, 4.0], [4.0, 4.0]]));
    }

    #[test]
    fn mul_scalar_is_wired_to_the_algorithm_layer() {
        let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(
            m.mul_scalar(2.0),
            StaticMatrix::new([[2.0, 4.0], [6.0, 8.0]])
        );
    }

    #[test]
    fn mul_vector_is_wired_to_the_algorithm_layer() {
        let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let v = StaticVector::new([1.0, 1.0]);

        assert_eq!(m.mul_vector(&v), StaticVector::new([3.0, 7.0]));
    }

    #[test]
    fn mul_matrix_is_wired_to_the_algorithm_layer() {
        let a = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);

        assert_eq!(
            a.mul_matrix(&b),
            StaticMatrix::new([[19.0, 22.0], [43.0, 50.0]])
        );
    }

    #[test]
    fn transpose_is_wired_to_the_algorithm_layer() {
        let m = StaticMatrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        assert_eq!(
            m.transpose(),
            StaticMatrix::new([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        );
    }

    #[test]
    fn determinant_is_wired_to_the_algorithm_layer() {
        let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(m.determinant(), -2.0);
    }

    #[test]
    fn rank_is_wired_to_the_algorithm_layer() {
        let m = StaticMatrix::new([[1.0, 2.0], [2.0, 4.0]]);

        assert_eq!(m.rank(), 1);
    }
}

use crate::algorithm::matrix::{
    self as algorithm, CholeskyError, ConditionNumberError, DeterminantError, DimensionMismatch,
};
use crate::scalar::{FloatTolerance, Scalar};
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

/// `(u, sigma, v)`, the result of [`StaticMatrix::svd`]'s singular value decomposition.
type SvdResult<T, const R: usize, const C: usize> = (
    StaticMatrix<T, R, C>,
    StaticVector<T, C>,
    StaticMatrix<T, C, C>,
);

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
        T: FloatTolerance + PartialOrd,
    {
        let mut scratch = [[T::zero(); C]; R];
        algorithm::rank(self, R, C, scratch.as_flattened_mut()).unwrap_or(0)
    }

    /// Computes the QR decomposition of `self`: factors it as `q * r`, where `q` is an `R x
    /// R` orthogonal matrix and `r` is an `R x C` upper triangular matrix.
    ///
    /// Unlike [`Self::rank`] or [`Self::add`], `R >= C` (required by
    /// [`crate::algorithm::matrix::qr`]) isn't something the type system can rule out here —
    /// stable Rust has no way to bound one const generic against another — so this returns a
    /// `Result` rather than swallowing the dimension check.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` if `R < C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    /// use rustebra::storage::Storage;
    ///
    /// let m = StaticMatrix::new([[3.0_f64, 5.0], [4.0, 0.0]]);
    /// let (q, r) = m.qr().unwrap();
    ///
    /// // q * r reconstructs m (checked within tolerance: q's entries involve dividing by
    /// // an irrational square root for this m).
    /// let reconstructed = q.mul_matrix(&r);
    /// for (i, expected) in [3.0, 5.0, 4.0, 0.0].into_iter().enumerate() {
    ///     let actual = *reconstructed.get(i).unwrap();
    ///     assert!((actual - expected).abs() < 1e-9);
    /// }
    /// ```
    pub fn qr(&self) -> Result<(StaticMatrix<T, R, R>, StaticMatrix<T, R, C>), DimensionMismatch>
    where
        T: PartialOrd,
    {
        let mut q = [[T::zero(); R]; R];
        let mut r = [[T::zero(); C]; R];
        let mut scratch = [T::zero(); R];
        algorithm::qr(
            self,
            R,
            C,
            q.as_flattened_mut(),
            r.as_flattened_mut(),
            &mut scratch,
        )?;
        Ok((StaticMatrix::new(q), StaticMatrix::new(r)))
    }

    /// Computes the singular value decomposition of `self`: factors it as `u * diag(sigma) *
    /// vᵗ`, where `u` is an `R x C` matrix with orthonormal columns, `sigma` is a length-`C`
    /// vector of non-negative singular values sorted descending, and `v` is a `C x C`
    /// orthogonal matrix.
    ///
    /// Unlike [`Self::lu`] or [`Self::cholesky`], `self` doesn't need to be square — the
    /// singular value decomposition exists for any matrix.
    ///
    /// `scratch` must have exactly `5 * C * C + C + R` elements — see
    /// [`crate::algorithm::matrix::svd`]. Unlike every other method here, this can't build its
    /// own scratch buffer internally: that length combines two different const generics
    /// (`R` and `C`), and stable Rust has no way to size a fixed-size array from an expression
    /// over more than one const generic parameter. The caller picks how to provide it — a
    /// stack array sized with the formula above to stay allocation-free, or a `Vec` if the
    /// `alloc` feature is enabled and sizing it by hand isn't worth the trouble.
    ///
    /// # Errors
    ///
    /// Returns `Err(DimensionMismatch)` if `scratch.len() != 5 * C * C + C + R`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    /// use rustebra::storage::Storage;
    ///
    /// let m = StaticMatrix::new([[2.0_f64, 0.0], [0.0, 1.0]]);
    /// let mut scratch = [0.0; 5 * 2 * 2 + 2 + 2];
    /// let (u, sigma, v) = m.svd(&mut scratch).unwrap();
    /// assert!(sigma.get(0) >= sigma.get(1));
    /// assert!(*sigma.get(1).unwrap() >= 0.0);
    /// let _ = (u, v);
    /// ```
    pub fn svd(&self, scratch: &mut [T]) -> Result<SvdResult<T, R, C>, DimensionMismatch>
    where
        T: FloatTolerance + PartialOrd,
    {
        let mut u = [[T::zero(); C]; R];
        let mut sigma = [T::zero(); C];
        let mut v = [[T::zero(); C]; C];
        algorithm::svd(
            self,
            R,
            C,
            u.as_flattened_mut(),
            &mut sigma,
            v.as_flattened_mut(),
            scratch,
        )?;
        Ok((
            StaticMatrix::new(u),
            StaticVector::new(sigma),
            StaticMatrix::new(v),
        ))
    }
}

impl<T: Scalar, const N: usize> StaticMatrix<T, N, N> {
    /// Computes the determinant of `self`.
    ///
    /// `self` is a `StaticMatrix<T, N, N>`, so it's guaranteed by the type system to be
    /// square; `Err(DeterminantError::DimensionMismatch)` is therefore unreachable. However,
    /// `Err(DeterminantError::MatrixTooLargeWithoutAlloc)` is returned when the `alloc`
    /// feature is disabled and `N > 4`; in that case use
    /// [`crate::algorithm::matrix::determinant_lu`] with a caller-provided scratch buffer.
    ///
    /// # Errors
    ///
    /// Returns `Err(DeterminantError::MatrixTooLargeWithoutAlloc)` if the `alloc` feature is
    /// disabled and `N > 4`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.determinant(), Ok(-2.0));
    /// ```
    pub fn determinant(&self) -> Result<T, DeterminantError>
    where
        T: PartialOrd,
    {
        algorithm::determinant(self, N, N)
    }

    /// Computes the LU decomposition of `self`: factors it as `l * u`, where `l` is unit
    /// lower triangular and `u` is upper triangular, up to a row permutation recorded as a
    /// swap count (see [`crate::algorithm::matrix::lu`]) rather than materialized as its own
    /// matrix.
    ///
    /// `self` is a `StaticMatrix<T, N, N>`, so it's guaranteed by the type system to be
    /// square; the dimension mismatch [`crate::algorithm::matrix::lu`] can report is
    /// unreachable here.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// // Row 0 already holds the largest-magnitude entry (6) in column 0, so no swap is
    /// // needed.
    /// let m = StaticMatrix::new([[6.0, 3.0], [4.0, 3.0]]);
    /// let (l, u, swap_count) = m.lu();
    /// assert_eq!(swap_count, 0);
    /// assert_eq!(l, StaticMatrix::new([[1.0, 0.0], [4.0 / 6.0, 1.0]]));
    /// assert_eq!(u, StaticMatrix::new([[6.0, 3.0], [0.0, 1.0]]));
    /// ```
    pub fn lu(&self) -> (StaticMatrix<T, N, N>, StaticMatrix<T, N, N>, usize)
    where
        T: PartialOrd,
    {
        let mut l = [[T::zero(); N]; N];
        let mut u = [[T::zero(); N]; N];
        let swap_count = algorithm::lu(self, N, N, l.as_flattened_mut(), u.as_flattened_mut())
            .unwrap_or_default();
        (StaticMatrix::new(l), StaticMatrix::new(u), swap_count)
    }

    /// Computes the Cholesky decomposition of `self`: factors it as `l * lᵗ`, where `l` is
    /// lower triangular with positive diagonal entries.
    ///
    /// `self` is a `StaticMatrix<T, N, N>`, so it's guaranteed by the type system to be
    /// square; the only dimension mismatch [`crate::algorithm::matrix::cholesky`] can report
    /// is unreachable here. `self` must still be symmetric positive-definite, which the type
    /// system can't guarantee — see [`crate::algorithm::matrix::CholeskyError`].
    ///
    /// # Errors
    ///
    /// Returns `Err(CholeskyError::NotPositiveDefinite)` if `self` is not positive-definite.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// // Symmetric positive-definite: [[4, 2], [2, 2]].
    /// let m = StaticMatrix::new([[4.0, 2.0], [2.0, 2.0]]);
    /// let l = m.cholesky().unwrap();
    /// assert_eq!(l, StaticMatrix::new([[2.0, 0.0], [1.0, 1.0]]));
    /// ```
    pub fn cholesky(&self) -> Result<StaticMatrix<T, N, N>, CholeskyError>
    where
        T: FloatTolerance + PartialOrd,
    {
        let mut l = [[T::zero(); N]; N];
        algorithm::cholesky(self, N, N, l.as_flattened_mut())?;
        Ok(StaticMatrix::new(l))
    }

    /// Computes the condition number of `self`: `kappa(self) = sigma_max / sigma_min`, the
    /// ratio of its largest to smallest singular value.
    ///
    /// `scratch` must have exactly `7 * N * N + 3 * N` elements — see
    /// [`crate::algorithm::matrix::condition_number`]. As with [`Self::svd`], this can't build
    /// its own scratch buffer internally: that length is a polynomial in `N`, and stable Rust
    /// has no way to size a fixed-size array from anything but a const generic parameter used
    /// standalone. The caller picks how to provide it — a stack array sized with the formula
    /// above to stay allocation-free, or a `Vec` if the `alloc` feature is enabled.
    ///
    /// # Errors
    ///
    /// Returns `Err(ConditionNumberError::DimensionMismatch)` if `scratch.len() != 7 * N * N +
    /// 3 * N`. Returns `Err(ConditionNumberError::Singular)` if `self`'s smallest singular
    /// value is negligible relative to its largest.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::matrix::StaticMatrix;
    ///
    /// let m = StaticMatrix::new([[1.0_f64, 0.0], [0.0, 1.0]]);
    /// let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];
    /// let kappa = m.condition_number(&mut scratch).unwrap();
    /// assert!((kappa - 1.0).abs() < 1e-9);
    /// ```
    pub fn condition_number(&self, scratch: &mut [T]) -> Result<T, ConditionNumberError>
    where
        T: FloatTolerance + PartialOrd,
    {
        algorithm::condition_number(self, N, N, scratch)
    }
}

#[cfg(test)]
mod tests {
    use super::StaticMatrix;
    use crate::storage::Storage;
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

        assert_eq!(m.determinant(), Ok(-2.0));
    }

    #[test]
    fn rank_is_wired_to_the_algorithm_layer() {
        let m = StaticMatrix::new([[1.0, 2.0], [2.0, 4.0]]);

        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn lu_is_wired_to_the_algorithm_layer() {
        let m = StaticMatrix::new([[6.0, 3.0], [4.0, 3.0]]);

        let (l, u, swap_count) = m.lu();
        assert_eq!(swap_count, 0);
        assert_eq!(l, StaticMatrix::new([[1.0, 0.0], [4.0 / 6.0, 1.0]]));
        assert_eq!(u, StaticMatrix::new([[6.0, 3.0], [0.0, 1.0]]));
    }

    #[test]
    fn qr_is_wired_to_the_algorithm_layer() {
        let m = StaticMatrix::new([[3.0_f64, 5.0], [4.0, 0.0]]);

        let (q, r) = m.qr().unwrap();
        let reconstructed = q.mul_matrix(&r);
        for (actual, expected) in [
            reconstructed.data[0][0],
            reconstructed.data[0][1],
            reconstructed.data[1][0],
            reconstructed.data[1][1],
        ]
        .into_iter()
        .zip([3.0, 5.0, 4.0, 0.0])
        {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_of_matrix_with_more_columns_than_rows_is_an_error_not_a_panic() {
        let m = StaticMatrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        assert_eq!(m.qr(), Err(crate::algorithm::matrix::DimensionMismatch));
    }

    #[test]
    fn cholesky_is_wired_to_the_algorithm_layer() {
        let m = StaticMatrix::new([[4.0, 2.0], [2.0, 2.0]]);

        assert_eq!(
            m.cholesky(),
            Ok(StaticMatrix::new([[2.0, 0.0], [1.0, 1.0]]))
        );
    }

    #[test]
    fn cholesky_of_non_positive_definite_matrix_is_an_error_not_a_panic() {
        // [[1, 2], [2, 1]]; not positive-definite (its second leading principal minor,
        // 1*1 - 2*2 = -3, is negative).
        let m = StaticMatrix::new([[1.0, 2.0], [2.0, 1.0]]);

        assert_eq!(
            m.cholesky(),
            Err(crate::algorithm::matrix::CholeskyError::NotPositiveDefinite)
        );
    }

    #[test]
    fn svd_is_wired_to_the_algorithm_layer() {
        // [[1, 1], [0, 1]]; a non-diagonal, well-conditioned case (golden-ratio singular
        // values).
        let m = StaticMatrix::new([[1.0_f64, 1.0], [0.0, 1.0]]);
        let mut scratch = [0.0; 5 * 2 * 2 + 2 + 2];

        let (_, sigma, _) = m.svd(&mut scratch).unwrap();
        assert!(sigma.get(0) >= sigma.get(1));
        assert!(*sigma.get(1).unwrap() >= 0.0);
    }

    #[test]
    fn svd_mismatched_scratch_length_is_an_error_not_a_panic() {
        let m = StaticMatrix::new([[1.0, 1.0], [0.0, 1.0]]);
        let mut scratch = [0.0; 4];

        assert_eq!(
            m.svd(&mut scratch),
            Err(crate::algorithm::matrix::DimensionMismatch)
        );
    }

    #[test]
    fn condition_number_is_wired_to_the_algorithm_layer() {
        let m = StaticMatrix::new([[100.0_f64, 0.0], [0.0, 1.0]]);
        let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

        let kappa = m.condition_number(&mut scratch).unwrap();
        assert!((kappa - 100.0).abs() < 1e-6);
    }

    #[test]
    fn condition_number_of_singular_matrix_is_an_error() {
        // [[1, 2], [2, 4]]; row 1 is twice row 0, so this is singular (rank 1).
        let m = StaticMatrix::new([[1.0_f64, 2.0], [2.0, 4.0]]);
        let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

        assert_eq!(
            m.condition_number(&mut scratch),
            Err(crate::algorithm::matrix::ConditionNumberError::Singular)
        );
    }
}

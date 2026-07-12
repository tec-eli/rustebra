/// The `K x K` symmetric tridiagonal matrix produced by [`crate::krylov::lanczos`]: the
/// projection `Qᵗ * A * Q` of a symmetric matrix onto the orthonormal Krylov basis `Q`.
///
/// Only the diagonal (`K` entries) and the first sub/super-diagonal (`K - 1` entries, shared
/// by symmetry) are stored; every other entry is zero by construction. Its eigenvalues
/// approximate eigenvalues of the projected matrix — exactly its spectrum when `K` equals the
/// matrix dimension and no breakdown occurred.
///
/// # Examples
///
/// ```
/// use rustebra::krylov::lanczos;
/// use rustebra::storage::{Basis, StaticStorage};
///
/// // Already tridiagonal: [[2, 1], [1, 2]]. Starting from e1, Lanczos reproduces it.
/// let a = StaticStorage::new([2.0_f64, 1.0, 1.0, 2.0]);
/// let v0 = StaticStorage::new([1.0, 0.0]);
/// let mut buffer = [0.0; 4];
/// let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
/// let mut scratch = [0.0; 2];
/// let t = lanczos(&a, 2, &v0, 1e-12, &mut basis, &mut scratch).unwrap();
///
/// assert!((t.diagonal()[0] - 2.0).abs() < 1e-12);
/// assert!((t.diagonal()[1] - 2.0).abs() < 1e-12);
/// assert_eq!(t.off_diagonal().len(), 1);
/// assert!((t.off_diagonal()[0] - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TridiagonalMatrix<T, const K: usize> {
    diagonal: [T; K],
    // `[T; K - 1]` isn't expressible on stable Rust, so the array carries one padding slot;
    // `off_diagonal()` hides it.
    off_diagonal: [T; K],
}

impl<T, const K: usize> TridiagonalMatrix<T, K> {
    /// Assembles the result of a Lanczos run; `off_diagonal[K - 1]` is padding, never read.
    pub(super) fn new(diagonal: [T; K], off_diagonal: [T; K]) -> Self {
        Self {
            diagonal,
            off_diagonal,
        }
    }

    /// The `K` diagonal entries `α_0 ..= α_{K-1}`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::krylov::lanczos;
    /// use rustebra::storage::{Basis, StaticStorage};
    ///
    /// let a = StaticStorage::new([3.0_f64]);
    /// let v0 = StaticStorage::new([1.0]);
    /// let mut buffer = [0.0; 1];
    /// let mut basis = Basis::<f64, 1>::new(&mut buffer, 1).unwrap();
    /// let mut scratch = [0.0; 1];
    /// let t = lanczos(&a, 1, &v0, 1e-12, &mut basis, &mut scratch).unwrap();
    ///
    /// assert_eq!(t.diagonal(), &[3.0]);
    /// ```
    pub fn diagonal(&self) -> &[T; K] {
        &self.diagonal
    }

    /// The `K - 1` sub/super-diagonal entries `β_0 ..= β_{K-2}` (shared by symmetry):
    /// `off_diagonal()[j]` couples rows `j` and `j + 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::krylov::lanczos;
    /// use rustebra::storage::{Basis, StaticStorage};
    ///
    /// let a = StaticStorage::new([3.0_f64]);
    /// let v0 = StaticStorage::new([1.0]);
    /// let mut buffer = [0.0; 1];
    /// let mut basis = Basis::<f64, 1>::new(&mut buffer, 1).unwrap();
    /// let mut scratch = [0.0; 1];
    /// let t = lanczos(&a, 1, &v0, 1e-12, &mut basis, &mut scratch).unwrap();
    ///
    /// // A 1x1 tridiagonal matrix has no off-diagonal at all.
    /// assert!(t.off_diagonal().is_empty());
    /// ```
    pub fn off_diagonal(&self) -> &[T] {
        &self.off_diagonal[..K.saturating_sub(1)]
    }
}

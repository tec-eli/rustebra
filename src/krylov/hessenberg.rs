use crate::scalar::Scalar;

/// The `K x K` upper Hessenberg matrix produced by [`crate::krylov::arnoldi`]: the projection
/// `Qᵗ * A * Q` of a general (possibly non-symmetric) operator onto the orthonormal Krylov
/// basis `Q`.
///
/// Every entry below the first subdiagonal (`row > col + 1`) is zero by construction. Unlike
/// [`crate::krylov::TridiagonalMatrix`], which packs its two nonzero diagonals into a pair of
/// `K`-length arrays, this stores the full `K x K` block densely: a non-symmetric projection
/// fills its entire upper triangle, so there is no sparser layout to exploit, and `[T; K]`
/// nested `K` times (rather than an expression like `[T; K * K]`) is what stable Rust's
/// const generics allow.
///
/// # Examples
///
/// ```
/// use rustebra::krylov::arnoldi;
/// use rustebra::storage::{Basis, StaticStorage};
///
/// // Already upper Hessenberg: [[2, 1, 0], [3, 2, 1], [0, 1, 2]]. Starting from e1, Arnoldi
/// // walks the matrix's own rows and reproduces it exactly.
/// let a = StaticStorage::new([2.0_f64, 1.0, 0.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0]);
/// let v0 = StaticStorage::new([1.0, 0.0, 0.0]);
/// let mut buffer = [0.0; 9];
/// let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
/// let mut scratch = [0.0; 3];
///
/// let (h, reached) = arnoldi(&a, 3, &v0, 1e-12, &mut basis, &mut scratch).unwrap();
///
/// assert_eq!(reached, 3);
/// assert!((h.entry(0, 0).unwrap() - 2.0).abs() < 1e-12);
/// assert!((h.entry(1, 0).unwrap() - 3.0).abs() < 1e-12);
/// assert!((h.entry(0, 2).unwrap() - 0.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HessenbergMatrix<T, const K: usize> {
    data: [[T; K]; K],
}

impl<T: Scalar, const K: usize> HessenbergMatrix<T, K> {
    /// Assembles the result of an Arnoldi run from its dense `K x K` row-major storage.
    pub(super) fn new(data: [[T; K]; K]) -> Self {
        Self { data }
    }

    /// The entry at `(row, col)`, or `None` if either index is `>= K`.
    ///
    /// Structurally zero entries (`row > col + 1`) read back as `T::zero()`, the same as any
    /// entry that happens to vanish numerically; the type carries no separate record of which
    /// zeros are structural.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::krylov::arnoldi;
    /// use rustebra::storage::{Basis, StaticStorage};
    ///
    /// let a = StaticStorage::new([2.0_f64]);
    /// let v0 = StaticStorage::new([1.0]);
    /// let mut buffer = [0.0; 1];
    /// let mut basis = Basis::<f64, 1>::new(&mut buffer, 1).unwrap();
    /// let mut scratch = [0.0; 1];
    /// let (h, _) = arnoldi(&a, 1, &v0, 1e-12, &mut basis, &mut scratch).unwrap();
    ///
    /// assert_eq!(h.entry(0, 0), Some(2.0));
    /// assert_eq!(h.entry(1, 0), None);
    /// ```
    pub fn entry(&self, row: usize, col: usize) -> Option<T> {
        if row >= K || col >= K {
            return None;
        }
        Some(self.data[row][col])
    }
}

#[cfg(test)]
mod tests {
    use super::HessenbergMatrix;

    #[test]
    fn entry_reads_back_the_stored_value() {
        let h = HessenbergMatrix::<f64, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(h.entry(0, 0), Some(1.0));
        assert_eq!(h.entry(0, 1), Some(2.0));
        assert_eq!(h.entry(1, 0), Some(3.0));
        assert_eq!(h.entry(1, 1), Some(4.0));
    }

    #[test]
    fn entry_out_of_bounds_is_none() {
        let h = HessenbergMatrix::<f64, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(h.entry(2, 0), None);
        assert_eq!(h.entry(0, 2), None);
    }
}

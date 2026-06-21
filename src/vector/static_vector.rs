use core::fmt;

use crate::algorithm::vector as algorithm;
use crate::algorithm::vector::LengthMismatch;
use crate::scalar::Scalar;
use crate::storage::{StaticStorage, Storage};

/// A stack-allocated vector of exactly `N` elements.
///
/// This is the public API layer (ADR 0006, layer 4) for vectors backed by static storage:
/// it wires [`StaticStorage`] together with the generic functions in
/// [`crate::algorithm::vector`] into a concrete, ergonomic type, so callers don't need to
/// work with `Storage`/`Scalar` generics directly.
///
/// # Examples
///
/// ```
/// use rustebra::vector::StaticVector;
///
/// let a = StaticVector::new([1.0, 2.0, 3.0]);
/// let b = StaticVector::new([4.0, 5.0, 6.0]);
/// assert_eq!(a.add(&b), StaticVector::new([5.0, 7.0, 9.0]));
/// ```
pub struct StaticVector<T, const N: usize> {
    storage: StaticStorage<T, N>,
}

impl<T, const N: usize> Storage for StaticVector<T, N> {
    type Item = T;

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.storage.get(index)
    }
}

impl<T: Scalar, const N: usize> StaticVector<T, N> {
    /// Creates a new `StaticVector` from a fixed-size array of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::StaticVector;
    ///
    /// let v = StaticVector::new([1.0, 2.0, 3.0]);
    /// ```
    pub fn new(data: [T; N]) -> Self {
        Self {
            storage: StaticStorage::new(data),
        }
    }

    /// Computes the element-wise sum of `self` and `other`.
    ///
    /// `self` and `other` are both `StaticVector<T, N>`, so they're guaranteed by the type
    /// system to have exactly `N` elements each; the dimension mismatch
    /// [`crate::algorithm::vector::add`] can report is unreachable here.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::StaticVector;
    ///
    /// let a = StaticVector::new([1.0, 2.0, 3.0]);
    /// let b = StaticVector::new([4.0, 5.0, 6.0]);
    /// assert_eq!(a.add(&b), StaticVector::new([5.0, 7.0, 9.0]));
    /// ```
    pub fn add(&self, other: &Self) -> Self {
        let mut data = [T::zero(); N];
        match algorithm::add(&self.storage, &other.storage, &mut data) {
            Ok(()) | Err(LengthMismatch) => {}
        }
        Self::new(data)
    }

    /// Computes the element-wise difference of `self` and `other`.
    ///
    /// `self` and `other` are both `StaticVector<T, N>`, so they're guaranteed by the type
    /// system to have exactly `N` elements each; the dimension mismatch
    /// [`crate::algorithm::vector::sub`] can report is unreachable here.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::StaticVector;
    ///
    /// let a = StaticVector::new([4.0, 5.0, 6.0]);
    /// let b = StaticVector::new([1.0, 2.0, 3.0]);
    /// assert_eq!(a.sub(&b), StaticVector::new([3.0, 3.0, 3.0]));
    /// ```
    pub fn sub(&self, other: &Self) -> Self {
        let mut data = [T::zero(); N];
        match algorithm::sub(&self.storage, &other.storage, &mut data) {
            Ok(()) | Err(LengthMismatch) => {}
        }
        Self::new(data)
    }

    /// Computes the element-wise product of `self` and `factor`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::StaticVector;
    ///
    /// let v = StaticVector::new([1.0, 2.0, 3.0]);
    /// assert_eq!(v.scale(2.0), StaticVector::new([2.0, 4.0, 6.0]));
    /// ```
    pub fn scale(&self, factor: T) -> Self {
        let mut data = [T::zero(); N];
        match algorithm::scale(&self.storage, factor, &mut data) {
            Ok(()) | Err(LengthMismatch) => {}
        }
        Self::new(data)
    }

    /// Computes the dot (inner) product of `self` and `other`.
    ///
    /// `self` and `other` are both `StaticVector<T, N>`, so they're guaranteed by the type
    /// system to have exactly `N` elements each; the dimension mismatch
    /// [`crate::algorithm::vector::dot`] can report is unreachable here.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::StaticVector;
    ///
    /// let a = StaticVector::new([1.0, 2.0, 3.0]);
    /// let b = StaticVector::new([4.0, 5.0, 6.0]);
    /// assert_eq!(a.dot(&b), 32.0);
    /// ```
    pub fn dot(&self, other: &Self) -> T {
        match algorithm::dot(&self.storage, &other.storage) {
            Ok(value) => value,
            // Same-`N` invariant from the type system makes this unreachable.
            Err(LengthMismatch) => T::zero(),
        }
    }

    /// Computes the Euclidean (L2) norm of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::StaticVector;
    ///
    /// let v = StaticVector::new([3.0, 4.0]);
    /// assert_eq!(v.norm(), 5.0);
    /// ```
    pub fn norm(&self) -> T {
        algorithm::norm(&self.storage)
    }
}

impl<T, const N: usize> PartialEq for StaticVector<T, N>
where
    T: Scalar + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        (0..N).all(|i| self.storage.get(i) == other.storage.get(i))
    }
}

impl<T, const N: usize> fmt::Debug for StaticVector<T, N>
where
    T: Scalar + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..N).filter_map(|i| self.storage.get(i)))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::StaticVector;

    #[test]
    fn constructs_from_array() {
        let v = StaticVector::new([1.0, 2.0, 3.0]);
        assert_eq!(v, StaticVector::new([1.0, 2.0, 3.0]));
    }

    #[test]
    fn add_is_wired_to_the_algorithm_layer() {
        let a = StaticVector::new([1.0, 2.0, 3.0]);
        let b = StaticVector::new([4.0, 5.0, 6.0]);

        assert_eq!(a.add(&b), StaticVector::new([5.0, 7.0, 9.0]));
    }

    #[test]
    fn sub_is_wired_to_the_algorithm_layer() {
        let a = StaticVector::new([4.0, 5.0, 6.0]);
        let b = StaticVector::new([1.0, 2.0, 3.0]);

        assert_eq!(a.sub(&b), StaticVector::new([3.0, 3.0, 3.0]));
    }

    #[test]
    fn scale_is_wired_to_the_algorithm_layer() {
        let v = StaticVector::new([1.0, 2.0, 3.0]);

        assert_eq!(v.scale(2.0), StaticVector::new([2.0, 4.0, 6.0]));
    }

    #[test]
    fn dot_is_wired_to_the_algorithm_layer() {
        let a = StaticVector::new([1.0, 2.0, 3.0]);
        let b = StaticVector::new([4.0, 5.0, 6.0]);

        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn norm_is_wired_to_the_algorithm_layer() {
        let v = StaticVector::new([3.0, 4.0]);

        assert_eq!(v.norm(), 5.0);
    }
}

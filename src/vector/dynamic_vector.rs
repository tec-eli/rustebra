use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::algorithm::vector as algorithm;
use crate::algorithm::vector::LengthMismatch;
use crate::scalar::Scalar;
use crate::storage::{DynamicStorage, Storage};

/// A heap-allocated vector of runtime-determined length. Requires the `alloc` feature.
///
/// This is the public API layer (ADR 0006, layer 4) for vectors backed by dynamic storage:
/// it wires [`DynamicStorage`] together with the generic functions in
/// [`crate::algorithm::vector`] into a concrete, ergonomic type, so callers don't need to
/// work with `Storage`/`Scalar` generics directly.
///
/// Unlike [`crate::vector::StaticVector`], two `DynamicVector`s aren't guaranteed by the
/// type system to have the same length, so operations on a pair of them can genuinely fail
/// with [`LengthMismatch`] at runtime, rather than that case being statically unreachable.
///
/// # Examples
///
/// ```
/// use rustebra::vector::DynamicVector;
///
/// let a = DynamicVector::new(vec![1.0, 2.0, 3.0]);
/// let b = DynamicVector::new(vec![4.0, 5.0, 6.0]);
/// assert_eq!(a.add(&b), Ok(DynamicVector::new(vec![5.0, 7.0, 9.0])));
/// ```
pub struct DynamicVector<T> {
    storage: DynamicStorage<T>,
}

impl<T> Storage for DynamicVector<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.storage.get(index)
    }
}

impl<T: Scalar> DynamicVector<T> {
    /// Creates a new `DynamicVector` from a `Vec` of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::DynamicVector;
    ///
    /// let v = DynamicVector::new(vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn new(data: Vec<T>) -> Self {
        Self {
            storage: DynamicStorage::new(data),
        }
    }

    /// Computes the element-wise sum of `self` and `other`.
    ///
    /// # Errors
    ///
    /// Returns `Err(LengthMismatch)` if `self` and `other` don't have the same length,
    /// rather than panicking, per ADR 0004.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::DynamicVector;
    ///
    /// let a = DynamicVector::new(vec![1.0, 2.0, 3.0]);
    /// let b = DynamicVector::new(vec![4.0, 5.0, 6.0]);
    /// assert_eq!(a.add(&b), Ok(DynamicVector::new(vec![5.0, 7.0, 9.0])));
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self, LengthMismatch> {
        let mut data = vec![T::zero(); self.storage.len()];
        algorithm::add(&self.storage, &other.storage, &mut data)?;
        Ok(Self::new(data))
    }

    /// Computes the element-wise difference of `self` and `other`.
    ///
    /// # Errors
    ///
    /// Returns `Err(LengthMismatch)` if `self` and `other` don't have the same length,
    /// rather than panicking, per ADR 0004.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::DynamicVector;
    ///
    /// let a = DynamicVector::new(vec![4.0, 5.0, 6.0]);
    /// let b = DynamicVector::new(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(a.sub(&b), Ok(DynamicVector::new(vec![3.0, 3.0, 3.0])));
    /// ```
    pub fn sub(&self, other: &Self) -> Result<Self, LengthMismatch> {
        let mut data = vec![T::zero(); self.storage.len()];
        algorithm::sub(&self.storage, &other.storage, &mut data)?;
        Ok(Self::new(data))
    }

    /// Computes the element-wise product of `self` and `factor`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::DynamicVector;
    ///
    /// let v = DynamicVector::new(vec![1.0, 2.0, 3.0]);
    /// assert_eq!(v.scale(2.0), DynamicVector::new(vec![2.0, 4.0, 6.0]));
    /// ```
    pub fn scale(&self, factor: T) -> Self {
        let mut data = vec![T::zero(); self.storage.len()];
        // `data` is constructed with exactly `self.storage.len()` elements, so this can
        // never disagree in length.
        match algorithm::scale(&self.storage, factor, &mut data) {
            Ok(()) | Err(LengthMismatch) => {}
        }
        Self::new(data)
    }

    /// Computes the dot (inner) product of `self` and `other`.
    ///
    /// # Errors
    ///
    /// Returns `Err(LengthMismatch)` if `self` and `other` don't have the same length,
    /// rather than panicking, per ADR 0004.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::DynamicVector;
    ///
    /// let a = DynamicVector::new(vec![1.0, 2.0, 3.0]);
    /// let b = DynamicVector::new(vec![4.0, 5.0, 6.0]);
    /// assert_eq!(a.dot(&b), Ok(32.0));
    /// ```
    pub fn dot(&self, other: &Self) -> Result<T, LengthMismatch> {
        algorithm::dot(&self.storage, &other.storage)
    }

    /// Computes the Euclidean (L2) norm of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::vector::DynamicVector;
    ///
    /// let v = DynamicVector::new(vec![3.0, 4.0]);
    /// assert_eq!(v.norm(), 5.0);
    /// ```
    pub fn norm(&self) -> T {
        algorithm::norm(&self.storage)
    }
}

impl<T> PartialEq for DynamicVector<T>
where
    T: Scalar + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.storage.len() == other.storage.len()
            && (0..self.storage.len()).all(|i| self.storage.get(i) == other.storage.get(i))
    }
}

impl<T> fmt::Debug for DynamicVector<T>
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
    use super::{DynamicVector, LengthMismatch};

    #[test]
    fn constructs_from_vec() {
        let v = DynamicVector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v, DynamicVector::new(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn add_is_wired_to_the_algorithm_layer() {
        let a = DynamicVector::new(vec![1.0, 2.0, 3.0]);
        let b = DynamicVector::new(vec![4.0, 5.0, 6.0]);

        assert_eq!(a.add(&b), Ok(DynamicVector::new(vec![5.0, 7.0, 9.0])));
    }

    #[test]
    fn add_mismatched_lengths_is_an_error_not_a_panic() {
        let a = DynamicVector::new(vec![1.0, 2.0]);
        let b = DynamicVector::new(vec![1.0, 2.0, 3.0]);

        assert_eq!(a.add(&b), Err(LengthMismatch));
    }

    #[test]
    fn sub_is_wired_to_the_algorithm_layer() {
        let a = DynamicVector::new(vec![4.0, 5.0, 6.0]);
        let b = DynamicVector::new(vec![1.0, 2.0, 3.0]);

        assert_eq!(a.sub(&b), Ok(DynamicVector::new(vec![3.0, 3.0, 3.0])));
    }

    #[test]
    fn sub_mismatched_lengths_is_an_error_not_a_panic() {
        let a = DynamicVector::new(vec![1.0, 2.0]);
        let b = DynamicVector::new(vec![1.0, 2.0, 3.0]);

        assert_eq!(a.sub(&b), Err(LengthMismatch));
    }

    #[test]
    fn scale_is_wired_to_the_algorithm_layer() {
        let v = DynamicVector::new(vec![1.0, 2.0, 3.0]);

        assert_eq!(v.scale(2.0), DynamicVector::new(vec![2.0, 4.0, 6.0]));
    }

    #[test]
    fn dot_is_wired_to_the_algorithm_layer() {
        let a = DynamicVector::new(vec![1.0, 2.0, 3.0]);
        let b = DynamicVector::new(vec![4.0, 5.0, 6.0]);

        assert_eq!(a.dot(&b), Ok(32.0));
    }

    #[test]
    fn dot_mismatched_lengths_is_an_error_not_a_panic() {
        let a = DynamicVector::new(vec![1.0, 2.0]);
        let b = DynamicVector::new(vec![1.0, 2.0, 3.0]);

        assert_eq!(a.dot(&b), Err(LengthMismatch));
    }

    #[test]
    fn norm_is_wired_to_the_algorithm_layer() {
        let v = DynamicVector::new(vec![3.0, 4.0]);

        assert_eq!(v.norm(), 5.0);
    }
}

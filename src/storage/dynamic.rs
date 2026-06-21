use alloc::vec::Vec;

use super::Storage;

/// Heap-allocated [`Storage`] backed by a `Vec<T>`. Requires the `alloc` feature.
///
/// # Examples
///
/// ```
/// use rustebra::storage::{Storage, DynamicStorage};
///
/// let storage = DynamicStorage::new(vec![1, 2, 3]);
/// assert_eq!(storage.len(), 3);
/// assert_eq!(storage.get(1), Some(&2));
/// assert_eq!(storage.get(3), None);
/// ```
pub struct DynamicStorage<T> {
    data: Vec<T>,
}

impl<T> DynamicStorage<T> {
    /// Creates a new `DynamicStorage` from a `Vec` of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::storage::DynamicStorage;
    ///
    /// let storage = DynamicStorage::new(vec![1, 2, 3]);
    /// ```
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }
}

impl<T> Storage for DynamicStorage<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::{DynamicStorage, Storage};

    #[test]
    fn len_and_is_empty() {
        let storage = DynamicStorage::new(vec![1, 2, 3]);
        assert_eq!(storage.len(), 3);
        assert!(!storage.is_empty());

        let empty: DynamicStorage<i32> = DynamicStorage::new(vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn get_in_and_out_of_bounds() {
        let storage = DynamicStorage::new(vec![1, 2, 3]);
        assert_eq!(storage.get(0), Some(&1));
        assert_eq!(storage.get(2), Some(&3));
        assert_eq!(storage.get(3), None);
    }
}

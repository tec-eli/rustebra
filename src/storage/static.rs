use super::Storage;

/// Stack-allocated [`Storage`] backed by a const-generic array.
///
/// # Examples
///
/// ```
/// use rustebra::storage::{Storage, StaticStorage};
///
/// let storage = StaticStorage::new([1, 2, 3]);
/// assert_eq!(storage.len(), 3);
/// assert_eq!(storage.get(1), Some(&2));
/// assert_eq!(storage.get(3), None);
/// ```
pub struct StaticStorage<T, const N: usize> {
    data: [T; N],
}

impl<T, const N: usize> StaticStorage<T, N> {
    /// Creates a new `StaticStorage` from an array of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::storage::StaticStorage;
    ///
    /// let storage = StaticStorage::new([1, 2, 3]);
    /// ```
    pub fn new(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T, const N: usize> Storage for StaticStorage<T, N> {
    type Item = T;

    fn len(&self) -> usize {
        N
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::{StaticStorage, Storage};

    #[test]
    fn len_and_is_empty() {
        let storage = StaticStorage::new([1, 2, 3]);
        assert_eq!(storage.len(), 3);
        assert!(!storage.is_empty());

        let empty: StaticStorage<i32, 0> = StaticStorage::new([]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn get_in_and_out_of_bounds() {
        let storage = StaticStorage::new([1, 2, 3]);
        assert_eq!(storage.get(0), Some(&1));
        assert_eq!(storage.get(2), Some(&3));
        assert_eq!(storage.get(3), None);
    }
}

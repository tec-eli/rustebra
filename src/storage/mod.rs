mod basis;
pub use self::basis::Basis;

mod r#static;
pub use self::r#static::StaticStorage;

#[cfg(feature = "alloc")]
mod dynamic;
#[cfg(feature = "alloc")]
pub use self::dynamic::DynamicStorage;

/// A fixed-layout collection of elements that algorithms in this crate read from.
///
/// Defines the minimal capability the algorithm layer needs: element access and a length,
/// independent of whether the backing memory lives on the stack or the heap. Kept
/// intentionally minimal so algorithms written against it work the same way regardless of
/// storage strategy, without exposing anything storage-specific that would leak into the
/// generic algorithm layer above it.
///
/// # Examples
///
/// ```
/// use rustebra::storage::Storage;
///
/// fn first<S: Storage>(storage: &S) -> Option<&S::Item> {
///     storage.get(0)
/// }
/// ```
pub trait Storage {
    /// The type of element held by this storage.
    type Item;

    /// The number of elements in this storage.
    fn len(&self) -> usize;

    /// Returns `true` if this storage holds no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the element at `index`, or `None` if `index` is out of bounds.
    fn get(&self, index: usize) -> Option<&Self::Item>;
}

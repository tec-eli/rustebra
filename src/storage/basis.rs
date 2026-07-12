use super::Storage;

/// A mutable view over caller-provided memory holding `K` vectors of `n` elements each — the
/// orthonormal basis a Krylov method (Lanczos, Arnoldi) builds up, sized at compile time.
///
/// The basis size `K` is a const generic following the crate-wide convention for Krylov
/// basis sizes, while the vector length `n` stays a runtime value so the same type serves
/// both static and dynamic problem sizes. The backing memory is a flat caller-provided
/// slice of exactly `K * n` elements (vector `k` occupies `data[k * n .. (k + 1) * n]`),
/// because [`Storage`] exposes no way to allocate memory internally — the same constraint
/// every scratch and output buffer in this crate works around.
///
/// # Examples
///
/// ```
/// use rustebra::storage::Basis;
///
/// let mut buffer = [0.0_f64; 6];
/// let basis = Basis::<f64, 2>::new(&mut buffer, 3).unwrap();
/// assert_eq!(basis.vector_len(), 3);
/// assert_eq!(basis.vector(0), Some(&[0.0, 0.0, 0.0][..]));
/// assert_eq!(basis.vector(2), None);
/// ```
#[derive(Debug)]
pub struct Basis<'a, T, const K: usize> {
    data: &'a mut [T],
    n: usize,
}

impl<'a, T, const K: usize> Basis<'a, T, K> {
    /// Creates a basis of `K` vectors of `n` elements each over `data`, or `None` if `data`
    /// doesn't hold exactly `K * n` elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::storage::Basis;
    ///
    /// let mut buffer = [0.0_f64; 6];
    /// assert!(Basis::<f64, 2>::new(&mut buffer, 3).is_some());
    ///
    /// let mut too_short = [0.0_f64; 5];
    /// assert!(Basis::<f64, 2>::new(&mut too_short, 3).is_none());
    /// ```
    pub fn new(data: &'a mut [T], n: usize) -> Option<Self> {
        if data.len() != K * n {
            return None;
        }
        Some(Self { data, n })
    }

    /// The length `n` of each basis vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::storage::Basis;
    ///
    /// let mut buffer = [0.0_f64; 6];
    /// let basis = Basis::<f64, 2>::new(&mut buffer, 3).unwrap();
    /// assert_eq!(basis.vector_len(), 3);
    /// ```
    pub fn vector_len(&self) -> usize {
        self.n
    }

    /// Returns basis vector `index` as a slice, or `None` if `index >= K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::storage::Basis;
    ///
    /// let mut buffer = [1.0_f64, 2.0, 3.0, 4.0];
    /// let basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
    /// assert_eq!(basis.vector(1), Some(&[3.0, 4.0][..]));
    /// assert_eq!(basis.vector(2), None);
    /// ```
    pub fn vector(&self, index: usize) -> Option<&[T]> {
        if index >= K {
            return None;
        }
        self.data.get(index * self.n..(index + 1) * self.n)
    }

    /// Mutable counterpart of [`Basis::vector`], for the algorithms that fill the basis in.
    pub(crate) fn vector_mut(&mut self, index: usize) -> Option<&mut [T]> {
        if index >= K {
            return None;
        }
        self.data.get_mut(index * self.n..(index + 1) * self.n)
    }
}

impl<T, const K: usize> Storage for Basis<'_, T, K> {
    type Item = T;

    fn len(&self) -> usize {
        K * self.n
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::{Basis, Storage};

    #[test]
    fn rejects_a_buffer_whose_length_disagrees_with_k_times_n() {
        let mut buffer = [0.0_f64; 5];
        assert!(Basis::<f64, 2>::new(&mut buffer, 3).is_none());
        assert!(Basis::<f64, 2>::new(&mut buffer[..4], 2).is_some());
    }

    #[test]
    fn vectors_are_consecutive_rows_of_the_flat_buffer() {
        let mut buffer = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let basis = Basis::<f64, 3>::new(&mut buffer, 2).unwrap();

        assert_eq!(basis.vector(0), Some(&[1.0, 2.0][..]));
        assert_eq!(basis.vector(1), Some(&[3.0, 4.0][..]));
        assert_eq!(basis.vector(2), Some(&[5.0, 6.0][..]));
        assert_eq!(basis.vector(3), None);
    }

    #[test]
    fn vector_mut_writes_through_to_the_backing_buffer() {
        let mut buffer = [0.0_f64; 4];
        {
            let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
            basis.vector_mut(1).unwrap()[0] = 7.0;
            assert!(basis.vector_mut(2).is_none());
        }
        assert_eq!(buffer, [0.0, 0.0, 7.0, 0.0]);
    }

    #[test]
    fn storage_view_is_flat_over_all_vectors() {
        let mut buffer = [1.0, 2.0, 3.0, 4.0];
        let basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();

        assert_eq!(basis.len(), 4);
        assert!(!basis.is_empty());
        assert_eq!(basis.get(2), Some(&3.0));
        assert_eq!(basis.get(4), None);
    }

    #[test]
    fn zero_k_basis_is_empty() {
        let mut buffer: [f64; 0] = [];
        let basis = Basis::<f64, 0>::new(&mut buffer, 3).unwrap();

        assert_eq!(basis.len(), 0);
        assert!(basis.is_empty());
        assert_eq!(basis.vector(0), None);
    }
}

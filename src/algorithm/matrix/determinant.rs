use super::DimensionMismatch;
use crate::scalar::Scalar;
use crate::storage::Storage;

/// A read-only [`Storage`] view over the `(n-1) x (n-1)` minor of an `n x n`, row-major
/// matrix `Storage`, obtained by removing row `skip_row` and column `skip_col` — used by
/// [`determinant`]'s cofactor expansion so each minor is a zero-copy view into `a` rather
/// than a freshly materialized submatrix (this crate is `no_std`-first, and `n` is only
/// known at runtime, so there's no fixed-size buffer to materialize one into).
///
/// Holds `storage` as `&dyn Storage<Item = T>` rather than a generic parameter: see
/// [`cofactor_expansion`] for why.
struct Minor<'a, T> {
    storage: &'a dyn Storage<Item = T>,
    n: usize,
    skip_row: usize,
    skip_col: usize,
}

impl<T> Storage for Minor<'_, T> {
    type Item = T;

    fn len(&self) -> usize {
        let m = self.n - 1;
        m * m
    }

    fn get(&self, index: usize) -> Option<&Self::Item> {
        let m = self.n - 1;
        if index >= m * m {
            return None;
        }
        let r = index / m;
        let c = index % m;
        let orig_r = if r < self.skip_row { r } else { r + 1 };
        let orig_c = if c < self.skip_col { c } else { c + 1 };
        self.storage.get(orig_r * self.n + orig_c)
    }
}

/// Recursively computes the determinant of the `n x n` matrix `a` via cofactor expansion
/// along the first row. Assumes `a.len() == n * n` and `n >= 1`; callers are responsible
/// for checking this (see [`determinant`]).
///
/// Takes `a` as `&dyn Storage<Item = T>` rather than a generic `Storage` parameter: each
/// recursive call wraps `a` in another [`Minor`], so a generic parameter would make every
/// recursion level its own type (`Minor<Minor<Minor<...>>>`), which the compiler must
/// monomorphize separately — but the recursion depth is only known at runtime (it's `n`),
/// so that monomorphization never terminates at compile time. A trait object keeps every
/// recursive call at the same concrete type, which is why this is one of the exceptions to
/// preferring generics over `dyn Trait` in this crate.
fn cofactor_expansion<T: Scalar>(a: &dyn Storage<Item = T>, n: usize) -> T {
    if n == 1 {
        return match a.get(0) {
            Some(&x) => x,
            None => T::zero(),
        };
    }

    let mut sum = T::zero();
    for col in 0..n {
        // `col < n == a.len() / n`, the length of row 0, so `get` is always `Some`; handled
        // explicitly rather than panicking, per ADR 0004.
        let Some(&entry) = a.get(col) else {
            return sum;
        };
        let minor = Minor {
            storage: a,
            n,
            skip_row: 0,
            skip_col: col,
        };
        let term = entry.mul(cofactor_expansion(&minor, n - 1));
        sum = if col % 2 == 0 {
            sum.add(term)
        } else {
            sum.sub(term)
        };
    }
    sum
}

/// Computes the determinant of the `rows x cols` matrix `a` via cofactor expansion:
/// recursively expanding along the first row,
/// `det(a) = sum_j (-1)^j * a[0][j] * det(minor(0, j))`, where `minor(0, j)` is the
/// `(n-1) x (n-1)` matrix obtained by removing row 0 and column `j` from `a`. The base case
/// is a single element, whose determinant is itself.
///
/// Only defined for square matrices, since the determinant itself is only defined for
/// square matrices.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a` is not square (`rows != cols`), or if `a` doesn't
/// have exactly `rows * cols` elements, rather than panicking, per ADR 0004.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::determinant;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[1, 2], [3, 4]]; det = 1*4 - 2*3 = -2.
/// let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(determinant(&a, 2, 2), Ok(-2.0));
/// ```
pub fn determinant<S, T>(a: &S, rows: usize, cols: usize) -> Result<T, DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar,
{
    if rows != cols || a.len() != rows * cols {
        return Err(DimensionMismatch);
    }
    Ok(cofactor_expansion(a, rows))
}

#[cfg(test)]
mod tests {
    use super::{DimensionMismatch, determinant};
    use crate::storage::StaticStorage;

    #[test]
    fn determinant_of_known_2x2_matrix() {
        // [[1, 2], [3, 4]]; det = 1*4 - 2*3 = -2.
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);

        assert_eq!(determinant(&a, 2, 2), Ok(-2.0));
    }

    #[test]
    fn determinant_of_known_3x3_matrix() {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 10]]; det = -3.
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);

        assert_eq!(determinant(&a, 3, 3), Ok(-3.0));
    }

    #[test]
    fn determinant_of_singular_matrix_with_a_zero_row_is_zero() {
        // [[1, 2, 3], [0, 0, 0], [7, 8, 9]]; a zero row makes the matrix singular.
        let a = StaticStorage::new([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0]);

        assert_eq!(determinant(&a, 3, 3), Ok(0.0));
    }

    #[test]
    fn determinant_of_non_square_matrix_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(determinant(&a, 2, 3), Err(DimensionMismatch));
    }
}

use crate::algorithm::vector;
use crate::scalar::Scalar;
use crate::storage::Storage;

/// Error returned by matrix operations in this module when operand dimensions don't agree.
///
/// `Storage` (see ADR 0003) only knows about a flat element count, not rows and columns —
/// matrices are stored row-major in a flat `Storage`, with their shape (`rows`, `cols`)
/// passed alongside each operand rather than queried from it. This error covers both that
/// shape disagreeing between operands, and a flat length not actually matching its claimed
/// `rows * cols` (the same role [`crate::algorithm::vector::LengthMismatch`] plays for
/// vectors, generalized to two dimensions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionMismatch;

/// Computes the element-wise sum of the `a_rows x a_cols` matrix `a` and the
/// `b_rows x b_cols` matrix `b`, writing the result into `out`.
///
/// `a`, `b`, and `out` all hold their matrix in row-major order: row `r`, column `c` lives
/// at flat index `r * cols + c`.
///
/// `a` and `b` may be different `Storage` implementations (e.g. one static, one dynamic),
/// as long as they hold the same `Scalar` type.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a_rows != b_rows`, `a_cols != b_cols`, or any of
/// `a`, `b`, `out` doesn't have exactly `a_rows * a_cols` elements, rather than panicking,
/// per ADR 0004.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::add;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrices: [[1, 2], [3, 4]] and [[5, 6], [7, 8]].
/// let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
/// let b = StaticStorage::new([5.0, 6.0, 7.0, 8.0]);
/// let mut out = [0.0; 4];
/// add(&a, 2, 2, &b, 2, 2, &mut out).unwrap();
/// assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
/// ```
pub fn add<S1, S2, T>(
    a: &S1,
    a_rows: usize,
    a_cols: usize,
    b: &S2,
    b_rows: usize,
    b_cols: usize,
    out: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S1: Storage<Item = T>,
    S2: Storage<Item = T>,
    T: Scalar,
{
    if a_rows != b_rows || a_cols != b_cols {
        return Err(DimensionMismatch);
    }
    let len = a_rows * a_cols;
    if a.len() != len || b.len() != len || out.len() != len {
        return Err(DimensionMismatch);
    }
    for (i, slot) in out.iter_mut().enumerate() {
        // `i < len == a.len() == b.len()`, so both `get` calls below are always `Some`;
        // handled explicitly rather than panicking, per ADR 0004.
        let (Some(&x), Some(&y)) = (a.get(i), b.get(i)) else {
            return Err(DimensionMismatch);
        };
        *slot = x.add(y);
    }
    Ok(())
}

/// Computes the element-wise difference of the `a_rows x a_cols` matrix `a` and the
/// `b_rows x b_cols` matrix `b`, writing the result into `out`.
///
/// `a`, `b`, and `out` all hold their matrix in row-major order: row `r`, column `c` lives
/// at flat index `r * cols + c`.
///
/// `a` and `b` may be different `Storage` implementations (e.g. one static, one dynamic),
/// as long as they hold the same `Scalar` type.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a_rows != b_rows`, `a_cols != b_cols`, or any of
/// `a`, `b`, `out` doesn't have exactly `a_rows * a_cols` elements, rather than panicking,
/// per ADR 0004.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::sub;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrices: [[5, 6], [7, 8]] and [[1, 2], [3, 4]].
/// let a = StaticStorage::new([5.0, 6.0, 7.0, 8.0]);
/// let b = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
/// let mut out = [0.0; 4];
/// sub(&a, 2, 2, &b, 2, 2, &mut out).unwrap();
/// assert_eq!(out, [4.0, 4.0, 4.0, 4.0]);
/// ```
pub fn sub<S1, S2, T>(
    a: &S1,
    a_rows: usize,
    a_cols: usize,
    b: &S2,
    b_rows: usize,
    b_cols: usize,
    out: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S1: Storage<Item = T>,
    S2: Storage<Item = T>,
    T: Scalar,
{
    if a_rows != b_rows || a_cols != b_cols {
        return Err(DimensionMismatch);
    }
    let len = a_rows * a_cols;
    if a.len() != len || b.len() != len || out.len() != len {
        return Err(DimensionMismatch);
    }
    for (i, slot) in out.iter_mut().enumerate() {
        // `i < len == a.len() == b.len()`, so both `get` calls below are always `Some`;
        // handled explicitly rather than panicking, per ADR 0004.
        let (Some(&x), Some(&y)) = (a.get(i), b.get(i)) else {
            return Err(DimensionMismatch);
        };
        *slot = x.sub(y);
    }
    Ok(())
}

/// Computes the element-wise product of the `a_rows x a_cols` matrix `a` and `factor`,
/// writing the result into `out`.
///
/// There's only one `Storage` operand here, so there's no pair of operands to disagree in
/// shape with each other — but `out` is still a separate buffer the caller provides, so it
/// can still disagree in length with `a`'s claimed `a_rows * a_cols`.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a` or `out` doesn't have exactly `a_rows * a_cols`
/// elements, rather than panicking, per ADR 0004.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::mul_scalar;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[1, 2], [3, 4]].
/// let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
/// let mut out = [0.0; 4];
/// mul_scalar(&a, 2, 2, 2.0, &mut out).unwrap();
/// assert_eq!(out, [2.0, 4.0, 6.0, 8.0]);
/// ```
pub fn mul_scalar<S, T>(
    a: &S,
    a_rows: usize,
    a_cols: usize,
    factor: T,
    out: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar,
{
    let len = a_rows * a_cols;
    if a.len() != len || out.len() != len {
        return Err(DimensionMismatch);
    }
    for (i, slot) in out.iter_mut().enumerate() {
        // `i < len == a.len()`, so `get` below is always `Some`; handled explicitly rather
        // than panicking, per ADR 0004.
        let Some(&x) = a.get(i) else {
            return Err(DimensionMismatch);
        };
        *slot = x.mul(factor);
    }
    Ok(())
}

/// A read-only [`Storage`] view over row `start / cols` of a flat, row-major matrix
/// `Storage`, so [`crate::algorithm::vector::dot`] can be reused for matrix-vector and
/// matrix-matrix multiplication instead of re-deriving the summation.
struct Row<'a, S> {
    storage: &'a S,
    start: usize,
    len: usize,
}

impl<S: Storage> Storage for Row<'_, S> {
    type Item = S::Item;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> Option<&Self::Item> {
        if index >= self.len {
            return None;
        }
        self.storage.get(self.start + index)
    }
}

/// A read-only [`Storage`] view over one column of a flat, row-major matrix `Storage`, so
/// [`crate::algorithm::vector::dot`] can be reused for matrix-matrix multiplication instead
/// of re-deriving the summation.
struct Column<'a, S> {
    storage: &'a S,
    start: usize,
    stride: usize,
    len: usize,
}

impl<S: Storage> Storage for Column<'_, S> {
    type Item = S::Item;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> Option<&Self::Item> {
        if index >= self.len {
            return None;
        }
        self.storage.get(self.start + index * self.stride)
    }
}

/// Computes the matrix-vector product `a * v`: each element `out[i]` is the dot product of
/// row `i` of the `a_rows x a_cols` matrix `a` and the vector `v`.
///
/// Reuses [`crate::algorithm::vector::dot`] rather than re-deriving the summation.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `v`'s length doesn't match `a_cols` (the "inner
/// dimension" matrix-vector multiplication requires), or if `a` or `out` doesn't have a
/// length matching its claimed shape (`a_rows * a_cols` and `a_rows` respectively), rather
/// than panicking, per ADR 0004.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::mul_vector;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[1, 2], [3, 4]].
/// let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
/// let v = StaticStorage::new([1.0, 1.0]);
/// let mut out = [0.0; 2];
/// mul_vector(&a, 2, 2, &v, &mut out).unwrap();
/// assert_eq!(out, [3.0, 7.0]);
/// ```
pub fn mul_vector<S, V, T>(
    a: &S,
    a_rows: usize,
    a_cols: usize,
    v: &V,
    out: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S: Storage<Item = T>,
    V: Storage<Item = T>,
    T: Scalar,
{
    if a.len() != a_rows * a_cols || v.len() != a_cols || out.len() != a_rows {
        return Err(DimensionMismatch);
    }
    for i in 0..a_rows {
        let row = Row {
            storage: a,
            start: i * a_cols,
            len: a_cols,
        };
        // `row.len() == a_cols == v.len()`, so `dot` can never disagree here.
        let Ok(value) = vector::dot(&row, v) else {
            return Err(DimensionMismatch);
        };
        let Some(slot) = out.get_mut(i) else {
            return Err(DimensionMismatch);
        };
        *slot = value;
    }
    Ok(())
}

/// Computes the matrix-matrix product `a * b`: each element `out[i * b_cols + j]` (row-major)
/// is the dot product of row `i` of the `a_rows x a_cols` matrix `a` and column `j` of the
/// `b_rows x b_cols` matrix `b`.
///
/// Reuses [`crate::algorithm::vector::dot`] rather than re-deriving the summation.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a_cols != b_rows` (the "inner dimension" matrix
/// multiplication requires), or if `a`, `b`, or `out` doesn't have a length matching its
/// claimed shape, rather than panicking, per ADR 0004.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::mul_matrix;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrices: [[1, 2], [3, 4]] and [[5, 6], [7, 8]].
/// let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
/// let b = StaticStorage::new([5.0, 6.0, 7.0, 8.0]);
/// let mut out = [0.0; 4];
/// mul_matrix(&a, 2, 2, &b, 2, 2, &mut out).unwrap();
/// assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
/// ```
pub fn mul_matrix<S1, S2, T>(
    a: &S1,
    a_rows: usize,
    a_cols: usize,
    b: &S2,
    b_rows: usize,
    b_cols: usize,
    out: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S1: Storage<Item = T>,
    S2: Storage<Item = T>,
    T: Scalar,
{
    if a_cols != b_rows {
        return Err(DimensionMismatch);
    }
    if a.len() != a_rows * a_cols || b.len() != b_rows * b_cols || out.len() != a_rows * b_cols {
        return Err(DimensionMismatch);
    }
    for i in 0..a_rows {
        let row = Row {
            storage: a,
            start: i * a_cols,
            len: a_cols,
        };
        for j in 0..b_cols {
            let col = Column {
                storage: b,
                start: j,
                stride: b_cols,
                len: b_rows,
            };
            // `row.len() == a_cols == b_rows == col.len()`, so `dot` can never disagree here.
            let Ok(value) = vector::dot(&row, &col) else {
                return Err(DimensionMismatch);
            };
            let Some(slot) = out.get_mut(i * b_cols + j) else {
                return Err(DimensionMismatch);
            };
            *slot = value;
        }
    }
    Ok(())
}

/// Computes the transpose of the `a_rows x a_cols` matrix `a`, writing the
/// `a_cols x a_rows` result into `out`.
///
/// Pure reindexing — no `Scalar` arithmetic involved, so `T` only needs to be `Copy`.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a` or `out` doesn't have exactly `a_rows * a_cols`
/// elements, rather than panicking, per ADR 0004.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::transpose;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x3 matrix: [[1, 2, 3], [4, 5, 6]].
/// let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let mut out = [0.0; 6];
/// transpose(&a, 2, 3, &mut out).unwrap();
/// // [[1, 4], [2, 5], [3, 6]]
/// assert_eq!(out, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
/// ```
pub fn transpose<S, T>(
    a: &S,
    a_rows: usize,
    a_cols: usize,
    out: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Copy,
{
    let len = a_rows * a_cols;
    if a.len() != len || out.len() != len {
        return Err(DimensionMismatch);
    }
    for r in 0..a_rows {
        for c in 0..a_cols {
            let Some(&value) = a.get(r * a_cols + c) else {
                return Err(DimensionMismatch);
            };
            let Some(slot) = out.get_mut(c * a_rows + r) else {
                return Err(DimensionMismatch);
            };
            *slot = value;
        }
    }
    Ok(())
}

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
    use super::{
        DimensionMismatch, add, determinant, mul_matrix, mul_scalar, mul_vector, sub, transpose,
    };
    use crate::storage::StaticStorage;

    #[test]
    fn adds_matching_dimensions_element_wise() {
        // [[1, 2], [3, 4]] + [[5, 6], [7, 8]] = [[6, 8], [10, 12]]
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let b = StaticStorage::new([5.0, 6.0, 7.0, 8.0]);
        let mut out = [0.0; 4];

        assert_eq!(add(&a, 2, 2, &b, 2, 2, &mut out), Ok(()));
        assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn mismatched_row_counts_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let mut out = [0.0; 4];

        assert_eq!(add(&a, 3, 2, &b, 2, 2, &mut out), Err(DimensionMismatch));
    }

    #[test]
    fn mismatched_column_counts_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut out = [0.0; 6];

        assert_eq!(add(&a, 2, 3, &b, 3, 2, &mut out), Err(DimensionMismatch));
    }

    #[test]
    fn mismatched_output_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let b = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let mut out = [0.0; 3];

        assert_eq!(add(&a, 2, 2, &b, 2, 2, &mut out), Err(DimensionMismatch));
    }

    #[test]
    fn subs_matching_dimensions_element_wise() {
        // [[5, 6], [7, 8]] - [[1, 2], [3, 4]] = [[4, 4], [4, 4]]
        let a = StaticStorage::new([5.0, 6.0, 7.0, 8.0]);
        let b = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let mut out = [0.0; 4];

        assert_eq!(sub(&a, 2, 2, &b, 2, 2, &mut out), Ok(()));
        assert_eq!(out, [4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn sub_mismatched_dimensions_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let mut out = [0.0; 4];

        assert_eq!(sub(&a, 3, 2, &b, 2, 2, &mut out), Err(DimensionMismatch));
    }

    #[test]
    fn mul_scalar_by_known_factor() {
        // [[1, 2], [3, 4]] * 2 = [[2, 4], [6, 8]]
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let mut out = [0.0; 4];

        assert_eq!(mul_scalar(&a, 2, 2, 2.0, &mut out), Ok(()));
        assert_eq!(out, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn mul_scalar_mismatched_dimensions_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let mut out = [0.0; 3];

        assert_eq!(mul_scalar(&a, 2, 2, 2.0, &mut out), Err(DimensionMismatch));
    }

    #[test]
    fn mul_vector_of_known_matrix_and_vector() {
        // [[1, 2], [3, 4]] * [1, 1] = [3, 7]
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let v = StaticStorage::new([1.0, 1.0]);
        let mut out = [0.0; 2];

        assert_eq!(mul_vector(&a, 2, 2, &v, &mut out), Ok(()));
        assert_eq!(out, [3.0, 7.0]);
    }

    #[test]
    fn mul_vector_mismatched_inner_dimension_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let v = StaticStorage::new([1.0, 1.0, 1.0]);
        let mut out = [0.0; 2];

        assert_eq!(mul_vector(&a, 2, 2, &v, &mut out), Err(DimensionMismatch));
    }

    #[test]
    fn mul_matrix_of_known_matrices() {
        // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let b = StaticStorage::new([5.0, 6.0, 7.0, 8.0]);
        let mut out = [0.0; 4];

        assert_eq!(mul_matrix(&a, 2, 2, &b, 2, 2, &mut out), Ok(()));
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn mul_matrix_mismatched_inner_dimension_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // 2x3
        let b = StaticStorage::new([1.0, 2.0, 3.0, 4.0]); // 2x2, but inner dim needs 3
        let mut out = [0.0; 4];

        assert_eq!(
            mul_matrix(&a, 2, 3, &b, 2, 2, &mut out),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn transposes_known_matrix() {
        // [[1, 2, 3], [4, 5, 6]] transposed = [[1, 4], [2, 5], [3, 6]]
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut out = [0.0; 6];

        assert_eq!(transpose(&a, 2, 3, &mut out), Ok(()));
        assert_eq!(out, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_mismatched_output_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut out = [0.0; 5];

        assert_eq!(transpose(&a, 2, 3, &mut out), Err(DimensionMismatch));
    }

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

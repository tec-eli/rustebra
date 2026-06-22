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

/// Computes the rank of the `rows x cols` matrix `a`: the number of linearly independent
/// rows, found by reducing `a` to row echelon form via Gaussian elimination (with partial
/// pivoting) and counting the rows that aren't entirely zero.
///
/// Unlike [`determinant`], `a` doesn't need to be square.
///
/// `Storage` is read-only, so the elimination can't mutate `a` in place; `scratch` is a
/// caller-provided buffer that this function copies `a` into and reduces, leaving `a`
/// itself untouched.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a` or `scratch` doesn't have exactly `rows * cols`
/// elements, rather than panicking.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::rank;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[1, 2], [2, 4]]; row 1 is twice row 0, so rank is 1.
/// let a = StaticStorage::new([1.0, 2.0, 2.0, 4.0]);
/// let mut scratch = [0.0; 4];
/// assert_eq!(rank(&a, 2, 2, &mut scratch), Ok(1));
/// ```
pub fn rank<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    scratch: &mut [T],
) -> Result<usize, DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialEq,
{
    let len = rows * cols;
    if a.len() != len || scratch.len() != len {
        return Err(DimensionMismatch);
    }
    for (i, slot) in scratch.iter_mut().enumerate() {
        // `i < len == a.len()`, so `get` below is always `Some`; handled explicitly rather
        // than panicking.
        let Some(&x) = a.get(i) else {
            return Err(DimensionMismatch);
        };
        *slot = x;
    }

    let zero = T::zero();
    let mut pivot_row = 0;
    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }
        let Some(found) = (pivot_row..rows).find(|&r| scratch[r * cols + col] != zero) else {
            continue;
        };
        if found != pivot_row {
            for c in 0..cols {
                scratch.swap(found * cols + c, pivot_row * cols + c);
            }
        }
        let pivot_value = scratch[pivot_row * cols + col];
        for r in (pivot_row + 1)..rows {
            let factor = scratch[r * cols + col].div(pivot_value);
            for c in col..cols {
                let term = factor.mul(scratch[pivot_row * cols + c]);
                scratch[r * cols + c] = scratch[r * cols + c].sub(term);
            }
        }
        pivot_row += 1;
    }

    let rank = (0..rows)
        .filter(|&r| (0..cols).any(|c| scratch[r * cols + c] != zero))
        .count();
    Ok(rank)
}

/// Computes the LU decomposition of the `rows x cols` matrix `a`: factors it as `l * u`,
/// where `l` is unit lower triangular (1s on the diagonal) and `u` is upper triangular, up to
/// a row permutation recorded as a swap count rather than materialized as its own matrix —
/// `l * u == p * a`, where `p` is the permutation built from applying that many row swaps, in
/// order, to the identity.
///
/// This is the high-level entry point: it always delegates to [`lu_partial_pivot`], which
/// documents the pivoting strategy. A non-pivoting variant isn't offered, since plain
/// Gaussian elimination fails outright on inputs with a zero pivot that pivoting would
/// otherwise route around (see [`lu_partial_pivot`]'s examples).
///
/// Only defined for square matrices, like [`determinant`].
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a` is not square (`rows != cols`), or if `a`,
/// `out_l`, or `out_u` doesn't have exactly `rows * cols` elements, rather than panicking,
/// per ADR 0004.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::lu;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[4, 3], [6, 3]].
/// let a = StaticStorage::new([4.0, 3.0, 6.0, 3.0]);
/// let mut l = [0.0; 4];
/// let mut u = [0.0; 4];
/// let swap_count = lu(&a, 2, 2, &mut l, &mut u).unwrap();
/// assert_eq!(swap_count, 0);
/// assert_eq!(l, [1.0, 0.0, 1.5, 1.0]);
/// assert_eq!(u, [4.0, 3.0, 0.0, -1.5]);
/// ```
pub fn lu<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_l: &mut [T],
    out_u: &mut [T],
) -> Result<usize, DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialEq,
{
    lu_partial_pivot(a, rows, cols, out_l, out_u)
}

/// Computes the LU decomposition of the `rows x cols` matrix `a` via Gaussian elimination
/// with partial pivoting: at each step `k`, before eliminating column `k`, the first row at
/// or below `k` with a nonzero entry in column `k` is swapped into position `k` (the same
/// "swap to the first nonzero" pivoting [`rank`] already uses) — this is what lets the
/// decomposition succeed on inputs plain (non-pivoting) Gaussian elimination would fail on,
/// such as a zero already sitting on the diagonal.
///
/// `out_l` and `out_u` start as the identity and a copy of `a`, respectively. Each swap
/// exchanges the full row of `out_u` (which still holds every column not yet eliminated), but
/// only columns `0..k` of `out_l` (the multipliers already computed for earlier columns) —
/// `out_l`'s diagonal and everything to its right is left untouched by the swap, which is
/// what keeps `out_l` unit lower triangular despite the row reordering. If no nonzero entry
/// exists in column `k` at or below row `k`, `a` is singular along that column; elimination
/// for `k` is skipped (there is no pivot to divide by) and decomposition continues — `out_u`
/// ends up with a zero on that diagonal entry instead of returning an error, since a zero
/// pivot is a property of `a`, not a malformed call.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `a` is not square (`rows != cols`), or if `a`,
/// `out_l`, or `out_u` doesn't have exactly `rows * cols` elements, rather than panicking,
/// per ADR 0004.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::lu_partial_pivot;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[0, 1], [1, 1]]; a zero already sits on the diagonal, so this
/// // would fail without pivoting.
/// let a = StaticStorage::new([0.0, 1.0, 1.0, 1.0]);
/// let mut l = [0.0; 4];
/// let mut u = [0.0; 4];
/// let swap_count = lu_partial_pivot(&a, 2, 2, &mut l, &mut u).unwrap();
/// assert_eq!(swap_count, 1);
/// // l * u == p * a, where p swapped rows 0 and 1: [[1, 1], [0, 1]].
/// assert_eq!(l, [1.0, 0.0, 0.0, 1.0]);
/// assert_eq!(u, [1.0, 1.0, 0.0, 1.0]);
/// ```
pub fn lu_partial_pivot<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_l: &mut [T],
    out_u: &mut [T],
) -> Result<usize, DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialEq,
{
    if rows != cols {
        return Err(DimensionMismatch);
    }
    let n = rows;
    let len = n * n;
    if a.len() != len || out_l.len() != len || out_u.len() != len {
        return Err(DimensionMismatch);
    }

    for (i, slot) in out_u.iter_mut().enumerate() {
        // `i < len == a.len()`, so `get` below is always `Some`; handled explicitly rather
        // than panicking, per ADR 0004.
        let Some(&x) = a.get(i) else {
            return Err(DimensionMismatch);
        };
        *slot = x;
    }

    let zero = T::zero();
    let one = T::one();
    for slot in out_l.iter_mut() {
        *slot = zero;
    }
    for i in 0..n {
        out_l[i * n + i] = one;
    }

    let mut swap_count = 0;
    for k in 0..n {
        if let Some(p) = (k..n).find(|&r| out_u[r * n + k] != zero)
            && p != k
        {
            for c in 0..n {
                out_u.swap(k * n + c, p * n + c);
            }
            for c in 0..k {
                out_l.swap(k * n + c, p * n + c);
            }
            swap_count += 1;
        }

        let pivot = out_u[k * n + k];
        if pivot == zero {
            continue;
        }

        for i in (k + 1)..n {
            let factor = out_u[i * n + k].div(pivot);
            out_l[i * n + k] = factor;
            for c in k..n {
                let term = factor.mul(out_u[k * n + c]);
                out_u[i * n + c] = out_u[i * n + c].sub(term);
            }
        }
    }

    Ok(swap_count)
}

/// Computes the QR decomposition of the `rows x cols` matrix `a` via Householder
/// reflections: factors it as `q * r`, where `q` is a `rows x rows` orthogonal matrix
/// (`qᵗ * q` is the identity) and `r` is a `rows x cols` upper triangular matrix.
///
/// For each column `k`, the subvector `x = a[k.., k]` (the entries of column `k` from row
/// `k` down) is reflected onto a multiple of the first coordinate axis by the Householder
/// reflector `h = i - 2 * v * vᵗ / (vᵗ * v)`, where `v = x + sign(x₁) * ‖x‖ * e₁` (`e₁` the
/// first standard basis vector, `x₁` the first entry of `x`). Choosing the sign of the
/// `‖x‖` term to match `x₁` avoids subtracting two nearly equal numbers when `x₁` and `‖x‖`
/// are close in magnitude (catastrophic cancellation); `x₁ == 0` is treated as positive,
/// since there is no cancellation to avoid either way.
///
/// Applying `h` to `a` (restricted to rows/columns `k..`) zeroes out every entry below the
/// diagonal in column `k`, without disturbing the zeros already produced in earlier
/// columns. Repeating this for every column, `hₙ * ... * h₂ * h₁ * a = r`; since each `h` is
/// symmetric and its own inverse, `a = h₁ * h₂ * ... * hₙ * r`, so `q = h₁ * h₂ * ... * hₙ`.
/// `out_q` starts as the identity and is right-multiplied by each `h` as it's built — both
/// `out_r` and `out_q` are updated directly via the reflection formula applied to their
/// columns (for `out_r`) and rows (for `out_q`), rather than materializing every `h` as its
/// own `rows x rows` matrix and calling [`mul_matrix`] repeatedly.
///
/// `scratch` is a caller-provided buffer of length `rows` this function uses to hold `v` for
/// the column currently being eliminated; `Storage` exposes no way to allocate a scratch
/// buffer internally (the same constraint [`rank`]'s `scratch` parameter works around).
///
/// Only defined for `rows >= cols`: `q` is always square (`rows x rows`), so `rows < cols`
/// would force `r` to have more nonzero rows than columns, which an upper triangular matrix
/// cannot represent.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `rows < cols`, if `a` or `out_r` doesn't have exactly
/// `rows * cols` elements, if `out_q` doesn't have exactly `rows * rows` elements, or if
/// `scratch` doesn't have exactly `rows` elements, rather than panicking.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::{mul_matrix, qr_householder};
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[3, 5], [4, 0]].
/// let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
/// let mut q = [0.0; 4];
/// let mut r = [0.0; 4];
/// let mut scratch = [0.0; 2];
/// qr_householder(&a, 2, 2, &mut q, &mut r, &mut scratch).unwrap();
///
/// // q * r reconstructs a (checked within tolerance: q's entries involve dividing by
/// // ‖x‖, an irrational square root for this a).
/// let mut qr = [0.0; 4];
/// mul_matrix(
///     &StaticStorage::new(q),
///     2,
///     2,
///     &StaticStorage::new(r),
///     2,
///     2,
///     &mut qr,
/// )
/// .unwrap();
/// for (actual, expected) in qr.iter().zip([3.0, 5.0, 4.0, 0.0]) {
///     assert!((actual - expected).abs() < 1e-9);
/// }
/// ```
pub fn qr_householder<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_q: &mut [T],
    out_r: &mut [T],
    scratch: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    if rows < cols {
        return Err(DimensionMismatch);
    }
    if a.len() != rows * cols
        || out_r.len() != rows * cols
        || out_q.len() != rows * rows
        || scratch.len() != rows
    {
        return Err(DimensionMismatch);
    }

    for (i, slot) in out_r.iter_mut().enumerate() {
        // `i < rows * cols == a.len()`, so `get` below is always `Some`;
        let Some(&x) = a.get(i) else {
            return Err(DimensionMismatch);
        };
        *slot = x;
    }

    let zero = T::zero();
    let one = T::one();
    let two = one.add(one);
    for slot in out_q.iter_mut() {
        *slot = zero;
    }
    for i in 0..rows {
        out_q[i * rows + i] = one;
    }

    for k in 0..cols {
        let mut norm_sq = zero;
        for i in k..rows {
            let xi = out_r[i * cols + k];
            norm_sq = norm_sq.add(xi.mul(xi));
        }
        let norm = norm_sq.sqrt();
        if norm == zero {
            // Column `k` is already zero from row `k` down; no reflection needed.
            continue;
        }

        let x1 = out_r[k * cols + k];
        let alpha = if x1 < zero { zero.sub(norm) } else { norm };

        scratch[k] = x1.add(alpha);
        for i in (k + 1)..rows {
            scratch[i] = out_r[i * cols + k];
        }

        let mut v_dot_v = zero;
        for &v_i in &scratch[k..rows] {
            v_dot_v = v_dot_v.add(v_i.mul(v_i));
        }
        if v_dot_v == zero {
            continue;
        }

        for c in k..cols {
            let mut dot = zero;
            for i in k..rows {
                dot = dot.add(scratch[i].mul(out_r[i * cols + c]));
            }
            let factor = two.mul(dot).div(v_dot_v);
            for i in k..rows {
                out_r[i * cols + c] = out_r[i * cols + c].sub(factor.mul(scratch[i]));
            }
        }

        for i in 0..rows {
            let mut dot = zero;
            for j in k..rows {
                dot = dot.add(scratch[j].mul(out_q[i * rows + j]));
            }
            let factor = two.mul(dot).div(v_dot_v);
            for j in k..rows {
                out_q[i * rows + j] = out_q[i * rows + j].sub(factor.mul(scratch[j]));
            }
        }
    }

    Ok(())
}

/// Computes the QR decomposition of the `rows x cols` matrix `a` via modified Gram-Schmidt
/// orthogonalization: factors it as `q * r`, where `q` is a `rows x cols` matrix with
/// orthonormal columns (`qᵗ * q` is the identity) and `r` is a `cols x cols` upper
/// triangular matrix.
///
/// For each column `j` of `a`, `v` starts as a copy of that column, then has its projection
/// onto every already-computed `qᵢ` (`i < j`) subtracted out in turn —
/// `v ← v − (qᵢ · v) · qᵢ` — before normalizing what's left to get `qⱼ = v / ‖v‖`. Projecting
/// against the running `v` rather than the original column (the "modified" part of modified
/// Gram-Schmidt) uses the most up-to-date, already-orthogonalized `v` for each projection,
/// which is what keeps rounding error from compounding as badly as plain (classical)
/// Gram-Schmidt does. The projection coefficients `qᵢ · v` become `r[i][j]`, and `‖v‖`
/// (computed after the last projection) becomes `r[j][j]`; `r[i][j]` for `i > j` is left at
/// its initial `0`, which is what makes `r` upper triangular.
///
/// `out_q` is `rows x cols` rather than `rows x rows` (unlike [`qr_householder`]'s `out_q`):
/// Gram-Schmidt only ever produces as many orthonormal columns as `a` has, with no further
/// columns to complete `q` into a square, fully orthogonal matrix.
///
/// `scratch` is a caller-provided buffer of length `rows` this function uses to hold `v` for
/// the column currently being orthogonalized; `Storage` exposes no way to allocate a scratch
/// buffer internally (the same constraint [`rank`]'s `scratch` parameter works around).
///
/// If column `j` of `a` lies entirely in the span of the previous columns (or is zero to
/// begin with), `v` reduces to the zero vector and there is no direction left to normalize
/// into `qⱼ`; column `j` of `out_q` is left at `0` instead of dividing by a zero norm, the
/// same "leave a zero instead of erroring" choice [`lu_partial_pivot`] makes for a zero
/// pivot, since linear dependence is a property of `a`, not a malformed call.
///
/// Only defined for `rows >= cols`, the same requirement [`qr_householder`] has: `q`'s
/// columns can only be made pairwise orthogonal if there are at least as many rows as
/// columns to spread them across.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` if `rows < cols`, if `a` doesn't have exactly
/// `rows * cols` elements, if `out_q` doesn't have exactly `rows * cols` elements, if `out_r`
/// doesn't have exactly `cols * cols` elements, or if `scratch` doesn't have exactly `rows`
/// elements, rather than panicking.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::{mul_matrix, qr_gram_schmidt};
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[3, 5], [4, 0]].
/// let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
/// let mut q = [0.0; 4];
/// let mut r = [0.0; 4];
/// let mut scratch = [0.0; 2];
/// qr_gram_schmidt(&a, 2, 2, &mut q, &mut r, &mut scratch).unwrap();
///
/// // q * r reconstructs a (checked within tolerance: q's entries involve dividing by
/// // ‖v‖, an irrational square root for this a).
/// let mut qr = [0.0; 4];
/// mul_matrix(
///     &StaticStorage::new(q),
///     2,
///     2,
///     &StaticStorage::new(r),
///     2,
///     2,
///     &mut qr,
/// )
/// .unwrap();
/// for (actual, expected) in qr.iter().zip([3.0, 5.0, 4.0, 0.0]) {
///     assert!((actual - expected).abs() < 1e-9);
/// }
/// ```
pub fn qr_gram_schmidt<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_q: &mut [T],
    out_r: &mut [T],
    scratch: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialEq,
{
    if rows < cols {
        return Err(DimensionMismatch);
    }
    if a.len() != rows * cols
        || out_q.len() != rows * cols
        || out_r.len() != cols * cols
        || scratch.len() != rows
    {
        return Err(DimensionMismatch);
    }

    let zero = T::zero();
    for slot in out_r.iter_mut() {
        *slot = zero;
    }

    for j in 0..cols {
        for (r, slot) in scratch.iter_mut().enumerate() {
            // `r < rows` and `j < cols`, so `r * cols + j < rows * cols == a.len()`; `get`
            // below is always `Some`, handled explicitly rather than panicking.
            let Some(&x) = a.get(r * cols + j) else {
                return Err(DimensionMismatch);
            };
            *slot = x;
        }

        for i in 0..j {
            let mut dot = zero;
            for r in 0..rows {
                dot = dot.add(out_q[r * cols + i].mul(scratch[r]));
            }
            out_r[i * cols + j] = dot;
            for r in 0..rows {
                scratch[r] = scratch[r].sub(dot.mul(out_q[r * cols + i]));
            }
        }

        let mut norm_sq = zero;
        for &v_r in scratch.iter() {
            norm_sq = norm_sq.add(v_r.mul(v_r));
        }
        let norm = norm_sq.sqrt();
        out_r[j * cols + j] = norm;

        if norm == zero {
            // Column `j` of `a` is linearly dependent on the earlier columns (or zero);
            // there's no direction left to normalize into `q`'s column `j`.
            for r in 0..rows {
                out_q[r * cols + j] = zero;
            }
            continue;
        }
        for r in 0..rows {
            out_q[r * cols + j] = scratch[r].div(norm);
        }
    }

    Ok(())
}

/// Computes the QR decomposition of the `rows x cols` matrix `a`: factors it as `q * r`,
/// where `q` is a `rows x rows` orthogonal matrix and `r` is a `rows x cols` upper
/// triangular matrix.
///
/// This is the high-level entry point: it always delegates to [`qr_householder`], since
/// Householder reflections are more numerically stable than [`qr_gram_schmidt`] for general
/// use (see [`qr_householder`]'s docs for why), and there's no observable property of `a`
/// this function currently uses to decide Gram-Schmidt would do better instead.
///
/// Note this returns the same shapes [`qr_householder`] does (`out_q` is `rows x rows`),
/// which differs from [`qr_gram_schmidt`]'s `rows x cols` `out_q` — a caller switching
/// between this function and [`qr_gram_schmidt`] directly needs differently-sized buffers.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` under the same conditions as [`qr_householder`].
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::matrix::qr;
/// use rustebra::storage::StaticStorage;
///
/// // Row-major 2x2 matrix: [[3, 5], [4, 0]].
/// let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
/// let mut q = [0.0; 4];
/// let mut r = [0.0; 4];
/// let mut scratch = [0.0; 2];
/// qr(&a, 2, 2, &mut q, &mut r, &mut scratch).unwrap();
/// assert_eq!(r[1 * 2 + 0], 0.0); // r is upper triangular.
/// ```
pub fn qr<S, T>(
    a: &S,
    rows: usize,
    cols: usize,
    out_q: &mut [T],
    out_r: &mut [T],
    scratch: &mut [T],
) -> Result<(), DimensionMismatch>
where
    S: Storage<Item = T>,
    T: Scalar + PartialOrd,
{
    qr_householder(a, rows, cols, out_q, out_r, scratch)
}

#[cfg(test)]
mod tests {
    use super::{
        DimensionMismatch, add, determinant, lu, lu_partial_pivot, mul_matrix, mul_scalar,
        mul_vector, qr, qr_gram_schmidt, qr_householder, rank, sub, transpose,
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

    #[test]
    fn rank_of_full_rank_matrix() {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 10]]; linearly independent rows, rank 3.
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let mut scratch = [0.0; 9];

        assert_eq!(rank(&a, 3, 3, &mut scratch), Ok(3));
    }

    #[test]
    fn rank_of_rank_deficient_matrix() {
        // [[1, 2, 3], [2, 4, 6], [0, 1, 1]]; row 1 is twice row 0, so rank is 2, not 3.
        let a = StaticStorage::new([1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 1.0, 1.0]);
        let mut scratch = [0.0; 9];

        assert_eq!(rank(&a, 3, 3, &mut scratch), Ok(2));
    }

    #[test]
    fn rank_of_non_square_matrix() {
        // [[1, 2, 3], [4, 5, 6]]; linearly independent rows, rank 2.
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut scratch = [0.0; 6];

        assert_eq!(rank(&a, 2, 3, &mut scratch), Ok(2));
    }

    #[test]
    fn rank_mismatched_scratch_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let mut scratch = [0.0; 3];

        assert_eq!(rank(&a, 2, 2, &mut scratch), Err(DimensionMismatch));
    }

    #[test]
    fn lu_of_known_matrix_with_no_pivoting_needed() {
        // [[4, 3], [6, 3]]; the diagonal is already nonzero, so no row swap is needed.
        let a = StaticStorage::new([4.0, 3.0, 6.0, 3.0]);
        let mut l = [0.0; 4];
        let mut u = [0.0; 4];

        assert_eq!(lu(&a, 2, 2, &mut l, &mut u), Ok(0));
        assert_eq!(l, [1.0, 0.0, 1.5, 1.0]);
        assert_eq!(u, [4.0, 3.0, 0.0, -1.5]);

        // l * u == p * a; p is the identity here, since no swap occurred.
        let mut lu_product = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(l),
            2,
            2,
            &StaticStorage::new(u),
            2,
            2,
            &mut lu_product,
        )
        .unwrap();
        assert_eq!(lu_product, [4.0, 3.0, 6.0, 3.0]);
    }

    #[test]
    fn lu_pivots_when_a_zero_sits_on_the_diagonal() {
        // [[0, 1], [1, 1]]; plain Gaussian elimination would divide by the zero pivot at
        // (0, 0), so this only succeeds because of partial pivoting.
        let a = StaticStorage::new([0.0, 1.0, 1.0, 1.0]);
        let mut l = [0.0; 4];
        let mut u = [0.0; 4];

        assert_eq!(lu_partial_pivot(&a, 2, 2, &mut l, &mut u), Ok(1));
        assert_eq!(l, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(u, [1.0, 1.0, 0.0, 1.0]);

        // l * u == p * a, where p swapped rows 0 and 1 of a: [[1, 1], [0, 1]].
        let mut lu_product = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(l),
            2,
            2,
            &StaticStorage::new(u),
            2,
            2,
            &mut lu_product,
        )
        .unwrap();
        assert_eq!(lu_product, [1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn lu_of_singular_matrix_leaves_a_zero_pivot_instead_of_erroring() {
        // [[0, 0], [0, 5]]; column 0 is entirely zero, so there's no pivot to swap in for
        // column 0 at all (singular along that column), not just a zero on the diagonal.
        let a = StaticStorage::new([0.0, 0.0, 0.0, 5.0]);
        let mut l = [0.0; 4];
        let mut u = [0.0; 4];

        assert_eq!(lu(&a, 2, 2, &mut l, &mut u), Ok(0));
        assert_eq!(l, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(u, [0.0, 0.0, 0.0, 5.0]);

        let mut lu_product = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(l),
            2,
            2,
            &StaticStorage::new(u),
            2,
            2,
            &mut lu_product,
        )
        .unwrap();
        assert_eq!(lu_product, [0.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn lu_of_non_square_matrix_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut l = [0.0; 9];
        let mut u = [0.0; 9];

        assert_eq!(lu(&a, 2, 3, &mut l, &mut u), Err(DimensionMismatch));
    }

    #[test]
    fn lu_mismatched_output_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
        let mut l = [0.0; 3];
        let mut u = [0.0; 4];

        assert_eq!(lu(&a, 2, 2, &mut l, &mut u), Err(DimensionMismatch));
    }

    #[test]
    fn qr_householder_of_square_matrix_q_is_orthogonal() {
        // [[3, 5], [4, 0]]; column 0 has norm 5, chosen so the reflection lands on nice
        // integers in r, even though q itself doesn't.
        let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_householder(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut q_t = [0.0; 4];
        transpose(&StaticStorage::new(q), 2, 2, &mut q_t).unwrap();
        let mut q_t_q = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(q_t),
            2,
            2,
            &StaticStorage::new(q),
            2,
            2,
            &mut q_t_q,
        )
        .unwrap();
        for (actual, expected) in q_t_q.iter().zip([1.0, 0.0, 0.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_householder_of_square_matrix_q_times_r_reconstructs_a() {
        // Same matrix as the orthogonality test above.
        let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_householder(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut qr = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(q),
            2,
            2,
            &StaticStorage::new(r),
            2,
            2,
            &mut qr,
        )
        .unwrap();
        for (actual, expected) in qr.iter().zip([3.0, 5.0, 4.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_householder_of_non_square_matrix_with_more_rows_than_columns() {
        // Row-major 3x2 matrix: [[12, 1], [6, 2], [-4, 3]]; column 0 has norm 14, chosen so
        // the first reflection lands on nice integers.
        let a = StaticStorage::new([12.0_f64, 1.0, 6.0, 2.0, -4.0, 3.0]);
        let mut q = [0.0; 9];
        let mut r = [0.0; 6];
        let mut scratch = [0.0; 3];

        assert_eq!(
            qr_householder(&a, 3, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut q_t = [0.0; 9];
        transpose(&StaticStorage::new(q), 3, 3, &mut q_t).unwrap();
        let mut q_t_q = [0.0; 9];
        mul_matrix(
            &StaticStorage::new(q_t),
            3,
            3,
            &StaticStorage::new(q),
            3,
            3,
            &mut q_t_q,
        )
        .unwrap();
        #[rustfmt::skip]
        let identity = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        for (actual, expected) in q_t_q.iter().zip(identity) {
            assert!((actual - expected).abs() < 1e-9);
        }

        let mut qr = [0.0; 6];
        mul_matrix(
            &StaticStorage::new(q),
            3,
            3,
            &StaticStorage::new(r),
            3,
            2,
            &mut qr,
        )
        .unwrap();
        let expected_a = [12.0, 1.0, 6.0, 2.0, -4.0, 3.0];
        for (actual, expected) in qr.iter().zip(expected_a) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_householder_of_matrix_with_more_columns_than_rows_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 6];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_householder(&a, 2, 3, &mut q, &mut r, &mut scratch),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn qr_householder_mismatched_scratch_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([3.0, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 3];

        assert_eq!(
            qr_householder(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn qr_gram_schmidt_of_square_matrix_q_is_orthogonal() {
        // Same matrix as the qr_householder orthogonality test above.
        let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_gram_schmidt(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut q_t = [0.0; 4];
        transpose(&StaticStorage::new(q), 2, 2, &mut q_t).unwrap();
        let mut q_t_q = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(q_t),
            2,
            2,
            &StaticStorage::new(q),
            2,
            2,
            &mut q_t_q,
        )
        .unwrap();
        for (actual, expected) in q_t_q.iter().zip([1.0, 0.0, 0.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_gram_schmidt_of_square_matrix_q_times_r_reconstructs_a() {
        // Same matrix as the orthogonality test above.
        let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_gram_schmidt(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut qr_product = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(q),
            2,
            2,
            &StaticStorage::new(r),
            2,
            2,
            &mut qr_product,
        )
        .unwrap();
        for (actual, expected) in qr_product.iter().zip([3.0, 5.0, 4.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_gram_schmidt_of_non_square_matrix_with_more_rows_than_columns() {
        // Same matrix as the qr_householder non-square test above.
        let a = StaticStorage::new([12.0_f64, 1.0, 6.0, 2.0, -4.0, 3.0]);
        let mut q = [0.0; 6];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 3];

        assert_eq!(
            qr_gram_schmidt(&a, 3, 2, &mut q, &mut r, &mut scratch),
            Ok(())
        );

        let mut q_t = [0.0; 6];
        transpose(&StaticStorage::new(q), 3, 2, &mut q_t).unwrap();
        let mut q_t_q = [0.0; 4];
        mul_matrix(
            &StaticStorage::new(q_t),
            2,
            3,
            &StaticStorage::new(q),
            3,
            2,
            &mut q_t_q,
        )
        .unwrap();
        for (actual, expected) in q_t_q.iter().zip([1.0, 0.0, 0.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }

        let mut qr_product = [0.0; 6];
        mul_matrix(
            &StaticStorage::new(q),
            3,
            2,
            &StaticStorage::new(r),
            2,
            2,
            &mut qr_product,
        )
        .unwrap();
        let expected_a = [12.0, 1.0, 6.0, 2.0, -4.0, 3.0];
        for (actual, expected) in qr_product.iter().zip(expected_a) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn qr_gram_schmidt_of_matrix_with_more_columns_than_rows_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 9];
        let mut scratch = [0.0; 2];

        assert_eq!(
            qr_gram_schmidt(&a, 2, 3, &mut q, &mut r, &mut scratch),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn qr_gram_schmidt_mismatched_scratch_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([3.0, 5.0, 4.0, 0.0]);
        let mut q = [0.0; 4];
        let mut r = [0.0; 4];
        let mut scratch = [0.0; 3];

        assert_eq!(
            qr_gram_schmidt(&a, 2, 2, &mut q, &mut r, &mut scratch),
            Err(DimensionMismatch)
        );
    }

    #[test]
    fn qr_matches_qr_householder() {
        let a = StaticStorage::new([3.0_f64, 5.0, 4.0, 0.0]);

        let mut q_high_level = [0.0; 4];
        let mut r_high_level = [0.0; 4];
        let mut scratch_high_level = [0.0; 2];
        assert_eq!(
            qr(
                &a,
                2,
                2,
                &mut q_high_level,
                &mut r_high_level,
                &mut scratch_high_level
            ),
            Ok(())
        );

        let mut q_explicit = [0.0; 4];
        let mut r_explicit = [0.0; 4];
        let mut scratch_explicit = [0.0; 2];
        assert_eq!(
            qr_householder(
                &a,
                2,
                2,
                &mut q_explicit,
                &mut r_explicit,
                &mut scratch_explicit
            ),
            Ok(())
        );

        assert_eq!(q_high_level, q_explicit);
        assert_eq!(r_high_level, r_explicit);
    }
}

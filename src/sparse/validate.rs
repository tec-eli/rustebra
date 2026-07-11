use crate::scalar::Scalar;

use super::{CscMatrix, CsrMatrix};

/// Error returned by [`validate_csr`] and [`validate_csc`] when a matrix violates the
/// canonical compressed-format invariants.
///
/// "Pointer array" means `row_ptr` for CSR and `col_ptr` for CSC; "index array" means
/// `col_indices` for CSR and `row_indices` for CSC.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, ValidateError, validate_csr};
///
/// // A stored zero is constructible, but not canonical.
/// let zero = CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![0.0_f64]).unwrap();
/// assert_eq!(validate_csr(&zero), Err(ValidateError::ExplicitZero));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidateError {
    /// The index and value arrays have different lengths.
    LengthMismatch,
    /// The pointer array does not have exactly `rows + 1` (CSR) or `cols + 1` (CSC) entries.
    PtrLengthMismatch,
    /// The pointer array does not start at `0`, is not non-decreasing, or its last entry
    /// does not equal `nnz`.
    PtrInvalid,
    /// A stored index is >= the dimension it indexes into (`cols` for CSR, `rows` for CSC).
    IndexOutOfBounds,
    /// Indices within a row (CSR) or column (CSC) are not in ascending order.
    UnsortedIndices,
    /// The same index appears more than once within a row (CSR) or column (CSC).
    DuplicateIndex,
    /// A stored value is exactly zero.
    ExplicitZero,
}

/// Checks that `m` satisfies the canonical CSR invariants, returning the first violation found.
///
/// [`CsrMatrix::new`] already rejects structurally malformed input (array lengths, `row_ptr`
/// monotonicity, index bounds), so this function is chiefly useful for the invariants the
/// constructor deliberately does **not** enforce: column indices sorted ascending within each
/// row, no duplicate column index within a row, and no stored entry equal to exactly zero.
/// The structural invariants are re-checked anyway as defense in depth.
///
/// The zero check is exact; use [`prune_csr`] with a caller-supplied tolerance to remove
/// near-zero entries.
///
/// Checks run structural-first: [`ValidateError::LengthMismatch`], then
/// [`ValidateError::PtrLengthMismatch`], [`ValidateError::PtrInvalid`],
/// [`ValidateError::IndexOutOfBounds`], then per-row ordering
/// ([`ValidateError::DuplicateIndex`] / [`ValidateError::UnsortedIndices`]), and finally
/// [`ValidateError::ExplicitZero`].
///
/// [`prune_csr`]: crate::sparse::prune_csr
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, ValidateError, validate_csr};
///
/// // Canonical: sorted columns, no duplicates, no stored zeros.
/// let ok = CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
/// assert_eq!(validate_csr(&ok), Ok(()));
///
/// // Row 0 has columns [2, 0] — constructible, but not canonical.
/// let unsorted =
///     CsrMatrix::new(1, 3, vec![0, 2], vec![2, 0], vec![1.0_f64, 2.0]).unwrap();
/// assert_eq!(validate_csr(&unsorted), Err(ValidateError::UnsortedIndices));
///
/// // A stored zero is constructible, but not canonical.
/// let zero = CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![0.0_f64]).unwrap();
/// assert_eq!(validate_csr(&zero), Err(ValidateError::ExplicitZero));
/// ```
pub fn validate_csr<T: Scalar>(m: &CsrMatrix<T>) -> Result<(), ValidateError> {
    validate_compressed(m.rows(), m.cols(), m.row_ptr(), m.col_indices(), m.values())
}

/// Checks that `m` satisfies the canonical CSC invariants, returning the first violation found.
///
/// This is the CSC counterpart to [`validate_csr`]: same checks, same order, with `col_ptr`
/// as the pointer array and `row_indices` (sorted ascending within each column, no
/// duplicates, bounded by `rows`) as the index array.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CscMatrix, ValidateError, validate_csc};
///
/// // Canonical: sorted rows within each column, no duplicates, no stored zeros.
/// let ok = CscMatrix::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
/// assert_eq!(validate_csc(&ok), Ok(()));
///
/// // Column 0 stores row 1 twice — constructible, but not canonical.
/// let dup = CscMatrix::new(3, 1, vec![0, 2], vec![1, 1], vec![1.0_f64, 2.0]).unwrap();
/// assert_eq!(validate_csc(&dup), Err(ValidateError::DuplicateIndex));
/// ```
pub fn validate_csc<T: Scalar>(m: &CscMatrix<T>) -> Result<(), ValidateError> {
    validate_compressed(m.cols(), m.rows(), m.col_ptr(), m.row_indices(), m.values())
}

/// Shared CSR/CSC validation over the compressed-format triplet. `outer` is the compressed
/// dimension (rows for CSR, cols for CSC); `inner` bounds the stored indices.
fn validate_compressed<T: Scalar>(
    outer: usize,
    inner: usize,
    ptr: &[u32],
    indices: &[u32],
    values: &[T],
) -> Result<(), ValidateError> {
    if indices.len() != values.len() {
        return Err(ValidateError::LengthMismatch);
    }
    if ptr.len() != outer + 1 {
        return Err(ValidateError::PtrLengthMismatch);
    }
    let nnz = indices.len();
    if ptr[0] != 0 || ptr[outer] as usize != nnz {
        return Err(ValidateError::PtrInvalid);
    }
    for i in 1..=outer {
        if ptr[i] < ptr[i - 1] {
            return Err(ValidateError::PtrInvalid);
        }
    }
    for &idx in indices {
        if idx as usize >= inner {
            return Err(ValidateError::IndexOutOfBounds);
        }
    }
    for s in 0..outer {
        for k in ptr[s] as usize + 1..ptr[s + 1] as usize {
            if indices[k] == indices[k - 1] {
                return Err(ValidateError::DuplicateIndex);
            }
            if indices[k] < indices[k - 1] {
                return Err(ValidateError::UnsortedIndices);
            }
        }
    }
    for v in values {
        if *v == T::zero() {
            return Err(ValidateError::ExplicitZero);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ValidateError, validate_csr};
    use crate::sparse::CsrMatrix;
    use alloc::vec;

    #[test]
    fn canonical_matrix_passes() {
        let m =
            CsrMatrix::new(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
        assert_eq!(validate_csr(&m), Ok(()));
    }

    #[test]
    fn empty_matrix_passes() {
        let m = CsrMatrix::<f64>::new(3, 4, vec![0, 0, 0, 0], vec![], vec![]).unwrap();
        assert_eq!(validate_csr(&m), Ok(()));
    }

    #[test]
    fn length_mismatch_detected() {
        let m = CsrMatrix::new_raw(1, 2, vec![0, 2], vec![0, 1], vec![1.0_f64]);
        assert_eq!(validate_csr(&m), Err(ValidateError::LengthMismatch));
    }

    #[test]
    fn ptr_length_mismatch_detected() {
        let m = CsrMatrix::new_raw(2, 2, vec![0, 1], vec![1], vec![1.0_f64]);
        assert_eq!(validate_csr(&m), Err(ValidateError::PtrLengthMismatch));
    }

    #[test]
    fn ptr_not_starting_at_zero_detected() {
        let m = CsrMatrix::new_raw(1, 2, vec![1, 1], vec![0], vec![1.0_f64]);
        assert_eq!(validate_csr(&m), Err(ValidateError::PtrInvalid));
    }

    #[test]
    fn ptr_not_monotone_detected() {
        let m = CsrMatrix::new_raw(2, 3, vec![0, 2, 1], vec![0, 1], vec![1.0_f64, 2.0]);
        assert_eq!(validate_csr(&m), Err(ValidateError::PtrInvalid));
    }

    #[test]
    fn ptr_last_entry_mismatch_detected() {
        let m = CsrMatrix::new_raw(1, 2, vec![0, 5], vec![0, 1], vec![1.0_f64, 2.0]);
        assert_eq!(validate_csr(&m), Err(ValidateError::PtrInvalid));
    }

    #[test]
    fn index_out_of_bounds_detected() {
        let m = CsrMatrix::new_raw(1, 2, vec![0, 2], vec![0, 3], vec![1.0_f64, 2.0]);
        assert_eq!(validate_csr(&m), Err(ValidateError::IndexOutOfBounds));
    }

    #[test]
    fn unsorted_indices_detected() {
        let m = CsrMatrix::new(1, 3, vec![0, 2], vec![2, 0], vec![1.0_f64, 2.0]).unwrap();
        assert_eq!(validate_csr(&m), Err(ValidateError::UnsortedIndices));
    }

    #[test]
    fn duplicate_index_detected() {
        let m = CsrMatrix::new(1, 3, vec![0, 2], vec![1, 1], vec![1.0_f64, 2.0]).unwrap();
        assert_eq!(validate_csr(&m), Err(ValidateError::DuplicateIndex));
    }

    #[test]
    fn duplicates_do_not_leak_across_row_boundaries() {
        // Row 0 ends with column 1 and row 1 starts with column 1: legal, not a duplicate.
        let m =
            CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
        assert_eq!(validate_csr(&m), Ok(()));
    }

    #[test]
    fn explicit_zero_detected() {
        let m = CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![1.0_f64, 0.0]).unwrap();
        assert_eq!(validate_csr(&m), Err(ValidateError::ExplicitZero));
    }

    #[test]
    fn nan_entry_is_not_reported_as_zero() {
        let m = CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![f64::NAN]).unwrap();
        assert_eq!(validate_csr(&m), Ok(()));
    }
}

#[cfg(test)]
mod tests_csc {
    use super::{ValidateError, validate_csc};
    use crate::sparse::CscMatrix;
    use alloc::vec;

    #[test]
    fn csc_canonical_matrix_passes() {
        let m =
            CscMatrix::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
        assert_eq!(validate_csc(&m), Ok(()));
    }

    #[test]
    fn csc_length_mismatch_detected() {
        let m = CscMatrix::new_raw(2, 1, vec![0, 2], vec![0, 1], vec![1.0_f64]);
        assert_eq!(validate_csc(&m), Err(ValidateError::LengthMismatch));
    }

    #[test]
    fn csc_ptr_length_mismatch_detected() {
        let m = CscMatrix::new_raw(2, 2, vec![0, 1], vec![1], vec![1.0_f64]);
        assert_eq!(validate_csc(&m), Err(ValidateError::PtrLengthMismatch));
    }

    #[test]
    fn csc_ptr_invalid_detected() {
        let m = CscMatrix::new_raw(3, 2, vec![0, 2, 1], vec![0, 1], vec![1.0_f64, 2.0]);
        assert_eq!(validate_csc(&m), Err(ValidateError::PtrInvalid));
    }

    #[test]
    fn csc_index_out_of_bounds_detected() {
        let m = CscMatrix::new_raw(2, 1, vec![0, 2], vec![0, 3], vec![1.0_f64, 2.0]);
        assert_eq!(validate_csc(&m), Err(ValidateError::IndexOutOfBounds));
    }

    #[test]
    fn csc_unsorted_indices_detected() {
        let m = CscMatrix::new(3, 1, vec![0, 2], vec![2, 0], vec![1.0_f64, 2.0]).unwrap();
        assert_eq!(validate_csc(&m), Err(ValidateError::UnsortedIndices));
    }

    #[test]
    fn csc_duplicate_index_detected() {
        let m = CscMatrix::new(3, 1, vec![0, 2], vec![1, 1], vec![1.0_f64, 2.0]).unwrap();
        assert_eq!(validate_csc(&m), Err(ValidateError::DuplicateIndex));
    }

    #[test]
    fn csc_explicit_zero_detected() {
        let m = CscMatrix::new(2, 1, vec![0, 2], vec![0, 1], vec![1.0_f64, 0.0]).unwrap();
        assert_eq!(validate_csc(&m), Err(ValidateError::ExplicitZero));
    }
}

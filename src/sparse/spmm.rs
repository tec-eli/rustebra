use alloc::vec;
use alloc::vec::Vec;

use crate::scalar::Scalar;

use super::CsrMatrix;
use super::add::DimensionMismatch;
use super::sorted_csr::SortedCsrMatrix;

/// Multiplies two CSR sparse matrices (`a × b`), returning the product as a
/// [`SortedCsrMatrix`].
///
/// `a` is `m × k` and `b` is `k × n`; the result is `m × n`.  The algorithm uses a
/// dense row accumulator of size `n` so the working memory is `O(n)` per output row.
/// Output column indices within each row are guaranteed to be in ascending order, which
/// is why the result is a [`SortedCsrMatrix`] rather than a plain `CsrMatrix`.
///
/// # Errors
///
/// Returns `Err(DimensionMismatch)` when `a.cols() != b.rows()`.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, spmm_csr};
///
/// // [1  0]   [1  2]   [1  2]
/// // [0  2] × [3  4] = [6  8]
/// let a = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 2.0]).unwrap();
/// let b = CsrMatrix::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1],
///                        vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
/// let c = spmm_csr(&a, &b).unwrap();
/// assert_eq!(c.row_ptr(),     &[0, 2, 4]);
/// assert_eq!(c.col_indices(), &[0, 1, 0, 1]);
/// assert_eq!(c.values(),      &[1.0, 2.0, 6.0, 8.0]);
/// ```
pub fn spmm_csr<T: Scalar>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
) -> Result<SortedCsrMatrix<T>, DimensionMismatch> {
    if a.cols() != b.rows() {
        return Err(DimensionMismatch);
    }
    let m = a.rows();
    let n = b.cols();

    let mut out_row_ptr = vec![0usize; m + 1];
    let mut out_col: Vec<usize> = Vec::new();
    let mut out_val: Vec<T> = Vec::new();

    // Dense accumulator: `dense[j]` holds the accumulated value for column j in the
    // current output row.  `touched` tracks which columns are non-zero so we can reset
    // only those entries after each row, keeping the overall work proportional to nnz.
    let mut dense = vec![T::zero(); n];
    let mut touched: Vec<usize> = Vec::new();
    // `in_touched[j]` is true when column j has been placed in `touched`.
    let mut in_touched = vec![false; n];

    let a_row_ptr = a.row_ptr();
    let a_col = a.col_indices();
    let a_val = a.values();
    let b_row_ptr = b.row_ptr();
    let b_col = b.col_indices();
    let b_val = b.values();

    for r in 0..m {
        for ka in a_row_ptr[r]..a_row_ptr[r + 1] {
            let k = a_col[ka];
            let av = a_val[ka];
            for kb in b_row_ptr[k]..b_row_ptr[k + 1] {
                let j = b_col[kb];
                let bv = b_val[kb];
                let prev = dense[j];
                dense[j] = prev.add(av.mul(bv));
                if !in_touched[j] {
                    touched.push(j);
                    in_touched[j] = true;
                }
            }
        }
        // Emit output entries sorted by column index.
        touched.sort_unstable();
        for &col in &touched {
            out_col.push(col);
            out_val.push(dense[col]);
            dense[col] = T::zero();
            in_touched[col] = false;
        }
        touched.clear();
        out_row_ptr[r + 1] = out_col.len();
    }

    Ok(SortedCsrMatrix::from_sorted_unchecked(CsrMatrix::new_raw(
        m,
        n,
        out_row_ptr,
        out_col,
        out_val,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s2_spmm_exact_cancellation_zeros() {
        let a = CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![1.0_f64, -1.0]).unwrap();
        let b = CsrMatrix::new(2, 2, vec![0, 1, 1], vec![0], vec![1.0_f64]).unwrap();
        let c = spmm_csr(&a, &b).unwrap();
        assert_eq!(c.nnz(), 0);
    }
}

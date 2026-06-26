use alloc::vec;
use alloc::vec::Vec;

use crate::scalar::Scalar;

use super::sorted_csr::SortedCsrMatrix;
use super::{CooMatrix, CscMatrix, CsrMatrix};

/// Converts a COO sparse matrix to sorted CSR format.
///
/// Triplets that share the same `(row, col)` position are **summed** together, so the
/// resulting matrix contains at most one entry per unique position.  Within each row,
/// column indices are in ascending order — a side-effect of the sort pass — which is why
/// this function returns a [`SortedCsrMatrix`] rather than a plain `CsrMatrix`.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CooMatrix, SortedCsrMatrix, coo_to_csr};
///
/// // 2×2 matrix; position (0, 0) appears twice and the values are summed.
/// let coo = CooMatrix::new(
///     2, 2,
///     vec![0, 0, 1],
///     vec![0, 0, 1],
///     vec![3.0_f64, 4.0, 5.0],
/// )
/// .unwrap();
///
/// let csr: SortedCsrMatrix<f64> = coo_to_csr(coo);
/// assert_eq!(csr.nnz(), 2);
/// assert_eq!(csr.row_ptr(),    &[0, 1, 2]);
/// assert_eq!(csr.col_indices(), &[0, 1]);
/// assert_eq!(csr.values(),      &[7.0, 5.0]); // 3 + 4 = 7
/// ```
pub fn coo_to_csr<T: Scalar>(coo: CooMatrix<T>) -> SortedCsrMatrix<T> {
    let (rows, cols, coo_row, coo_col, coo_val) = coo.into_raw_parts();
    let nnz_in = coo_val.len();

    // Sort entry positions by (row, col) so duplicates become adjacent.
    let mut order: Vec<usize> = (0..nnz_in).collect();
    order.sort_by_key(|&i| (coo_row[i], coo_col[i]));

    let mut out_row: Vec<usize> = Vec::new();
    let mut out_col: Vec<usize> = Vec::new();
    let mut out_val: Vec<T> = Vec::new();

    for &i in &order {
        let r = coo_row[i];
        let c = coo_col[i];
        let v = coo_val[i];

        let is_dup = out_row.last().copied() == Some(r) && out_col.last().copied() == Some(c);
        if is_dup {
            if let Some(last) = out_val.last_mut() {
                *last = last.add(v);
            }
        } else {
            out_row.push(r);
            out_col.push(c);
            out_val.push(v);
        }
    }

    // Prefix-sum of per-row entry counts to build row_ptr.
    let mut row_ptr = vec![0usize; rows + 1];
    for &r in &out_row {
        row_ptr[r + 1] += 1;
    }
    for i in 1..=rows {
        row_ptr[i] += row_ptr[i - 1];
    }

    SortedCsrMatrix::from_sorted_unchecked(CsrMatrix::new_raw(
        rows, cols, row_ptr, out_col, out_val,
    ))
}

/// Converts a CSR sparse matrix to COO format by expanding the row-pointer array into
/// explicit per-entry row indices.
///
/// The output preserves the column indices and values in the same order they appear in
/// the source `CsrMatrix`.  If the source was produced by [`coo_to_csr`] (which deduplicates
/// and sorts within rows), the output `CooMatrix` will have no duplicate `(row, col)`
/// entries and ascending column indices within each row — but the general contract does
/// not guarantee uniqueness.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, csr_to_coo};
///
/// // 3×3 identity in CSR, converted to COO.
/// let csr = CsrMatrix::new(
///     3, 3,
///     vec![0, 1, 2, 3],
///     vec![0, 1, 2],
///     vec![1.0_f64, 1.0, 1.0],
/// )
/// .unwrap();
///
/// let coo = csr_to_coo(csr);
/// assert_eq!(coo.nnz(), 3);
/// assert_eq!(coo.row_indices(), &[0, 1, 2]);
/// assert_eq!(coo.col_indices(), &[0, 1, 2]);
/// assert_eq!(coo.values(),      &[1.0, 1.0, 1.0]);
/// ```
pub fn csr_to_coo<T>(csr: CsrMatrix<T>) -> CooMatrix<T> {
    let (rows, cols, row_ptr, col_indices, values) = csr.into_raw_parts();
    let nnz = values.len();

    let mut row_indices = Vec::with_capacity(nnz);
    for r in 0..rows {
        let count = row_ptr[r + 1] - row_ptr[r];
        for _ in 0..count {
            row_indices.push(r);
        }
    }

    CooMatrix::new_raw(rows, cols, row_indices, col_indices, values)
}

/// Converts a CSR sparse matrix to CSC format.
///
/// All entries are preserved; the output is sorted by column (then row within each column).
/// The `values` order in the output reflects column-major traversal order.
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CsrMatrix, csr_to_csc};
///
/// // 3×3 identity in CSR → CSC.
/// let csr = CsrMatrix::new(
///     3, 3,
///     vec![0, 1, 2, 3],
///     vec![0, 1, 2],
///     vec![1.0_f64, 1.0, 1.0],
/// )
/// .unwrap();
///
/// let csc = csr_to_csc(csr);
/// assert_eq!(csc.col_ptr(),     &[0, 1, 2, 3]);
/// assert_eq!(csc.row_indices(), &[0, 1, 2]);
/// assert_eq!(csc.values(),      &[1.0, 1.0, 1.0]);
/// ```
pub fn csr_to_csc<T: Scalar>(m: CsrMatrix<T>) -> CscMatrix<T> {
    let (rows, cols, row_ptr, col_indices, values) = m.into_raw_parts();
    let nnz = values.len();

    // Collect (col, row, val) triples from the CSR layout.
    let mut triples: Vec<(usize, usize, T)> = Vec::with_capacity(nnz);
    for r in 0..rows {
        for k in row_ptr[r]..row_ptr[r + 1] {
            triples.push((col_indices[k], r, values[k]));
        }
    }
    // Sort by (col, row) so entries are in column-major order.
    triples.sort_by_key(|&(c, r, _)| (c, r));

    // Build col_ptr via prefix-sum of per-column counts.
    let mut col_ptr = vec![0usize; cols + 1];
    for &(c, _, _) in &triples {
        col_ptr[c + 1] += 1;
    }
    for i in 1..=cols {
        col_ptr[i] += col_ptr[i - 1];
    }

    let mut out_row: Vec<usize> = Vec::with_capacity(nnz);
    let mut out_val: Vec<T> = Vec::with_capacity(nnz);
    for (_, r, v) in triples {
        out_row.push(r);
        out_val.push(v);
    }

    CscMatrix::new_raw(rows, cols, col_ptr, out_row, out_val)
}

/// Converts a CSC sparse matrix to CSR format.
///
/// All entries are preserved; the output is sorted by row (then column within each row).
///
/// # Examples
///
/// ```
/// use rustebra::sparse::{CscMatrix, csc_to_csr};
///
/// // 3×3 identity in CSC → CSR.
/// let csc = CscMatrix::new(
///     3, 3,
///     vec![0, 1, 2, 3],
///     vec![0, 1, 2],
///     vec![1.0_f64, 1.0, 1.0],
/// )
/// .unwrap();
///
/// let csr = csc_to_csr(csc);
/// assert_eq!(csr.row_ptr(),     &[0, 1, 2, 3]);
/// assert_eq!(csr.col_indices(), &[0, 1, 2]);
/// assert_eq!(csr.values(),      &[1.0, 1.0, 1.0]);
/// ```
pub fn csc_to_csr<T: Scalar>(m: CscMatrix<T>) -> CsrMatrix<T> {
    let (rows, cols, col_ptr, row_indices, values) = m.into_raw_parts();
    let nnz = values.len();

    // Collect (row, col, val) triples from the CSC layout.
    let mut triples: Vec<(usize, usize, T)> = Vec::with_capacity(nnz);
    for c in 0..cols {
        for k in col_ptr[c]..col_ptr[c + 1] {
            triples.push((row_indices[k], c, values[k]));
        }
    }
    // Sort by (row, col) so entries are in row-major order.
    triples.sort_by_key(|&(r, c, _)| (r, c));

    // Build row_ptr via prefix-sum of per-row counts.
    let mut row_ptr = vec![0usize; rows + 1];
    for &(r, _, _) in &triples {
        row_ptr[r + 1] += 1;
    }
    for i in 1..=rows {
        row_ptr[i] += row_ptr[i - 1];
    }

    let mut out_col: Vec<usize> = Vec::with_capacity(nnz);
    let mut out_val: Vec<T> = Vec::with_capacity(nnz);
    for (_, c, v) in triples {
        out_col.push(c);
        out_val.push(v);
    }

    CsrMatrix::new_raw(rows, cols, row_ptr, out_col, out_val)
}

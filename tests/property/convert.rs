#![cfg(feature = "alloc")]

use proptest::prelude::*;
use rustebra::sparse::{CooMatrix, CsrMatrix, coo_to_csr, csc_to_csr, csr_to_coo, csr_to_csc};
use std::collections::HashMap;

proptest! {
    /// Property test: COO → CSR → COO round-trip preserves logical matrix values.
    ///
    /// Generates random sparse matrices, converts COO → CSR → COO, and verifies that:
    /// 1. Each unique (row, col) position in the output has the correct summed value
    /// 2. Dimensions are preserved
    /// 3. No spurious entries appear
    #[test]
    fn coo_csr_coo_roundtrip_preserves_values(
        rows in 1usize..=10,
        cols in 1usize..=10,
        entries in prop::collection::vec(
            (0usize..10, 0usize..10, -100.0..100.0f64),
            0..50,
        ),
    ) {
        // Filter entries to be in bounds
        let valid_entries: Vec<_> = entries
            .into_iter()
            .filter(|&(r, c, _)| r < rows && c < cols)
            .collect();

        prop_assume!(!valid_entries.is_empty());

        let row_idx: Vec<usize> = valid_entries.iter().map(|e| e.0).collect();
        let col_idx: Vec<usize> = valid_entries.iter().map(|e| e.1).collect();
        let values: Vec<_> = valid_entries.iter().map(|e| e.2).collect();
        let row_idx_u32: Vec<u32> = row_idx.iter().map(|&r| r as u32).collect();
        let col_idx_u32: Vec<u32> = col_idx.iter().map(|&c| c as u32).collect();

        // Create COO matrix
        let coo = CooMatrix::new(rows, cols, row_idx_u32, col_idx_u32, values.clone())
            .expect("entries should be in bounds");

        // Convert COO → CSR → COO
        let sorted_csr = coo_to_csr(coo).expect("dimensions fit within test-generated bounds");
        let csr: CsrMatrix<f64> = sorted_csr.into();
        let coo_roundtrip = csr_to_coo(csr).expect("dimensions fit within test-generated bounds");

        // Verify dimensions are preserved
        prop_assert_eq!(coo_roundtrip.rows(), rows);
        prop_assert_eq!(coo_roundtrip.cols(), cols);

        // Build a reference map of (row, col) → summed value from original entries
        let mut expected: HashMap<(usize, usize), f64> = HashMap::new();
        for i in 0..row_idx.len() {
            *expected.entry((row_idx[i], col_idx[i])).or_insert(0.0) += values[i];
        }

        // Build a map from output entries
        let mut actual: HashMap<(usize, usize), f64> = HashMap::new();
        for i in 0..coo_roundtrip.nnz() {
            let r = coo_roundtrip.row_indices()[i] as usize;
            let c = coo_roundtrip.col_indices()[i] as usize;
            let v = coo_roundtrip.values()[i];
            *actual.entry((r, c)).or_insert(0.0) += v;
        }

        // Verify every expected entry matches
        for (&(r, c), &expected_val) in &expected {
            let actual_val = actual.get(&(r, c)).copied().unwrap_or(0.0);
            prop_assert!(
                (actual_val - expected_val).abs() < 1e-10,
                "Value mismatch at ({}, {}): expected {}, got {}",
                r,
                c,
                expected_val,
                actual_val
            );
        }

        // Verify no spurious entries
        prop_assert_eq!(
            actual.len(),
            expected.len(),
            "Output has wrong number of entries"
        );
    }

    /// Property test: CSR → CSC → CSR round-trip preserves all values.
    ///
    /// Builds a valid, duplicate-free `CsrMatrix` via `coo_to_csr` (whose output is already
    /// sorted row-major with ascending column indices within each row), then converts it
    /// CSR → CSC → CSR and verifies that:
    /// 1. Dimensions are preserved
    /// 2. The row-pointer, column-index, and value arrays are unchanged, since both
    ///    conversions only reorder entries (by (col, row) then back by (row, col)) without
    ///    performing any arithmetic on the stored values.
    #[test]
    fn csr_csc_csr_roundtrip_preserves_values(
        rows in 1usize..=10,
        cols in 1usize..=10,
        entries in prop::collection::vec(
            (0usize..10, 0usize..10, -100.0..100.0f64),
            0..50,
        ),
    ) {
        // Filter entries to be in bounds
        let valid_entries: Vec<_> = entries
            .into_iter()
            .filter(|&(r, c, _)| r < rows && c < cols)
            .collect();

        prop_assume!(!valid_entries.is_empty());

        let row_idx_u32: Vec<u32> = valid_entries.iter().map(|e| e.0 as u32).collect();
        let col_idx_u32: Vec<u32> = valid_entries.iter().map(|e| e.1 as u32).collect();
        let values: Vec<f64> = valid_entries.iter().map(|e| e.2).collect();

        // Build a valid, already-sorted CSR matrix via the COO → CSR conversion.
        let coo = CooMatrix::new(rows, cols, row_idx_u32, col_idx_u32, values)
            .expect("entries should be in bounds");
        let sorted_csr = coo_to_csr(coo).expect("dimensions fit within test-generated bounds");
        let csr: CsrMatrix<f64> = sorted_csr.into();

        // Capture the expected layout before `csr` is consumed by the conversion below.
        let expected_rows = csr.rows();
        let expected_cols = csr.cols();
        let expected_row_ptr = csr.row_ptr().to_vec();
        let expected_col_indices = csr.col_indices().to_vec();
        let expected_values = csr.values().to_vec();

        // Convert CSR → CSC → CSR
        let csc = csr_to_csc(csr).expect("dimensions fit within test-generated bounds");
        let csr_roundtrip =
            csc_to_csr(csc.into()).expect("dimensions fit within test-generated bounds");

        prop_assert_eq!(csr_roundtrip.rows(), expected_rows);
        prop_assert_eq!(csr_roundtrip.cols(), expected_cols);
        prop_assert_eq!(csr_roundtrip.row_ptr(), expected_row_ptr.as_slice());
        prop_assert_eq!(csr_roundtrip.col_indices(), expected_col_indices.as_slice());
        prop_assert_eq!(csr_roundtrip.values(), expected_values.as_slice());
    }
}

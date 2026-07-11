//! Fixed-matrix edge cases for the documented NaN/Inf policy: single dense operations
//! propagate non-finite values per IEEE 754 rather than erroring or panicking, and
//! `prune_csr` treats a non-finite tolerance as an always-false comparison rather than a
//! panic. Krylov's detect-and-error behavior is covered separately in `krylov.rs`.

use rustebra::matrix::StaticMatrix;
use rustebra::storage::Storage;
use rustebra::vector::StaticVector;

#[test]
fn add_propagates_nan_and_infinity_to_the_affected_entries_only() {
    let a = StaticMatrix::new([[1.0_f64, f64::NAN], [f64::INFINITY, 4.0]]);
    let b = StaticMatrix::new([[10.0_f64, 20.0], [30.0, 40.0]]);

    let sum = a.add(&b);

    assert_eq!(*sum.get(0).unwrap(), 11.0);
    assert!(sum.get(1).unwrap().is_nan());
    assert_eq!(*sum.get(2).unwrap(), f64::INFINITY);
    assert_eq!(*sum.get(3).unwrap(), 44.0);
}

#[test]
fn sub_propagates_nan_and_infinity_to_the_affected_entries_only() {
    let a = StaticMatrix::new([[1.0_f64, f64::NAN], [f64::INFINITY, 4.0]]);
    let b = StaticMatrix::new([[10.0_f64, 20.0], [30.0, 40.0]]);

    let diff = a.sub(&b);

    assert_eq!(*diff.get(0).unwrap(), -9.0);
    assert!(diff.get(1).unwrap().is_nan());
    assert_eq!(*diff.get(2).unwrap(), f64::INFINITY);
    assert_eq!(*diff.get(3).unwrap(), -36.0);
}

#[test]
fn mul_scalar_propagates_nan_and_infinity() {
    let a = StaticMatrix::new([[1.0_f64, f64::NAN], [f64::NEG_INFINITY, 4.0]]);

    let scaled = a.mul_scalar(2.0);

    assert_eq!(*scaled.get(0).unwrap(), 2.0);
    assert!(scaled.get(1).unwrap().is_nan());
    assert_eq!(*scaled.get(2).unwrap(), f64::NEG_INFINITY);
    assert_eq!(*scaled.get(3).unwrap(), 8.0);

    // Multiplying by zero collapses a NaN to NaN and an infinity to NaN (IEEE 754
    // `inf * 0 = NaN`), not to a finite zero.
    let zeroed = a.mul_scalar(0.0);
    assert!(zeroed.get(1).unwrap().is_nan());
    assert!(zeroed.get(2).unwrap().is_nan());
}

#[test]
fn mul_vector_propagates_nan_through_the_dot_product_sum() {
    // Every output entry that sums a NaN contribution becomes NaN; entries whose row never
    // touches the poisoned column stay finite.
    let a = StaticMatrix::new([[1.0_f64, f64::NAN], [2.0, 3.0]]);
    let v = StaticVector::new([1.0, 1.0]);

    let result = a.mul_vector(&v);

    assert!(result.get(0).unwrap().is_nan());
    assert_eq!(*result.get(1).unwrap(), 5.0);
}

#[test]
fn mul_matrix_propagates_nan_through_the_accumulated_sum() {
    let a = StaticMatrix::new([[1.0_f64, f64::NAN], [2.0, 3.0]]);
    let b = StaticMatrix::new([[1.0_f64, 0.0], [0.0, 1.0]]);

    let product = a.mul_matrix(&b);

    assert!(product.get(1).unwrap().is_nan());
    assert_eq!(*product.get(2).unwrap(), 2.0);
    assert_eq!(*product.get(3).unwrap(), 3.0);
}

#[test]
fn transpose_moves_nan_and_infinity_without_altering_them() {
    let a = StaticMatrix::new([[1.0_f64, f64::NAN, f64::INFINITY]]);

    let transposed = a.transpose();

    assert_eq!(*transposed.get(0).unwrap(), 1.0);
    assert!(transposed.get(1).unwrap().is_nan());
    assert_eq!(*transposed.get(2).unwrap(), f64::INFINITY);
}

#[cfg(feature = "alloc")]
mod prune_csr_nan_threshold {
    use rustebra::sparse::{CsrMatrix, prune_csr};

    // `prune_csr` keeps an entry `v` when it fails the "negligible" test
    // `v <= tolerance && v >= -tolerance`. A NaN tolerance makes every comparison against it
    // false per IEEE 754 (NaN compares unordered with everything, including itself), so the
    // negligibility test can never be satisfied: no entry is ever pruned, not even exact
    // zeros. This documents that behavior rather than treating it as an error case, matching
    // `prune_csr`'s existing "NaN entries are always kept" contract extended to the
    // threshold argument itself.
    #[test]
    fn nan_tolerance_prunes_nothing_not_even_exact_zeros() {
        let m = CsrMatrix::new(1, 3, vec![0, 3], vec![0, 1, 2], vec![0.0_f64, 5.0, 0.0]).unwrap();

        let pruned = prune_csr(m, f64::NAN).unwrap();

        assert_eq!(pruned.nnz(), 3);
        assert_eq!(pruned.values(), &[0.0, 5.0, 0.0]);
    }

    #[test]
    fn nan_valued_entries_survive_a_nan_tolerance_too() {
        let m = CsrMatrix::new(1, 2, vec![0, 2], vec![0, 1], vec![f64::NAN, 5.0]).unwrap();

        let pruned = prune_csr(m, f64::NAN).unwrap();

        assert_eq!(pruned.nnz(), 2);
        assert!(pruned.values()[0].is_nan());
        assert_eq!(pruned.values()[1], 5.0);
    }
}

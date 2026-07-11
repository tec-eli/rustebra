//! Differential property tests: this crate's decompositions checked against independently
//! implemented ones from `nalgebra` on random inputs, rather than against hand-worked
//! examples.

mod cholesky;

/// Whether `a` and `b` differ by no more than `tol` in absolute value.
pub(crate) fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[cfg(test)]
mod tests {
    use super::approx_eq;

    #[test]
    fn equal_values_are_approx_eq_at_zero_tolerance() {
        assert!(approx_eq(1.0, 1.0, 0.0));
    }

    #[test]
    fn values_within_tolerance_are_approx_eq() {
        assert!(approx_eq(1.0, 1.0 + 5e-10, 1e-9));
    }

    #[test]
    fn values_outside_tolerance_are_not_approx_eq() {
        assert!(!approx_eq(1.0, 1.1, 1e-9));
    }
}

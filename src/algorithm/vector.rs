use crate::scalar::Scalar;
use crate::storage::Storage;

/// Error returned by vector operations in this module when operand lengths don't agree.
///
/// `Storage` (see ADR 0003) exposes only read access and a length; it has no way to
/// construct a new instance generically, so these functions write their result into a
/// caller-provided output slice instead of returning a new `Storage`. That slice's length
/// is one more thing that must agree with the operands', alongside the operands' lengths
/// agreeing with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthMismatch;

/// Computes the element-wise sum of `a` and `b`, writing the result into `out`.
///
/// `a` and `b` may be different `Storage` implementations (e.g. one static, one dynamic),
/// as long as they hold the same `Scalar` type.
///
/// # Errors
///
/// Returns `Err(LengthMismatch)` if `a`, `b`, and `out` don't all have the same length.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::vector::add;
/// use rustebra::storage::StaticStorage;
///
/// let a = StaticStorage::new([1.0, 2.0, 3.0]);
/// let b = StaticStorage::new([4.0, 5.0, 6.0]);
/// let mut out = [0.0; 3];
/// add(&a, &b, &mut out).unwrap();
/// assert_eq!(out, [5.0, 7.0, 9.0]);
/// ```
pub fn add<S1, S2, T>(a: &S1, b: &S2, out: &mut [T]) -> Result<(), LengthMismatch>
where
    S1: Storage<Item = T>,
    S2: Storage<Item = T>,
    T: Scalar,
{
    let len = a.len();
    if b.len() != len || out.len() != len {
        return Err(LengthMismatch);
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let (Some(&x), Some(&y)) = (a.get(i), b.get(i)) else {
            return Err(LengthMismatch);
        };
        *slot = x.add(y);
    }
    Ok(())
}

/// Computes the element-wise difference of `a` and `b`, writing the result into `out`.
///
/// `a` and `b` may be different `Storage` implementations (e.g. one static, one dynamic),
/// as long as they hold the same `Scalar` type.
///
/// # Errors
///
/// Returns `Err(LengthMismatch)` if `a`, `b`, and `out` don't all have the same length.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::vector::sub;
/// use rustebra::storage::StaticStorage;
///
/// let a = StaticStorage::new([4.0, 5.0, 6.0]);
/// let b = StaticStorage::new([1.0, 2.0, 3.0]);
/// let mut out = [0.0; 3];
/// sub(&a, &b, &mut out).unwrap();
/// assert_eq!(out, [3.0, 3.0, 3.0]);
/// ```
pub fn sub<S1, S2, T>(a: &S1, b: &S2, out: &mut [T]) -> Result<(), LengthMismatch>
where
    S1: Storage<Item = T>,
    S2: Storage<Item = T>,
    T: Scalar,
{
    let len = a.len();
    if b.len() != len || out.len() != len {
        return Err(LengthMismatch);
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let (Some(&x), Some(&y)) = (a.get(i), b.get(i)) else {
            return Err(LengthMismatch);
        };
        *slot = x.sub(y);
    }
    Ok(())
}

/// Computes the element-wise product of `a` and `factor`, writing the result into `out`.
///
/// There's only one `Storage` operand here, so there's no pair of operands to disagree in
/// length with each other — but `out` is still a separate buffer the caller provides, so it
/// can still disagree in length with `a`.
///
/// # Errors
///
/// Returns `Err(LengthMismatch)` if `out` doesn't have the same length as `a`.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::vector::scale;
/// use rustebra::storage::StaticStorage;
///
/// let a = StaticStorage::new([1.0, 2.0, 3.0]);
/// let mut out = [0.0; 3];
/// scale(&a, 2.0, &mut out).unwrap();
/// assert_eq!(out, [2.0, 4.0, 6.0]);
/// ```
pub fn scale<S, T>(a: &S, factor: T, out: &mut [T]) -> Result<(), LengthMismatch>
where
    S: Storage<Item = T>,
    T: Scalar,
{
    let len = a.len();
    if out.len() != len {
        return Err(LengthMismatch);
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let Some(&x) = a.get(i) else {
            return Err(LengthMismatch);
        };
        *slot = x.mul(factor);
    }
    Ok(())
}

/// Computes the dot (inner) product of `a` and `b`: `a[0] * b[0] + a[1] * b[1] + ...`.
///
/// `a` and `b` may be different `Storage` implementations (e.g. one static, one dynamic),
/// as long as they hold the same `Scalar` type.
///
/// This is also the building block for the Euclidean (L2) norm of a vector `v`:
/// `‖v‖ = sqrt(dot(v, v))`.
///
/// # Errors
///
/// Returns `Err(LengthMismatch)` if `a` and `b` don't have the same length.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::vector::dot;
/// use rustebra::storage::StaticStorage;
///
/// let a = StaticStorage::new([1.0, 2.0, 3.0]);
/// let b = StaticStorage::new([4.0, 5.0, 6.0]);
/// assert_eq!(dot(&a, &b), Ok(32.0));
/// ```
pub fn dot<S1, S2, T>(a: &S1, b: &S2) -> Result<T, LengthMismatch>
where
    S1: Storage<Item = T>,
    S2: Storage<Item = T>,
    T: Scalar,
{
    let len = a.len();
    if b.len() != len {
        return Err(LengthMismatch);
    }
    let mut sum = T::zero();
    for i in 0..len {
        let (Some(&x), Some(&y)) = (a.get(i), b.get(i)) else {
            return Err(LengthMismatch);
        };
        sum = sum.add(x.mul(y));
    }
    Ok(sum)
}

/// Computes the Euclidean (L2) norm of `a`: `‖a‖ = sqrt(dot(a, a))`.
///
/// Implemented in terms of [`dot`] and [`Scalar::sqrt`] rather than re-deriving the sum of
/// squares separately.
///
/// Unlike the other functions in this module, `norm` has no failure mode to report: it
/// takes a single operand compared only against itself (via `dot(a, a)`), so there's no
/// pair of lengths that can ever disagree. It returns `T` directly rather than a `Result`.
///
/// # Examples
///
/// ```
/// use rustebra::algorithm::vector::norm;
/// use rustebra::storage::StaticStorage;
///
/// let v = StaticStorage::new([3.0, 4.0]);
/// assert_eq!(norm(&v), 5.0);
/// ```
pub fn norm<S, T>(a: &S) -> T
where
    S: Storage<Item = T>,
    T: Scalar,
{
    match dot(a, a) {
        Ok(sum_of_squares) => sum_of_squares.sqrt(),
        // `dot(a, a)` compares `a.len()` with itself, which can never disagree.
        Err(LengthMismatch) => T::zero(),
    }
}

#[cfg(test)]
mod tests {
    use super::{LengthMismatch, add, dot, norm, scale, sub};
    use crate::storage::StaticStorage;

    #[test]
    fn adds_matching_lengths_element_wise() {
        let a = StaticStorage::new([1.0, 2.0, 3.0]);
        let b = StaticStorage::new([4.0, 5.0, 6.0]);
        let mut out = [0.0; 3];

        assert_eq!(add(&a, &b, &mut out), Ok(()));
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn mismatched_operand_lengths_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0]);
        let b = StaticStorage::new([1.0, 2.0, 3.0]);
        let mut out = [0.0; 3];

        assert_eq!(add(&a, &b, &mut out), Err(LengthMismatch));
    }

    #[test]
    fn mismatched_output_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0]);
        let b = StaticStorage::new([1.0, 2.0, 3.0]);
        let mut out = [0.0; 2];

        assert_eq!(add(&a, &b, &mut out), Err(LengthMismatch));
    }

    #[test]
    fn subs_matching_lengths_element_wise() {
        let a = StaticStorage::new([4.0, 5.0, 6.0]);
        let b = StaticStorage::new([1.0, 2.0, 3.0]);
        let mut out = [0.0; 3];

        assert_eq!(sub(&a, &b, &mut out), Ok(()));
        assert_eq!(out, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn sub_mismatched_operand_lengths_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0]);
        let b = StaticStorage::new([1.0, 2.0, 3.0]);
        let mut out = [0.0; 3];

        assert_eq!(sub(&a, &b, &mut out), Err(LengthMismatch));
    }

    #[test]
    fn sub_mismatched_output_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0]);
        let b = StaticStorage::new([1.0, 2.0, 3.0]);
        let mut out = [0.0; 2];

        assert_eq!(sub(&a, &b, &mut out), Err(LengthMismatch));
    }

    #[test]
    fn scales_by_known_factor() {
        let a = StaticStorage::new([1.0, 2.0, 3.0]);
        let mut out = [0.0; 3];

        assert_eq!(scale(&a, 2.0, &mut out), Ok(()));
        assert_eq!(out, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn scales_by_zero() {
        let a = StaticStorage::new([1.0, 2.0, 3.0]);
        let mut out = [1.0, 1.0, 1.0];

        assert_eq!(scale(&a, 0.0, &mut out), Ok(()));
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn scale_mismatched_output_length_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0, 3.0]);
        let mut out = [0.0; 2];

        assert_eq!(scale(&a, 2.0, &mut out), Err(LengthMismatch));
    }

    #[test]
    fn dot_of_known_vectors() {
        let a = StaticStorage::new([1.0, 2.0, 3.0]);
        let b = StaticStorage::new([4.0, 5.0, 6.0]);

        assert_eq!(dot(&a, &b), Ok(32.0));
    }

    #[test]
    fn dot_mismatched_lengths_is_an_error_not_a_panic() {
        let a = StaticStorage::new([1.0, 2.0]);
        let b = StaticStorage::new([1.0, 2.0, 3.0]);

        assert_eq!(dot(&a, &b), Err(LengthMismatch));
    }

    #[test]
    fn norm_of_known_vector_is_exact() {
        let v = StaticStorage::new([3.0, 4.0]);

        assert_eq!(norm(&v), 5.0);
    }

    #[test]
    fn norm_of_irrational_is_within_tolerance() {
        let v = StaticStorage::new([1.0, 1.0]);

        let result = norm(&v);
        let expected = core::f64::consts::SQRT_2;
        assert!((result - expected).abs() < 1e-9);
    }
}

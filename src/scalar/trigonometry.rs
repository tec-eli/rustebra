use core::ops::{Add, Div, Mul, Sub};

use super::taylor::taylor_series;

/// Computes the sine of `value` (in radians) via [`taylor_series`]:
/// `sin(x) = x - x^3/3! + x^5/5! - ...`.
///
/// `zero` and `one` are passed in by the caller, since this is generic over any type with
/// the right arithmetic operations rather than over [`super::Scalar`] itself (`Scalar::sin`
/// is implemented in terms of this function, so the function itself can't depend on
/// `Scalar`), the same reason [`super::sqrt::newton_raphson`] takes `zero`/`two` explicitly.
pub(super) fn sin<T>(value: T, zero: T, one: T) -> T
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    let neg_x2 = zero.sub(value.mul(value));
    let two = one.add(one);
    let three = two.add(one);

    taylor_series(value, neg_x2, two, three, two, two)
}

/// Computes the cosine of `value` (in radians) via [`taylor_series`]:
/// `cos(x) = 1 - x^2/2! + x^4/4! - ...`.
///
/// Same `zero`/`one` parameter convention as [`sin`], for the same reason.
pub(super) fn cos<T>(value: T, zero: T, one: T) -> T
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    let neg_x2 = zero.sub(value.mul(value));
    let two = one.add(one);

    taylor_series(one, neg_x2, one, two, two, two)
}

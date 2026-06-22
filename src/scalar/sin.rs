use core::ops::{Add, Div, Mul, Sub};

/// Fixed iteration count for [`taylor_series`]; see the precision note on
/// [`super::Scalar::sin`] for the trade-off this implies for inputs far from zero.
const ITERATIONS: u32 = 20;

/// Computes `sin(value)` via a fixed-iteration Taylor series expansion around zero:
/// `sin(x) = x - x^3/3! + x^5/5! - ...`, computed through the recurrence
/// `term_{k+1} = term_k * (-x^2) / ((2k+2)(2k+3))` rather than recomputing factorials and
/// powers from scratch for every term.
///
/// `zero` and `one` are passed in by the caller, since this is generic over any type with
/// the right arithmetic operations rather than over [`super::Scalar`] itself (`Scalar::sin`
/// is implemented in terms of this function, so the function itself can't depend on
/// `Scalar`).
pub(super) fn taylor_series<T>(value: T, zero: T, one: T) -> T
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    let neg_x2 = zero.sub(value.mul(value));
    let two = one.add(one);

    let mut term = value;
    let mut sum = term;
    let mut a = two;
    let mut b = two.add(one);
    for _ in 0..ITERATIONS {
        term = term.mul(neg_x2).div(a.mul(b));
        sum = sum.add(term);
        a = a.add(two);
        b = b.add(two);
    }
    sum
}

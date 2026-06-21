use core::ops::{Add, Div};

/// Fixed iteration count for [`newton_raphson`]. Chosen generously enough to converge for
/// normal-range `f32`/`f64` inputs; see the precision note on [`super::Scalar::sqrt`] for the
/// trade-off this implies at extreme magnitudes.
const ITERATIONS: u32 = 50;

/// Computes the square root of `value` via fixed-iteration Newton-Raphson (Babylonian)
/// iteration: `x_{n+1} = (x_n + value / x_n) / 2`.
///
/// `zero` and `two` are passed in by the caller, since this is generic over any type with
/// the right arithmetic operations rather than over [`super::Scalar`] itself (`Scalar::sqrt`
/// is implemented in terms of this function, so the function itself can't depend on
/// `Scalar`). Returns `zero` immediately for `value <= zero`: see [`super::Scalar::sqrt`] for
/// why zero and negative inputs share that behavior.
pub(super) fn newton_raphson<T>(value: T, zero: T, two: T) -> T
where
    T: Copy + PartialOrd + Add<Output = T> + Div<Output = T>,
{
    if value <= zero {
        return zero;
    }
    let mut x = value;
    for _ in 0..ITERATIONS {
        x = (x + value / x) / two;
    }
    x
}

use core::ops::{Add, Div, Mul, Sub};

use super::taylor::taylor_series;

/// Reduces `value` (in radians) to a remainder in `[-pi/4, pi/4]` plus a quadrant `0..4`,
/// by subtracting the nearest multiple of `pi/2`.
///
/// [`sin`]/[`cos`] converge to full precision only for arguments near zero, since their
/// underlying [`taylor_series`] runs a fixed number of terms; reducing to the nearest
/// quadrant (rather than only to `[-pi, pi]`) keeps that fixed budget accurate across the
/// whole documented `[-2*pi, 2*pi]` domain, including near a zero crossing of `sin`/`cos`
/// itself — reducing only to `[-pi, pi]` can leave the remainder close to `pi`, where the
/// series needs far more terms than it does near zero for the same precision.
///
/// `quadrant` tells the caller which of `sin(r)`/`cos(r)`/`-sin(r)`/`-cos(r)` reconstructs
/// `sin(value)` (and the complementary choice for `cos(value)`).
///
/// The multiple is found via a truncating cast to `i64` (rounded to nearest by biasing the
/// quotient by +-0.5 first) rather than a `libm` `floor`/`round` call, so this stays
/// available in `no_std` builds without a `libm` dependency.
///
/// `pi/2` itself is irrational and only approximated by `PIO2_HI`, so subtracting it in one
/// step would bake that single-`f64`-ulp approximation error into the remainder — small in
/// absolute terms, but large relative to the remainder near a zero crossing of `sin`/`cos`
/// (where the remainder itself is close to zero). Splitting `pi/2` into a leading `PIO2_HI`
/// (exactly representable, so `value - k*PIO2_HI` loses no bits beyond ordinary
/// cancellation) and a trailing correction `PIO2_LO` — a Cody-Waite reduction — recovers
/// most of that precision.
pub(super) fn reduce_f64(value: f64) -> (f64, i32) {
    const PIO2_HI: f64 = core::f64::consts::FRAC_PI_2;
    const PIO2_LO: f64 = 1.2246467991473532e-16 / 2.0;

    let quotient = value / PIO2_HI;
    let half = if value >= 0.0 { 0.5 } else { -0.5 };
    let k = (quotient + half) as i64;
    let k_f = k as f64;

    let remainder = (value - k_f * PIO2_HI) - k_f * PIO2_LO;
    let quadrant = k.rem_euclid(4) as i32;

    (remainder, quadrant)
}

/// `f32` counterpart of [`reduce_f64`].
///
/// The subtraction that isolates the remainder is done in `f64`, not `f32`: `f32`'s ~7
/// decimal digits of precision aren't enough to represent `PIO2_HI`/`PIO2_LO` (and thus the
/// remainder near a zero crossing) as tightly as `f64` can, so reducing in `f32` arithmetic
/// directly reintroduces the same cancellation error `f64`'s reduction avoids. Widening to
/// `f64` for this one step, then narrowing the (already small) remainder back down, costs
/// nothing at runtime worth avoiding in exchange for using the wider type's precision where
/// it matters most.
pub(super) fn reduce_f32(value: f32) -> (f32, i32) {
    let (remainder, quadrant) = reduce_f64(value as f64);
    (remainder as f32, quadrant)
}

/// Computes the sine of `value` (in radians) via [`taylor_series`]:
/// `sin(x) = x - x^3/3! + x^5/5! - ...`.
///
/// `zero` and `one` are passed in by the caller, since this is generic over any type with
/// the right arithmetic operations rather than over [`super::Scalar`] itself (`Scalar::sin`
/// is implemented in terms of this function, so the function itself can't depend on
/// `Scalar`), the same reason [`super::newton_raphson::newton_raphson`] takes `zero`/`two`
/// explicitly.
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

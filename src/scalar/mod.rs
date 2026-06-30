mod f32;
mod f64;
mod float_tolerance;
mod newton_raphson;
mod taylor;
mod trigonometry;

pub use self::float_tolerance::FloatTolerance;

/// A numeric type that algorithms in this crate operate on.
///
/// Defines the minimal arithmetic surface required by the algorithm layer: additive and
/// multiplicative identities, and the four basic arithmetic operations. See ADR 0005 for why
/// the scalar type is generic rather than hardcoded, and ADR 0006 for how this trait fits into
/// the crate's layered architecture.
///
/// # Examples
///
/// ```
/// use rustebra::scalar::Scalar;
///
/// fn double<T: Scalar>(x: T) -> T {
///     x.add(x)
/// }
/// ```
pub trait Scalar: Copy + PartialEq {
    /// The additive identity, `0`.
    fn zero() -> Self;
    /// The multiplicative identity, `1`.
    fn one() -> Self;
    /// Adds `self` and `rhs`.
    fn add(self, rhs: Self) -> Self;
    /// Subtracts `rhs` from `self`.
    fn sub(self, rhs: Self) -> Self;
    /// Multiplies `self` and `rhs`.
    fn mul(self, rhs: Self) -> Self;
    /// Divides `self` by `rhs`.
    fn div(self, rhs: Self) -> Self;
    /// Returns the square root of `self`, via fixed-iteration Newton-Raphson (Babylonian)
    /// iteration: `x_{n+1} = (x_n + self / x_n) / 2`.
    ///
    /// `self == 0` returns `0` immediately, without iterating, since the formula divides by
    /// the previous iterate and would otherwise divide by zero.
    ///
    /// `self < 0` has no real square root. This implementation returns `0` for negative
    /// inputs rather than panicking or propagating a sentinel like `NaN`: `Scalar` is a
    /// generic, infallible trait that must also support future non-float implementations
    /// (e.g. fixed-point) with no `NaN` representation, so the contract is defined purely
    /// in terms of values every `Scalar` implementor can produce.
    ///
    /// The iteration count is fixed rather than convergence-checked, so behavior is
    /// predictable in `no_std` contexts (the same amount of work runs regardless of the
    /// input). This converges to the correctly-rounded result for the magnitudes typical of
    /// vector norms, but may lose precision for inputs at the extreme ends of the type's
    /// exponent range, where more iterations would be needed to leave the initial guess's
    /// slow-convergence region.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::scalar::Scalar;
    ///
    /// // Called via `Scalar::sqrt`, not `.sqrt()`: `f64` has its own inherent `sqrt` in
    /// // `std`, which would shadow the trait method `.sqrt()` resolves to.
    /// assert_eq!(Scalar::sqrt(4.0f64), 2.0);
    /// assert_eq!(Scalar::sqrt(0.0f64), 0.0);
    /// ```
    fn sqrt(self) -> Self;

    /// Returns the sine of `self` (in radians), via a fixed-iteration Taylor series
    /// expansion around zero: `sin(x) = x - x^3/3! + x^5/5! - ...`.
    ///
    /// The iteration count is fixed rather than convergence-checked, so behavior is
    /// predictable in `no_std` contexts (the same amount of work runs regardless of the
    /// input). This converges to the correctly-rounded result near zero, but loses
    /// precision as `self` grows away from zero, since this implementation performs no
    /// range reduction (e.g. reducing `self` modulo `2*pi` first).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::scalar::Scalar;
    ///
    /// assert_eq!(Scalar::sin(0.0f64), 0.0);
    ///
    /// let result = Scalar::sin(core::f64::consts::FRAC_PI_2);
    /// assert!((result - 1.0).abs() < 1e-9);
    /// ```
    fn sin(self) -> Self;

    /// Returns the cosine of `self` (in radians), via a fixed-iteration Taylor series
    /// expansion around zero: `cos(x) = 1 - x^2/2! + x^4/4! - ...`.
    ///
    /// Same fixed-iteration, no-range-reduction trade-off as [`Scalar::sin`]: predictable,
    /// bounded work in `no_std` contexts, at the cost of precision for `self` far from
    /// zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::scalar::Scalar;
    ///
    /// assert_eq!(Scalar::cos(0.0f64), 1.0);
    ///
    /// let result = Scalar::cos(core::f64::consts::PI);
    /// assert!((result - (-1.0)).abs() < 1e-9);
    /// ```
    fn cos(self) -> Self;
}

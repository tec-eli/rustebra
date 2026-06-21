mod f32;
mod f64;
mod sqrt;

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
pub trait Scalar: Copy {
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
}

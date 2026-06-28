use super::Scalar;

/// A [`Scalar`] type that can report its own machine epsilon, so callers don't have to supply
/// a numerical tolerance by hand for every approximate-zero comparison.
///
/// Kept separate from [`Scalar`] itself, rather than as a method on it, so that a future
/// `Scalar` implementor with no meaningful notion of machine epsilon (e.g. a fixed-point or
/// exact-rational type) isn't forced to answer a question that doesn't apply to it just to
/// satisfy this convenience: it can still implement `Scalar` alone and use every function that
/// takes an explicit tolerance.
///
/// # Examples
///
/// ```
/// use rustebra::scalar::FloatTolerance;
///
/// assert_eq!(f64::epsilon(), f64::EPSILON);
/// ```
pub trait FloatTolerance: Scalar {
    /// The smallest positive value `e` such that `Self::one().add(e) != Self::one()` —
    /// i.e. machine epsilon for this type.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustebra::scalar::FloatTolerance;
    ///
    /// assert_eq!(f32::epsilon(), f32::EPSILON);
    /// ```
    fn epsilon() -> Self;
}

#[cfg(test)]
mod tests {
    use super::FloatTolerance;

    #[test]
    fn f32_epsilon_matches_core() {
        assert_eq!(f32::epsilon(), f32::EPSILON);
    }

    #[test]
    fn f64_epsilon_matches_core() {
        assert_eq!(f64::epsilon(), f64::EPSILON);
    }
}

use super::Scalar;
use super::sqrt::newton_raphson;

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn sqrt(self) -> Self {
        newton_raphson(self, 0.0, 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Scalar;

    #[test]
    fn identities() {
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
    }

    #[test]
    fn arithmetic() {
        assert_eq!(2.0f64.add(3.0), 5.0);
        assert_eq!(5.0f64.sub(3.0), 2.0);
        assert_eq!(2.0f64.mul(3.0), 6.0);
        assert_eq!(6.0f64.div(3.0), 2.0);
    }

    #[test]
    fn sqrt_of_perfect_square_is_exact() {
        // Called via `Scalar::sqrt`, not `.sqrt()`: in `std` test builds, `f64` has its own
        // inherent `sqrt`, which would shadow the trait method `.sqrt()` resolves to.
        assert_eq!(Scalar::sqrt(4.0f64), 2.0);
        assert_eq!(Scalar::sqrt(0.0f64), 0.0);
    }

    #[test]
    fn sqrt_of_irrational_is_within_tolerance() {
        let result = Scalar::sqrt(2.0f64);
        let expected = core::f64::consts::SQRT_2;
        assert!((result - expected).abs() < 1e-9);
    }

    #[test]
    fn sqrt_of_negative_returns_zero() {
        assert_eq!(Scalar::sqrt(-4.0f64), 0.0);
    }
}

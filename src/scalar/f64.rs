use super::FloatTolerance;
use super::Scalar;
use super::newton_raphson::newton_raphson;
use super::trigonometry::{cos, sin};

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

    fn sin(self) -> Self {
        sin(self, Self::zero(), Self::one())
    }

    fn cos(self) -> Self {
        cos(self, Self::zero(), Self::one())
    }
}

impl FloatTolerance for f64 {
    fn epsilon() -> Self {
        f64::EPSILON
    }
}

#[cfg(test)]
mod tests {
    use super::{FloatTolerance, Scalar};

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

    #[test]
    fn sin_of_known_angles() {
        assert_eq!(Scalar::sin(0.0f64), 0.0);

        let result = Scalar::sin(core::f64::consts::FRAC_PI_2);
        assert!((result - 1.0).abs() < 1e-9);

        let result = Scalar::sin(core::f64::consts::PI);
        assert!((result - 0.0).abs() < 1e-9);
    }

    #[test]
    fn cos_of_known_angles() {
        assert_eq!(Scalar::cos(0.0f64), 1.0);

        let result = Scalar::cos(core::f64::consts::FRAC_PI_2);
        assert!((result - 0.0).abs() < 1e-9);

        let result = Scalar::cos(core::f64::consts::PI);
        assert!((result - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn epsilon_matches_core() {
        assert_eq!(f64::epsilon(), f64::EPSILON);
    }
}

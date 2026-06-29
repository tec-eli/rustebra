use rustebra::scalar::Scalar;

#[test]
fn f64_scalar_operations() {
    assert_eq!(f64::zero(), 0.0);
    assert_eq!(f64::one(), 1.0);
    assert_eq!(2.0f64.add(3.0), 5.0);
    assert_eq!(5.0f64.sub(3.0), 2.0);
    assert_eq!(2.0f64.mul(3.0), 6.0);
    assert_eq!(6.0f64.div(3.0), 2.0);
    assert_eq!(Scalar::sqrt(4.0f64), 2.0);

    let sin = Scalar::sin(core::f64::consts::FRAC_PI_2);
    assert!((sin - 1.0).abs() < 1e-9);

    let cos = Scalar::cos(core::f64::consts::PI);
    assert!((cos - (-1.0)).abs() < 1e-9);
}

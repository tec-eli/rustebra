use rustebra::scalar::Scalar;

#[test]
fn f32_scalar_operations() {
    assert_eq!(f32::zero(), 0.0);
    assert_eq!(f32::one(), 1.0);
    assert_eq!(2.0f32.add(3.0), 5.0);
    assert_eq!(5.0f32.sub(3.0), 2.0);
    assert_eq!(2.0f32.mul(3.0), 6.0);
    assert_eq!(6.0f32.div(3.0), 2.0);
    assert_eq!(Scalar::sqrt(4.0f32), 2.0);

    let sin = Scalar::sin(core::f32::consts::FRAC_PI_2);
    assert!((sin - 1.0).abs() < 1e-6);

    let cos = Scalar::cos(core::f32::consts::PI);
    assert!((cos - (-1.0)).abs() < 1e-6);
}

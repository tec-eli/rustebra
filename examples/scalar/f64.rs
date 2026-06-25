use rustebra::scalar::Scalar;

pub(crate) fn run() {
    println!("\n== f64 ==");
    println!("f64::zero() = {:?}", f64::zero());
    println!("f64::one() = {:?}", f64::one());
    println!("2.0 + 3.0 = {:?}", 2.0f64.add(3.0));
    println!("5.0 - 3.0 = {:?}", 5.0f64.sub(3.0));
    println!("2.0 * 3.0 = {:?}", 2.0f64.mul(3.0));
    println!("6.0 / 3.0 = {:?}", 6.0f64.div(3.0));
}

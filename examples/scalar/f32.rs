use rustebra::scalar::Scalar;

pub(crate) fn run() {
    println!("== f32 ==");
    println!("f32::zero() = {:?}", f32::zero());
    println!("f32::one() = {:?}", f32::one());
    println!("2.0 + 3.0 = {:?}", 2.0f32.add(3.0));
    println!("5.0 - 3.0 = {:?}", 5.0f32.sub(3.0));
    println!("2.0 * 3.0 = {:?}", 2.0f32.mul(3.0));
    println!("6.0 / 3.0 = {:?}", 6.0f32.div(3.0));
}

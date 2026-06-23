use rustebra::scalar::Scalar;

pub(crate) fn run() {
    println!("\n== sqrt ==");
    println!("sqrt(2.0) [f32] = {:?}", Scalar::sqrt(2.0f32));
    println!("sqrt(2.0) [f64] = {:?}", Scalar::sqrt(2.0f64));
}

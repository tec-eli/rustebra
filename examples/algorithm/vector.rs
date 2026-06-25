use rustebra::algorithm::vector::{add, dot, norm, scale, sub};
use rustebra::storage::StaticStorage;

pub(crate) fn run() {
    println!("\n== vector ==");
    let a = StaticStorage::new([1.0, 2.0, 3.0]);
    let b = StaticStorage::new([4.0, 5.0, 6.0]);

    let mut sum = [0.0; 3];
    add(&a, &b, &mut sum).unwrap();
    println!("a + b = {sum:?}");

    let mut diff = [0.0; 3];
    sub(&a, &b, &mut diff).unwrap();
    println!("a - b = {diff:?}");

    let mut scaled = [0.0; 3];
    scale(&a, 2.0, &mut scaled).unwrap();
    println!("a * 2 = {scaled:?}");

    println!("a . b = {:?}", dot(&a, &b).unwrap());
    println!("||a|| = {:.4}", norm(&a));
}

use rustebra::vector::StaticVector;

pub(crate) fn run() {
    println!("== StaticVector ==");
    let a = StaticVector::new([1.0, 2.0, 3.0]);
    let b = StaticVector::new([4.0, 5.0, 6.0]);

    println!("a = {a:?}");
    println!("b = {b:?}");
    println!("a + b = {:?}", a.add(&b));
    println!("a - b = {:?}", a.sub(&b));
    println!("a scaled by 2 = {:?}", a.scale(2.0));
    println!("a . b = {}", a.dot(&b));
    println!("||a|| = {:.4}", a.norm());
}

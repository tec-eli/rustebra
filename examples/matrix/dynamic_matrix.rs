use rustebra::matrix::DynamicMatrix;
use rustebra::vector::DynamicVector;

pub(crate) fn run() {
    println!("\n== DynamicMatrix ==");
    let a = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("2x2 data");
    let b = DynamicMatrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).expect("2x2 data");
    let v = DynamicVector::new(vec![1.0, 1.0]);

    println!("a = {a:?}");
    println!("b = {b:?}");

    let sum = a.add(&b).expect("a and b have the same shape");
    println!("a + b = {sum:?}");

    let diff = a.sub(&b).expect("a and b have the same shape");
    println!("a - b = {diff:?}");

    println!("a scaled by 2 = {:?}", a.mul_scalar(2.0));

    let product_v = a
        .mul_vector(&v)
        .expect("v's length matches a's column count");
    println!("a * v = {product_v:?}");

    let product_m = a
        .mul_matrix(&b)
        .expect("a's column count matches b's row count");
    println!("a * b = {product_m:?}");

    println!("a^T = {:?}", a.transpose());

    let det = a.determinant().expect("a is square");
    println!("det(a) = {det:?}");

    println!("rank(a) = {:?}", a.rank());
}

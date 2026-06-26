use rustebra::matrix::StaticMatrix;
use rustebra::vector::StaticVector;

pub(crate) fn run() {
    println!("== StaticMatrix ==");
    let a = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let b = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
    let v = StaticVector::new([1.0, 1.0]);

    println!("a = {a:?}");
    println!("b = {b:?}");
    println!("a + b = {:?}", a.add(&b));
    println!("a - b = {:?}", a.sub(&b));
    println!("a scaled by 2 = {:?}", a.mul_scalar(2.0));
    println!("a * v = {:?}", a.mul_vector(&v));
    println!("a * b = {:?}", a.mul_matrix(&b));
    println!("a^T = {:?}", a.transpose());
    println!("det(a) = {:?}", a.determinant());
    println!("rank(a) = {:?}", a.rank());

    let (l, u, swap_count) = a.lu();
    println!("lu(a) = (l = {l:?}, u = {u:?}, swap_count = {swap_count})");

    let (q, r) = a.qr().expect("a has at least as many rows as columns");
    println!("qr(a) = (q = {q:?}, r = {r:?})");

    let spd = StaticMatrix::new([[4.0, 2.0], [2.0, 2.0]]);
    println!("cholesky(spd) = {:?}", spd.cholesky());

    let mut svd_scratch = [0.0; 5 * 2 * 2 + 2 + 2];
    let (svd_u, sigma, v) = a.svd(&mut svd_scratch).expect("scratch is correctly sized");
    println!("svd(a) = (u = {svd_u:?}, sigma = {sigma:?}, v = {v:?})");

    let mut condition_scratch = [0.0; 7 * 2 * 2 + 3 * 2];
    println!(
        "condition_number(a) = {:?}",
        a.condition_number(&mut condition_scratch)
    );
}

//! Tour `algorithm::matrix`'s functions directly, the layer `StaticMatrix`/`DynamicMatrix`
//! are built on top of: each function takes `Storage` operands and caller-provided output
//! buffers instead of a matrix type, and several operations (LU, QR, Cholesky, SVD, condition
//! number) aren't wrapped by either matrix type at all yet. Every operation with more than
//! one algorithm shows both its high-level entry point and the explicit algorithm it (or an
//! alternative) names directly.
//!
//! Run with: `cargo run --example algorithm_matrix`

use rustebra::algorithm::matrix::{
    add, cholesky, cholesky_decompose, condition_number, condition_number_svd, determinant,
    determinant_cofactor, determinant_lu, lu, lu_partial_pivot, mul_matrix, mul_scalar, mul_vector,
    qr, qr_gram_schmidt, qr_householder, rank, sub, svd, svd_qr_iteration, transpose,
};
use rustebra::storage::StaticStorage;

fn main() {
    arithmetic();
    determinant_example();
    rank_example();
    lu_example();
    qr_example();
    cholesky_example();
    svd_example();
    condition_number_example();
}

fn arithmetic() {
    println!("== arithmetic ==");
    // Row-major 2x2 matrices: [[1, 2], [3, 4]] and [[5, 6], [7, 8]].
    let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0]);
    let b = StaticStorage::new([5.0, 6.0, 7.0, 8.0]);

    let mut sum = [0.0; 4];
    add(&a, 2, 2, &b, 2, 2, &mut sum).unwrap();
    println!("a + b = {sum:?}");

    let mut diff = [0.0; 4];
    sub(&a, 2, 2, &b, 2, 2, &mut diff).unwrap();
    println!("a - b = {diff:?}");

    let mut scaled = [0.0; 4];
    mul_scalar(&a, 2, 2, 2.0, &mut scaled).unwrap();
    println!("a * 2 = {scaled:?}");

    let v = StaticStorage::new([1.0, 1.0]);
    let mut product_v = [0.0; 2];
    mul_vector(&a, 2, 2, &v, &mut product_v).unwrap();
    println!("a * v = {product_v:?}");

    let mut product_m = [0.0; 4];
    mul_matrix(&a, 2, 2, &b, 2, 2, &mut product_m).unwrap();
    println!("a * b = {product_m:?}");

    let mut transposed = [0.0; 4];
    transpose(&a, 2, 2, &mut transposed).unwrap();
    println!("a^T = {transposed:?}");
}

fn determinant_example() {
    println!("\n== determinant ==");
    // [[1, 2, 3], [4, 5, 6], [7, 8, 10]].
    let a = StaticStorage::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);

    println!("det(a) = {:?}", determinant(&a, 3, 3).unwrap());
    println!(
        "det(a) via cofactor expansion (explicit) = {:?}",
        determinant_cofactor(&a, 3, 3).unwrap()
    );

    let mut scratch = [0.0; 2 * 3 * 3];
    println!(
        "det(a) via LU decomposition (explicit) = {:?}",
        determinant_lu(&a, 3, 3, &mut scratch).unwrap()
    );
}

fn rank_example() {
    println!("\n== rank ==");
    // [[1, 2], [2, 4]]; row 1 is twice row 0, so this is rank-deficient.
    let a = StaticStorage::new([1.0, 2.0, 2.0, 4.0]);
    let mut scratch = [0.0; 4];

    println!("rank(a) = {:?}", rank(&a, 2, 2, &mut scratch).unwrap());
}

fn lu_example() {
    println!("\n== LU decomposition ==");
    // [[4, 3], [6, 3]].
    let a = StaticStorage::new([4.0, 3.0, 6.0, 3.0]);
    let mut l = [0.0; 4];
    let mut u = [0.0; 4];

    let swaps = lu(&a, 2, 2, &mut l, &mut u).unwrap();
    println!("l = {l:?}, u = {u:?} (row swaps: {swaps})");

    let mut l_explicit = [0.0; 4];
    let mut u_explicit = [0.0; 4];
    lu_partial_pivot(&a, 2, 2, &mut l_explicit, &mut u_explicit).unwrap();
    println!("l (explicit partial pivot) = {l_explicit:?}, u = {u_explicit:?}");
}

fn qr_example() {
    println!("\n== QR decomposition ==");
    // [[3, 5], [4, 0]].
    let a = StaticStorage::new([3.0, 5.0, 4.0, 0.0]);
    let mut scratch = [0.0; 2];

    let mut q = [0.0; 4];
    let mut r = [0.0; 4];
    qr(&a, 2, 2, &mut q, &mut r, &mut scratch).unwrap();
    println!("q = {q:?}, r = {r:?}");

    let mut q_householder = [0.0; 4];
    let mut r_householder = [0.0; 4];
    qr_householder(
        &a,
        2,
        2,
        &mut q_householder,
        &mut r_householder,
        &mut scratch,
    )
    .unwrap();
    println!("q (explicit householder) = {q_householder:?}, r = {r_householder:?}");

    let mut q_gram_schmidt = [0.0; 4];
    let mut r_gram_schmidt = [0.0; 4];
    qr_gram_schmidt(
        &a,
        2,
        2,
        &mut q_gram_schmidt,
        &mut r_gram_schmidt,
        &mut scratch,
    )
    .unwrap();
    println!("q (explicit gram-schmidt) = {q_gram_schmidt:?}, r = {r_gram_schmidt:?}");
}

fn cholesky_example() {
    println!("\n== Cholesky decomposition ==");
    // [[4, 2], [2, 2]], symmetric positive-definite.
    let a = StaticStorage::new([4.0, 2.0, 2.0, 2.0]);

    let mut l = [0.0; 4];
    cholesky(&a, 2, 2, &mut l).unwrap();
    println!("l = {l:?}");

    let mut l_explicit = [0.0; 4];
    cholesky_decompose(&a, 2, 2, &mut l_explicit).unwrap();
    println!("l (explicit) = {l_explicit:?}");
}

fn svd_example() {
    println!("\n== Singular Value Decomposition (SVD) ==");
    // [[2, 0], [0, 1]].
    let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
    let mut scratch = [0.0; 5 * 2 * 2 + 2 + 2];

    let mut u = [0.0; 4];
    let mut sigma = [0.0; 2];
    let mut v = [0.0; 4];
    svd(&a, 2, 2, &mut u, &mut sigma, &mut v, &mut scratch).unwrap();
    println!("sigma = {sigma:?}, u = {u:?}, v = {v:?}");

    let mut sigma_explicit = [0.0; 2];
    svd_qr_iteration(&a, 2, 2, &mut u, &mut sigma_explicit, &mut v, &mut scratch).unwrap();
    println!("sigma (explicit QR iteration) = {sigma_explicit:?}");
}

fn condition_number_example() {
    println!("\n== Condition number ==");
    // [[100, 0], [0, 1]]: ill-conditioned, kappa = 100.
    let a = StaticStorage::new([100.0, 0.0, 0.0, 1.0]);
    let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

    println!(
        "kappa(a) = {:?}",
        condition_number(&a, 2, 2, &mut scratch).unwrap()
    );
    println!(
        "kappa(a) via SVD (explicit) = {:?}",
        condition_number_svd(&a, 2, 2, &mut scratch).unwrap()
    );
}

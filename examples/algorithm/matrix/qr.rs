use rustebra::algorithm::matrix::{qr, qr_gram_schmidt, qr_householder};
use rustebra::storage::StaticStorage;

pub(crate) fn run() {
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

use super::{TOL, assert_all_close, assert_upper_triangular};
use rustebra::algorithm::matrix::{determinant, mul_matrix, qr, transpose};
use rustebra::storage::StaticStorage;

#[test]
fn qr_of_square_matrix_is_orthogonal_and_reconstructs_a_and_matches_determinant() {
    // Symmetric tridiagonal matrix, det = 4 by direct cofactor expansion.
    #[rustfmt::skip]
    let data = [
        2.0, -1.0, 0.0,
        -1.0, 2.0, -1.0,
        0.0, -1.0, 2.0,
    ];
    let a = StaticStorage::new(data);
    let mut q = [0.0; 9];
    let mut r = [0.0; 9];
    let mut scratch = [0.0; 3];

    assert_eq!(qr(&a, 3, 3, &mut q, &mut r, &mut scratch), Ok(()));
    assert_upper_triangular(&r, 3, 3);

    let mut q_t = [0.0; 9];
    transpose(&StaticStorage::new(q), 3, 3, &mut q_t).unwrap();
    let mut q_t_q = [0.0; 9];
    mul_matrix(
        &StaticStorage::new(q_t),
        3,
        3,
        &StaticStorage::new(q),
        3,
        3,
        &mut q_t_q,
    )
    .unwrap();
    #[rustfmt::skip]
    let identity = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    assert_all_close(&q_t_q, &identity, TOL);

    let mut qr_product = [0.0; 9];
    mul_matrix(
        &StaticStorage::new(q),
        3,
        3,
        &StaticStorage::new(r),
        3,
        3,
        &mut qr_product,
    )
    .unwrap();
    assert_all_close(&qr_product, &data, TOL);

    // det(a) = det(q) * det(r); q is orthogonal so |det(q)| == 1, leaving |det(r)| (the
    // product of its diagonal, since r is upper triangular) equal to |det(a)|.
    let det_via_r_diagonal = (r[0] * r[4] * r[8]).abs();
    let det_via_cofactor = determinant(&a, 3, 3).unwrap().abs();
    assert!((det_via_r_diagonal - det_via_cofactor).abs() < TOL);
}

#[test]
fn qr_of_matrix_with_more_rows_than_columns_is_orthogonal_and_reconstructs_a() {
    #[rustfmt::skip]
    let data = [
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        1.0, -1.0,
    ];
    let a = StaticStorage::new(data);
    let mut q = [0.0; 16];
    let mut r = [0.0; 8];
    let mut scratch = [0.0; 4];

    assert_eq!(qr(&a, 4, 2, &mut q, &mut r, &mut scratch), Ok(()));
    assert_upper_triangular(&r, 4, 2);

    let mut q_t = [0.0; 16];
    transpose(&StaticStorage::new(q), 4, 4, &mut q_t).unwrap();
    let mut q_t_q = [0.0; 16];
    mul_matrix(
        &StaticStorage::new(q_t),
        4,
        4,
        &StaticStorage::new(q),
        4,
        4,
        &mut q_t_q,
    )
    .unwrap();
    #[rustfmt::skip]
    let identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    assert_all_close(&q_t_q, &identity, TOL);

    let mut qr_product = [0.0; 8];
    mul_matrix(
        &StaticStorage::new(q),
        4,
        4,
        &StaticStorage::new(r),
        4,
        2,
        &mut qr_product,
    )
    .unwrap();
    assert_all_close(&qr_product, &data, TOL);
}

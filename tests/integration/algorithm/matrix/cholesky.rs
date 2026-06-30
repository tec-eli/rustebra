use super::{TOL, assert_all_close, assert_lower_triangular};
use rustebra::algorithm::matrix::{CholeskyError, cholesky, determinant, mul_matrix, transpose};
use rustebra::storage::StaticStorage;

#[test]
fn cholesky_of_known_spd_matrix_reconstructs_a_and_matches_determinant() {
    // [[25, 15, -5], [15, 18, 0], [-5, 0, 11]]; a standard worked Cholesky example with an
    // exact integer factor l = [[5, 0, 0], [3, 3, 0], [-1, 1, 3]], but verified here purely
    // through the mathematical contract (triangularity, positive diagonal, reconstruction)
    // rather than that specific factor.
    #[rustfmt::skip]
    let data = [
        25.0, 15.0, -5.0,
        15.0, 18.0, 0.0,
        -5.0, 0.0, 11.0,
    ];
    let a = StaticStorage::new(data);
    let mut l = [0.0; 9];

    assert_eq!(cholesky(&a, 3, 3, &mut l), Ok(()));
    assert_lower_triangular(&l, 3);
    for i in 0..3 {
        assert!(l[i * 3 + i] > 0.0);
    }

    let mut l_t = [0.0; 9];
    transpose(&StaticStorage::new(l), 3, 3, &mut l_t).unwrap();
    let mut l_l_t = [0.0; 9];
    mul_matrix(
        &StaticStorage::new(l),
        3,
        3,
        &StaticStorage::new(l_t),
        3,
        3,
        &mut l_l_t,
    )
    .unwrap();
    assert_all_close(&l_l_t, &data, TOL);

    // det(a) = det(l) * det(l^t) = det(l)^2, since l is triangular, det(l) is its diagonal
    // product.
    let det_via_l = l[0] * l[4] * l[8];
    let det_via_cofactor = determinant(&a, 3, 3).unwrap();
    assert!((det_via_l * det_via_l - det_via_cofactor).abs() < TOL);
}

#[test]
fn cholesky_of_indefinite_symmetric_matrix_is_not_positive_definite() {
    // [[2, 4], [4, 3]]; det = 6 - 16 = -10 < 0, so this symmetric matrix cannot be
    // positive-definite (a positive-definite matrix always has a positive determinant).
    let a = StaticStorage::new([2.0, 4.0, 4.0, 3.0]);
    let mut l = [0.0; 4];

    assert_eq!(
        cholesky(&a, 2, 2, &mut l),
        Err(CholeskyError::NotPositiveDefinite)
    );
}

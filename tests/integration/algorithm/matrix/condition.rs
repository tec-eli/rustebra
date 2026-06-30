use super::TOL_ITERATIVE;
use rustebra::algorithm::matrix::{ConditionNumberError, condition_number, transpose};
use rustebra::storage::StaticStorage;

#[test]
fn condition_number_of_a_rotation_matrix_is_one() {
    // Equal singular values (see the SVD test above) mean sigma_max / sigma_min == 1.
    let a = StaticStorage::new([0.0_f64, -1.0, 1.0, 0.0]);
    let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

    let kappa = condition_number(&a, 2, 2, &mut scratch).unwrap();
    assert!((kappa - 1.0).abs() < TOL_ITERATIVE);
}

#[test]
fn condition_number_of_symmetric_matrix_matches_ratio_of_known_eigenvalues() {
    // [[3, 1], [1, 3]] is symmetric, so its singular values equal the absolute values of its
    // eigenvalues, which are exactly 4 and 2 here (eigenvectors [1, 1] and [1, -1]).
    let a = StaticStorage::new([3.0_f64, 1.0, 1.0, 3.0]);
    let mut scratch = [0.0; 7 * 2 * 2 + 3 * 2];

    let kappa = condition_number(&a, 2, 2, &mut scratch).unwrap();
    assert!((kappa - 2.0).abs() < TOL_ITERATIVE);
}

#[test]
fn condition_number_of_singular_matrix_is_an_error() {
    // Every row is a multiple of [1, 2, 3], so this 3x3 matrix is rank-deficient (rank 1).
    #[rustfmt::skip]
    let a = StaticStorage::new([
        2.0_f64, 4.0, 6.0,
        1.0, 2.0, 3.0,
        3.0, 6.0, 9.0,
    ]);
    let mut scratch = [0.0; 7 * 3 * 3 + 3 * 3];

    assert_eq!(
        condition_number(&a, 3, 3, &mut scratch),
        Err(ConditionNumberError::Singular)
    );
}

#[test]
fn condition_number_is_invariant_under_transpose() {
    // a and a^t always share the same singular values, so they must share the same
    // condition number, even though a itself is neither symmetric nor diagonal.
    let a = StaticStorage::new([2.0_f64, 1.0, 0.0, 3.0]);
    let mut a_t = [0.0; 4];
    transpose(&a, 2, 2, &mut a_t).unwrap();

    let mut scratch_a = [0.0; 7 * 2 * 2 + 3 * 2];
    let kappa_a = condition_number(&a, 2, 2, &mut scratch_a).unwrap();

    let mut scratch_a_t = [0.0; 7 * 2 * 2 + 3 * 2];
    let kappa_a_t = condition_number(&StaticStorage::new(a_t), 2, 2, &mut scratch_a_t).unwrap();

    assert!((kappa_a - kappa_a_t).abs() < TOL_ITERATIVE);
}

use super::{TOL_ITERATIVE, assert_all_close};
use rustebra::algorithm::matrix::{determinant, rank, svd};
use rustebra::storage::StaticStorage;

#[test]
fn svd_of_a_rotation_matrix_has_all_singular_values_equal_to_one() {
    // A 90-degree rotation is orthogonal, so a*a^t is the identity and every singular value
    // is exactly 1, even though the algorithm has no special-case for orthogonal input.
    let a = StaticStorage::new([0.0_f64, -1.0, 1.0, 0.0]);
    let mut u = [0.0; 4];
    let mut sigma = [0.0; 2];
    let mut v = [0.0; 4];
    let mut scratch = [0.0; 5 * 2 * 2 + 2 + 2];

    assert_eq!(
        svd(&a, 2, 2, &mut u, &mut sigma, &mut v, &mut scratch),
        Ok(())
    );
    assert_all_close(&sigma, &[1.0, 1.0], TOL_ITERATIVE);
}

#[test]
fn svd_nonzero_singular_value_count_matches_independently_computed_rank() {
    // Every row is a multiple of [1, 2, 3], so this is rank 1 regardless of how many columns
    // it has.
    #[rustfmt::skip]
    let data = [
        2.0, 4.0, 6.0,
        1.0, 2.0, 3.0,
        3.0, 6.0, 9.0,
    ];
    let a = StaticStorage::new(data);
    let mut u = [0.0; 9];
    let mut sigma = [0.0; 3];
    let mut v = [0.0; 9];
    let mut scratch = [0.0; 5 * 3 * 3 + 3 + 3];

    assert_eq!(
        svd(&a, 3, 3, &mut u, &mut sigma, &mut v, &mut scratch),
        Ok(())
    );
    let nonzero_count = sigma.iter().filter(|&&s| s > 1e-6).count();

    let mut rank_scratch = [0.0; 9];
    let independent_rank = rank(&a, 3, 3, &mut rank_scratch).unwrap();

    assert_eq!(nonzero_count, independent_rank);
    assert_eq!(independent_rank, 1);
}

#[test]
fn svd_singular_value_product_matches_determinant_for_a_square_matrix() {
    // For a square matrix, |det(a)| = product of its singular values; this holds regardless
    // of what u and v actually are.
    #[rustfmt::skip]
    let data = [
        2.0, -1.0, 0.0,
        -1.0, 2.0, -1.0,
        0.0, -1.0, 2.0,
    ];
    let a = StaticStorage::new(data);
    let mut u = [0.0; 9];
    let mut sigma = [0.0; 3];
    let mut v = [0.0; 9];
    let mut scratch = [0.0; 5 * 3 * 3 + 3 + 3];

    assert_eq!(
        svd(&a, 3, 3, &mut u, &mut sigma, &mut v, &mut scratch),
        Ok(())
    );
    let sigma_product: f64 = sigma.iter().product();
    let det = determinant(&a, 3, 3).unwrap().abs();

    assert!((sigma_product - det).abs() < TOL_ITERATIVE);
}

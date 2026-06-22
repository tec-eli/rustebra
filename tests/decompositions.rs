//! Black-box mathematical tests for `algorithm::matrix`'s decompositions and derived
//! quantities (LU, QR, Cholesky, SVD, condition number). Each test checks a property the
//! decomposition is mathematically required to satisfy (orthogonality, triangularity,
//! reconstruction, or agreement with an independently-computed quantity from this crate's
//! own public API) on inputs distinct from the ones already exercised by the unit tests
//! colocated with the implementation, rather than re-asserting specific algorithm output.

use rustebra::algorithm::matrix::{
    CholeskyError, ConditionNumberError, cholesky, condition_number, determinant, lu, mul_matrix,
    qr, rank, svd, transpose,
};
use rustebra::storage::StaticStorage;

const TOL: f64 = 1e-9;
const TOL_ITERATIVE: f64 = 1e-6;

fn assert_all_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < tol, "expected {e}, got {a}");
    }
}

fn assert_lower_triangular(m: &[f64], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            assert_eq!(m[i * n + j], 0.0, "({i}, {j}) should be 0");
        }
    }
}

fn assert_unit_lower_triangular(m: &[f64], n: usize) {
    assert_lower_triangular(m, n);
    for i in 0..n {
        assert_eq!(m[i * n + i], 1.0, "({i}, {i}) should be 1");
    }
}

fn assert_upper_triangular(m: &[f64], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols.min(i) {
            assert_eq!(m[i * cols + j], 0.0, "({i}, {j}) should be 0");
        }
    }
}

// ---------- LU ----------

#[test]
fn lu_of_matrix_with_nonzero_diagonal_needs_no_pivoting_and_reconstructs_a() {
    // Every diagonal entry is already nonzero, so the "first nonzero" pivoting rule never
    // swaps: l * u == a directly, with no permutation to account for.
    #[rustfmt::skip]
    let data = [
        10.0, 2.0, 1.0,
        1.0, 8.0, 2.0,
        2.0, 1.0, 9.0,
    ];
    let a = StaticStorage::new(data);
    let mut l = [0.0; 9];
    let mut u = [0.0; 9];

    assert_eq!(lu(&a, 3, 3, &mut l, &mut u), Ok(0));
    assert_unit_lower_triangular(&l, 3);
    assert_upper_triangular(&u, 3, 3);

    let mut product = [0.0; 9];
    mul_matrix(
        &StaticStorage::new(l),
        3,
        3,
        &StaticStorage::new(u),
        3,
        3,
        &mut product,
    )
    .unwrap();
    assert_all_close(&product, &data, TOL);
}

#[test]
fn lu_of_singular_matrix_agrees_with_determinant_via_cofactor_expansion() {
    // Row 1 is twice row 0, so this is singular regardless of which row partial pivoting
    // happens to swap in.
    #[rustfmt::skip]
    let data = [
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0,
        1.0, 0.0, 1.0,
    ];
    let a = StaticStorage::new(data);
    let mut l = [0.0; 9];
    let mut u = [0.0; 9];

    let swaps = lu(&a, 3, 3, &mut l, &mut u).unwrap();
    assert_unit_lower_triangular(&l, 3);
    assert_upper_triangular(&u, 3, 3);

    // det(a) = (-1)^swaps * product(diag(u)); compare against the independent cofactor
    // expansion that `determinant` uses for 3x3 input.
    let mut det_via_lu = u[0] * u[4] * u[8];
    if swaps % 2 == 1 {
        det_via_lu = -det_via_lu;
    }
    let det_via_cofactor = determinant(&a, 3, 3).unwrap();
    assert!((det_via_lu - det_via_cofactor).abs() < TOL);
    assert!(det_via_cofactor.abs() < TOL); // singular, so determinant is 0.
}

// ---------- QR ----------

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

// ---------- Cholesky ----------

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

// ---------- SVD ----------

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

// ---------- Condition number ----------

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

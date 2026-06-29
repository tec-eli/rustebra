use super::{TOL, assert_all_close, assert_unit_lower_triangular, assert_upper_triangular};
use rustebra::algorithm::matrix::{determinant, lu, mul_matrix};
use rustebra::storage::StaticStorage;

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

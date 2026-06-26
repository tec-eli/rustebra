use rustebra::matrix::StaticMatrix;
use rustebra::storage::Storage;
use rustebra::vector::StaticVector;

#[test]
fn static_matrix_construction_and_operations() {
    let a = StaticMatrix::new([[1.0_f64, 2.0], [3.0, 4.0]]);
    let b = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
    let v = StaticVector::new([1.0, 1.0]);

    assert_eq!(a.add(&b), StaticMatrix::new([[6.0, 8.0], [10.0, 12.0]]));
    assert_eq!(b.sub(&a), StaticMatrix::new([[4.0, 4.0], [4.0, 4.0]]));
    assert_eq!(
        a.mul_scalar(2.0),
        StaticMatrix::new([[2.0, 4.0], [6.0, 8.0]])
    );
    assert_eq!(a.mul_vector(&v), StaticVector::new([3.0, 7.0]));
    assert_eq!(
        a.mul_matrix(&b),
        StaticMatrix::new([[19.0, 22.0], [43.0, 50.0]])
    );
    assert_eq!(
        StaticMatrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).transpose(),
        StaticMatrix::new([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    );
    assert_eq!(a.determinant(), Ok(-2.0));
    assert_eq!(a.rank(), 2);

    let (l, u, swap_count) = a.lu();
    assert_eq!(swap_count, 1);
    // l * u == p * a, where p swapped rows 0 and 1 of a (row 1 holds the larger-magnitude
    // pivot in column 0).
    assert_eq!(
        l.mul_matrix(&u),
        StaticMatrix::new([[3.0, 4.0], [1.0, 2.0]])
    );

    let (q, r) = a.qr().unwrap();
    let reconstructed = q.mul_matrix(&r);
    for i in 0..4 {
        let actual = *reconstructed.get(i).unwrap();
        let expected = *a.get(i).unwrap();
        assert!((actual - expected).abs() < 1e-9);
    }

    let spd = StaticMatrix::new([[4.0, 2.0], [2.0, 2.0]]);
    assert_eq!(
        spd.cholesky(),
        Ok(StaticMatrix::new([[2.0, 0.0], [1.0, 1.0]]))
    );

    let mut svd_scratch = [0.0; 5 * 2 * 2 + 2 + 2];
    let (_, sigma, _) = a.svd(&mut svd_scratch).unwrap();
    assert!(sigma.get(0) >= sigma.get(1));
    assert!(*sigma.get(1).unwrap() >= 0.0);

    let mut condition_scratch = [0.0; 7 * 2 * 2 + 3 * 2];
    let kappa = a.condition_number(&mut condition_scratch).unwrap();
    assert!(kappa > 1.0);
}

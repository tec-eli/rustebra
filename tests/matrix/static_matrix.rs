use rustebra::matrix::StaticMatrix;
use rustebra::vector::StaticVector;

#[test]
fn static_matrix_construction_and_operations() {
    let a = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
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
    assert_eq!(a.determinant(), -2.0);
    assert_eq!(a.rank(), 2);
}

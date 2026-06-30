use rustebra::vector::StaticVector;

#[test]
fn static_vector_construction_and_operations() {
    let a = StaticVector::new([1.0, 2.0, 3.0]);
    let b = StaticVector::new([4.0, 5.0, 6.0]);

    assert_eq!(a.add(&b), StaticVector::new([5.0, 7.0, 9.0]));
    assert_eq!(b.sub(&a), StaticVector::new([3.0, 3.0, 3.0]));
    assert_eq!(a.scale(2.0), StaticVector::new([2.0, 4.0, 6.0]));
    assert_eq!(a.dot(&b), 32.0);
    assert_eq!(StaticVector::new([3.0, 4.0]).norm(), 5.0);
}

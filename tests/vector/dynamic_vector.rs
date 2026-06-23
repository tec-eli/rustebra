#[cfg(feature = "alloc")]
#[test]
fn dynamic_vector_construction_and_operations() {
    use rustebra::vector::DynamicVector;

    let a = DynamicVector::new(vec![1.0, 2.0, 3.0]);
    let b = DynamicVector::new(vec![4.0, 5.0, 6.0]);

    assert_eq!(a.add(&b), Ok(DynamicVector::new(vec![5.0, 7.0, 9.0])));
    assert_eq!(b.sub(&a), Ok(DynamicVector::new(vec![3.0, 3.0, 3.0])));
    assert_eq!(a.scale(2.0), DynamicVector::new(vec![2.0, 4.0, 6.0]));
    assert_eq!(a.dot(&b), Ok(32.0));
    assert_eq!(DynamicVector::new(vec![3.0, 4.0]).norm(), 5.0);
}

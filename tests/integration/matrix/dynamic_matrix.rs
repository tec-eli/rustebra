#[cfg(feature = "alloc")]
#[test]
fn dynamic_matrix_construction_and_operations() {
    use rustebra::matrix::DynamicMatrix;
    use rustebra::storage::Storage;
    use rustebra::vector::DynamicVector;

    let a = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = DynamicMatrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let v = DynamicVector::new(vec![1.0, 1.0]);

    assert_eq!(
        a.add(&b),
        Ok(DynamicMatrix::new(2, 2, vec![6.0, 8.0, 10.0, 12.0]).unwrap())
    );
    assert_eq!(
        b.sub(&a),
        Ok(DynamicMatrix::new(2, 2, vec![4.0, 4.0, 4.0, 4.0]).unwrap())
    );
    assert_eq!(
        a.mul_scalar(2.0),
        DynamicMatrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]).unwrap()
    );
    assert_eq!(a.mul_vector(&v), Ok(DynamicVector::new(vec![3.0, 7.0])));
    assert_eq!(
        a.mul_matrix(&b),
        Ok(DynamicMatrix::new(2, 2, vec![19.0, 22.0, 43.0, 50.0]).unwrap())
    );
    assert_eq!(
        DynamicMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .transpose(),
        DynamicMatrix::new(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap()
    );
    assert_eq!(a.determinant(), Ok(-2.0));
    assert_eq!(a.rank(), 2);

    let (l, u, swap_count) = a.lu().unwrap();
    assert_eq!(swap_count, 1);
    // l * u == p * a, where p swapped rows 0 and 1 of a (row 1 holds the larger-magnitude
    // pivot in column 0).
    assert_eq!(
        l.mul_matrix(&u).unwrap(),
        DynamicMatrix::new(2, 2, vec![3.0, 4.0, 1.0, 2.0]).unwrap()
    );

    let (q, r) = a.qr().unwrap();
    assert_eq!((q.rows(), q.cols()), (2, 2));
    assert_eq!((r.rows(), r.cols()), (2, 2));

    let spd = DynamicMatrix::new(2, 2, vec![4.0, 2.0, 2.0, 2.0]).unwrap();
    assert_eq!(
        spd.cholesky(),
        Ok(DynamicMatrix::new(2, 2, vec![2.0, 0.0, 1.0, 1.0]).unwrap())
    );

    let (_, sigma, _) = a.svd().unwrap();
    assert_eq!(sigma.len(), 2);

    let kappa = a.condition_number().unwrap();
    assert!(kappa > 1.0);
}

#[cfg(feature = "alloc")]
#[test]
fn zero_sized_matrix_0x0_construction_and_basic_ops() {
    use rustebra::matrix::DynamicMatrix;

    let m = DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap();
    assert_eq!(m.rows(), 0);
    assert_eq!(m.cols(), 0);

    let m_scaled = m.mul_scalar(2.0);
    assert_eq!(m_scaled.rows(), 0);
    assert_eq!(m_scaled.cols(), 0);

    let m_transposed = m.transpose();
    assert_eq!(m_transposed.rows(), 0);
    assert_eq!(m_transposed.cols(), 0);
}

#[cfg(feature = "alloc")]
#[test]
fn zero_sized_matrix_0x0_arithmetic() {
    use rustebra::matrix::DynamicMatrix;

    let a = DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap();
    let b = DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap();

    assert_eq!(
        a.add(&b),
        Ok(DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap())
    );
    assert_eq!(
        a.sub(&b),
        Ok(DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap())
    );
    assert_eq!(
        a.mul_matrix(&b),
        Ok(DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap())
    );
}

#[cfg(feature = "alloc")]
#[test]
fn zero_sized_matrix_0x0_algorithms() {
    use rustebra::matrix::DynamicMatrix;
    use rustebra::storage::Storage;

    let m = DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap();

    assert_eq!(m.rank(), 0);
    assert_eq!(m.determinant(), Ok(1.0_f64));
    assert_eq!(
        m.qr(),
        Ok((
            DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap(),
            DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap()
        ))
    );
    assert_eq!(
        m.lu(),
        Ok((
            DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap(),
            DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap(),
            0
        ))
    );
    assert_eq!(
        m.cholesky(),
        Ok(DynamicMatrix::new(0_usize, 0_usize, vec![] as Vec<f64>).unwrap())
    );

    let (u, sigma, v) = m.svd().unwrap();
    assert_eq!(u.rows(), 0);
    assert_eq!(u.cols(), 0);
    assert_eq!(sigma.len(), 0);
    assert_eq!(v.rows(), 0);
    assert_eq!(v.cols(), 0);
}

#[cfg(feature = "alloc")]
#[test]
fn zero_sized_matrix_0xn_construction() {
    use rustebra::matrix::DynamicMatrix;

    let m = DynamicMatrix::new(0_usize, 3_usize, vec![] as Vec<f64>).unwrap();
    assert_eq!(m.rows(), 0);
    assert_eq!(m.cols(), 3);

    let m_scaled = m.mul_scalar(2.0);
    assert_eq!(m_scaled.rows(), 0);
    assert_eq!(m_scaled.cols(), 3);

    let m_transposed = m.transpose();
    assert_eq!(m_transposed.rows(), 3);
    assert_eq!(m_transposed.cols(), 0);
}

#[cfg(feature = "alloc")]
#[test]
fn zero_sized_matrix_0xn_arithmetic_fails_on_dimension_mismatch() {
    use rustebra::matrix::DynamicMatrix;

    let a = DynamicMatrix::new(0_usize, 3_usize, vec![] as Vec<f64>).unwrap();
    let b = DynamicMatrix::new(0_usize, 2_usize, vec![] as Vec<f64>).unwrap();

    assert!(a.add(&b).is_err());
    assert!(a.sub(&b).is_err());
}

#[cfg(feature = "alloc")]
#[test]
fn zero_sized_matrix_0xn_same_cols_arithmetic() {
    use rustebra::matrix::DynamicMatrix;

    let a = DynamicMatrix::new(0_usize, 3_usize, vec![] as Vec<f64>).unwrap();
    let b = DynamicMatrix::new(0_usize, 3_usize, vec![] as Vec<f64>).unwrap();

    assert_eq!(
        a.add(&b),
        Ok(DynamicMatrix::new(0_usize, 3_usize, vec![] as Vec<f64>).unwrap())
    );
    assert_eq!(
        a.sub(&b),
        Ok(DynamicMatrix::new(0_usize, 3_usize, vec![] as Vec<f64>).unwrap())
    );
}

#[cfg(feature = "alloc")]
#[test]
fn zero_sized_matrix_nx0_construction() {
    use rustebra::matrix::DynamicMatrix;

    let m = DynamicMatrix::new(3_usize, 0_usize, vec![] as Vec<f64>).unwrap();
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 0);

    let m_scaled = m.mul_scalar(2.0);
    assert_eq!(m_scaled.rows(), 3);
    assert_eq!(m_scaled.cols(), 0);

    let m_transposed = m.transpose();
    assert_eq!(m_transposed.rows(), 0);
    assert_eq!(m_transposed.cols(), 3);
}

#[cfg(feature = "alloc")]
#[test]
fn zero_sized_matrix_nx0_arithmetic_fails_on_dimension_mismatch() {
    use rustebra::matrix::DynamicMatrix;

    let a = DynamicMatrix::new(3_usize, 0_usize, vec![] as Vec<f64>).unwrap();
    let b = DynamicMatrix::new(2_usize, 0_usize, vec![] as Vec<f64>).unwrap();

    assert!(a.add(&b).is_err());
    assert!(a.sub(&b).is_err());
}

#[cfg(feature = "alloc")]
#[test]
fn zero_sized_matrix_nx0_same_rows_arithmetic() {
    use rustebra::matrix::DynamicMatrix;

    let a = DynamicMatrix::new(3_usize, 0_usize, vec![] as Vec<f64>).unwrap();
    let b = DynamicMatrix::new(3_usize, 0_usize, vec![] as Vec<f64>).unwrap();

    assert_eq!(
        a.add(&b),
        Ok(DynamicMatrix::new(3_usize, 0_usize, vec![] as Vec<f64>).unwrap())
    );
    assert_eq!(
        a.sub(&b),
        Ok(DynamicMatrix::new(3_usize, 0_usize, vec![] as Vec<f64>).unwrap())
    );
}

#[cfg(feature = "alloc")]
#[test]
fn zero_sized_matrix_0xn_rank() {
    use rustebra::matrix::DynamicMatrix;

    let m = DynamicMatrix::new(0_usize, 5_usize, vec![] as Vec<f64>).unwrap();
    assert_eq!(m.rank(), 0);
}

#[cfg(feature = "alloc")]
#[test]
fn zero_sized_matrix_nx0_rank() {
    use rustebra::matrix::DynamicMatrix;

    let m = DynamicMatrix::new(5_usize, 0_usize, vec![] as Vec<f64>).unwrap();
    assert_eq!(m.rank(), 0);
}

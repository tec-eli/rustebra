# Matrices

`rustebra` provides two matrix types: `StaticMatrix<T, R, C>`, stack-allocated with
compile-time-known shape, and `DynamicMatrix<T>`, heap-allocated behind the `alloc` feature
with a runtime-known shape. This section covers how to construct them and the operations —
arithmetic, matrix-vector and matrix-matrix products, transpose, rank, and the LU/QR/SVD
decompositions — they support.

# Decompositions

`rustebra` factors matrices via LU (Gaussian elimination with partial pivoting), QR
(orthogonal-triangular factorization), Cholesky (for symmetric positive-definite matrices),
and SVD (singular value decomposition). Each is available both as a low-level function in
`rustebra::algorithm::matrix` operating on `Storage`, and as an ergonomic method on
`StaticMatrix`/`DynamicMatrix`. This section covers LU, QR, and SVD in detail.

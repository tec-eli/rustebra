# Operations

`StaticMatrix` and `DynamicMatrix` support the same core set of operations: element-wise
addition and subtraction, scalar scaling, matrix-vector and matrix-matrix products,
transpose, rank, and the LU, QR, Cholesky, and SVD decompositions, plus condition number.
The example below runs each of them on a `StaticMatrix`.

```rust
{{#include ../../../examples/matrix/static_matrix.rs}}
```

## Gotchas

- On `StaticMatrix`, operations between two matrices (`add`, `sub`, `mul_matrix`,
  `mul_vector`) return a plain value, not a `Result` — the const generics guarantee
  compatible shapes at compile time. The `DynamicMatrix` equivalents return
  `Result<_, DimensionMismatch>` instead, since two `DynamicMatrix`s aren't guaranteed to
  match in shape.
- `qr` returns `Result<_, DimensionMismatch>` on *both* types, because `R >= C` isn't
  something the type system can enforce even for `StaticMatrix` — stable Rust has no way to
  bound one const generic against another.
- `svd` and `condition_number` on `StaticMatrix` take a caller-provided `scratch` buffer
  sized by an explicit formula (`5 * C * C + C + R` for `svd`, `7 * N * N + 3 * N` for
  `condition_number`) — get the size wrong and you get `Err(DimensionMismatch)` back rather
  than a panic. `DynamicMatrix`'s `svd`/`condition_number` allocate their own scratch space
  internally and don't take this parameter.
- `determinant` on `StaticMatrix<T, N, N>` returns
  `Err(DeterminantError::MatrixTooLargeWithoutAlloc)` if the `alloc` feature is disabled and
  `N > 4` — for larger matrices without `alloc`, use
  `rustebra::algorithm::matrix::determinant_lu` directly with your own scratch buffer.
- `cholesky` requires `self` to be symmetric positive-definite; it returns
  `Err(CholeskyError::NotPositiveDefinite)` rather than a garbage result or a panic when
  that doesn't hold.

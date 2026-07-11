# SVD

Singular value decomposition factors any `rows x cols` matrix `a` as `u * diag(sigma) *
vᵗ`, where `u` has orthonormal columns, `sigma` is a length-`cols` vector of non-negative
singular values sorted in descending order, and `v` is a `cols x cols` orthogonal matrix.
Unlike LU or Cholesky, `a` doesn't need to be square — the SVD exists for any matrix.

`rustebra` computes it via `svd_qr_iteration`: it eigendecomposes `aᵗ * a` using a
fixed-iteration QR algorithm (100 iterations, rather than convergence-checked, so behavior
stays predictable in `no_std` contexts) to get `v` and `sigma`, then recovers `u` from
`a * v` scaled by `sigma`. `svd` is the general-purpose entry point, delegating to
`svd_qr_iteration` with an automatically-computed tolerance so callers don't have to pick
one. Both are shown below, alongside the ergonomic
`StaticMatrix`/`DynamicMatrix::svd()` method used in
[Matrix Operations](../04-matrices/operations.md).

```rust
{{#include ../../../examples/algorithm/matrix/svd.rs}}
```

## Gotchas

- `svd`/`svd_qr_iteration` need a caller-provided `scratch` buffer sized `5 * cols * cols +
  cols + rows` — get the length wrong and you get `Err(DimensionMismatch)` back, not a
  panic. `DynamicMatrix::svd()` allocates this internally and doesn't expose the parameter.
- The QR iteration count inside `svd_qr_iteration` is fixed (100), not convergence-checked
  — for matrices with closely clustered singular values, more iterations might be needed
  for full precision than this budget provides.

# QR

QR decomposition factors a `rows x cols` matrix `a` (with `rows >= cols`) as `q * r`,
where `q` has orthonormal columns and `r` is upper triangular. `rustebra` provides two
algorithms:

- `qr_householder` — builds `q` as a full `rows x rows` orthogonal matrix via Householder
  reflections, zeroing out the sub-diagonal of each column in turn.
- `qr_gram_schmidt` — modified Gram-Schmidt orthogonalization, producing a `rows x cols`
  `q` (only as many orthonormal columns as `a` has, not a full square orthogonal matrix).
  Projecting each column against the running, already-orthogonalized vector (rather than
  the original column) is what keeps rounding error from compounding as badly as classical
  Gram-Schmidt does.

`qr` is the general-purpose entry point and currently delegates to `qr_householder`. Both
algorithms, plus the entry point, are demonstrated below, alongside the ergonomic
`StaticMatrix`/`DynamicMatrix::qr()` method used in
[Matrix Operations](../04-matrices/operations.md).

```rust
{{#include ../../../examples/algorithm/matrix/qr.rs}}
```

## Gotchas

- `qr_householder`'s `q` is `rows x rows`; `qr_gram_schmidt`'s `q` is only `rows x cols`.
  Don't assume both produce a same-shaped `q` if you swap between them.
- If a column of `a` lies entirely in the span of the previous columns (or is zero), Gram-
  Schmidt has no direction left to normalize into that column of `q` — rather than erroring,
  it leaves that column as `0`, since linear dependence is a property of the input, not a
  malformed call.
- Both algorithms require `rows >= cols`; this can't be checked by the type system on
  `StaticMatrix` (stable Rust can't bound one const generic against another), so `qr()`
  returns `Result` on both matrix types even though most other `StaticMatrix` operations
  don't need to.

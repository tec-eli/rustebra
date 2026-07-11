# Construction

Like vectors, rustebra matrices come in a stack-allocated and a heap-allocated flavor.

`StaticMatrix<T, R, C>` stores its shape as const generics, so `R` and `C` are part of the
type and known at compile time. It's constructed from an array of rows — `[[T; C]; R]` —
and requires no allocator:

```rust
use rustebra::matrix::StaticMatrix;

let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
```

`DynamicMatrix<T>` is heap-allocated (available behind the `alloc` feature) and stores its
shape — `rows` and `cols` — as runtime fields rather than type parameters. It's constructed
from a flat, row-major `Vec<T>` plus the explicit row and column counts, and returns a
`Result` since the data's length isn't statically guaranteed to match `rows * cols`:

```rust
use rustebra::matrix::DynamicMatrix;

let m = DynamicMatrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
assert_eq!(m.rows(), 2);
assert_eq!(m.cols(), 2);
```

## Gotchas

- `StaticMatrix<T, 2, 3>` and `StaticMatrix<T, 3, 2>` are unrelated types, just like
  differently-sized `StaticVector`s — the compiler rejects shape-mismatched operations at
  compile time rather than at runtime.
- `DynamicMatrix::new` returns `Err(DimensionMismatch)` rather than panicking if
  `data.len() != rows * cols` — check the result rather than assuming construction always
  succeeds.

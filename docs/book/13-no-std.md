# no_std & Embedded

`rustebra` is `no_std` by default: `src/lib.rs` declares `#![cfg_attr(not(test), no_std)]`,
so the crate has no dependency on the standard library (outside test builds) and no
allocator requirement unless you opt in.

Without any feature flags, only stack-allocated, compile-time-sized types are available:
`StaticVector<T, N>` and `StaticMatrix<T, R, C>`, whose lengths/shapes are const generics.
These work on embedded targets with no heap at all.

Enabling the `alloc` feature pulls in `extern crate alloc` and unlocks heap-allocated,
runtime-sized types: `DynamicVector<T>`, `DynamicMatrix<T>`, and every sparse matrix type in
`rustebra::sparse` (`CooMatrix`, `CsrMatrix`, `CscMatrix`, and their sorted variants), all
of which are gated behind `#[cfg(feature = "alloc")]` and simply don't exist in the crate's
public API without it.

```toml
# no_std, no allocator: only StaticVector/StaticMatrix are available.
[dependencies]
rustebra = "0.3"

# adds DynamicVector/DynamicMatrix and the sparse module.
[dependencies]
rustebra = { version = "0.3", features = ["alloc"] }
```

## Gotchas

- Code that references `DynamicVector`, `DynamicMatrix`, or anything in `rustebra::sparse`
  without the `alloc` feature enabled won't compile — these items aren't hidden behind a
  runtime check, they're absent from the crate entirely.
- `StaticMatrix::determinant` on matrices larger than `4x4` returns
  `Err(DeterminantError::MatrixTooLargeWithoutAlloc)` when the `alloc` feature is disabled —
  cofactor expansion beyond that size needs heap-allocated scratch space that isn't
  available in a pure `no_std` build.

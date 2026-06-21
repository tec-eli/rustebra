# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `Scalar` trait (zero, one, add, sub, mul, div) defining the scalar layer.
- `Scalar` implementation for `f32`.
- `Scalar` implementation for `f64`.
- Unit tests for the `f32` and `f64` `Scalar` implementations.
- `Storage` trait (element access, length) defining the storage layer.
- `StaticStorage<T, const N: usize>`, a stack-allocated `Storage` backed by a const-generic array.
- `alloc` feature flag (per ADR 0001), and `DynamicStorage<T>`, a heap-allocated `Storage` backed by `Vec<T>`, gated behind it.
- Unit tests for the `StaticStorage` and `DynamicStorage` implementations.
- Verified the crate builds, lints, and tests clean with `--no-default-features` (no `alloc`).
- `Scalar::sqrt`, computed via fixed-iteration Newton-Raphson (Babylonian) iteration, since `no_std` has no `f32`/`f64` `sqrt`; returns `0` for `self <= 0` (no real square root, and no `NaN` sentinel since `Scalar` must stay infallible for future non-float implementors).
- Unit tests for `Scalar::sqrt` on `f32` and `f64` (exact perfect squares, tolerance-bounded irrational results, negative-input behavior).
- `algorithm::vector::add`, the first algorithm-layer function: element-wise vector addition generic over `Storage` + `Scalar`, writing into a caller-provided output slice (since `Storage` has no way to construct a new instance) and returning `Result<(), LengthMismatch>` instead of panicking on mismatched lengths, per ADR 0004.
- Unit tests for `algorithm::vector::add` (matching-length addition, mismatched operand lengths, mismatched output length).
- `algorithm::vector::sub`, element-wise vector subtraction, following the same `Storage` + `Scalar` generic, output-slice, `Result<(), LengthMismatch>` pattern as `add`.
- Unit tests for `algorithm::vector::sub` (matching-length subtraction, mismatched operand lengths, mismatched output length).
- `algorithm::vector::scale`, element-wise scalar multiplication; still returns `Result<(), LengthMismatch>` since the output buffer's length can disagree with the input's even with only one `Storage` operand.
- Unit tests for `algorithm::vector::scale` (known factor, scaling by zero, mismatched output length).
- `algorithm::vector::dot`, the inner product, returning `Result<T, LengthMismatch>` instead of panicking on mismatched lengths; documented as the building block for the upcoming `norm` (`‖v‖ = sqrt(dot(v, v))`).
- Unit tests for `algorithm::vector::dot` (known vectors, mismatched lengths).
- `algorithm::vector::norm`, the Euclidean (L2) norm, implemented as `dot(a, a).sqrt()`; infallible (returns `T` directly), since comparing `a` against itself can never produce a length mismatch.
- Unit tests for `algorithm::vector::norm` (exact known case, tolerance-bounded irrational case).
- `StaticVector<T, const N: usize>`, the first public API layer type: wires `StaticStorage` together with `algorithm::vector`'s `add`/`sub`/`scale`/`dot`/`norm` into ergonomic methods (`v1.add(&v2)`, etc.), constructed from a fixed-size array via `StaticVector::new`.
- `PartialEq` and `Debug` implementations for `StaticVector`, to support equality assertions and printing in tests without exposing raw element access.
- Unit tests for `StaticVector` construction and each operation (`add`, `sub`, `scale`, `dot`, `norm`), confirming the wiring to the algorithm layer.
- `DynamicVector<T>`, the heap-allocated counterpart to `StaticVector`, gated behind the `alloc` feature: wires `DynamicStorage` together with `algorithm::vector`'s functions the same way, but since two `DynamicVector`s aren't guaranteed by the type system to share a length, `add`/`sub`/`dot` return `Result<_, LengthMismatch>` for real (not just defensively) rather than panicking, per ADR 0004.
- `PartialEq` and `Debug` implementations for `DynamicVector`, accounting for operands of different lengths.
- Unit tests for `DynamicVector` construction and each operation, including mismatched-length cases for `add`, `sub`, and `dot`.
- `examples/static_vector.rs` and `examples/dynamic_vector.rs`, runnable end-to-end demonstrations of `StaticVector` and `DynamicVector` (the latter gated via `required-features = ["alloc"]` in `Cargo.toml`, so `cargo build`/`run --examples` skip it cleanly without the feature).
- `tests/vector.rs`, a black-box integration test exercising the public `StaticVector`/`DynamicVector` API end-to-end (construction plus `add`/`sub`/`scale`/`dot`/`norm`) using only `pub` items, with the `DynamicVector` case gated behind `alloc`.
- `algorithm::matrix::add`, the first matrix-layer function: element-wise matrix addition generic over `Storage` + `Scalar`. Matrices are stored row-major in a flat `Storage` (which has no concept of rows/columns itself, per ADR 0003), so `rows`/`cols` are passed explicitly alongside each operand; returns `Result<(), DimensionMismatch>` instead of panicking on a row, column, or flat-length disagreement, per ADR 0004.
- Unit tests for `algorithm::matrix::add` (matching dimensions, mismatched rows, mismatched columns, mismatched output length).
- `algorithm::matrix::sub`, element-wise matrix subtraction, following the same row-major, explicit-shape, `Result<(), DimensionMismatch>` pattern as `add`.
- `algorithm::matrix::mul_scalar`, element-wise scalar multiplication of a matrix.
- `algorithm::matrix::mul_vector`, the matrix-vector product: each output element is `dot(row_i, v)`, reusing `algorithm::vector::dot` via a private zero-copy `Row` `Storage` view instead of re-deriving the summation.
- `algorithm::matrix::mul_matrix`, the matrix-matrix product: each output element is `dot(row_i of a, col_j of b)`, reusing `algorithm::vector::dot` via the `Row` view and a new private `Column` `Storage` view; validates `a`'s column count against `b`'s row count (the multiplication's inner dimension).
- `algorithm::matrix::transpose`, pure reindexing with no `Scalar` arithmetic, so it only requires `T: Copy` rather than `T: Scalar`.
- Unit tests for `sub`, `mul_scalar`, `mul_vector`, `mul_matrix`, and `transpose` (known-value cases plus dimension-mismatch error cases).
- `Storage` implementations for `StaticVector` and `DynamicVector` (delegating to their internal storage), so they can be passed directly to generic `Storage`-based functions — needed for `StaticMatrix`/`DynamicMatrix` to multiply against them.
- `StaticMatrix<T, const R: usize, const C: usize>`, the first public-API matrix type: stored as `[[T; C]; R]` (since `R * C` can't be a single const-generic array length on stable Rust, unlike `StaticVector`, it implements `Storage` directly over that field rather than wrapping `StaticStorage`). Wires `add`/`sub`/`mul_scalar`/`mul_vector`/`mul_matrix`/`transpose` to `algorithm::matrix`; `mul_matrix<const C2: usize>` changes shape (`R x C` times `C x C2` gives `R x C2`), with the inner-dimension check statically guaranteed (and thus unreachable) by the type system, same as the other same-shape operations.
- Unit tests for `StaticMatrix` construction and each operation, confirming the wiring to the algorithm layer.
- `DynamicMatrix<T>`, the heap-allocated counterpart to `StaticMatrix`, gated behind the `alloc` feature: shape (`rows`/`cols`) lives in its fields rather than its type, so `add`/`sub`/`mul_vector`/`mul_matrix` return `Result<_, DimensionMismatch>` for real (not just defensively) rather than panicking, per ADR 0004; `mul_scalar`/`transpose` stay infallible, sized from `self` alone. Constructed via `DynamicMatrix::new(rows, cols, Vec<T>)`, itself fallible since the flat data's length can disagree with the claimed shape.
- `PartialEq` and `Debug` implementations for `DynamicMatrix`, accounting for operands of different shapes.
- Unit tests for `DynamicMatrix` construction and each operation, including mismatched-shape/inner-dimension error cases for `add`, `sub`, `mul_vector`, and `mul_matrix`.
- `examples/static_matrix.rs` and `examples/dynamic_matrix.rs`, runnable end-to-end demonstrations of `StaticMatrix` and `DynamicMatrix` (the latter gated via `required-features = ["alloc"]` in `Cargo.toml`, matching the vector examples' pattern).
- `tests/matrix.rs`, a black-box integration test exercising the public `StaticMatrix`/`DynamicMatrix` API end-to-end (construction plus `add`/`sub`/`mul_scalar`/`mul_vector`/`mul_matrix`/`transpose`) using only `pub` items, with the `DynamicMatrix` case gated behind `alloc`.
- `algorithm::matrix::determinant`, computed via recursive cofactor expansion along the first row; only defined for square matrices, returning `Result<T, DimensionMismatch>` instead of panicking on non-square input, per ADR 0004. Each minor is a zero-copy `Storage` view (a private `Minor` type) rather than a materialized submatrix, since submatrix sizes are only known at runtime in this `no_std`-first crate; `Minor` holds its source as `&dyn Storage<Item = T>` rather than a generic parameter, since a generic parameter would make every recursion level its own type (`Minor<Minor<Minor<...>>>`) with no compile-time bound on recursion depth — the one exception in this module to preferring generics over `dyn Trait`.
- Unit tests for `algorithm::matrix::determinant` (known 2x2 and 3x3 values, a singular matrix with a zero row, and non-square input as an error case).

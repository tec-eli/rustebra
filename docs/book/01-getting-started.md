# Getting Started

`rustebra` is a hybrid `no_std`/`alloc` linear algebra crate for Rust, scaling from embedded
targets to dynamic Krylov subspace solvers. The crate is `no_std` by default (`#![cfg_attr(not(test),
no_std)]` in `lib.rs`), so it has no dependency on the standard library and no allocator
requirement unless you opt in.

## Adding the dependency

Add `rustebra` to `Cargo.toml`:

```toml
[dependencies]
rustebra = "0.3"
```

## `no_std` vs. `alloc`

By default, only stack-allocated, compile-time-sized types are available — for example
`StaticVector<T, N>`, whose length `N` is a const generic. This works on embedded targets
with no heap.

Enabling the `alloc` feature pulls in `extern crate alloc` and unlocks heap-allocated,
runtime-sized types such as `DynamicVector<T>`:

```toml
[dependencies]
rustebra = { version = "0.3", features = ["alloc"] }
```

## A first example

The core building block is `Scalar`, a trait implemented for `f32` and `f64` that defines the
arithmetic surface (`add`, `sub`, `mul`, `div`, `sqrt`, `sin`, `cos`) the rest of the crate is
generic over. Vector types like `StaticVector` are built on top of it:

```rust
use rustebra::vector::StaticVector;

let a = StaticVector::new([1.0, 2.0, 3.0]);
let b = StaticVector::new([4.0, 5.0, 6.0]);

println!("a + b = {:?}", a.add(&b));
println!("||a|| = {:.4}", a.norm());
```

`StaticVector::new` infers `N` from the array literal's length, so `a` and `b` here are both
`StaticVector<f64, 3>`. See [Vectors](./03-vectors/README.md) for the full construction and
operations surface.

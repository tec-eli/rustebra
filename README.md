[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# rustebra

A linear algebra and sparse matrix library for Rust, designed to run on strict
`#![no_std]` embedded targets by default, while scaling up to dynamic, allocation-backed
algorithms — including Krylov subspace methods such as Lanczos and Arnoldi iteration — when
a heap is available.

## Status

Early development. The architecture and scope are being defined before implementation; see
`docs/adr/` for the decisions made so far. No version has been published yet.

## Why this exists

Rust currently lacks a linear algebra library that is simultaneously serious about `no_std`
support and complete enough to cover sparse matrices and iterative solvers. Existing options
tend to assume a heap is always available, or only provide a partial set of operations for
constrained environments. This project aims to close that gap.

## Design principles

- **No allocator required by default.** The core of the library works entirely on the stack,
  using const generics to fix sizes at compile time.
- **Allocation is opt-in.** Dynamic, heap-backed data structures and algorithms are available
  behind a feature flag, for use in environments with an operating system.
- **Generic over numeric precision.** Operations are written to work with different
  floating-point types, reflecting the range of hardware this library targets — from
  microcontrollers without double-precision floating-point units to desktop and server
  systems.
- **Explicit error handling.** Recoverable failures are reported through `Result`, not
  panics, since an uncontrolled abort is often unacceptable in embedded contexts.

## Usage

This is a library crate, so "running" it means building it and running its test suite:

```sh
cargo build
cargo test
```

By default, the crate builds `#![no_std]` with no allocator. To opt into the heap-backed
data structures and algorithms (Krylov subspace methods, dynamic matrices, etc.), enable the
`alloc` feature:

```sh
cargo build --features alloc
cargo test --features alloc
```

## Documentation

- `docs/` — architecture decision records, documenting why the project is built the
  way it is.

To generate and view the API documentation locally:

```sh
cargo doc --open
```

## License

Apache License 2.0.
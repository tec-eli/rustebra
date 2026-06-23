---
layout: default
title: rustebra
---

<div align="center">
  <img src="assets/banner.png" alt="rustebra" width="100%"/>
</div>

<div align="center">

[![Crates.io](https://img.shields.io/crates/v/rustebra?style=flat-square&color=fc8d62)](https://crates.io/crates/rustebra)
[![docs.rs](https://img.shields.io/docsrs/rustebra?style=flat-square&label=docs.rs)](https://docs.rs/rustebra)
[![CI](https://img.shields.io/github/actions/workflow/status/tec-eli/rustebra/ci.yml?style=flat-square&label=CI)](https://github.com/tec-eli/rustebra/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![no_std](https://img.shields.io/badge/no__std-compatible-success?style=flat-square)](https://docs.rust-embedded.org/book/)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-orange?style=flat-square)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0.html)

**A hybrid `no_std`/`alloc` linear algebra library for Rust.**
Stack-first by default. Scales to sparse matrices and Krylov subspace solvers when a heap is available.

[API Reference](api/rustebra/) · [Architecture Decisions](adr/) · [Algorithms](algorithms/) · [Contributing](https://github.com/tec-eli/rustebra/blob/main/CONTRIBUTING.md)

</div>

---

## Status

Early development. The architecture and scope are being defined before implementation; see
[Architecture Decisions](adr/) for the decisions made so far. No version has been published yet.

## Why this exists

Rust currently lacks a linear algebra library that is simultaneously serious about `no_std`
support and complete enough to cover sparse matrices and iterative solvers. Existing options
tend to assume a heap is always available, or only provide a partial set of operations for
constrained environments. This project aims to close that gap.

## Design principles

- **No allocator required by default.** The core of the library works entirely on the stack,
  using const generics to fix sizes at compile time.
- **Allocation is opt-in.** Dynamic, heap-backed data structures and algorithms are available
  behind the `alloc` feature flag, for use in environments with an operating system.
- **Generic over numeric precision.** Operations are written to work with different
  floating-point types, reflecting the range of hardware this library targets — from
  microcontrollers without double-precision floating-point units to desktop and server
  systems.
- **Explicit error handling.** Recoverable failures are reported through `Result`, not
  panics, since an uncontrolled abort is often unacceptable in embedded contexts.

## Usage

```toml
[dependencies]
rustebra = "0.1"

# Optional: heap-backed structures and Krylov solvers
rustebra = { version = "0.1", features = ["alloc"] }
```

Build and test locally:

```sh
# no_std build (default)
cargo build
cargo test

# with alloc feature
cargo build --features alloc
cargo test --features alloc
```

## Documentation

- **[API reference](api/rustebra/)** — generated from `cargo doc`.
- **[Architecture decisions](adr/)** — records of the key design choices made during development.
- **[Algorithms](algorithms/)** — mathematical reference for every algorithm implemented in this project.

To generate and browse the API docs locally:

```sh
cargo doc --open
```

## Contributing

Contributions are welcome. Please read
[CONTRIBUTING.md](https://github.com/tec-eli/rustebra/blob/main/CONTRIBUTING.md) before
opening a pull request.

---

## License

Licensed under the [Apache License 2.0](https://github.com/tec-eli/rustebra/blob/main/LICENSE).

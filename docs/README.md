# rustebra

![rustebra](assets/banner.png)

[![Crates.io](https://img.shields.io/crates/v/rustebra?style=flat-square&color=fc8d62)](https://crates.io/crates/rustebra)
[![docs.rs](https://img.shields.io/docsrs/rustebra?style=flat-square&label=docs.rs)](https://docs.rs/rustebra)
[![CI](https://img.shields.io/github/actions/workflow/status/tec-eli/rustebra/ci.yml?style=flat-square&label=CI)](https://github.com/tec-eli/rustebra/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![no_std](https://img.shields.io/badge/no__std-compatible-success?style=flat-square)](https://docs.rust-embedded.org/book/)

**A hybrid `no_std`/`alloc` linear algebra library for Rust.**
Stack-first by default. Scales to sparse matrices and Krylov subspace solvers when a heap
is available.

<p>
  <a class="rb-cta-inline" href="rustdoc/rustebra/index.html">Go to the API Reference &rarr;</a>
</p>

## Why this exists

Rust currently lacks a linear algebra library that is simultaneously serious about `no_std`
support and complete enough to cover sparse matrices and iterative solvers. Existing options
tend to assume a heap is always available, or only provide a partial set of operations for
constrained environments. rustebra aims to close that gap.

## Design principles

- **No allocator required by default.** The core of the library works entirely on the
  stack, using const generics to fix sizes at compile time.
- **Allocation is opt-in.** Dynamic, heap-backed data structures and algorithms are
  available behind the `alloc` feature flag.
- **Generic over numeric precision.** Operations work across floating-point types, from
  microcontrollers without double-precision units to desktop and server systems.
- **Explicit error handling.** Recoverable failures are reported through `Result`, not
  panics.

## Usage

```toml
[dependencies]
rustebra = "0.2"

# Optional: heap-backed structures and Krylov solvers
rustebra = { version = "0.2", features = ["alloc"] }
```

```sh
# no_std build (default)
cargo build
cargo test

# with the alloc feature
cargo build --features alloc
cargo test --features alloc
```

## Where to go next

- **[API Reference](rustdoc/rustebra/index.html)** — generated from `cargo doc`.
- **[Algorithms](algorithms/index.md)** — mathematical reference for every algorithm
  implemented in this project.
- **[Architecture Decisions](adr/index.md)** — records of the key design choices made
  during development.

## Contributing

Contributions are welcome. See
[CONTRIBUTING.md](https://github.com/tec-eli/rustebra/blob/main/CONTRIBUTING.md) before
opening a pull request.

## License

Licensed under the [Apache License 2.0](https://github.com/tec-eli/rustebra/blob/main/LICENSE.md).

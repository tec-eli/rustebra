# Contributing to rustebra

Thank you for your interest in contributing! This document explains how to get started.

## Before you open a PR

- Check the [open issues](https://github.com/tec-eli/rustebra/issues) to see if your idea is already being tracked.
- For significant changes, open an issue first to discuss the approach. Architecture decisions are recorded in [`docs/adr/`](docs/adr/).
- All contributions must be compatible with `#![no_std]` unless explicitly gated behind the `alloc` feature.

## Development setup

```sh
git clone https://github.com/tec-eli/rustebra
cd rustebra

# Default no_std build
cargo build
cargo test

# With heap-backed structures
cargo build --features alloc
cargo test --features alloc

# Generate API docs
cargo doc --open
```

Rust edition 2024 is required. The minimum supported Rust version (MSRV) is 1.85.

## Code conventions

- Recoverable errors use `Result`, not `panic!` or `unwrap`.
- Public items must have doc comments (`///`). Include at least one `# Examples` block for public functions.
- Feature-gated code goes behind `#[cfg(feature = "alloc")]`.
- Run `cargo clippy` and `cargo fmt` before committing.

## Pull request checklist

- [ ] `cargo test` passes (default and `--features alloc`)
- [ ] `cargo clippy -- -D warnings` is clean
- [ ] `cargo fmt --check` passes
- [ ] New public API has doc comments and an example
- [ ] `CHANGELOG.md` entry added under `[Unreleased]` if applicable

## Adding yourself as a contributor

This project uses [all-contributors](https://allcontributors.org/). After your PR is merged,
a maintainer will add you to the contributors table in `README.md`. If you'd like to add
yourself, install the CLI and run:

```sh
npx all-contributors add <your-github-username> <contribution-type>
```

Contribution types: `code`, `doc`, `bug`, `test`, `design`, `ideas`, `review`.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

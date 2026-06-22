# TODO

Each version below is a real, publishable release — not a sub-version. A checkbox is small
enough to be one focused unit of work (implement, test, commit), but it belongs to the
version it ships in, nothing finer-grained than that. The project stays on `0.x` until every
group below is implemented and the public API has been reviewed as a whole — `1.0.0` is an
explicit future decision, not a default endpoint.

---

## v0.1.0 — Initial release

- [x] Define the `Scalar` trait (minimal: zero, one, add, sub, mul, div)
- [x] Implement `Scalar` for `f32`
- [x] Implement `Scalar` for `f64`
- [x] Unit tests for `Scalar` implementations
- [x] Define the minimal `Storage` trait (element access, dimensions)
- [x] Implement static storage backed by a const-generic array
- [x] Implement dynamic storage backed by `Vec<T>`, behind the `alloc` feature
- [x] Unit tests for both storage implementations
- [x] Smoke test confirming the crate builds with `--no-default-features` (no `alloc`)
- [x] `sqrt` on the `Scalar` trait (fixed-iteration Newton-Raphson, no `std` dependency)
- [x] Vector operations: `add`, `sub`, `scale`, `dot`, `norm`, each with doc-comment +
  compiling example
- [x] `StaticVector<T, const N: usize>` and `DynamicVector<T>` (behind `alloc`), each with an
  example in `examples/`
- [x] Matrix operations: `add`, `sub`, `mul_scalar`, `mul` (matrix×vector and matrix×matrix),
  `transpose`, each with doc-comment + compiling example
- [x] `StaticMatrix<T, const R: usize, const C: usize>` and `DynamicMatrix<T>` (behind
  `alloc`), each with an example in `examples/`
- [x] External tests in `tests/` covering the public vector and matrix API end-to-end
- [x] `Cargo.toml` metadata complete: `license`, `description`, `repository`
- [x] `LICENSE` file present at repo root with the copyright line filled in
- [ ] Working tree clean (no uncommitted changes) — `cargo publish --dry-run` passes without
  `--allow-dirty`
- [x] `cargo fmt --check`, `cargo clippy --all-features -- -D warnings`,
  `cargo test --no-default-features`, `cargo test --all-features` all clean
- [x] `CHANGELOG.md` has a `[0.1.0]` section (moved out of `[Unreleased]`)
- [x] Tag `v0.1.0`

---

## v0.2.0 — Matrix decompositions and structural properties

- [x] Determinant
- [x] sin
- [x] cos
- [x] Rank
- [x] LU decomposition
- [x] QR decomposition
- [x] Cholesky decomposition
- [ ] Singular Value Decomposition (SVD)
- [ ] Condition number
- [x] no-alloc end-to-end test (custom panicking allocator confirms StaticVector and
  StaticMatrix never touch the heap)

## v0.3.0 — Sparse matrices

- [ ] COO representation
- [ ] CSR representation
- [ ] CSC representation
- [ ] Conversions between representations
- [ ] Sparse operations: add, scalar multiply, sparse×dense-vector, sparse×sparse,
  sparse×dense-matrix

## v0.4.0 — Krylov subspace methods

- [ ] Power iteration / inverse power iteration
- [ ] Lanczos iteration
- [ ] Conjugate Gradient (CG)
- [ ] Arnoldi iteration
- [ ] GMRES

## v0.5.0 — Numerical calculus

- [ ] Numerical differentiation (finite differences)
- [ ] Numerical integration (trapezoidal, Simpson's, Gaussian quadrature)
- [ ] Root finding (bisection, Newton-Raphson, secant)
- [ ] Interpolation (linear, polynomial, cubic spline)
- [ ] Basic optimization (gradient descent, line search)

## v0.6.0 — Dynamical systems

- [ ] ODE solvers (Euler, Runge-Kutta, adaptive step-size, implicit methods)
- [ ] Systems of coupled ODEs
- [ ] Canonical examples (logistic map, Lorenz system, Rössler system)
- [ ] Fixed points and stability analysis
- [ ] Lyapunov exponents
- [ ] Bifurcation analysis

## v0.7.0 — Signal processing

- [ ] FIR / IIR filtering
- [ ] Convolution / correlation
- [ ] DFT / FFT
- [ ] Window functions
- [ ] Signal analysis (RMS, peak detection, zero-crossing rate, spectral analysis)
- [ ] Resampling (decimation, interpolation)

---

## 1.0.0 — not yet scoped

To be defined once every group above is implemented and the public API has been reviewed as
a whole.
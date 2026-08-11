# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Released]

## [0.5.0] - 2026-08-03

### Added

- **Lanczos iteration.** `krylov::lanczos` tridiagonalizes a real symmetric matrix over a compile-time-sized Krylov
  subspace, returning the projection as a new `TridiagonalMatrix<T, K>` and filling an orthonormal basis. Uses full
  reorthogonalization, so the basis stays orthonormal to working precision.
- **Arnoldi iteration.** `krylov::arnoldi` reduces a general, possibly non-symmetric matrix to upper Hessenberg form
  over a compile-time-sized Krylov subspace, returning the projection as a new `HessenbergMatrix<T, K>` and filling an
  orthonormal basis. Unlike Lanczos, there's no three-term recurrence to exploit, so every step orthogonalizes
  (via modified Gram-Schmidt) against the entire basis built so far, not just the two previous vectors.
- **Conjugate Gradient.** `krylov::conjugate_gradient` solves a symmetric positive-definite (SPD) linear system
  `A x = b` iteratively, without factorizing `A`, converging in at most `n` steps in exact arithmetic. Detects
  non-SPD input operationally — a search direction whose curvature `p · (A p)` isn't positive — rather than assuming
  the caller already validated it.
- **GMRES(m), restarted.** `krylov::gmres`, gated behind the `alloc` feature, solves a general (possibly
  non-symmetric) linear system `A x = b`: Arnoldi builds an `M`-dimensional Krylov basis from the current residual,
  the resulting least-squares problem is solved via Givens rotations, and the cycle restarts from the improved
  iterate until the residual meets tolerance or the restart budget runs out. Takes `A` as a `SparseLinearOp` so
  restart cycles reuse the same workspace instead of allocating on every application.
- `storage::Basis<T, K>`, a const-generic view over caller-provided memory holding the `K` basis vectors a Krylov
  method builds up — the storage-layer piece Arnoldi and GMRES(m) reuse.
- `ConvergenceError::Breakdown`, reported when the Krylov subspace turns out to be invariant before reaching the
  requested dimension (e.g. a repeated eigenvalue, or a starting vector inside a small invariant subspace). For
  Arnoldi and GMRES specifically, this is instead reported as a successful, smaller-than-requested basis, since the
  invariant subspace found is still exact and useful.
- Property tests for `qr`, `cholesky_decompose`, and `svd` that check each decomposition's defining mathematical
  invariants directly (orthogonality, triangularity, reconstruction, positive-definiteness) instead of comparing
  against nalgebra, so they also cover shapes nalgebra wouldn't be a fair oracle for.
- Edge-case tests for dimension extremes — degenerate `0xn`/`nx0`/`0x0` shapes, the smallest nontrivial shape
  (`1x1`), and very rectangular shapes — across the dense decompositions and sparse matrix construction.

## [Released]

## [0.4.0] - 2026-07-11

### Added

- **Eigenvalue solvers.** A new `krylov` module finds a matrix's dominant eigenvalue (`power_iteration`) or its
  smallest-magnitude eigenvalue (`inverse_power_iteration`), returning the eigenvalue with its eigenvector. Works with
  sparse or dense matrices, with configurable iteration budget and precision.
- `ConvergenceError::NonFinite`, reported when a solver's numbers blow up (infinity or NaN) instead of quietly burning
  through the rest of its iteration budget.
- `sparse::validate`, a helper for checking that a sparse matrix implementation is wired up correctly.
- `SortedCscMatrix`, matching the existing `SortedCsrMatrix`.
- Differential property tests checking `qr_householder`, `svd`, and `cholesky_decompose` against nalgebra's
  independently computed results.
- Edge-case tests for NaN/Infinity propagation in dense matrix ops, unusual sparse matrix shapes (empty, fully dense,
  diagonal-only), and CSR↔CSC round-trips.
- `# Examples` docs for the public error types that were still missing one.
- Specs with Architecture decision.
- `book` section with the mathematics of the library.

### Changed

- **Breaking:** `add_csr`/`add_csc` and `csr_to_csc`/`csc_to_csr` now return `SortedCsrMatrix`/`SortedCscMatrix`
  instead of a plain matrix, making their already-sorted output visible in the type instead of just the docs.
- **Breaking:** `SparseLinearOp::apply` now writes into a caller-provided buffer instead of allocating on every call,
  so iterative solvers can reuse it across iterations. Use `matvec_csr`/`matvec_csc` if you want a plain `Vec` back.
- Archived ADRs.

### Fixed

- `sin`/`cos` could be noticeably inaccurate for inputs far from zero (close to `2π`, or near their own zero
  crossings).
- `svd`'s tolerance for negligible singular values was scaled by `cols` only, which was too tight for tall matrices;
  now scaled by `max(rows, cols)`.

## [Released]

## [0.3.2] - 2026-06-30

### Added

- A property test for converting between sparse matrix formats.

### Changed

- Small internal cleanup to avoid a repeated allocation inside sparse matrix addition.
- Reorganized where tests live: firmware tests moved to their own folder, and the old black-box tests moved under a
  general `integration` folder.
- Added `prune_csc`, the column-major counterpart to `prune_csr` (dropping near-zero entries from a sparse matrix).

### Fixed

- Multiplying a sparse matrix by a wide dense matrix could silently overflow (and panic or give a wrong answer) on
  32-bit targets; the size calculation is now checked for overflow.
- `prune_csr`/`prune_csc` treated a negative tolerance in a way that inverted their keep/drop logic instead of just
  ignoring it; a negative tolerance is now clamped to zero.
- Adding or multiplying sparse matrices could leave explicit zero entries in the result when values happened to cancel
  out exactly; those are no longer stored.

## [0.3.1] - 2026-06-28

### Added

- Documentation covering the `x_cols == 0` edge case for sparse-times-dense multiplication.
- New examples covering the sparse matrix module: construction, converting between formats, and core operations.

### Changed

- Updated the list of known issues.

### Fixed

- The publish workflow now passes the crates.io token as an environment variable instead of a command-line flag.

## [0.3.0] - 2026-06-28

### Added

- **A whole new sparse matrix module.** Three storage formats — COO, CSR, and CSC — each validating their input (no
  panics on malformed data), plus functions to convert between them, add two sparse matrices, multiply a sparse matrix
  by a vector or by a dense matrix, multiply two sparse matrices together, scale a sparse matrix by a constant, and drop
  near-zero entries. A `SortedCsrMatrix` type marks, in the type system, matrices whose entries are known to be sorted —
  useful for algorithms that need that guarantee.
- A `SparseLinearOp` trait so solver code can work generically with "any matrix-like thing I can multiply by a vector,"
  without caring whether it's backed by CSR, CSC, or something else.
- A shared `DimensionMismatch` error, used consistently across every sparse operation that can fail because of
  mismatched shapes.
- **Hardware examples.** A separate workspace under `firmware/` for running rustebra on real embedded targets, starting
  with an ARM Cortex-M3 example (runnable in an emulator) doing low-pass filtering on simulated sensor data — zero heap
  allocation.
- `lu`, `qr`, `cholesky`, `svd`, and `condition_number` are now available directly as methods on
  `StaticMatrix`/`DynamicMatrix`, so you no longer need to reach for the lower-level algorithm functions and hand-size
  their scratch buffers to use them day-to-day.
- `FloatTolerance`, exposing machine epsilon for `f32`/`f64`, used to compute sensible default tolerances automatically
  for callers who don't want to think about numerical precision.
- `rank_with_tolerance`, for callers who *do* want to set that threshold explicitly.
- More edge-case tests exercising real floating-point rounding noise (not just tidy exact-zero inputs) around the
  tolerance boundary for `rank`, `cholesky`, `svd`, and `condition_number`.

### Changed

- Partial pivoting (used by `lu`, `rank`, and friends) now genuinely picks the largest-magnitude candidate at each step,
  instead of just the first nonzero one it found — the actual guarantee "partial pivoting" is supposed to provide for
  keeping rounding error in check, which the old approach didn't really deliver despite the name.
- `cholesky`, `svd`, and `condition_number` now treat "close enough to zero" using a proper tolerance instead of exact
  equality, so tiny floating-point noise no longer causes a valid input to be misclassified. The tolerance is computed
  automatically for you; the lower-level functions (`cholesky_decompose`, etc.) let you pass your own instead. The
  elimination-based functions (`lu`, `qr`) deliberately keep exact-zero checks — for those, zero is an algebraic fact
  about the input, not a judgment call.
- Bumped to `0.3.0` and updated crate keywords/categories.

### Fixed

- Computing a determinant for a large matrix without the `alloc` feature enabled now returns a clear error instead of
  silently falling back to a factorial-time algorithm.
- Cholesky decomposition now rejects non-symmetric input with a clear error, instead of silently using only half the
  matrix and producing a wrong answer with no warning.

### Removed

- A brittle "no accidental heap allocation" test that could mask its own real failures behind an unrelated crash. The
  cross-compilation check to a heap-less embedded target (added above) already covers what this test was for, more
  reliably.

## [0.2.1] - 2026-06-25

### Added

- More runnable examples: touring the vector algorithms directly, and a new `examples/storage` showing both the stack-
  and heap-allocated storage backends.
- One example file per elementary math function (`sqrt`, etc.) instead of bundling them together.
- A GitHub Actions workflow that publishes new versions to crates.io automatically on release.

### Changed

- Cleaned up duplicated `sin`/`cos` code between the `f32` and `f64` implementations by sharing one generic
  implementation.
- Reorganized the examples folder to mirror the tests folder's structure, and migrated the documentation site from
  Jekyll to mdBook with a custom theme.

## [0.2.0] - 2026-06-23

### Added

- **Matrix decompositions:** `determinant`, `rank`, `lu` (LU decomposition), `qr` (QR decomposition, via Householder
  reflections or Gram-Schmidt), `cholesky`, `svd` (singular value decomposition), and `condition_number` — the core
  toolkit for solving linear systems, analyzing a matrix's numerical stability, and related tasks. Each comes with both
  a high-level "just do the sensible thing" entry point and, where relevant, lower-level functions for explicit control
  over which algorithm runs.
- `Scalar::sin` and `Scalar::cos`, computed via a fixed-effort Taylor series (accurate near zero; this version doesn't
  yet correct for inputs far from zero — see the "Fixed" entry above in Unreleased for that follow-up).
- A `no_std` (heap-less) build is now verified in CI by actually cross-compiling to a real embedded target, catching
  accidental dependencies on the standard library that a normal desktop build wouldn't catch.
- A CI pipeline covering both build configurations (with and without heap allocation), linting, formatting, and
  publishing generated docs to GitHub Pages.
- Initial GitHub Pages documentation site, plus standard project scaffolding (PR template, contributors config).

### Changed

- `determinant` now automatically picks a faster algorithm (LU-based) for larger matrices instead of always using the
  much slower recursive method, which becomes impractically slow as matrices grow.
- Reorganized the test suite to mirror the source tree's structure, one file per thing under test, instead of a handful
  of large flat files.

## [0.1.0] - 2026-06-21

### Added

- **The foundational building blocks of the crate:** a `Scalar` trait (implemented for `f32` and `f64`) defining basic
  arithmetic, and a `Storage` trait with both a stack-allocated (`StaticStorage`) and heap-allocated (`DynamicStorage`,
  behind the optional `alloc` feature) implementation.
- `Scalar::sqrt`, computed via Newton-Raphson iteration since embedded (`no_std`) targets have no built-in square root;
  returns `0` for negative input rather than erroring.
- **Vectors and matrices**, in both fixed-size (`StaticVector`/`StaticMatrix`, sized at compile time) and heap-allocated
  (`DynamicVector`/`DynamicMatrix`) flavors, supporting addition, subtraction, scalar multiplication, dot product, norm,
  matrix-vector and matrix-matrix multiplication, and transpose. Operations that could receive mismatched shapes return
  a clear error instead of panicking; operations where the type system already guarantees matching shapes (e.g. two
  `StaticVector`s of the same size) skip that check entirely.
- Runnable examples and black-box integration tests covering the full public API of this initial release, on both `f32`
  and `f64` where relevant.
- Verified the crate builds, lints, and passes its tests with the heap-allocated (`alloc`) feature turned off,
  confirming it's usable on targets with no heap at all.

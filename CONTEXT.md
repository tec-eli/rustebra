# Architecture Decision Records (ADR) Summaries

## ADR 0001: Hybrid Memory Allocation Strategy via Feature Flags
**Decision:** The project adopts a hybrid memory architecture consisting of a strict `no_std` core (stack-allocated via 
const generics) and an optional `"alloc"` feature flag for dynamic, heap-backed data structures.
**Rationale:** This maximizes portability, allowing the library to run safely on constrained embedded systems without a 
memory allocator, while scaling up to support advanced algorithms requiring dynamic memory when operating systems are available.

## ADR 0002: Single Crate with Incremental Feature Flags
**Decision:** The library will be published as a single crate rather than a multi-crate workspace. Feature flags (like 
the `alloc` flag) will be added incrementally as new mathematical domains are implemented.
**Rationale:** Distributing a single crate keeps maintenance and versioning simple during the early stages of development.
It avoids the upfront design burden of a complex workspace while remaining reversible if domains grow large enough to warrant separation later.

## ADR 0003: Minimal Shared Storage Trait Between Static and Dynamic Types
**Decision:** Static and dynamic storage types will share a minimal, internal storage trait scoped only to what generic algorithms need. 
The public-facing APIs for static and dynamic types will remain separate.
**Rationale:** An internal shared trait allows complex algorithms (e.g., Krylov methods) to be written once and reused
across both storage strategies. Keeping the public types separate prevents premature, leaky abstractions from complicating the user experience.

## ADR 0004: Per-Module Error Handling with Result
**Decision:** Recoverable failures are reported using `Result`. Each mathematical module defines its own scoped error 
type for the specific failures it can encounter, rather than using a massive crate-wide error enum or panicking.
**Rationale:** Panicking is unacceptable for bare-metal targets. Using per-module error types decouples independent 
mathematical domains from one another, allowing callers to handle errors with precise control.

## ADR 0005: Generic Scalar Type
**Decision:** All mathematical operations are written generically over a scalar type (e.g., `f32`, `f64`) rather than 
hardcoding a single numeric type. Required capabilities are added incrementally as needed.
**Rationale:** This accommodates varying hardware limits—such as microcontrollers lacking double-precision floating-point 
units—preventing a massive refactor later while keeping the library highly adaptable.

## ADR 0006: Trait-Based Generic Layered Architecture
**Decision:** The crate uses a 4-layer architecture: Scalar, Storage, Algorithm, and Public API. Each layer only depends 
on the ones below it, with algorithms written generically against traits rather than concrete storage types.
**Rationale:** This layered approach clarifies dependencies and prevents duplicating algorithms. It naturally fits the 
internal structure of numerical libraries, avoiding forced external metaphors like Hexagonal architecture.

## ADR 0007: Public Elementary Scalar Functions, Scoped to Internal Needs
**Decision:** The library implements and publicly exposes elementary math functions (like `sqrt`, `sin`) for `#![no_std]` 
environments, but only implements the ones the library actually needs internally.
**Rationale:** This prevents users from needing a separate `libm` dependency for basic math already used by the crate,
without burdening the project with becoming a complete, general-purpose math library.

## ADR 0008: Two-Level Public API for Matrix Operations
**Decision:** Matrix operations are exposed via a two-level API: high-level functions that auto-select the best algorithm 
based on input properties, and explicit functions that run a specifically named algorithm.
**Rationale:** This serves two audiences: general users get a simple, ergonomic interface that "just works," while researchers 
and mathematical users retain strict control over which algorithms are executed.

## ADR 0009: Numerical Tolerance for Approximate-Zero Comparisons
**Decision:** Functions making threshold judgments (like rank, negligibility, or positive-definiteness) now require an 
explicit `tolerance` parameter from the caller, while general-user entry points auto-compute a safe default based on machine epsilon.
**Rationale:** Exact comparisons to `0.0` fail on real-world floating-point data due to rounding noise. Thresholds prevent 
silent failures on non-toy matrices while maintaining exact algorithms where mathematically appropriate.

## ADR 0010: Sparse Matrix Public API Shape
**Decision:** Sparse operations are implemented as free functions. The COO format is restricted to matrix assembly only. 
Shape mismatch errors are unified, and solvers interact with matrices via a format-agnostic `SparseLinearOp` trait.
**Rationale:** This establishes clear semantic boundaries and prevents misuse (like unexpected duplicate accumulations 
in COO). The unified error types and trait abstractions provide zero-overhead, format-agnostic compatibility for solvers.

## ADR 0011: SortedCsrMatrix — Type-Level Column-Sort InvariantOriginal File: 0011-sorted-csr-matrix.mdDecision: 
The library introduces SortedCsrMatrix<T>, a newtype wrapper around CsrMatrix<T> that provides a compile-time guarantee 
that column indices within every row are sorted in ascending order. This wrapper implements Deref for transparent 
read-only access to the underlying matrix.  Rationale: Certain sparse algorithms, such as coo_to_csr and spmm_csr, 
naturally produce sorted matrices, while future operations like sparse Cholesky factorization strictly require them. 
Encoding this invariant at the type level with zero runtime cost eliminates the need for callers to defensively sort 
matrices or rely on fragile, undocumented conventions.  

## ADR 0012: Sparse Entry Pruning and Explicit Zero SemanticsOriginal 
Decision: A prune_csr function is introduced to remove numerically negligible entries from a sparse matrix based 
entirely on an explicit, caller-supplied tolerance threshold. The function evaluates the condition |v| > tolerance using 
only PartialOrd and existing Scalar arithmetic (such as negating the tolerance) to avoid adding an abs() method to the 
generic Scalar trait.  Rationale: Over the course of iterative computations, sparse matrices accumulate near-zero fill-in 
entries that degrade memory usage and arithmetic throughput. Because the correct pruning threshold depends completely on 
the specific scale of the caller's problem, providing an auto-computed machine-epsilon default is impossible and incorrect 
for this specific operation.  
# Sparse Matrix Public API Shape

## Summary

Sparse matrix operations (COO, CSR, CSC) are exposed as free functions rather than methods,
COO is a build-only format with no arithmetic of its own, shape failures across every sparse
operation share one error type, matvec/matmat return owned allocations, and a narrow trait
lets iterative solvers call into any sparse format without knowing which one they're holding.

## Scope

Applies to the three sparse storage formats — coordinate (COO), compressed sparse row (CSR),
and compressed sparse column (CSC) — and to every operation defined over them: construction,
conversion between formats, matrix-vector and matrix-matrix products, addition, scaling, and
the interface iterative solvers use to call into a sparse matrix without depending on its
concrete format.

## Decision

**Free functions, not methods.** All sparse operations are free functions, following the
same convention used throughout the algorithm layer elsewhere in the crate. This keeps
operation logic decoupled from the format type definitions, so operations can be added
incrementally without reopening format modules — and stays consistent with the crate's
avoidance of trait-object-based dynamic dispatch, which a method surface on the types
themselves would not by itself require, but which the free-function convention sidesteps
entirely regardless.

**COO is build-only.** COO is the natural way to build a sparse matrix incrementally — it
requires no ordering invariant, and duplicate entries at the same position are valid and
expected while building. Arithmetic directly on COO is either expensive or semantically
ambiguous (adding two COO matrices without first sorting both has no efficient
implementation, and naively concatenating entry lists produces a result with unresolved
duplicate positions that behaves differently from the equivalent CSR operation). COO
supports construction and entry-by-entry manipulation; arithmetic requires converting to CSR
or CSC first.

**One shared shape-mismatch error.** Every sparse operation that can fail due to
incompatible dimensions — matvec's vector-length-versus-column-count check, binary
operations' equal-dimension check, matmat's inner-dimension-agreement check — returns the
same single error type rather than a module-specific one. All of these failures represent
the same underlying fact from a caller's perspective, and a single type means one import and
one `?`-target regardless of which sparse operation is being called.

**Owned output.** Matrix-vector and matrix-matrix products return an owned, newly allocated
vector or matrix. Every sparse format already requires an allocator to exist at all, so
there is no `no_std`-without-`alloc` path through this API to preserve, and an owned return
keeps call sites simple.

**Solver interface without dynamic dispatch.** Iterative solvers need only a matrix-vector
product and the operand dimensions, and should not be forced to pick a specific format at
the call site. A trait exposing `rows`, `cols`, and an `apply`-style matrix-vector product
provides this indirection through generic, monomorphized dispatch — no trait objects, no
heap allocation for the indirection itself. Every sparse format type implements it, and
solver functions are written generically against it rather than against any one format.

**CSR/CSC parity.** Both formats expose the same operation set (scale, matvec, add, matmat),
because a caller who receives a matrix in one format should not be forced to convert before
using it just because an operation happened to be implemented for the other format first.

## Constraints

- A shape-mismatch failure anywhere in the sparse module must surface through the same
  error type, regardless of which operation detected it.
- The solver-facing trait must add no heap allocation of its own beyond what the concrete
  sparse format already requires, and must not rely on trait objects.
- Any per-iteration matrix-vector product called from inside a solver's loop must be able to
  write into a caller-supplied output buffer rather than allocate fresh on every call — a
  per-iteration allocation is both a throughput cost and, on real-time embedded targets, a
  determinism problem.
- COO must never be extended with arithmetic operations that produce ambiguous results (such
  as unresolved duplicate entries after an add); any operation with that risk requires
  converting to CSR or CSC first.

## Status

Implemented. All sparse operations are free functions; COO supports construction and
conversion only; a single shared error type covers every shape mismatch; the solver-facing
trait writes into a caller-supplied buffer for its matrix-vector product; CSR and CSC expose
matching operation sets.

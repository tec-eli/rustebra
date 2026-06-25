
# ADR 0008: Two-Level Public API for Matrix Operations

## Status
Accepted

## Context
Several matrix operations (solving linear systems, computing the determinant, decomposing a
matrix) can be carried out by more than one algorithm. The algorithms differ in
computational cost, numerical stability, and preconditions (e.g. Cholesky requires symmetric
positive-definite input; cofactor expansion is only practical for small matrices).

Two categories of users need to be served:

1. **General users** who want a correct result without choosing an algorithm — they call one
   function and expect the library to do the right thing.
2. **Mathematical users** (researchers, students, algorithm designers) who need control over
   which algorithm runs — to study its behavior, compare results, or exploit knowledge about
   their specific input that the library cannot infer automatically.

Exposing only a high-level API satisfies category 1 but blocks category 2. Exposing only
explicit algorithms satisfies category 2 but adds unnecessary friction for category 1.

## Decision
The public API has two levels for any operation where multiple algorithms exist:

1. **High-level functions** (e.g. `determinant`, `solve`) select an algorithm automatically
   based on observable properties of the input (matrix size, symmetry, positive-definiteness
   where detectable). The specific algorithm chosen is an implementation detail — it is not
   part of the public contract and may change between versions without a breaking change, as
   long as the mathematical result is correct.

2. **Explicit algorithm functions** (e.g. `determinant_cofactor`, `determinant_lu`,
   `solve_lu`, `solve_cholesky`) always run the named algorithm regardless of input
   properties. These are stable public API: the algorithm named in the function will not
   change.

Both levels are publicly exposed and documented. Neither is marked as internal or
discouraged.

## Consequences

### Positive
* General users get a single, ergonomic entry point per operation.
* Mathematical users retain full control and can rely on a specific algorithm by name.
* The high-level API can be improved (better selection heuristics, new algorithms) without
  breaking callers who use it.
* Explicit functions are self-documenting — the name states exactly what runs.

### Negative / Trade-offs
* More public surface area to maintain: every operation that has multiple algorithms needs
  both a high-level function and one explicit function per algorithm.
* The automatic selection logic in high-level functions must be documented clearly enough
  that users can predict which algorithm runs, even if it is not a formal guarantee.

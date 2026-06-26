
# ADR 0012: Sparse Entry Pruning and Explicit Zero Semantics

## Status
Accepted

## Context
Sparse matrices accumulate numerical near-zeros over the course of iterative
computation: fill-in from matrix products, cancellation from additions, and
ordinary floating-point rounding all produce entries that are structurally
non-zero but numerically negligible. Leaving them in place degrades both memory
use and arithmetic throughput proportionally to their count. A function that drops
entries below a caller-supplied threshold addresses this.

### What "below threshold" means

For a real-valued entry `v` and threshold `τ > 0`, the condition for pruning is

    |v| ≤ τ

i.e. keep `v` if and only if `|v| > τ`. This is the standard definition used by
MATLAB's `spfun`/`droptol` and SciPy's `eliminate_zeros`/`prune`.

`Scalar` (as defined in the crate) has no absolute value or ordering methods. Its
design reflects the minimal arithmetic surface that the algorithm layer needs;
adding `abs()` and `PartialOrd` to `Scalar` to support one pruning function would
couple all `Scalar` implementors to numeric-ordering concepts that may not apply to
them (e.g. a future complex-number scalar).

The correct approach is to add the required bounds directly to the pruning function:

```rust
pub fn prune_csr<T: Scalar + PartialOrd>(m: CsrMatrix<T>, tolerance: T) -> CsrMatrix<T>
```

The condition `|v| > τ` is then expressed without `abs()` by computing the
negation of the tolerance as `T::zero().sub(tolerance)` (which uses only `Scalar`
methods) and testing `v > tolerance || v < neg_tolerance`. This keeps the
predicate correct for negative entries without requiring an `abs()` method on `T`.

### Caller-supplied versus auto-computed threshold

[ADR 0009](0009-numerical-tolerance-for-approximate-zero.md) addresses
auto-computed defaults for operations where machine epsilon provides a sensible
baseline (rank, singular-value negligibility, positive-definiteness). Entry
pruning is different: the "right" threshold depends entirely on the caller's
domain — the scale of the physical quantity being modelled, the acceptable
accuracy loss for the downstream computation, and whether the caller is
compressing for memory or eliminating cancellation noise. There is no neutral
machine-epsilon-derived default that is correct across those use cases. A
caller who supplies `tolerance = 0.0` gets no pruning; one who supplies
`tolerance = 1e-12` gets aggressive pruning for a solver working in that scale.

The auto-computing path (`T: Scalar + FloatTolerance`) is therefore not provided.
`prune_csr` takes an explicit threshold and nothing else.

### Return type

`prune_csr` returns a plain `CsrMatrix<T>`, not `SortedCsrMatrix<T>`. Pruning
drops entries but does not reorder them. If the input had sorted column indices
within each row (e.g. it was a `SortedCsrMatrix`), the output also has sorted
indices — but encoding this in the type would require overloading `prune_csr`
or adding a second function, which is speculative complexity. Callers who need a
sorted result after pruning call `SortedCsrMatrix::from_csr` on the output.

### Format scope

Only CSR is provided. CSC pruning (`prune_csc`) is not part of this ADR because
no current operation requires it. It would be structurally identical with rows and
columns swapped. Adding it when a concrete use case arises, rather than now, is
consistent with the codebase's no-speculative-code rule.

## Decision

`prune_csr<T: Scalar + PartialOrd>(m: CsrMatrix<T>, tolerance: T) -> CsrMatrix<T>`
is added to `rustebra::sparse`. It drops every entry `v` satisfying `|v| ≤ tolerance`
(equivalently: keeps entries with `v > tolerance` or `v < -tolerance`) and
returns a new `CsrMatrix<T>` with the reduced entry set. The threshold is always
caller-supplied; no auto-computed default is provided.

## Consequences

### Positive
- Iterative algorithms can shed fill-in between steps without an external helper,
  keeping memory and per-iteration cost proportional to the entries that matter.
- The predicate is expressed using only `PartialOrd` and existing `Scalar`
  arithmetic; no new methods are added to `Scalar`.
- The threshold is explicit: callers who want aggressive pruning get it, and
  callers who want no pruning can pass `T::zero()` (though the more common idiom
  is simply not to call the function).

### Negative / Trade-offs
- Callers must choose their own threshold. For general users unfamiliar with their
  problem's scale this is non-trivial, but there is no correct machine-level
  default that could substitute for the judgement. Documentation must make this
  explicit.
- `prune_csc` is absent. Callers working in CSC can convert to CSR, prune, and
  convert back, at the cost of two format conversions. This is a deliberate
  deferral, not an oversight.
- The return type is `CsrMatrix<T>`, not `SortedCsrMatrix<T>`, even though pruning
  preserves sort order. Callers who need the stronger type must wrap the result
  with `SortedCsrMatrix::from_csr`, which re-sorts. A future `prune_sorted_csr`
  returning `SortedCsrMatrix<T>` could avoid that re-sort by using
  `from_sorted_unchecked`, but is deferred until there is a concrete caller.

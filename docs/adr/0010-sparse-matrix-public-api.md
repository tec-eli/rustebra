
# ADR 0010: Sparse Matrix Public API Shape

## Status
Accepted

## Context
The sparse matrix module introduces three storage formats — COO (coordinate),
CSR (compressed sparse row), and CSC (compressed sparse column) — each with
different computational characteristics. The initial implementation included
arithmetic operations (`add`, `scale`, matvec) on all three formats and defined
its own dimension-mismatch error type in the matvec module. Several design
questions were left open before the API stabilised:

1. **Surface**: should operations be free functions or methods on the format types?
2. **Format scope**: should every format support every operation, or should some
   formats be restricted to roles they are genuinely efficient at?
3. **Error unification**: each operation module defined its own error type for
   shape failures; is a single shared type correct?
4. **Return allocation**: should vector outputs be owned `Vec<T>` or written into
   a caller-supplied slice?
5. **Solver interface**: Krylov solvers need to call a matrix-vector product
   without knowing which sparse format holds the data. What abstraction covers this?
6. **API symmetry**: should CSR and CSC expose the same set of operations?

### Free functions versus methods

The library is `no_std`-first with an optional `alloc` feature. Dynamic dispatch
via trait objects (`Box<dyn Trait>`) is excluded by the architecture. A method
surface on the format types would not itself require dynamic dispatch, but it would
couple the operation logic to the type definition in the same module, making it
harder to add operations incrementally without re-opening format modules. The
algorithm layer throughout this codebase uses free functions (e.g. `add`,
`mul_matrix`) rather than methods for the same reason.

### COO scope

COO is the natural way to build a sparse matrix incrementally: it requires no
ordering invariant, and duplicate `(row, col)` entries are valid and common.
Arithmetic on COO, however, is expensive or semantically unusual: there is no
efficient way to add two COO matrices without first sorting both; scaling is
`O(nnz)` but so is converting to CSR and scaling there. The `add_coo` function
in the initial implementation concatenated the two entry lists and returned a
COO matrix with duplicate positions — a semantically surprising result that
differs from what `add_csr` does. Using COO as a build-only format, converted to
CSR/CSC before arithmetic, makes the contract unambiguous and removes an
API surface that encouraged misuse.

### Error unification

Sparse operations can fail in exactly one observable way from a caller's
perspective: the operand shapes are incompatible. Matvec checks that the vector
length matches the matrix column count; binary matrix operations check that both
matrices have equal dimensions; matmat checks that inner dimensions agree. All
of these are the same condition. Separate error types per module add naming
variance with no benefit: callers checking for shape errors must import a
different type from each module, and `?`-propagation from a mixed-operation
function requires mapping between types that represent the same fact. A single
`DimensionMismatch` unit struct, owned by `sparse::add` as the natural home for
binary-operation errors, is exported from `sparse` and re-used by every other
sparse operation module.

### Owned Vec<T> return

Returning a `Vec<T>` from matvec and matmat allocates unconditionally, which is
intentional: all sparse operations already require `alloc` (their format types
allocate), so there is no meaningful `no_std` path through this API anyway.
A caller-supplied output slice would be more flexible for zero-copy pipelines,
but that is speculative: no such pipeline exists in this codebase today.
Owned outputs keep the signatures simple and the caller code short.

### Krylov solver interface

A Krylov method (conjugate gradient, GMRES, etc.) needs only matrix-vector
multiplication and the matrix dimensions. It should not be parameterized by
format: forcing callers to pick CSR or CSC to pass to a solver is an unnecessary
coupling. A trait

```rust
pub trait SparseLinearOp<T: Scalar> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn apply(&self, x: &[T]) -> Result<Vec<T>, DimensionMismatch>;
}
```

provides this indirection without `Box<dyn Trait>`: solver functions are generic
over `impl SparseLinearOp<T>`, which compiles to a monomorphised call and carries
no heap allocation. Both `CsrMatrix<T>` and `CscMatrix<T>` implement the trait.

The trait lives in its own module (`sparse::linear_op`) rather than in the format
modules, so neither format module depends on the other — the same direction of
dependency isolation used throughout the algorithm layer.

### API symmetry

CSR is the natural format for row-oriented operations (matvec, spmm output);
CSC is natural for column-oriented ones. However, both formats must support all
common operations because a caller often receives a matrix in one format and should
not be forced to convert before using it. Exposing only `matvec_csr` and not
`matvec_csc` (or vice versa) would be an artificial restriction. All operations
that exist for CSR are also provided for CSC, with identical signatures modulo the
format type.

## Decision

- All sparse operations are free functions, following the existing algorithm-layer
  convention.
- COO is a build-only format. `add_coo`, `scale_coo`, and `matvec_coo` are not
  part of the public API. COO is constructed, optionally manipulated entry-by-entry,
  then converted to CSR or CSC for arithmetic.
- All sparse operations that can fail due to incompatible shapes return
  `Err(DimensionMismatch)`, where `DimensionMismatch` is a single, shared unit
  struct exported from `rustebra::sparse`.
- Matvec and matmat output is an owned `Vec<T>`.
- `SparseLinearOp<T>` is a trait with three methods (`rows`, `cols`, `apply`).
  It is implemented by `CsrMatrix<T>`, `CscMatrix<T>`, and `SortedCsrMatrix<T>`.
  Solver functions are generic over `impl SparseLinearOp<T>`.
- CSR and CSC expose the same operation set: `scale`, `matvec`, `add`, `matmat`.

## Consequences

### Positive
- The API has a single canonical error type for shape failures; one import, one
  match arm, one `?`-target across all sparse code.
- COO's semantics are clear: it is an accumulator, not an arithmetic type. Users
  who call `coo_to_csr` get a matrix ready for computation with no surprising
  duplicate entries.
- Solver code stays format-agnostic with zero runtime overhead: monomorphised
  dispatch, no heap allocation for the trait indirection.
- CSR/CSC parity means switching formats to improve cache behaviour for a
  specific workload requires no API changes at the call site.

### Negative / Trade-offs
- Callers who genuinely want `O(nnz)` in-place scaling of a freshly assembled COO
  matrix before converting it must convert first. This is a minor inconvenience
  rather than a real cost: `coo_to_csr` followed by `scale_csr` is one extra
  function call.
- `Vec<T>` output allocates on every call. A future zero-copy path (write into
  caller-supplied slice) would require a new function variant; no such path
  exists today, so the cost is accepted.
- `SparseLinearOp` adds a trait to learn. Its narrowness (three methods) keeps
  this cost low.

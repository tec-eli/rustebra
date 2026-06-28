
# ADR 0011: SortedCsrMatrix — Type-Level Column-Sort Invariant

## Status
Accepted

## Context
Several sparse algorithms either require or naturally produce CSR matrices whose
column indices within each row are in ascending order. This "sorted" invariant
matters in at least three places:

- **`coo_to_csr`**: the conversion sorts all entries by `(row, col)` to build
  the row-pointer array, so the output column indices are trivially sorted. No
  extra pass is needed to enforce it.
- **`spmm_csr`**: the Gustavson dense-accumulator algorithm emits output entries
  via a `touched.sort_unstable()` pass per row, so the output is again trivially
  sorted.
- **Future operations**: algorithms such as sparse triangular solve and sparse
  Cholesky factorisation require sorted column indices and would otherwise need
  to check or enforce the invariant at runtime.

With a plain `CsrMatrix<T>`, there is no way for a function to declare at the
type level that it requires a sorted matrix, nor for a caller to see from a
return type that the result is sorted. Callers who need sorted CSR either sort
defensively on every call or rely on undocumented conventions. Both options are
fragile.

The alternative of storing a boolean flag inside `CsrMatrix` was considered and
rejected: a flag only shifts the invariant to "the flag agrees with reality," which
is no stronger than the undocumented convention, and it prevents the type system
from catching misuse statically.

A newtype wrapper encodes the invariant at the type level with zero runtime cost:
`SortedCsrMatrix<T>` is structurally identical to `CsrMatrix<T>` but carries the
guarantee that every row's column indices are in ascending order. The newtype does
not duplicate the format's logic — it only restricts construction.

### Construction paths

Two constructors exist:

1. **`SortedCsrMatrix::from_csr(m: CsrMatrix<T>)`** — public, sorts each row in
   place in `O(nnz log nnz)` time. Safe for any caller with an arbitrary CSR matrix.
2. **`SortedCsrMatrix::from_sorted_unchecked(m: CsrMatrix<T>)`** — `pub(super)`,
   asserts the invariant without verifying it. Used only by `coo_to_csr` and
   `spmm_csr`, which already guarantee sorted output by construction, and which
   live in the same `sparse` module. No unsafe code is involved; the only cost of
   violating the invariant is incorrect behaviour in algorithms that require it,
   not memory unsafety.

### Transparent access

`SortedCsrMatrix<T>` implements `Deref<Target = CsrMatrix<T>>`. This means every
read-only method on `CsrMatrix<T>` — `rows()`, `cols()`, `nnz()`, `row_ptr()`,
`col_indices()`, `values()` — is automatically available on `SortedCsrMatrix<T>`
without duplication. Tests that call these methods on a `SortedCsrMatrix` value
compile without any explicit conversion or forwarding boilerplate.

### Conversions

- **`SortedCsrMatrix<T>: Deref<Target = CsrMatrix<T>>`** — read-only transparent access.
- **`From<SortedCsrMatrix<T>> for CsrMatrix<T>`** — consuming conversion for functions
  that accept an ordinary `CsrMatrix`. Callers who want to discard the sorted
  guarantee call `.into_inner()`.
- **`PartialEq<CsrMatrix<T>> for SortedCsrMatrix<T>`** and its mirror — cross-type
  equality so tests can compare a `SortedCsrMatrix` result directly against a
  `CsrMatrix` expected value.
- **`SparseLinearOp<T>`** — implemented on `SortedCsrMatrix<T>` by delegating to the
  inner `CsrMatrix<T>`, since Krylov solvers only need `apply`.

## Decision

`SortedCsrMatrix<T>` is a new type wrapping `CsrMatrix<T>` that carries a
compile-time guarantee that column indices within every row are in ascending order.

- `coo_to_csr` returns `SortedCsrMatrix<T>` instead of `CsrMatrix<T>`.
- `spmm_csr` returns `Result<SortedCsrMatrix<T>, DimensionMismatch>`.
- The type lives in `sparse::sorted_csr` and is re-exported from `rustebra::sparse`.
- `from_sorted_unchecked` is `pub(super)` and not part of the public API.

## Consequences

### Positive
- Future algorithms that require sorted CSR can express the requirement in their
  signature (`m: &SortedCsrMatrix<T>`) and rely on the compiler to enforce it,
  rather than calling a runtime check on every entry.
- `coo_to_csr` and `spmm_csr` communicate to their callers, at the type level,
  that the output is already sorted — no secondary sort step is needed before
  passing the result to an operation that requires sorted CSR.
- `Deref` means zero boilerplate at call sites: existing code that reads fields
  or calls accessors on the result of `coo_to_csr` works without any changes.

### Negative / Trade-offs
- Callers who need to pass the result of `coo_to_csr` to a function that takes
  `CsrMatrix<T>` by value must call `.into_inner()`. This is a one-word change
  and makes the conversion explicit rather than implicit.
- Two related types (`CsrMatrix` and `SortedCsrMatrix`) exist in the public API.
  The naming is self-explanatory, but it is one more concept to learn. Given
  that the sorted invariant is load-bearing for future algorithms, the additional
  type is the correct trade-off.
- `from_sorted_unchecked` is invisible to external callers, but internally it is a
  trust boundary: any `pub(super)` code that calls it must actually produce sorted
  output. This is enforced by code review, not the type system.

# Sparse Operations

Sparse matrices support scaling, matrix-vector and matrix-matrix products against a dense
buffer, sparse-sparse addition, sparse-sparse multiplication, and pruning near-zero
entries. Each operation is a free function taking the sparse matrix by value or reference,
rather than a method — and each is provided in both a CSR and a CSC variant.

Scaling every stored value by a constant:

```rust
{{#include ../../../examples/sparse/scale.rs}}
```

Multiplying a sparse matrix by a dense vector:

```rust
{{#include ../../../examples/sparse/matvec.rs}}
```

Multiplying a sparse matrix by a dense matrix:

```rust
{{#include ../../../examples/sparse/matmat.rs}}
```

Adding two sparse matrices of the same shape:

```rust
{{#include ../../../examples/sparse/add.rs}}
```

Sparse-sparse matrix multiplication (`spmm_csr`), which multiplies two `CsrMatrix`
operands and returns a sorted result:

```rust
{{#include ../../../examples/sparse/spmm.rs}}
```

Pruning entries within a tolerance of zero:

```rust
{{#include ../../../examples/sparse/prune.rs}}
```

Sparse matrices also implement `SparseLinearOp`, an abstraction for "apply this matrix to
a dense vector, writing into a caller-supplied buffer" — the interface Krylov solvers are
written against so they work with any sparse format:

```rust
use rustebra::sparse::{CsrMatrix, SparseLinearOp};

let eye = CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0_f64, 1.0]).unwrap();
let mut y = [0.0; 2];
eye.apply(&[3.0, 5.0], &mut y).unwrap();
assert_eq!(y, [3.0, 5.0]);
```

## Gotchas

- Every operation here returns `Result` (e.g. `DimensionMismatch` for shape-incompatible
  operands) rather than panicking — sparse matrices carry their shape as runtime fields,
  so there's no type-level guarantee two operands are compatible the way there is for
  `StaticMatrix`.
- `apply` (from `SparseLinearOp`) writes into a caller-supplied output buffer instead of
  allocating a new one, so solver loops built on top of it can reuse the same buffer across
  iterations without allocating per call.
- `validate_csr`/`validate_csc` (in `rustebra::sparse`) check the canonical format
  invariants — including that no stored value is an *explicit* zero — separately from
  construction. `CsrMatrix::new`/`CscMatrix::new` accept explicit zeros; only
  `validate_csr`/`validate_csc` flag them.

# Construction & Conversion

Sparse matrices can be built directly in CSR or CSC format, or assembled from unordered
`(row, col, value)` triplets via `CooMatrix` and converted afterward. The three examples
below construct a matrix in each of the three formats:

```rust
{{#include ../../../examples/sparse/coo.rs}}
```

```rust
{{#include ../../../examples/sparse/csr.rs}}
```

```rust
{{#include ../../../examples/sparse/csc.rs}}
```

`CooMatrix` tolerates duplicate `(row, col)` triplets — they're logically summed when the
matrix is later used, e.g. by conversion to CSR. Converting between formats is a distinct
step from construction; the example below covers all four conversion functions
(`coo_to_csr`, `csr_to_coo`, `csr_to_csc`, `csc_to_csr`):

```rust
{{#include ../../../examples/sparse/convert.rs}}
```

## Gotchas

- `coo_to_csr` sums duplicate `(row, col)` entries and sorts column indices within each
  row as a side effect, which is why it returns a `SortedCsrMatrix` rather than a plain
  `CsrMatrix` — you get the stronger guarantee "for free" from the conversion algorithm.
- `CsrMatrix::new`/`CscMatrix::new`/`CooMatrix::new` all return `Result` rather than
  panicking on malformed input (mismatched array lengths, out-of-bounds indices, an
  invalid pointer array) — check the result rather than assuming construction always
  succeeds.

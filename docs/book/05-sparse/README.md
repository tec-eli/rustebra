# Sparse Matrices

`rustebra` also supports sparse matrices — matrices where most entries are zero and only
the non-zeros are stored. This section covers the storage formats (COO, CSR, CSC), how to
construct and convert between them, and the operations (scaling, matrix-vector and
matrix-matrix products, addition, pruning) available on them. Sparse support requires the
`alloc` feature.

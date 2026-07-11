# Performance

The main performance-relevant design choice in `rustebra` is the split between `Static*`
and `Dynamic*` types (see [no_std & Embedded](./13-no-std.md)). `StaticVector<T, N>` and
`StaticMatrix<T, R, C>` are stack-allocated with their size fixed at compile time — no
heap allocation, no indirection through a `Vec`, and their length/shape checks (where the
type system can rule out a mismatch) are eliminated entirely rather than deferred to
runtime. `DynamicVector<T>` and `DynamicMatrix<T>` trade that for runtime-flexible sizing,
at the cost of a heap allocation per construction and a `Result`-returning runtime shape
check on every operation between two of them.

For algorithms with a caller-provided `scratch` buffer (`svd`, `condition_number`, `qr` on
`StaticMatrix`), the caller controls whether that buffer lives on the stack or the heap —
sizing it as a stack array keeps the whole computation allocation-free.

Beyond this structural choice, `rustebra` doesn't publish benchmark numbers or make
specific throughput/complexity claims — the underlying algorithms (Gaussian elimination for
LU, Householder/Gram-Schmidt for QR, QR iteration for SVD) use standard, textbook time
complexity for their respective operations.

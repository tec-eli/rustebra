# Vectors

`rustebra` provides two vector types: `StaticVector<T, N>`, stack-allocated with a
compile-time length, and `DynamicVector<T>`, heap-allocated behind the `alloc` feature with
a runtime length. This section covers how to construct them, the arithmetic and dot-product
operations they support, and how to compute their norm.

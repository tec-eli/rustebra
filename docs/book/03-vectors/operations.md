# Operations

`StaticVector` and `DynamicVector` support the same core set of operations: element-wise
addition and subtraction, scalar scaling, the dot product, and the Euclidean norm. The
example below constructs two `StaticVector`s and runs each of them in turn.

```rust
{{#include ../../../examples/vector/static_vector.rs}}
```

## Gotchas

- `add`, `sub`, and `dot` on `StaticVector` return a plain value, not a `Result` — the
  const generic `N` guarantees both operands have the same length at compile time, so
  there's nothing to fail at runtime. The `DynamicVector` equivalents return
  `Result<_, LengthMismatch>` instead, since two `DynamicVector`s aren't guaranteed to
  match in length.
- `scale` takes the factor by value (`T`, not `&T`), matching `Scalar`'s `Copy` bound —
  pass the number directly rather than a reference.

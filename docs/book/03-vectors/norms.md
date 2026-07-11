# Norms & Distances

Both `StaticVector` and `DynamicVector` expose a single `norm()` method: the Euclidean
(L2) norm, `‖v‖ = sqrt(v . v)`, computed as the square root of the vector's dot product
with itself.

```rust
use rustebra::vector::StaticVector;

let v = StaticVector::new([3.0, 4.0]);
assert_eq!(v.norm(), 5.0);
```

There's no separate `distance` method. The Euclidean distance between two vectors is just
the norm of their difference, and falls out of the existing `sub` and `norm` operations:

```rust
use rustebra::vector::StaticVector;

let a = StaticVector::new([1.0, 2.0, 3.0]);
let b = StaticVector::new([4.0, 6.0, 3.0]);

let distance = a.sub(&b).norm();
assert_eq!(distance, 5.0);
```

For `DynamicVector`, `sub` returns a `Result` (two `DynamicVector`s aren't guaranteed to
match in length at compile time), so the equivalent distance computation is
`a.sub(&b)?.norm()`.

## Gotchas

- `norm()` itself never fails or returns a `Result` — even on `DynamicVector` — since
  computing the norm of a single vector has no dimension-mismatch case to report. It's
  `sub`/`add`/`dot` between *two* `DynamicVector`s that can fail.
- `sqrt` under the hood is `Scalar::sqrt`, a fixed-iteration approximation rather than a
  hardware intrinsic (see [Scalars & Numeric Types](../02-scalars.md)) — norms of
  extremely large or small magnitude vectors may lose a little precision relative to
  `f64::sqrt`.

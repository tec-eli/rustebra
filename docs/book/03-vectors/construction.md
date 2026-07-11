# Construction

rustebra vectors come in two flavors, distinguished by where their elements live and how
their length is tracked.

`StaticVector<T, N>` is stack-allocated: `N` is a const generic, so the length is part of
the type itself and is known at compile time. Two `StaticVector`s with different `N` are
different types — the compiler rejects mismatched-length operations before your code ever
runs. This is the default choice in a `no_std` context, since it requires no allocator.

`DynamicVector<T>` is heap-allocated (it wraps a `Vec<T>` internally) and its length is
only known at runtime. It's available behind the `alloc` feature. Because two
`DynamicVector`s can't be proven to have the same length at compile time, operations that
combine two of them return a `Result` instead of a bare value.

Both types are constructed directly from their backing collection — `StaticVector::new`
takes a fixed-size array, `DynamicVector::new` takes a `Vec`:

```rust
use rustebra::vector::StaticVector;

let v = StaticVector::new([1.0, 2.0, 3.0]);
```

Here `N` is inferred from the array literal's length (`3`), so it rarely needs to be
written out explicitly.

## Gotchas

- **`StaticVector<T, 3>` and `StaticVector<T, 4>` are unrelated types.** If you find
  yourself needing to add vectors whose lengths aren't known until runtime, reach for
  `DynamicVector` instead — trying to force different `N`s together is a compile error,
  not something you can work around with a cast.
- **`DynamicVector` needs the `alloc` feature enabled.** Without it, `rustebra::vector`
  only exports `StaticVector`, and code referencing `DynamicVector` won't compile.

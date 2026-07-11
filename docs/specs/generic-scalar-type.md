# Generic Scalar Type

## Summary

Every operation in the library is written generically over a scalar type from the start,
rather than hardcoded against a single concrete numeric type such as `f64`.

## Scope

Applies to every function, algorithm, and data structure in the crate that operates on
numeric values — vectors, matrices, sparse formats, and the algorithms defined over them.

## Decision

Operations are written against a scalar type parameter rather than a fixed concrete type.
The set of capabilities that type parameter must provide — basic arithmetic, whichever
elementary functions are actually needed, identity values, and so on — is defined
incrementally, as each capability is actually required by the mathematics being
implemented, rather than specified upfront in full.

## Constraints

- The library must remain usable with `f32` on resource-constrained targets that lack
  double-precision floating-point hardware, and with `f64` elsewhere, without maintaining
  two separate codebases for the two cases.
- Generality must be established before substantial algorithm code exists, so that
  switching or adding a numeric type later does not force a disruptive rewrite of function
  signatures across the crate.
- The scalar abstraction should not grow capability the current mathematics doesn't need;
  each new requirement is added only when an algorithm actually needs it, in keeping with
  the same discipline applied to the crate's other core traits.

## Status

Implemented. All operations across the crate are generic over the scalar type; `f32` and
`f64` are the concrete types currently in use, with room for further scalar types (such as
fixed-point) if a real need for one arises.

# Krylov Basis-Size Const-Generic Convention

## Summary

Krylov subspace methods that build up a basis of a fixed maximum size expose that size as a
`const K: usize` (Lanczos, Arnoldi) or `const M: usize` (GMRES(m) restart size) type
parameter directly on the public function, following the notation standard in the numerical
linear algebra literature.

## Scope

Applies to any current or future Krylov subspace method whose basis (or restart) size is
known at compile time — Lanczos and Arnoldi iterations, and the restart parameter of
GMRES(m). Does not apply to `power_iteration`/`inverse_power_iteration`, which have no basis
of growing size to bound. Does not introduce a new trait; it extends the existing minimal
[[static-dynamic-storage-trait]] the same way every other generic algorithm in the crate
already does.

## Decision

`const K: usize` names the Lanczos/Arnoldi basis size, and `const M: usize` names the
GMRES(m) restart size — the same symbols the literature uses for each, rather than a single
shared name across both algorithm families or a crate-invented one.

This size is a direct const-generic type parameter on the public function signature, not
hidden behind a wrapper type. Matrix types already expose their dimensions as const generics
throughout the public API (`StaticMatrix<T, const R: usize, const C: usize>`), so a basis
size expressed the same way is consistent with how callers already read every other
compile-time size in this crate; wrapping it would introduce a second convention for the
same kind of information with no offsetting benefit.

The const generic is an extension of the existing narrow `Storage` trait used to write
generic algorithms once against both static and dynamic storage, not a new trait — consistent
with the crate's one-pattern-reused-across-algorithms approach already established for
Krylov and other generic numerical code.

## Constraints

- `K` and `M` name the basis/restart size specifically for Lanczos/Arnoldi and GMRES(m)
  respectively; a different Krylov method with a differently-named parameter in its own
  literature uses that name instead, rather than being forced onto `K`/`M` for consistency's
  own sake.
- The basis/restart size is exposed as a direct type parameter on the public function, never
  hidden behind a wrapper type, matching how matrix dimensions are already exposed elsewhere
  in the public API.
- No new trait is introduced to support this; any capability the algorithms need beyond what
  the existing storage trait exposes is added to that trait only when a concrete algorithm
  needs it, not speculatively.

## Status

Partially implemented. Lanczos now exists and exposes its basis size as `const K: usize`,
establishing the pattern this convention describes. Arnoldi and GMRES(m) do not exist in the
crate yet.

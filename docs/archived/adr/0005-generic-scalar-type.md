
# ADR 0005: Generic Scalar Type

## Status
Accepted

## Context
The target environments for this library range from desktop and server systems, where
double-precision floating point (`f64`) is cheap and standard, to microcontrollers, many of
which lack a double-precision floating-point unit or any hardware floating-point support at
all, making single precision (`f32`) or fixed-point representations far more practical.

Hardcoding a single numeric type would simplify early development, since algorithms could be
written without generic type parameters or trait bounds. However, it would not reflect the
embedded use case that motivates the project, and changing the numeric type later would
require revisiting the signature of nearly every function in the library.

## Decision
Operations are written generically over a scalar type from the start, rather than hardcoded
to a single concrete numeric type. The set of capabilities such a type must provide (basic
arithmetic, the relevant elementary functions, identity values, etc.) is defined incrementally
as it is actually needed by the mathematics being implemented, rather than designed upfront in
full.

## Consequences

### Positive
* **Reflects real hardware constraints.** The library can be used with `f32` on
  resource-constrained targets and `f64` elsewhere, without maintaining separate codebases.
* **No large future refactor.** Generality is established before substantial code exists,
  avoiding a disruptive rewrite later.
* **Extensible.** The same generic foundation can, in principle, extend to other numeric
  representations (such as fixed-point) if a real need arises.

### Negative / Trade-offs
* **More verbose code from the start.** Every generic function carries type parameters and
  trait bounds, which adds reading and writing overhead, particularly while the underlying
  mathematics is still being learned and implemented.
* **Less beginner-friendly compiler errors.** Type errors involving generics combined with
  const generics tend to produce harder-to-read compiler diagnostics than equivalent errors
  in non-generic code.

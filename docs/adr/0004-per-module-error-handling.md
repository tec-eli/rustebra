---
layout: default
title: "ADR 0004: Per-Module Error Handling with Result"
---

# ADR 0004: Per-Module Error Handling with Result

## Status
Accepted

## Context
Operations across the library can fail for different reasons depending on the domain: an
operation on dynamically-sized data may receive incompatible dimensions discovered only at
runtime; a sparse format conversion may encounter malformed input; an iterative solver may
fail to converge. A decision is needed on how these failures are represented and reported to
the caller, and whether error types are shared or domain-specific.

Two extremes were considered: panicking on invalid input, or a single crate-wide error enum
covering every possible failure across all domains. Panicking is unsuitable for a library
targeting bare-metal environments, where an uncontrolled abort may be unacceptable for the
calling application. A single crate-wide error enum, on the other hand, would tightly couple
otherwise-independent mathematical domains to each other through their error types alone.

## Decision
Recoverable failures are reported through `Result`, not through panics. Each module defines
its own error type, scoped to the kinds of failure that are actually possible within that
module. A shared module holds error-related code that is genuinely common across domains
(such as shared conversion traits or formatting helpers), but it does not define a single
catch-all error type that all modules must funnel into.

## Consequences

### Positive
* **Decoupling.** Mathematical domains do not need to depend on each other's error
  vocabulary; a change to how sparse-format errors are represented has no effect on, say,
  dynamical-systems error types.
* **Caller control.** Failure handling is left to the code that calls the library, which is
  appropriate for a library meant to run in bare-metal and other failure-sensitive
  environments.
* **Smaller, more precise error types.** Each module's error type only needs to represent
  failures that can actually occur in that module, rather than a large enum with many
  variants irrelevant to most call sites.

### Negative / Trade-offs
* **More error types to maintain.** Every module that can fail needs its own error type
  defined and documented, rather than reusing one general-purpose type.
* **Cross-module error conversion.** Code that calls into multiple modules and wants to
  propagate a single error type upward will need explicit conversions between module-specific
  error types.

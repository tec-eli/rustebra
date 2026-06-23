---
layout: default
title: "ADR 0001: Hybrid Memory Allocation Strategy via Feature Flags"
---

# ADR 0001: Hybrid Memory Allocation Strategy via Feature Flags

## Status
Accepted (Revised — see Revision History)

## Context
The Rust ecosystem currently lacks an advanced linear algebra and sparse matrix library
tailored for strict embedded systems (`#![no_std]`) that can also scale to complex Krylov
subspace algorithms (such as Arnoldi or Lanczos iterations) in environments with an operating
system.

Basic linear algebra operations can run efficiently on the stack using fixed sizes known at
compile time. However, advanced operations (such as multiplying two sparse matrices or
running iterative solvers) suffer from severe limitations if dynamic memory (the heap) is
completely unavailable.

## Decision
We will implement a hybrid memory architecture based on two pillars:

1. **Strict core (`no_std` by default).** All baseline data layouts and elementary operations
   are designed without depending on the `alloc` crate. Const generics (`<const N: usize>`)
   are used to define the size of vectors, buffers, and matrices directly on the stack.
2. **Optional feature (`"alloc"`).** A conditional compilation flag named `alloc` unlocks
   dynamic data structures backed by `Vec<T>` and algorithms that require workspace
   allocation whose size cannot be known at compile time.

## Consequences

### Positive
* **Maximum portability.** The library can run on microcontrollers with only a few kilobytes
  of RAM without requiring a global memory allocator.
* **Compile-time safety.** In baseline mode, internal memory overflows and heap-fragmentation
  crashes are mathematically impossible.
* **Predictable performance.** Stack-allocated operations guarantee deterministic latency.
* **Scalability.** Advanced users can opt into dynamic, allocation-backed functionality by
  enabling a single feature.

### Negative / Trade-offs
* **API complexity.** Internal code must deal with more complex signatures due to the use of
  const generics in static mode.
* **Conceptual duplication.** Static and dynamic data structures (e.g. a fixed-size and a
  growable matrix) are maintained as separate types. How much of their behavior is shared is
  addressed separately in ADR 0003.

## Revision History
* **Original version:** also stated that "the API will be unified through common traits to
  guarantee that algorithms requiring no internal allocations can operate seamlessly over
  both static and dynamic memory structures." This claim was removed: whether and how static
  and dynamic types share an API surface is its own decision, not a consequence of the memory
  strategy. It is now addressed in ADR 0003, which currently decides against a shared trait.
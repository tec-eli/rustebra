# Per-Module Error Handling with Result

## Summary

Recoverable failures are reported through `Result`, never through panics, and each module
defines its own error type scoped to the failures that are actually possible within it,
rather than funneling every domain into one shared, crate-wide error enum.

## Scope

Applies library-wide, to every module capable of failing: operations on dynamically-sized
data that discover incompatible dimensions at runtime, sparse format conversions given
malformed input, iterative solvers that fail to converge, and any future module with its own
class of recoverable failure.

## Decision

Every module that can fail defines its own error type, containing only the variants that
can actually occur within that module's own operations. A shared module holds error-related
code that is genuinely common across domains — such as conversion traits or formatting
helpers — but it does not define a single catch-all error type that every other module must
funnel into.

No panics are used for recoverable failure. An operation that can fail for a reason the
caller might reasonably want to handle returns `Result`, full stop.

## Constraints

- The library targets bare-metal and other failure-sensitive environments where an
  uncontrolled abort may be unacceptable to the calling application — panicking on invalid,
  merely-unexpected input is not an option.
- A module's error type must not force otherwise-independent mathematical domains to depend
  on each other's error vocabulary; a change to one domain's error representation must not
  ripple into an unrelated domain.
- Error types must stay precise: a module's error type should only contain variants that can
  actually occur in that module, not a large shared enum with variants irrelevant to most
  call sites.

## Status

Implemented. Each domain module (dense operations, sparse formats, iterative solvers, and
others) defines and exports its own error type; no crate-wide catch-all error type exists.

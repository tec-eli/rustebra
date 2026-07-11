# Numerical Tolerance for Approximate-Zero Comparisons

## Summary

Functions whose entire purpose is judging whether a computed floating-point value is "small
enough not to count" — numerical rank, singular-value negligibility, and positive-definite
rounding noise — take an explicit tolerance rather than comparing against exact zero;
functions that are instead asking "is there truly nothing left, given the best available
candidate was already chosen" keep exact-zero comparisons and get their real fix elsewhere.

## Scope

Applies to every function that must decide whether a computed value counts as zero: rank
computation, singular value decomposition and condition-number estimation (deciding whether
a singular value is negligible), and Cholesky decomposition (deciding whether a value that
should be non-negative before a square root is negative only due to rounding noise, or
genuinely violates positive-definiteness). Also applies, by contrast, to pivot selection and
reflector/normalization zero-norm checks in LU and QR decomposition, which are addressed
directly rather than given a tolerance parameter. How a default tolerance is computed for
callers who don't want to supply their own is a separate concern.

## Decision

Two categories of exact-zero comparison exist, and they are treated differently.

The first category — "is there truly nothing left to act on" — is an algebraic fact about
the matrix, correctly decided by comparing to exact zero, provided the algorithm is already
choosing the best available candidate at each step rather than merely the first nonzero one.
Pivot selection and zero-norm checks fall here: their real defect was picking the first
nonzero candidate instead of the largest-magnitude one, and that defect is fixed by
switching to true largest-magnitude selection, after which the existing exact-zero check is
correct on its own terms. No tolerance parameter is added to these functions.

The second category — "is this small enough not to count" — has no exact answer once real,
non-toy floating-point data is involved. Rank, singular-value negligibility, and
positive-definiteness-under-rounding-noise all fall here. Functions answering this kind of
question gain an explicit `tolerance` parameter, compared either relative to an
already-computed scale (where one exists as a natural side effect of the algorithm) or as an
absolute threshold the caller chooses relative to their own data (where no such natural
scale exists).

## Constraints

- A decomposition function (LU, QR) must remain mathematically faithful to its actual input;
  it must never silently approximate its result differently depending on a threshold the
  caller did not ask for and may not know was applied.
- The "is there truly nothing left" category must never gain a tolerance parameter as a
  substitute for fixing genuine pivot/candidate-selection defects — thresholding would mask
  the real bug instead of fixing it.
- The two tolerance semantics in use — relative to an existing computed scale, versus an
  absolute threshold — must be documented clearly enough at each function that a caller
  cannot mistake one for the other.

## Status

Implemented. Rank, singular value decomposition, condition-number estimation, and Cholesky
decomposition each take an explicit tolerance parameter, relative or absolute as
appropriate. LU and QR pivot/normalization checks use true largest-magnitude selection with
exact-zero comparison and take no tolerance parameter. See also
[[auto-tolerance-defaults]] for how a default tolerance is offered to callers who don't want
to supply their own.

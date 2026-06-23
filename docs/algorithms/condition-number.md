---
layout: default
title: Condition Number
---

# Condition Number

## What it computes

A single non-negative number κ(A) that measures how sensitive the solution of a linear
system Ax = b is to small changes in A or b.

```
κ(A) = σ₁ / σₙ
```

where σ₁ is the largest singular value of A and σₙ is the smallest.

Equivalently, for invertible matrices:
```
κ(A) = ‖A‖ · ‖A⁻¹‖
```

## Intuition

Suppose you solve Ax = b and get a solution x. Now perturb b by a tiny amount δb. How much
does the solution change?

The condition number bounds the answer:
```
‖δx‖ / ‖x‖  ≤  κ(A) · ‖δb‖ / ‖b‖
```

A condition number of 1 means the system is perfectly well-conditioned: a 1% error in b
causes at most a 1% error in x. A condition number of 10⁶ means a tiny relative error in
b can cause an error up to a million times larger in x — the system is nearly impossible
to solve accurately with floating-point arithmetic.

Geometrically: a large condition number means the matrix A is "almost singular" — it maps
vectors in some direction to nearly zero, making it nearly impossible to tell which input
produced a given output.

## Scale and interpretation

| κ(A)          | Interpretation                                    |
|---------------|---------------------------------------------------|
| 1             | Perfect — no amplification of errors              |
| 10 – 100      | Well-conditioned — safe for most computations     |
| 10³ – 10⁶     | Moderately ill-conditioned — results may lose     |
|               | several digits of precision                       |
| > 10⁸         | Severely ill-conditioned — floating-point results |
|               | may be essentially meaningless                    |
| ∞             | Singular matrix — no solution or infinitely many  |

As a rule of thumb: if κ(A) ≈ 10^k, you lose approximately k digits of precision in the
solution.

## Method

The most reliable method is via SVD:
```
κ(A) = σ_max / σ_min
```

For symmetric positive-definite matrices, the eigenvalues equal the squared singular values,
so:
```
κ(A) = λ_max / λ_min
```

A cheaper but less reliable estimate uses the LU decomposition — several condition number
estimators (LAPACK-style) exist that avoid a full SVD while giving a good approximation.

## When to use it

- Before solving a linear system, to anticipate how accurate the solution can be.
- When comparing different formulations of the same problem — a better-conditioned
  formulation gives more accurate results with the same arithmetic.
- After computing a decomposition, as a sanity check on the result.
- In iterative methods (Krylov), a large condition number is why preconditioning is needed:
  it transforms the problem into a better-conditioned one.

## Relationship to other algorithms

- Computed exactly via SVD (most reliable).
- Estimated cheaply after LU decomposition (less reliable but faster).
- Directly motivates preconditioning in Krylov methods.
- A condition number of ∞ (or very large) is the formal definition of a matrix being
  singular (or nearly singular).

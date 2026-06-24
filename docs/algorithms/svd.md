
# Singular Value Decomposition (SVD)

## What it computes

A factorization of any m×n matrix A into three matrices:

```
A = U · Σ · Vᵀ
```

where:
- **U** is an m×m orthogonal matrix (the left singular vectors).
- **Σ** is an m×n diagonal matrix with non-negative entries σ₁ ≥ σ₂ ≥ ... ≥ 0 on the
  diagonal (the singular values).
- **V** is an n×n orthogonal matrix (the right singular vectors).

## Intuition

Every linear transformation — no matter how complex — can be broken into three simple steps:
1. **Rotate/reflect the input space** (Vᵀ).
2. **Scale each axis independently** (Σ — the singular values are the scale factors).
3. **Rotate/reflect the output space** (U).

The singular values tell you how much A stretches or compresses space in each "direction".
Large singular values correspond to directions where A amplifies; small ones where it
barely affects the input; zero singular values correspond to directions that get completely
collapsed (those are the nullspace of A).

Visually: SVD reveals the "principal axes" of the transformation.

## Singular values and their meaning

- The number of non-zero singular values equals the rank of A.
- The largest singular value σ₁ is the maximum amount A can stretch any unit vector.
- The smallest non-zero singular value σᵣ is the minimum stretch in the column space.
- If any singular value is zero, the matrix is rank-deficient (same as having determinant 0
  for square matrices).

## Method

Computing SVD directly is a multi-step process:

1. **Form AᵀA** (an n×n symmetric positive-semidefinite matrix).
2. **Compute its eigendecomposition:** AᵀA = V · D · Vᵀ, where D is diagonal with the
   eigenvalues on the diagonal.
3. **Singular values:** σᵢ = √λᵢ where λᵢ are the eigenvalues of AᵀA.
4. **Left singular vectors:** uᵢ = A·vᵢ / σᵢ for each non-zero singular value.

In practice, the eigendecomposition in step 2 is computed via the **QR algorithm** (iterative
QR decompositions applied repeatedly until the matrix converges to diagonal form), not by
solving the characteristic polynomial directly.

## Truncated SVD

Often only the k largest singular values are needed (k ≪ min(m,n)). Computing only those
(truncated SVD) is far cheaper than the full decomposition and sufficient for most
applications.

## Computational cost

O(min(m,n) · m · n) — the most expensive decomposition in this library. For large matrices,
iterative methods (Lanczos, Arnoldi) are used to approximate only the needed singular values.

## When to use it

- **Least-squares problems:** the most robust method, handling rank-deficient cases where
  QR would struggle.
- **Pseudoinverse:** A⁺ = V · Σ⁺ · Uᵀ, where Σ⁺ inverts the non-zero diagonal entries.
- **Low-rank approximation:** keep only the k largest singular values and vectors to get the
  best rank-k approximation of A (used in data compression, PCA, and dimensionality
  reduction).
- **Condition number:** σ₁ / σₙ (see condition number document).
- **Rank determination:** count non-zero singular values (more numerically reliable than
  Gaussian elimination for ill-conditioned matrices).

## Limitations

- The most computationally expensive decomposition in this library.
- Forming AᵀA explicitly can amplify floating-point errors — bidiagonalization-based
  algorithms (Golub-Reinsch) avoid this but are more complex to implement.

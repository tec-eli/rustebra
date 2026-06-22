# Cholesky Decomposition

## What it computes

A factorization of a symmetric positive-definite matrix A into:

```
A = L · Lᵀ
```

where **L** is lower triangular with positive diagonal entries. It is the "square root" of
a matrix in the sense that multiplying L by its own transpose recovers A.

## Intuition

LU decomposition works for any invertible square matrix, but it doesn't exploit symmetry.
If you know A is symmetric and positive-definite, you can do roughly half the work: instead
of finding two different triangular matrices L and U, you find one (L) and observe that U
is simply Lᵀ.

The positive-definite condition (all eigenvalues positive) guarantees that the square root
exists and that all diagonal entries of L are real and positive — which is what makes the
algorithm work without pivoting.

## What "symmetric positive-definite" means

- **Symmetric:** Aᵢⱼ = Aⱼᵢ for all i, j (the matrix equals its own transpose).
- **Positive-definite:** for every non-zero vector x, xᵀAx > 0. Intuitively, A doesn't
  flip any vector to point in an opposite direction.

Common sources: covariance matrices in statistics, stiffness matrices in structural
engineering, kernel matrices in machine learning.

## Method

Compute L column by column. For column j:

```
Lⱼⱼ = √(Aⱼⱼ − Σ Lⱼₖ²)        (k from 1 to j−1)

Lᵢⱼ = (Aᵢⱼ − Σ LᵢₖLⱼₖ) / Lⱼⱼ  (k from 1 to j−1, for i > j)
```

If at any point the expression inside the square root is negative or zero, the matrix is
not positive-definite and the decomposition does not exist.

**Example:**
```
A = | 4   2   2 |       L = | 2    0    0  |
    | 2   5   3 |           | 1    2    0  |
    | 2   3   6 |           | 1    1    √3 |
```

Verify: L · Lᵀ = A.

## Computational cost

O(n³/3) — approximately half the cost of LU decomposition for the same matrix size,
because the symmetry means only the lower triangle needs to be computed.

## When to use it

- Solving Ax = b when A is symmetric positive-definite — the most efficient general method
  in that case.
- Sampling from multivariate normal distributions (used in statistics and Monte Carlo
  methods).
- As a fast check for positive-definiteness: if Cholesky succeeds, the matrix is positive-
  definite; if it fails (negative under the square root), it is not.

## Limitations

- Only applicable to symmetric positive-definite matrices — will fail or produce incorrect
  results otherwise.
- Requires a square root operation at each diagonal step.
- Does not need pivoting (unlike LU), which simplifies the implementation but also means
  there is no fallback if the positive-definite condition is violated.

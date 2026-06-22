# LU Decomposition

## What it computes

A factorization of a square matrix A into two triangular matrices:

```
A = L · U
```

where **L** is lower triangular (zeros above the diagonal) and **U** is upper triangular
(zeros below the diagonal).

## Intuition

Solving a linear system Ax = b directly is expensive. But if A is already triangular,
solving is trivial — you just substitute forward or backward. LU decomposition transforms
the general problem into two easy triangular ones.

Once you have L and U, solving Ax = b becomes:
1. Solve Ly = b for y (forward substitution — trivial because L is lower triangular).
2. Solve Ux = y for x (back substitution — trivial because U is upper triangular).

The decomposition pays for itself when you need to solve multiple systems with the same A
but different right-hand sides b — compute L and U once, reuse them many times.

## Method

**Gaussian elimination, recorded as a matrix factorization.**

The same elimination steps used for row reduction can be captured as multiplications by
elementary lower-triangular matrices. Collecting those elimination steps gives L; the result
of the elimination is U.

At each step k, for each row i below row k:
```
multiplier mᵢₖ = aᵢₖ / aₖₖ
row i ← row i − mᵢₖ · row k
```

The multipliers mᵢₖ fill the lower triangle of L; the remaining matrix is U.

**Example:**
```
A = | 2  1  1 |       L = | 1    0   0 |    U = | 2  1   1  |
    | 4  3  3 |           | 2    1   0 |        | 0  1   1  |
    | 8  7  9 |           | 4    3   1 |        | 0  0   2  |
```

Verify: L · U = A.

## Pivoting

If a diagonal entry (pivot) is zero or very small, the division `aᵢₖ / aₖₖ` becomes
undefined or numerically unstable. **Partial pivoting** solves this by swapping rows before
each step to put the largest available entry on the diagonal:

```
P · A = L · U
```

where P is a permutation matrix recording the row swaps. Partial pivoting is the standard
in practice — LU without pivoting can fail even when the matrix is technically invertible.

## Computing the determinant via LU

Once U is available, the determinant is the product of its diagonal entries, adjusted for
the sign of the row permutations:

```
det(A) = (−1)^s · u₁₁ · u₂₂ · ... · uₙₙ
```

where s is the number of row swaps performed during pivoting. This is O(n²) — far cheaper
than cofactor expansion.

## Computational cost

O(n³) for the decomposition. O(n²) for each subsequent solve once L and U are known.

## When to use it

- Solving general square linear systems Ax = b, especially when solving multiple times with
  the same A.
- Computing the determinant efficiently for matrices larger than approximately 4×4.
- Computing the matrix inverse (though direct solve is usually preferable).
- As a building block for other algorithms.

## Limitations

- Only directly applicable to square matrices.
- Requires partial pivoting for numerical stability.
- Not the best choice when A has known special structure (symmetric positive-definite →
  Cholesky; orthogonal → QR).

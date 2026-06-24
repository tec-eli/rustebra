
# Rank

## What it computes

The rank of a matrix is the number of linearly independent rows (or equivalently, linearly
independent columns). It tells you the effective dimensionality of the information the matrix
carries.

## Intuition

If a matrix has 4 rows but one of them is a linear combination of the others (e.g. it equals
the sum of the first two rows), that row adds no new information. The rank counts only the
rows that are genuinely independent — the rest are redundant.

A matrix of size m×n has rank at most min(m, n). When the rank equals min(m, n), the matrix
is said to have **full rank**. When it is lower, the matrix is **rank-deficient**.

## Method

**Gaussian elimination to row echelon form.**

Apply a sequence of elementary row operations to reduce the matrix:
1. Find the leftmost column with a non-zero entry (the pivot column).
2. Swap rows if needed to bring a non-zero entry to the top of that column.
3. Subtract multiples of that row from all rows below it to zero out every entry below the
   pivot.
4. Repeat for the submatrix below and to the right of the current pivot.

The result is a matrix in **row echelon form**: each row either starts with a leading non-zero
entry (a pivot) further to the right than the row above it, or is entirely zero.

The rank is the number of non-zero rows in the result.

**Example:**
```
| 1  2  3 |       | 1  2  3 |
| 2  4  6 |  →    | 0  0  0 |   rank = 1
| 3  6  9 |       | 0  0  0 |
```

Every row was a multiple of the first; after elimination, only one non-zero row remains.

```
| 1  0  2 |       | 1  0  2 |
| 0  1  3 |  →    | 0  1  3 |   rank = 2
| 0  0  0 |       | 0  0  0 |
```

Two independent rows survive.

## Computational cost

O(m·n·min(m,n)) — efficient for matrices of practical size.

## When to use it

- Checking whether a linear system has a unique solution, infinitely many, or none.
- Detecting redundancy in a set of equations or vectors.
- As a prerequisite for understanding SVD and the condition number.

## Relationship to other algorithms

- A square matrix has full rank if and only if its determinant is non-zero.
- The rank is the number of non-zero singular values (see SVD).
- A rank-deficient matrix has no inverse.

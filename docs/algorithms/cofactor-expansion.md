
# Cofactor Expansion

## What it computes

The determinant of a square matrix.

## Intuition

The determinant measures how much a matrix scales areas (in 2D) or volumes (in 3D) when
applied as a linear transformation. A determinant of 2 means areas double; a determinant of
0 means the transformation collapses space into a lower dimension, losing information
irreversibly.

## Method

Pick any row or column. For each element in that row or column, multiply it by the
determinant of the submatrix obtained by removing that element's row and column (called the
**minor**), then alternate signs following the checkerboard pattern:

```
+ - + - ...
- + - + ...
+ - + - ...
```

The sum of those signed products is the determinant.

**Base cases:**

For a 1×1 matrix:
```
det([a]) = a
```

For a 2×2 matrix (the formula most people remember):
```
det | a  b | = a·d − b·c
    | c  d |
```

**General case — expanding along the first row:**
```
det(A) = Σ (−1)^(1+j) · a₁ⱼ · det(M₁ⱼ)
          j
```

where `M₁ⱼ` is the minor obtained by deleting row 1 and column j.

**3×3 example:**
```
det | a  b  c |
    | d  e  f | = a·(e·i − f·h) − b·(d·i − f·g) + c·(d·h − e·g)
    | g  h  i |
```

## Computational cost

O(n!) — grows factorially with matrix size. This is impractical for large matrices.
For matrices larger than approximately 4×4, LU decomposition computes the same result far
more efficiently.

## When to use it

- Small matrices (up to approximately 4×4) where the direct formula is exact and cheap.
- Educational contexts where the explicit formula is needed.
- As a building block for computing the inverse via the adjugate matrix.

## Limitations

- Impractical for large matrices due to O(n!) cost.
- Accumulates floating-point rounding errors as matrix size grows.
- Only defined for square matrices.

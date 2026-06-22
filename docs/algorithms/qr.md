# QR Decomposition

## What it computes

A factorization of any m×n matrix A (with m ≥ n) into:

```
A = Q · R
```

where **Q** is orthogonal (its columns are unit vectors, all perpendicular to each other,
and Qᵀ·Q = I) and **R** is upper triangular.

## Intuition

An orthogonal matrix Q is a pure rotation or reflection — it doesn't distort lengths or
angles. R is triangular, so easy to work with. Together they let you decompose any linear
transformation into "rotate/reflect first, then scale and shear".

Because Qᵀ = Q⁻¹ (a property unique to orthogonal matrices), solving Ax = b via QR becomes:
```
QRx = b  →  Rx = Qᵀb
```
which is a triangular system — solved immediately by back substitution.

## Methods

### Gram-Schmidt orthogonalization

Take the columns of A one by one. For each new column, subtract its projections onto all
previously processed columns to remove any component in their direction, then normalize to
unit length. The normalized columns form Q; the projection coefficients fill R.

For columns a₁, a₂, ..., aₙ:
```
u₁ = a₁
u₂ = a₂ − proj_{u₁}(a₂)
u₃ = a₃ − proj_{u₁}(a₃) − proj_{u₂}(a₃)
...

qᵢ = uᵢ / ‖uᵢ‖

where proj_u(a) = (a·u / u·u) · u
```

Simple to understand, but accumulates floating-point errors for nearly dependent columns.
**Modified Gram-Schmidt** (projecting against already-orthogonalized columns rather than the
originals) reduces this error significantly.

### Householder reflections

Rather than building Q column by column, apply a sequence of reflection matrices Hₖ that
zero out the entries below the diagonal one column at a time:

```
Hₙ · ... · H₂ · H₁ · A = R
Q = H₁ · H₂ · ... · Hₙ
```

Each Hₖ is a Householder reflector of the form:
```
H = I − 2·vvᵀ / (vᵀv)
```

chosen so that it maps a given vector onto a multiple of a coordinate axis. More
numerically stable than Gram-Schmidt and preferred in practice.

## Computational cost

O(mn²) for an m×n matrix — more expensive than LU for square systems, but applicable to
non-square matrices and more numerically stable.

## When to use it

- **Least-squares problems:** solving overdetermined systems (more equations than unknowns),
  where no exact solution exists and the best approximate solution is sought.
- **Eigenvalue algorithms:** QR iteration is the basis of the standard method for computing
  all eigenvalues of a matrix.
- **Arnoldi and Lanczos iterations:** QR is used internally to orthogonalize the Krylov
  basis.
- When numerical stability matters more than raw speed.

## Limitations

- More expensive than LU for square systems when stability is not a concern.
- The full Q matrix is large; often only Qᵀb is needed, not Q itself.

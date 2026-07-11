# Eigenvalues & Eigenvectors

An eigenvector of a square matrix `a` is a non-zero vector `v` whose direction `a` leaves
unchanged: `a * v = λ * v` for some scalar `λ`, its corresponding eigenvalue. Geometrically,
`a` only stretches or shrinks `v` (by a factor of `λ`) rather than rotating it into a
different direction. Eigenvalues show up throughout numerical linear algebra: a matrix's
condition number (see [Condition Number](../algorithms/condition-number.md)) is the ratio
of its largest to smallest singular value — themselves the square roots of the eigenvalues
of `aᵗ * a` — and stability analysis of iterative systems generally comes down to whether
relevant eigenvalues stay inside or outside the unit circle.

`rustebra` doesn't compute eigenvalues via a direct, closed-form decomposition (the way it
computes LU or QR). Instead, [Krylov Methods](./08-krylov/README.md) provides two iterative
methods: `power_iteration`, which converges to the dominant (largest-magnitude) eigenvalue
and a corresponding eigenvector, and `inverse_power_iteration`, which converges to whichever
eigenvalue lies nearest a caller-chosen shift — useful for targeting the smallest eigenvalue,
or any eigenvalue you already have a rough estimate for. See that section for the algorithms,
their convergence behavior, and worked examples.

## Gotchas

- These are iterative approximations, not exact decompositions — they return an error
  (`MaxIterationsExceeded`) rather than a result if convergence criteria aren't met within
  the given iteration budget, and convergence speed depends on how well-separated the
  target eigenvalue is from its neighbors. See [Power Iteration](./08-krylov/power-iteration.md)
  for the specifics.

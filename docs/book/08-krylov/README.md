# Krylov Methods

`rustebra` provides iterative eigenvalue methods in `rustebra::krylov`: power iteration for
the dominant (largest-magnitude) eigenvalue, inverse power iteration for the eigenvalue
nearest an arbitrary shift, and Lanczos iteration, which builds an orthonormal basis of a
Krylov subspace and the symmetric tridiagonal matrix a symmetric operator projects onto
within it. Unlike the direct decompositions in
[Decompositions](../06-decompositions/README.md), these refine an estimate (or a basis) over
many iterations and can fail to converge — or, for Lanczos, to extend the basis further —
within a given budget, in addition to the usual dimension and non-finite-value failure modes.

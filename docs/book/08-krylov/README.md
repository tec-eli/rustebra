# Krylov Methods

`rustebra` provides iterative solvers and eigenvalue methods in `rustebra::krylov`:

- **Power iteration** for the dominant (largest-magnitude) eigenvalue
- **Inverse power iteration** for the eigenvalue nearest an arbitrary shift
- **Conjugate Gradient (CG)** for solving symmetric positive-definite linear systems
- **Lanczos iteration**, which builds an orthonormal basis of a Krylov subspace and the
  symmetric tridiagonal matrix a symmetric operator projects onto within it
- **Arnoldi iteration**, the non-symmetric counterpart to Lanczos, which builds an orthonormal
  basis of a Krylov subspace and the upper Hessenberg matrix a general operator projects onto
  within it
- **GMRES(m)**, a restarted iterative solver for general (possibly non-symmetric) linear
  systems, built on top of Arnoldi iteration

Unlike the direct decompositions in [Decompositions](../06-decompositions/README.md), these
refine an estimate (or a basis) over many iterations and can fail to converge — or, for
Lanczos, to extend the basis further — within a given budget, in addition to the usual
dimension and non-finite-value failure modes.

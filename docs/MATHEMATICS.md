# Mathematical Landscape

A map of mathematical domains relevant to this project, grouped by how the concepts relate
to and depend on each other. Pure mathematics only — no implementation, no prioritization,
no versioning.

Groups are ordered by mathematical dependency: later groups build on earlier ones.

---

## Group A — Elementary functions

The scalar functions that everything else is built from.

- Powers and roots: square root, cube root, general power, hypotenuse (√(x²+y²))
- Exponential and logarithm: exponential, natural log, log base 2, log base 10
- Trigonometric: sine, cosine, tangent, and their inverses (arcsine, arccosine, arctangent,
  two-argument arctangent)
- Hyperbolic functions: sinh, cosh, tanh, and their inverses
- Rounding and classification: floor, ceiling, round, truncation, fractional part, absolute
  value, sign, detection of special values (not-a-number, infinity)
- Constants: π, e, τ, √2, etc.

These operate on single numbers, but every other group depends on them.

---

## Group B — Vectors

The smallest object with linear-algebraic structure: an ordered tuple of numbers with
addition and scalar multiplication.

- Arithmetic: addition, subtraction, scalar multiplication, element-wise multiplication and
  division
- Products: dot product (inner product), cross product (three dimensions only)
- Norms: L1 (sum of absolute values), L2 (Euclidean length), L∞ (maximum absolute value),
  general Lp norm
- Derived quantities: normalization (unit vector), angle between two vectors, projection of
  one vector onto another, distance between two vectors
- Comparisons: element-wise minimum/maximum, clamping to a range

Depends on Group A (norms require square roots; angles require inverse trigonometric
functions).

---

## Group C — Dense matrices

A matrix represents a linear map between vector spaces; this group builds directly on
vectors.

**Basic arithmetic**
- Addition, subtraction, scalar multiplication, element-wise multiplication
- Matrix–vector multiplication, matrix–matrix multiplication
- Transpose, trace

**Structural properties**
- Determinant
- Rank
- Matrix norms: Frobenius norm, induced 1-norm and ∞-norm, spectral norm
- Condition number

**Solving and inverting**
- Solving a linear system of equations
- Matrix inverse (well-defined and tractable mainly for small matrices)

**Decompositions** — each of these is itself a small algorithm with its own numerical
behavior, not a single operation
- LU decomposition, with and without pivoting
- QR decomposition (via Gram-Schmidt orthogonalization or Householder reflections)
- Cholesky decomposition (symmetric positive-definite matrices only; cheaper than LU when
  applicable)
- Eigendecomposition
- Singular Value Decomposition (the most general and most expensive; underlies
  least-squares fitting, principal component analysis, the pseudo-inverse, and the
  condition number)

Depends on Groups A and B.

---

## Group D — Sparse matrices

The same mathematical objects as Group C, in the case where most entries are zero. This is a
representation and access-pattern distinction, not new mathematics: the operations mirror
Group C, but the algorithms to carry them out differ substantially.

**Representations**
- Coordinate list: each non-zero stored as (row, column, value) — simple to construct,
  inefficient to compute with
- Compressed row representation — efficient for row-wise access and for multiplying by a
  vector
- Compressed column representation — the column-wise counterpart
- Conversion between representations

**Operations**
- Addition, scalar multiplication
- Multiplication by a dense vector
- Multiplication by another sparse matrix
- Multiplication by a dense matrix
- Measures of sparsity (proportion and pattern of non-zero entries)

Depends on Group C conceptually, and on Group A for any numeric reductions involved.

---

## Group E — Krylov subspace methods

For matrices too large to decompose directly, where forming the full matrix or its
decomposition is computationally infeasible. Instead of operating on the whole matrix, a
small subspace is built from repeated matrix–vector products, and the problem is solved
within that subspace.

- **Lanczos iteration** — applies to symmetric matrices; computationally cheap, since it only
  needs to retain the last couple of vectors generated. Underlies:
    - The Conjugate Gradient method, for solving linear systems with symmetric
      positive-definite matrices
    - Eigenvalue and eigenvector estimation for symmetric matrices
- **Arnoldi iteration** — applies to general (non-symmetric) matrices; more expensive, since
  it requires orthogonalizing against all previously generated vectors. Underlies:
    - GMRES, for solving general linear systems
    - Eigenvalue and eigenvector estimation for general matrices
- **Power iteration and inverse power iteration** — the simplest eigenvalue estimator,
  finding only the dominant eigenvalue; a natural stepping stone before Lanczos or Arnoldi
- **Preconditioning** — a family of techniques for transforming a problem so that iterative
  methods converge faster, rather than a single operation

Depends on Group B (vector operations) and on Group C or D as the underlying linear operator
— Krylov methods only require the ability to multiply that operator by a vector, regardless
of whether it is dense or sparse.

---

## Group F — Numerical calculus

The bridge between continuous mathematics and discrete computation.

- **Differentiation**: forward, backward, and central finite differences
- **Integration (quadrature)**: trapezoidal rule, Simpson's rule, Gaussian quadrature
- **Root finding**: bisection method, Newton-Raphson method, secant method
- **Interpolation**: linear interpolation, polynomial interpolation (Lagrange or Newton
  form), cubic splines
- **Optimization**: gradient descent, line search, as foundational methods

Depends on Group A (the functions typically being differentiated, integrated, or solved are
built from elementary functions), and on Group B for the multivariate case (gradients are
vectors).

---

## Group G — Dynamical systems

Models of systems that evolve over time according to differential equations, including
their long-term and chaotic behavior.

**Solving ordinary differential equations** (in increasing order of accuracy and cost)
- Euler's method (explicit) — simplest, least accurate
- Runge-Kutta methods (second order, fourth order) — standard general-purpose solvers
- Adaptive step-size methods (e.g. Runge-Kutta-Fehlberg / Dormand-Prince) — adjust the step
  size to control local error
- Implicit methods (e.g. backward Euler) — required for stiff systems, where explicit
  methods become numerically unstable

**Systems and their analysis**
- Systems of coupled differential equations, not just single equations
- Canonical examples used to study and validate solver behavior: the logistic map, the
  Lorenz system, the Rössler system
- Fixed points and their stability
- Lyapunov exponents — quantify sensitivity to initial conditions, the formal signature of
  chaotic behavior
- Bifurcation analysis — how qualitative system behavior changes as a parameter varies
- Phase-space trajectories and Poincaré sections, as tools for analyzing long-term behavior

Depends on Group A, on Group F (root finding for fixed points, differentiation for
linearization and stability analysis), on Group B (state vectors), and on Group C
(Jacobian matrices for stability analysis).

---

## Group H — Signal processing

The mathematics of processing discrete, sampled signals over time.

**Filtering**
- Finite impulse response filtering
- Infinite impulse response filtering (cheaper per sample, but can be unstable)
- Standard filter responses: low-pass, high-pass, band-pass, band-stop
- Convolution and correlation (cross-correlation, autocorrelation)

**Transforms**
- The Discrete Fourier Transform, as the conceptual baseline
- The Fast Fourier Transform, as its efficient computational form
- Window functions (Hamming, Hanning, Blackman), used to reduce spectral leakage before
  transforming a signal

**Signal analysis**
- Root mean square, peak detection, zero-crossing rate
- Spectral analysis: power spectrum, dominant frequency
- Resampling: decimation (reducing the sample rate) and interpolation (increasing it)

Depends on Group A (filters and transforms are built from trigonometric functions), Group B
(a signal is naturally represented as a vector of samples), and conceptually on Group F
(convolution is the discrete analogue of integration).


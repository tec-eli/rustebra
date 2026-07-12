<!-- MISSING EXAMPLE: expected at examples/krylov/conjugate_gradient.rs -->

# Conjugate Gradient (CG)

`conjugate_gradient` solves the symmetric positive-definite (SPD) linear system `A x = b` via
iterative refinement: building a sequence of conjugate directions from residuals, then stepping
along each direction to get closer to the solution. CG exploits the SPD structure for rapid
convergence — in exact arithmetic, it converges to the true solution in at most `n` iterations
for an `n × n` matrix, but in practice finite-precision arithmetic and early stopping via
residual tolerance reach an approximate solution much faster.

Each iteration does one matrix-vector product (`A * p_k`) and two dot products, with no
factorization and only 4 fixed-size working vectors (`r`, `p`, `Ap`, `x`). This makes CG
ideal for large systems where factorization is too expensive.

```rust
use rustebra::krylov::conjugate_gradient;
use rustebra::storage::StaticStorage;

// Symmetric positive-definite 2x2 matrix: [[2, 1], [1, 2]].
// System: 2x + y = 1, x + 2y = 2. Solution: x = [0, 1].
let a = StaticStorage::new([2.0_f64, 1.0, 1.0, 2.0]);
let b = [1.0, 2.0];
let x0 = [0.0, 0.0];
let mut x = [0.0; 2];
let mut r = [0.0; 2];
let mut p = [0.0; 2];
let mut ap = [0.0; 2];

conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap)
    .unwrap();

// Solution satisfies A x = b
assert!((x[0] - 0.0).abs() < 1e-8);
assert!((x[1] - 1.0).abs() < 1e-8);
```

## Convergence

The convergence rate depends on the condition number `κ(A) = λ_max / λ_min` of `a`: the
iteration converges to within factor `(√κ - 1) / (√κ + 1)` per step in the energy norm
`‖e‖_a = √(e^T A e)`. This means:

- **Well-conditioned matrices** (κ ≈ 1): Fast convergence, often within a handful of iterations.
- **Ill-conditioned matrices** (κ >> 1): Slow convergence, but still graceful degradation.

For example, a matrix with κ = 100 converges at rate roughly `(10 - 1) / (10 + 1) ≈ 0.82` per
iteration, so even pathological cases eventually reach a solution with patience.

## SPD Detection

A genuinely SPD matrix has all positive eigenvalues. CG detects a non-SPD input operationally:
if the dot product `p_k · (A p_k)` ever becomes zero, negative, or NaN mid-iteration, the
algorithm returns `ConvergenceError::NonFinite`. This happens automatically for:

- Negative-definite or indefinite matrices
- Singular or numerically singular matrices
- NaN/Inf from overflow or ill-conditioning

```rust
use rustebra::krylov::{conjugate_gradient, ConvergenceError};
use rustebra::storage::StaticStorage;

// Indefinite matrix: [[1, 0], [0, -1]] has eigenvalues 1 and -1.
let a = StaticStorage::new([1.0_f64, 0.0, 0.0, -1.0]);
let b = [1.0, 1.0];
let x0 = [0.0, 0.0];
let mut x = [0.0; 2];
let mut r = [0.0; 2];
let mut p = [0.0; 2];
let mut ap = [0.0; 2];

let result = conjugate_gradient(&a, 2, &b, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap);
assert_eq!(result, Err(ConvergenceError::NonFinite));
```

## Gotchas

- **Initial guess matters**: A good initial guess `x0` can reduce iterations significantly. If
  you have no prior estimate, `x0 = [0, ..., 0]` is standard.
- **Tolerance choice**: Requesting `tol = 1e-15` on `f64` (machine epsilon ~2.2e-16) can lead to
  `MaxIterationsExceeded`. Use `tol` relative to the scale of `b`.
- **Matrix must be SPD**: Non-symmetric or indefinite matrices cause immediate failure. If
  unsure, use a general-purpose solver like QR or SVD instead.
- **Nonfinite iterates**: If `A x_k` contains NaN or Inf, CG returns `ConvergenceError::NonFinite`
  on the next residual check — not a silent failure or panic, per the crate's policy.

## Example: Least-Squares via Normal Equations

While CG solves `A x = b` directly, you can solve overdetermined least-squares `min ‖A x - b‖`
via the normal equations: solve `A^T A x = A^T b`. The system is symmetric and positive-definite
(assuming full column rank), so CG applies:

```rust
use rustebra::krylov::conjugate_gradient;
use rustebra::storage::StaticStorage;

// Overdetermined system: 3 equations, 2 unknowns.
// A = [[1, 1], [1, 2], [1, 3]], b = [2, 3, 5]
// A^T A = [[3, 6], [6, 14]] (symmetric, positive-definite)
// A^T b = [10, 24]
let ata = StaticStorage::new([3.0_f64, 6.0, 6.0, 14.0]);
let atb = [10.0, 24.0];
let x0 = [0.0, 0.0];
let mut x = [0.0; 2];
let mut r = [0.0; 2];
let mut p = [0.0; 2];
let mut ap = [0.0; 2];

conjugate_gradient(&ata, 2, &atb, &x0, 100, 1e-10, &mut x, &mut r, &mut p, &mut ap)
    .unwrap();

// Verify solution approximately
assert!(x[0] > 0.5 && x[0] < 1.5);
assert!(x[1] > 1.0 && x[1] < 2.0);
```

Note: Normal equations can amplify rounding error for ill-conditioned matrices (κ(A^T A) ≈
κ(A)²). For numerical robustness, use QR decomposition or SVD instead.

<!-- MISSING EXAMPLE: expected at examples/krylov/power_iteration.rs -->

# Power Iteration

`power_iteration` computes the dominant (largest-magnitude) eigenvalue of an `n x n` matrix
`a`, along with a corresponding unit-length eigenvector. Starting from an initial guess
`v0` (normalized first), each iteration multiplies the current estimate by `a` and
renormalizes: `v_{k+1} = a * v_k / ‖a * v_k‖`. The eigenvalue estimate is the Rayleigh
quotient `λ_k = v_kᵗ * (a * v_k)`. Renormalizing every iteration is what keeps repeated
multiplication from overflowing or underflowing — only the iterate's direction evolves,
its magnitude is reset to `1` each round.

```rust
use rustebra::krylov::power_iteration;
use rustebra::storage::StaticStorage;

// [[2, 0], [0, 1]]; dominant eigenvalue 2, eigenvector (1, 0).
let a = StaticStorage::new([2.0, 0.0, 0.0, 1.0]);
let v0 = StaticStorage::new([1.0, 1.0]);
let mut eigenvector = [0.0; 2];
let mut scratch = [0.0; 2];

let eigenvalue = power_iteration(&a, 2, &v0, 100, 1e-10, &mut eigenvector, &mut scratch)
    .unwrap();
assert!((eigenvalue - 2.0).abs() < 1e-9);
```

Iteration stops once both the eigenvalue estimate and the eigenpair residual
(`‖a * v_k - λ_k * v_k‖`) fall within the requested tolerance, relative to the eigenvalue's
magnitude. Convergence speed depends on how well-separated the dominant eigenvalue is from
the second-largest: the eigenvector error shrinks by roughly `|λ2 / λ1|` per iteration.

## Inverse power iteration

Plain power iteration can only find the *largest*-magnitude eigenvalue. `inverse_power_iteration`
targets the eigenvalue nearest an arbitrary `shift` instead, by applying the same
direction-refinement loop to the operator `(a - shift * I)⁻¹` — computed via one partial-pivoted
LU factorization up front, then a triangular solve per iteration rather than a plain
matrix-vector product. With `shift == 0`, this finds the smallest-magnitude eigenvalue.
Convergence rate becomes `|λ1 - shift| / |λ2 - shift|` (nearest and second-nearest the
shift), so a shift close to a good eigenvalue estimate converges very fast.

## Gotchas

- Power iteration doesn't converge — and returns
  `ConvergenceError::MaxIterationsExceeded` — when the two largest-magnitude eigenvalues
  are exactly tied in magnitude but not value (e.g. `λ2 == -λ1`): the iterate oscillates
  forever rather than settling.
- `inverse_power_iteration` returns `ConvergenceError::SingularShift` if `shift` coincides
  with (or lands within `singular_tol` of) an actual eigenvalue, rather than solving
  against amplified numerical noise. A shift merely *near* an eigenvalue but past that
  check is not a problem — it converges faster, not worse.
- Both functions return `ConvergenceError::ZeroVector` if `v0` has zero norm, and
  `ConvergenceError::NonFinite` if an iterate goes `NaN`/infinite — errors, not panics or
  silent garbage output, per the crate's no-`unwrap`-in-library-code policy.

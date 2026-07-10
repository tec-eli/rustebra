//! Shared harness for the Krylov test suite: tolerance constants, known-spectrum matrix
//! generation, and the sign-insensitive eigenvector comparison every eigenvector check uses.

use nalgebra::{Matrix4, Vector4};
use proptest::prelude::*;

/// Side length of every matrix the property harness generates.
pub const N: usize = 4;

/// The `tol` passed into the algorithms under test.
pub const ALGORITHM_TOL: f64 = 1e-10;

/// What assertions check results against: two orders of magnitude of slack over
/// [`ALGORITHM_TOL`], because the requested tolerance and the achievable accuracy differ.
pub const ASSERTION_TOL: f64 = 1e-8;

/// The `singular_tol` passed to `inverse_power_iteration` in f64 tests.
pub const SINGULAR_TOL: f64 = 1e-12;

/// f32 counterpart of [`ALGORITHM_TOL`]: `sqrt(f32::EPSILON)` ≈ 3.45e-4, scaled from the
/// precision actually available in f32 rather than reused from the f64 constants.
pub fn algorithm_tol_f32() -> f32 {
    f32::EPSILON.sqrt()
}

/// f32 counterpart of [`ASSERTION_TOL`]: the same two orders of magnitude of slack over
/// [`algorithm_tol_f32`].
pub fn assertion_tol_f32() -> f32 {
    100.0 * f32::EPSILON.sqrt()
}

/// Normalized inner product `v · w / (‖v‖ ‖w‖)`, or `0` when either vector is degenerate.
pub fn overlap(v: &[f64], w: &[f64]) -> f64 {
    let dot: f64 = v.iter().zip(w.iter()).map(|(x, y)| x * y).sum();
    let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_w: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_v > 0.0 && norm_w > 0.0 {
        dot / (norm_v * norm_w)
    } else {
        0.0
    }
}

/// Eigenvector equality up to the sign ambiguity: `|v · w| / (‖v‖ ‖w‖) >= 1 - tol`. Both `v`
/// and `-v` are valid eigenvectors, so element-wise comparison is never correct.
pub fn approx_eq_eigenvector(v: &[f64], w: &[f64], tol: f64) -> bool {
    overlap(v, w).abs() >= 1.0 - tol
}

/// Iteration budget derived from the convergence ratio: `ceil(ln tol / ln ratio) * 2`, with a
/// small floor so a tiny ratio still leaves room for the mandatory two iterations and the
/// eigenvalue-stabilization double check.
pub fn max_iter_for(convergence_ratio: f64, tol: f64) -> usize {
    ((tol.ln() / convergence_ratio.ln()).ceil() as usize).max(5) * 2
}

/// The convergence ratio `max_i |λ_i / λ_1|` of a spectrum whose dominant eigenvalue sits at
/// index 0, as [`spectrum_with_gap`] guarantees.
pub fn dominant_ratio(eigenvalues: &[f64; N]) -> f64 {
    let dominant = eigenvalues[0].abs();
    eigenvalues[1..]
        .iter()
        .fold(0.0_f64, |acc, x| acc.max(x.abs() / dominant))
}

prop_compose! {
    /// Eigenvalues with a controlled spectral gap: the dominant eigenvalue `λ_1` (index 0) has
    /// magnitude in `[0.5, 50]` and random sign, every other eigenvalue satisfies
    /// `|λ_i / λ_1| <= 0.98` by construction, and no magnitude falls below `0.01` — the
    /// relative convergence criterion degenerates near zero, so near-zero targets are never
    /// generated.
    pub fn spectrum_with_gap()(
        magnitude in 0.5..50.0f64,
        negate in any::<bool>(),
        ratios in prop::array::uniform3(-0.98..0.98f64),
    ) -> [f64; N] {
        let lambda_1 = if negate { -magnitude } else { magnitude };
        let mut eigenvalues = [lambda_1, 0.0, 0.0, 0.0];
        for (slot, ratio) in eigenvalues[1..].iter_mut().zip(ratios) {
            let value = ratio * lambda_1;
            *slot = if value.abs() < 0.01 {
                if value >= 0.0 { 0.01 } else { -0.01 }
            } else {
                value
            };
        }
        eigenvalues
    }
}

/// Orthogonal matrix obtained by QR-decomposing the given random entries; Householder QR
/// yields an orthogonal `Q` for any input, so no rejection is needed.
fn orthogonal_from(entries: &[f64; 16]) -> Matrix4<f64> {
    Matrix4::from_row_slice(entries).qr().q()
}

fn row_major(m: &Matrix4<f64>) -> [f64; 16] {
    let mut out = [0.0; 16];
    for r in 0..N {
        for c in 0..N {
            out[r * N + c] = m[(r, c)];
        }
    }
    out
}

/// Column `index` of `m` as a plain array.
pub fn column(m: &Matrix4<f64>, index: usize) -> [f64; N] {
    [m[(0, index)], m[(1, index)], m[(2, index)], m[(3, index)]]
}

/// Symmetric matrix `A = Q D Qᵀ` (row-major) with the given eigenvalues; the eigenvector for
/// `eigenvalues[i]` is column `i` of the returned orthogonal `Q`.
pub fn symmetric_with_spectrum(
    eigenvalues: &[f64; N],
    q_seed: &[f64; 16],
) -> ([f64; 16], Matrix4<f64>) {
    let q = orthogonal_from(q_seed);
    let d = Matrix4::from_diagonal(&Vector4::from_row_slice(eigenvalues));
    let a = q * d * q.transpose();
    (row_major(&a), q)
}

/// Non-symmetric matrix `A = P D P⁻¹` (row-major) with the given eigenvalues, where
/// `P = Q · U` for an orthogonal `Q` and a unit upper-triangular `U` with small off-diagonal
/// entries — invertible by construction and well conditioned, so the spectrum stays an
/// accurate oracle. The eigenvector for `eigenvalues[i]` is column `i` of the returned `P`;
/// `P⁻¹` is returned for eigenbasis-coordinate checks.
pub fn nonsymmetric_with_spectrum(
    eigenvalues: &[f64; N],
    q_seed: &[f64; 16],
    upper: &[f64; 6],
) -> Option<([f64; 16], Matrix4<f64>, Matrix4<f64>)> {
    let q = orthogonal_from(q_seed);
    let mut u = Matrix4::identity();
    let mut next = 0;
    for r in 0..N {
        for c in (r + 1)..N {
            u[(r, c)] = upper[next];
            next += 1;
        }
    }
    let p = q * u;
    let p_inv = p.try_inverse()?;
    let d = Matrix4::from_diagonal(&Vector4::from_row_slice(eigenvalues));
    let a = p * d * p_inv;
    Some((row_major(&a), p, p_inv))
}

/// Coordinate of `v` along eigenvector `index` in the eigenbasis `P` (via the dual basis, the
/// rows of `P⁻¹`), normalized by `‖v‖` — the non-orthogonal analogue of [`overlap`], which
/// governs whether the iteration can reach that eigenpair from `v`.
pub fn eigenbasis_coordinate(p_inv: &Matrix4<f64>, v: &[f64; N], index: usize) -> f64 {
    let coefficient = (p_inv * Vector4::from_row_slice(v))[index];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 { coefficient / norm } else { 0.0 }
}

/// Fixed 3x3 orthogonal Householder matrix `H = I - 2 ŵ ŵᵀ` with `w = [1, 2, 3]`: symmetric,
/// orthogonal, and with no zero entries, so it mixes every coordinate. Used by the curated
/// edge-case and stress tests, where matrices are hand-picked rather than generated.
fn fixed_orthogonal_3() -> [[f64; 3]; 3] {
    let w = [1.0, 2.0, 3.0];
    let w_dot_w = 14.0;
    let mut h = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let identity = if r == c { 1.0 } else { 0.0 };
            h[r][c] = identity - 2.0 * w[r] * w[c] / w_dot_w;
        }
    }
    h
}

/// Fixed similarity transform `A = H · diag(eigenvalues) · Hᵀ` (row-major 3x3) built from
/// [`fixed_orthogonal_3`]; the eigenvector for `eigenvalues[i]` is [`fixed_eigenvector_3`]`(i)`.
pub fn fixed_similarity_3(eigenvalues: [f64; 3]) -> [f64; 9] {
    let h = fixed_orthogonal_3();
    let mut a = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            a[r * 3 + c] = (0..3).map(|k| h[r][k] * eigenvalues[k] * h[c][k]).sum();
        }
    }
    a
}

/// Column `index` of the fixed Householder matrix: the eigenvector paired with
/// `eigenvalues[index]` in [`fixed_similarity_3`].
pub fn fixed_eigenvector_3(index: usize) -> [f64; 3] {
    let h = fixed_orthogonal_3();
    [h[0][index], h[1][index], h[2][index]]
}

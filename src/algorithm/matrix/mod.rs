mod arithmetic;
mod cholesky;
mod condition;
mod determinant;
mod lu;
mod qr;
mod rank;
mod svd;

use crate::scalar::Scalar;

pub use arithmetic::{add, mul_matrix, mul_scalar, mul_vector, sub, transpose};
pub use cholesky::{CholeskyError, cholesky, cholesky_decompose};
pub use condition::{ConditionNumberError, condition_number, condition_number_svd};
pub use determinant::{DeterminantError, determinant, determinant_cofactor, determinant_lu};
pub use lu::{lu, lu_partial_pivot};
pub use qr::{qr, qr_gram_schmidt, qr_householder};
pub use rank::{rank, rank_with_tolerance};
pub(in crate::algorithm::matrix) use svd::QR_ITERATIONS;
pub use svd::{svd, svd_qr_iteration};

/// The absolute value of `x`, for any `Scalar` with an ordering — `Scalar`'s arithmetic
/// surface is deliberately minimal and has no `abs` of its own, and every
/// largest-magnitude-pivot or tolerance comparison in this module (and the convergence
/// checks in [`crate::krylov`]) needs one.
pub(crate) fn abs<T: Scalar + PartialOrd>(x: T) -> T {
    if x < T::zero() { T::zero().sub(x) } else { x }
}

/// Converts `n` to `T` by repeated addition from `T::one()` — `Scalar` has no numeric
/// conversion of its own, and this is the only operation the auto-computed default
/// tolerances in this module need from one. `n` is always a matrix dimension here, so this
/// is `O(n)` against an already `O(n^2)`-or-larger algorithm.
pub(super) fn n_as_scalar<T: Scalar>(n: usize) -> T {
    let mut value = T::zero();
    for _ in 0..n {
        value = value.add(T::one());
    }
    value
}

/// Error returned by matrix operations in this module when operand dimensions don't agree.
///
/// `Storage` (see ADR 0003) only knows about a flat element count, not rows and columns —
/// matrices are stored row-major in a flat `Storage`, with their shape (`rows`, `cols`)
/// passed alongside each operand rather than queried from it. This error covers both that
/// shape disagreeing between operands, and a flat length not actually matching its claimed
/// `rows * cols` (the same role [`crate::algorithm::vector::LengthMismatch`] plays for
/// vectors, generalized to two dimensions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionMismatch;

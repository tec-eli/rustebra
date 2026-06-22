mod arithmetic;
mod cholesky;
mod condition;
mod determinant;
mod lu;
mod qr;
mod rank;
mod svd;

pub use arithmetic::{add, mul_matrix, mul_scalar, mul_vector, sub, transpose};
pub use cholesky::{CholeskyError, cholesky, cholesky_decompose};
pub use condition::{ConditionNumberError, condition_number, condition_number_svd};
pub use determinant::determinant;
pub use lu::{lu, lu_partial_pivot};
pub use qr::{qr, qr_gram_schmidt, qr_householder};
pub use rank::rank;
pub use svd::{svd, svd_qr_iteration};

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

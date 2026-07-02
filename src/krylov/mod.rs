mod inverse_power_iteration;
mod power_iteration;

pub use self::inverse_power_iteration::inverse_power_iteration;
pub use self::power_iteration::power_iteration;

/// Error returned by iterative methods in this module.
///
/// Unlike the direct decompositions in [`crate::algorithm::matrix`], the methods here refine
/// an estimate over many iterations, so alongside the usual shape disagreements they can fail
/// by simply not reaching the requested tolerance, or by an iterate degenerating to the zero
/// vector (which can't be normalized into a direction for the next step).
///
/// # Examples
///
/// ```
/// use rustebra::krylov::{ConvergenceError, power_iteration};
/// use rustebra::storage::StaticStorage;
///
/// let a = StaticStorage::new([2.0_f64, 0.0, 0.0, 1.0]);
/// // The zero vector has no direction to refine.
/// let v0 = StaticStorage::new([0.0_f64, 0.0]);
/// let mut eigenvector = [0.0; 2];
/// let mut scratch = [0.0; 2];
/// let result = power_iteration(&a, 2, &v0, 100, 1e-10, &mut eigenvector, &mut scratch);
/// assert_eq!(result, Err(ConvergenceError::ZeroVector));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceError {
    /// Operand dimensions don't agree with each other or with their claimed shape.
    DimensionMismatch,
    /// The initial vector, or an iterate produced along the way, has zero norm and therefore
    /// no direction the iteration can continue from.
    ZeroVector,
    /// The convergence criteria were not met within the requested iteration budget.
    MaxIterationsExceeded,
    /// The shifted matrix `a - shift * I` is singular, or within the caller's singularity
    /// tolerance of it: the shift (numerically) coincides with an eigenvalue, so the linear
    /// system inverse iteration must solve each step has no reliable solution. Reported as a
    /// hard error rather than iterating on amplified noise; move the shift slightly and retry.
    SingularShift,
}

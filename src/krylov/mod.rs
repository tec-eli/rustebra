mod arnoldi;
mod conjugate_gradient;
mod hessenberg;
mod inverse_power_iteration;
mod lanczos;
mod power_iteration;
mod tridiagonal;

pub use self::arnoldi::arnoldi;
pub use self::conjugate_gradient::conjugate_gradient;
pub use self::hessenberg::HessenbergMatrix;
pub use self::inverse_power_iteration::inverse_power_iteration;
pub use self::lanczos::lanczos;
pub use self::power_iteration::power_iteration;
pub use self::tridiagonal::TridiagonalMatrix;

/// Error returned by iterative methods in this module.
///
/// Unlike the direct decompositions in [`crate::algorithm::matrix`], the methods here refine
/// an estimate over many iterations, so alongside the usual shape disagreements they can fail
/// by simply not reaching the requested tolerance, by an iterate degenerating to the zero
/// vector (which can't be normalized into a direction for the next step), or by an iterate
/// going non-finite (`NaN`/`Inf`).
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
    /// An iterate produced along the way contains a `NaN` or infinite value.
    ///
    /// This crate otherwise lets `NaN`/`Inf` propagate through arithmetic rather than
    /// checking for it. Krylov methods are the exception: because they loop, a poisoned
    /// iterate doesn't just produce one bad result, it burns the entire remaining iteration
    /// budget computing on garbage — a real cost on embedded targets, where that budget is
    /// bounded and can't be spent elsewhere. Detecting the condition once and stopping is
    /// cheaper than paying for `max_iter` more rounds of `NaN` arithmetic.
    NonFinite,
    /// The convergence criteria were not met within the requested iteration budget.
    MaxIterationsExceeded,
    /// A basis-building iteration (Lanczos) found the Krylov subspace to be invariant before
    /// reaching the requested dimension `K`: the candidate for the next basis vector fell to
    /// zero norm, within the caller's tolerance, so no new direction exists to extend the
    /// basis with. Distinct from [`ConvergenceError::ZeroVector`] (a degenerate input or
    /// iterate) because it is a property of the pairing of matrix and starting vector — a
    /// repeated eigenvalue, or a `v0` inside a small invariant subspace — and the remedy is
    /// different: retry with a smaller `K` or a different starting vector.
    Breakdown,
    /// The shifted matrix `a - shift * I` is singular, or within the caller's singularity
    /// tolerance of it: the shift (numerically) coincides with an eigenvalue, so the linear
    /// system inverse iteration must solve each step has no reliable solution. Reported as a
    /// hard error rather than iterating on amplified noise; move the shift slightly and retry.
    SingularShift,
}

#[cfg(feature = "alloc")]
mod coo;
#[cfg(feature = "alloc")]
pub use self::coo::{CooError, CooMatrix};

#[cfg(feature = "alloc")]
mod csr;
#[cfg(feature = "alloc")]
pub use self::csr::{CsrError, CsrMatrix};

#[cfg(feature = "alloc")]
mod csc;
#[cfg(feature = "alloc")]
pub use self::csc::{CscError, CscMatrix};

#[cfg(feature = "alloc")]
mod sorted_csr;
#[cfg(feature = "alloc")]
pub use self::sorted_csr::SortedCsrMatrix;

#[cfg(feature = "alloc")]
mod linear_op;
#[cfg(feature = "alloc")]
pub use self::linear_op::SparseLinearOp;

#[cfg(feature = "alloc")]
mod convert;
#[cfg(feature = "alloc")]
pub use self::convert::{coo_to_csr, csc_to_csr, csr_to_coo, csr_to_csc};

#[cfg(feature = "alloc")]
mod scale;
#[cfg(feature = "alloc")]
pub use self::scale::{scale_csc, scale_csr};

#[cfg(feature = "alloc")]
mod matvec;
#[cfg(feature = "alloc")]
pub use self::matvec::{matvec_csc, matvec_csr};

#[cfg(feature = "alloc")]
mod add;
#[cfg(feature = "alloc")]
pub use self::add::{DimensionMismatch, add_csc, add_csr};

#[cfg(feature = "alloc")]
mod matmat;
#[cfg(feature = "alloc")]
pub use self::matmat::{matmat_csc, matmat_csr};

#[cfg(feature = "alloc")]
mod spmm;
#[cfg(feature = "alloc")]
pub use self::spmm::spmm_csr;

#[cfg(feature = "alloc")]
mod prune;
#[cfg(feature = "alloc")]
pub use self::prune::{prune_csc, prune_csr};

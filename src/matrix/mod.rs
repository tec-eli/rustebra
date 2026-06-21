mod static_matrix;
pub use self::static_matrix::StaticMatrix;

#[cfg(feature = "alloc")]
mod dynamic_matrix;
#[cfg(feature = "alloc")]
pub use self::dynamic_matrix::DynamicMatrix;

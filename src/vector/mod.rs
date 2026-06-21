mod static_vector;
pub use self::static_vector::StaticVector;

#[cfg(feature = "alloc")]
mod dynamic_vector;
#[cfg(feature = "alloc")]
pub use self::dynamic_vector::DynamicVector;

#![cfg_attr(not(test), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod algorithm;
pub mod matrix;
pub mod scalar;
pub mod sparse;
pub mod storage;
pub mod vector;

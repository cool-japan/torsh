//! Core types and traits for the `ToRSh` deep learning framework
//!
//! This crate provides fundamental building blocks used throughout `ToRSh`,
//! including error types, device abstractions, and core traits.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod device;
pub mod dtype;
pub mod error;
pub mod shape;
pub mod storage;

// Re-export commonly used items
pub use device::{Device, DeviceType};
pub use dtype::{DType, TensorElement};
pub use error::{Result, TorshError};
pub use shape::{Shape, Stride};

// Re-export scirs2 for use by other crates
pub use scirs2;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::device::{Device, DeviceType};
    pub use crate::dtype::{DType, TensorElement};
    pub use crate::error::{Result, TorshError};
    pub use crate::shape::Shape;
}
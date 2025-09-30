//! Index-related error types for ToRSh
//!
//! This module contains error variants for index bounds checking,
//! indexing operations, and array access violations.

use crate::error::core::ErrorLocation;
use thiserror::Error;

/// Index-related error variants
#[derive(Error, Debug, Clone)]
pub enum IndexError {
    /// Index out of bounds
    #[error("Index out of bounds: index {index} is out of bounds for dimension with size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    /// Index error
    #[error("Index error: index {index} is out of bounds for size {size}")]
    IndexError { index: usize, size: usize },

    /// Index error with location
    #[error("Index error at {location}: index {index} is out of bounds for size {size}")]
    IndexErrorWithLocation {
        index: usize,
        size: usize,
        location: ErrorLocation,
    },
}

impl IndexError {
    pub fn out_of_bounds(index: usize, size: usize) -> Self {
        Self::IndexOutOfBounds { index, size }
    }

    pub fn category(&self) -> crate::error::core::ErrorCategory {
        crate::error::core::ErrorCategory::UserInput
    }
}

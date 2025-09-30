//! Common types and utilities for loss functions
//!
//! This module provides shared types and helper functions used across
//! different loss function categories.

use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// Reduction type for loss functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionType {
    /// No reduction applied
    None,
    /// Mean reduction
    Mean,
    /// Sum reduction
    Sum,
}

impl ReductionType {
    /// Apply the reduction to a tensor
    pub fn apply(&self, tensor: Tensor) -> TorshResult<Tensor> {
        match self {
            Self::None => Ok(tensor),
            Self::Mean => tensor.mean(None, false),
            Self::Sum => tensor.sum(),
        }
    }
}

//! Canonical reduction mode for loss functions.
//!
//! This enum is the single source of truth for reduction semantics
//! across all ToRSh crates. All loss functions should use this type
//! instead of string literals or crate-local enums.

use core::str::FromStr;

use crate::{Result, TorshError};

/// Specifies how to reduce per-element losses into a scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reduction {
    /// No reduction - return per-element losses.
    None,
    /// Mean over all elements (sum / numel). Matches PyTorch `"mean"`.
    Mean,
    /// Sum of all elements.
    Sum,
    /// Mean over batch dimension only (sum / batch_size).
    /// Equivalent to PyTorch `"batchmean"`.
    BatchMean,
}

impl FromStr for Reduction {
    type Err = TorshError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "mean" => Ok(Self::Mean),
            "sum" => Ok(Self::Sum),
            "batchmean" => Ok(Self::BatchMean),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown reduction mode: {}, try one of those: none, mean, sum, batchmean",
                s
            ))),
        }
    }
}

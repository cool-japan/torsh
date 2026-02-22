//! # StatisticalError - Trait Implementations
//!
//! This module contains trait implementations for `StatisticalError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::error::Error;

use super::types::StatisticalError;

impl fmt::Display for StatisticalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatisticalError::InsufficientData(msg) => {
                write!(f, "Insufficient data: {}", msg)
            }
            StatisticalError::InvalidParameters(msg) => {
                write!(f, "Invalid parameters: {}", msg)
            }
            StatisticalError::ConvergenceFailure(msg) => {
                write!(f, "Convergence failure: {}", msg)
            }
            StatisticalError::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {}", msg)
            }
            StatisticalError::DistributionMismatch(msg) => {
                write!(f, "Distribution mismatch: {}", msg)
            }
        }
    }
}

impl Error for StatisticalError {}


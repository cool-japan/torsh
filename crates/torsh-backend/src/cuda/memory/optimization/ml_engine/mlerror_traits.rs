//! # MLError - Trait Implementations
//!
//! This module contains trait implementations for `MLError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{MLError, Prediction};

impl std::fmt::Display for MLError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MLError::ModelNotFound(id) => write!(f, "Model not found: {}", id),
            MLError::ModelNotTrained(id) => write!(f, "Model not trained: {}", id),
            MLError::InsufficientData => write!(f, "Insufficient training data"),
            MLError::TrainingFailed(msg) => write!(f, "Training failed: {}", msg),
            MLError::PredictionFailed(msg) => write!(f, "Prediction failed: {}", msg),
            MLError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            MLError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            MLError::ResourceExhausted => write!(f, "Resource exhausted"),
        }
    }
}

impl std::error::Error for MLError {}


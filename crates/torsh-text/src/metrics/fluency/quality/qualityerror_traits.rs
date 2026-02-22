//! # QualityError - Trait Implementations
//!
//! This module contains trait implementations for `QualityError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::error::Error;

use super::types::QualityError;

impl fmt::Display for QualityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QualityError::AnalysisFailure(msg) => {
                write!(f, "Quality analysis failure: {}", msg)
            }
            QualityError::ValidationFailure(msg) => {
                write!(f, "Quality validation failure: {}", msg)
            }
            QualityError::BenchmarkingFailure(msg) => {
                write!(f, "Quality benchmarking failure: {}", msg)
            }
            QualityError::ReportingFailure(msg) => {
                write!(f, "Quality reporting failure: {}", msg)
            }
            QualityError::ConfigurationError(msg) => {
                write!(f, "Quality configuration error: {}", msg)
            }
        }
    }
}

impl Error for QualityError {}


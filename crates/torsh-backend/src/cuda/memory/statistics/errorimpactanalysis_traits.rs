//! # ErrorImpactAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `ErrorImpactAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ErrorImpactAnalysis, ErrorImpactLevel};

impl Default for ErrorImpactAnalysis {
    fn default() -> Self {
        Self {
            performance_impact: 0.0,
            user_impact: ErrorImpactLevel::Minimal,
            resource_waste: 0.0,
            recovery_cost: 0.0,
        }
    }
}

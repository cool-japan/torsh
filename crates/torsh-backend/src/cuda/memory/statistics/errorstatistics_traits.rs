//! # ErrorStatistics - Trait Implementations
//!
//! This module contains trait implementations for `ErrorStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::{ErrorImpactAnalysis, ErrorStatistics};

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            error_rate: 0.0,
            error_types: HashMap::new(),
            error_trends: Vec::new(),
            impact_analysis: ErrorImpactAnalysis::default(),
            state_correlations: Vec::new(),
        }
    }
}

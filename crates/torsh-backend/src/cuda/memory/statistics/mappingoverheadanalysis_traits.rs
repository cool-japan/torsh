//! # MappingOverheadAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `MappingOverheadAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::MappingOverheadAnalysis;

impl Default for MappingOverheadAnalysis {
    fn default() -> Self {
        Self {
            setup_overhead: Duration::from_secs(0),
            memory_overhead: 0,
            performance_overhead: 0.0,
            overall_overhead: 0.0,
        }
    }
}

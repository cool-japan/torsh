//! # CrossTypeCorrelations - Trait Implementations
//!
//! This module contains trait implementations for `CrossTypeCorrelations`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CorrelationSignificance, CrossTypeCorrelations};

impl Default for CrossTypeCorrelations {
    fn default() -> Self {
        Self {
            device_unified_correlation: 0.0,
            device_pinned_correlation: 0.0,
            unified_pinned_correlation: 0.0,
            significance_levels: CorrelationSignificance::default(),
        }
    }
}

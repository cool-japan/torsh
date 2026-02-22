//! # CorrelationSignificance - Trait Implementations
//!
//! This module contains trait implementations for `CorrelationSignificance`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CorrelationSignificance;

impl Default for CorrelationSignificance {
    fn default() -> Self {
        Self {
            device_unified: 0.0,
            device_pinned: 0.0,
            unified_pinned: 0.0,
        }
    }
}

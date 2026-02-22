//! # RetentionPolicy - Trait Implementations
//!
//! This module contains trait implementations for `RetentionPolicy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::RetentionPolicy;

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_retention: Duration::from_secs(30 * 24 * 3600),
            compress_after: Duration::from_secs(7 * 24 * 3600),
            summarize_after: Duration::from_secs(1 * 24 * 3600),
            cleanup_frequency: Duration::from_secs(3600),
        }
    }
}

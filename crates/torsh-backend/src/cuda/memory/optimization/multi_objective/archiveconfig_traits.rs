//! # ArchiveConfig - Trait Implementations
//!
//! This module contains trait implementations for `ArchiveConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ArchiveConfig, ArchiveUpdateStrategy, DuplicateHandling};

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            max_size: 200,
            update_strategy: ArchiveUpdateStrategy::CrowdingBased,
            duplicate_handling: DuplicateHandling::Reject,
            quality_threshold: 0.5,
            aging_strategy: None,
        }
    }
}


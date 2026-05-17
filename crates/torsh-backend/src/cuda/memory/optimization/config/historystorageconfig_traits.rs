//! # HistoryStorageConfig - Trait Implementations
//!
//! This module contains trait implementations for `HistoryStorageConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::HistoryStorageConfig;

impl Default for HistoryStorageConfig {
    fn default() -> Self {
        Self {
            max_evolution_points: 10_000,
            max_performance_records: 100_000,
            max_configuration_changes: 50_000,
        }
    }
}

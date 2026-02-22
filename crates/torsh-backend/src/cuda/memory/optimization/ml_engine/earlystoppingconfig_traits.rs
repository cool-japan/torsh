//! # EarlyStoppingConfig - Trait Implementations
//!
//! This module contains trait implementations for `EarlyStoppingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{EarlyStoppingConfig, EarlyStoppingMode};

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 10,
            min_delta: 0.001,
            monitor_metric: "validation_loss".to_string(),
            mode: EarlyStoppingMode::Minimize,
        }
    }
}


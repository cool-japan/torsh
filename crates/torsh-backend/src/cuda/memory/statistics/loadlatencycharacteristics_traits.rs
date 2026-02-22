//! # LoadLatencyCharacteristics - Trait Implementations
//!
//! This module contains trait implementations for `LoadLatencyCharacteristics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::LoadLatencyCharacteristics;

impl Default for LoadLatencyCharacteristics {
    fn default() -> Self {
        Self {
            low_load: Duration::from_secs(0),
            medium_load: Duration::from_secs(0),
            high_load: Duration::from_secs(0),
            load_sensitivity: 0.0,
        }
    }
}

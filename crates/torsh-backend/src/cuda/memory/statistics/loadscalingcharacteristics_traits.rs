//! # LoadScalingCharacteristics - Trait Implementations
//!
//! This module contains trait implementations for `LoadScalingCharacteristics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{LoadScalingCharacteristics, ScalingPattern};

impl Default for LoadScalingCharacteristics {
    fn default() -> Self {
        Self {
            scaling_pattern: ScalingPattern::Linear,
            degradation_threshold: 0.8,
            max_sustainable_load: 1.0,
            recovery_time: Duration::from_secs(10),
        }
    }
}

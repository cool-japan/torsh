//! # TransferOptimizationMetrics - Trait Implementations
//!
//! This module contains trait implementations for `TransferOptimizationMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use super::types::TransferOptimizationMetrics;

impl Default for TransferOptimizationMetrics {
    fn default() -> Self {
        Self {
            success_rate: 0.0,
            average_improvement: 0.0,
            cumulative_bandwidth_improvement: 0.0,
            latency_reduction: Duration::from_secs(0),
            improvements_by_type: HashMap::new(),
        }
    }
}

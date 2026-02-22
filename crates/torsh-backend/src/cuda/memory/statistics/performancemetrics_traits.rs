//! # PerformanceMetrics - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    AllocationPerformanceMetrics, EfficiencyMetrics, PerformanceMetrics, SystemPerformanceMetrics,
    TransferPerformanceMetrics,
};

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            allocation_performance: AllocationPerformanceMetrics::default(),
            transfer_performance: TransferPerformanceMetrics::default(),
            system_performance: SystemPerformanceMetrics::default(),
            efficiency_metrics: EfficiencyMetrics::default(),
        }
    }
}

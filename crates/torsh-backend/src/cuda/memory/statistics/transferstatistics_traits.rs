//! # TransferStatistics - Trait Implementations
//!
//! This module contains trait implementations for `TransferStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    TransferEfficiencyAnalysis, TransferPerformanceMetrics, TransferStatistics, VolumeStatistics,
};

impl Default for TransferStatistics {
    fn default() -> Self {
        Self {
            total_transfers: 0,
            volume_stats: VolumeStatistics::default(),
            performance_metrics: TransferPerformanceMetrics::default(),
            efficiency_analysis: TransferEfficiencyAnalysis::default(),
            patterns: Vec::new(),
        }
    }
}

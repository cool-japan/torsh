//! # PerformanceCharacteristics - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceCharacteristics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    ConsistencyMetrics, LatencyStatistics, LoadPerformanceMetrics, PerformanceCharacteristics,
    ThroughputMetrics,
};

impl Default for PerformanceCharacteristics {
    fn default() -> Self {
        Self {
            allocation_latency: LatencyStatistics::default(),
            deallocation_latency: LatencyStatistics::default(),
            throughput_metrics: ThroughputMetrics::default(),
            consistency_metrics: ConsistencyMetrics::default(),
            load_performance: LoadPerformanceMetrics::default(),
        }
    }
}

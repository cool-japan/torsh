//! # SystemPerformanceMetrics - Trait Implementations
//!
//! This module contains trait implementations for `SystemPerformanceMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    BottleneckAnalysis, ScalabilityMetrics, SystemLatencyCharacteristics, SystemPerformanceMetrics,
};

impl Default for SystemPerformanceMetrics {
    fn default() -> Self {
        Self {
            overall_throughput: 0.0,
            latency_characteristics: SystemLatencyCharacteristics::default(),
            resource_utilization: 0.0,
            scalability_metrics: ScalabilityMetrics::default(),
            bottleneck_analysis: BottleneckAnalysis::default(),
        }
    }
}

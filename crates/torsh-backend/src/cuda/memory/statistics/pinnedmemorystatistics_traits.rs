//! # PinnedMemoryStatistics - Trait Implementations
//!
//! This module contains trait implementations for `PinnedMemoryStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::allocation::AllocationStats;

use super::types::{
    CacheEffectiveness, MappingStatistics, PinnedMemoryStatistics, TransferOptimizationMetrics,
    TransferStatistics, UsagePatternAnalysis,
};

impl Default for PinnedMemoryStatistics {
    fn default() -> Self {
        Self {
            allocation_stats: AllocationStats::default(),
            transfer_stats: TransferStatistics::default(),
            mapping_stats: MappingStatistics::default(),
            cache_effectiveness: CacheEffectiveness::default(),
            optimization_metrics: TransferOptimizationMetrics::default(),
            usage_patterns: UsagePatternAnalysis::default(),
        }
    }
}

//! # UnifiedMemoryStatistics - Trait Implementations
//!
//! This module contains trait implementations for `UnifiedMemoryStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::allocation::AllocationStats;
use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::{
    AccessPatternAnalysis, AdviceImpactAnalysis, MigrationStatistics, OptimizationMetrics,
    PrefetchEffectiveness, UnifiedMemoryStatistics,
};

impl Default for UnifiedMemoryStatistics {
    fn default() -> Self {
        Self {
            allocation_stats: AllocationStats::default(),
            migration_stats: MigrationStatistics::default(),
            access_patterns: AccessPatternAnalysis::default(),
            prefetch_effectiveness: PrefetchEffectiveness::default(),
            advice_impact: AdviceImpactAnalysis::default(),
            optimization_metrics: OptimizationMetrics::default(),
            device_usage_distribution: HashMap::new(),
        }
    }
}

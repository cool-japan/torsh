//! # MultiObjectiveMetrics - Trait Implementations
//!
//! This module contains trait implementations for `MultiObjectiveMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CoverageMetrics, MultiObjectiveMetrics, PerformanceStats, QualityIndicators};

impl Default for MultiObjectiveMetrics {
    fn default() -> Self {
        Self {
            hypervolume: 0.0,
            spacing: 0.0,
            convergence: 0.0,
            diversity: 0.0,
            solution_count: 0,
            generations: 0,
            generational_distance: 0.0,
            inverted_generational_distance: 0.0,
            coverage_metrics: CoverageMetrics::default(),
            performance_stats: PerformanceStats::default(),
            quality_indicators: QualityIndicators::default(),
        }
    }
}


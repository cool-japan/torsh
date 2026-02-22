//! # SemanticMetricsConfig - Trait Implementations
//!
//! This module contains trait implementations for `SemanticMetricsConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SemanticMetricsConfig;

impl Default for SemanticMetricsConfig {
    fn default() -> Self {
        Self {
            enable_statistical_analysis: true,
            enable_distribution_analysis: true,
            enable_correlation_analysis: true,
            enable_trend_analysis: false,
            enable_performance_analysis: true,
            confidence_level: 0.95,
            max_clusters: 10,
            outlier_threshold: 2.0,
            min_sample_size: 30,
            enable_quality_assessment: true,
            historical_retention_days: 30,
        }
    }
}


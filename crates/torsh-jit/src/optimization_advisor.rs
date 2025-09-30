//! Intelligent optimization advisor with automated recommendations and analysis
//!
//! This module provides an AI-driven optimization advisor that analyzes
//! computation graphs, execution patterns, and performance characteristics
//! to provide intelligent optimization recommendations.
//!
//! # Architecture
//!
//! The optimization advisor is now built using a modular architecture:
//! - `advisor::config` - Configuration and core types
//! - `advisor::core` - Main advisor orchestration
//! - `advisor::patterns` - Pattern detection and analysis
//! - `advisor::performance` - Performance bottleneck analysis
//! - `advisor::cost` - Cost-benefit modeling
//! - `advisor::recommendations` - Recommendation generation
//! - `advisor::knowledge` - Learning and knowledge systems
//! - `advisor::utils` - Utility functions
//!
//! # Examples
//!
//! ```rust
//! use torsh_jit::optimization_advisor::{OptimizationAdvisor, AdvisorConfig, AnalysisInput};
//! use torsh_jit::{ComputationGraph, SystemConstraints, UserPreferences};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut advisor = OptimizationAdvisor::new(AdvisorConfig::default());
//! let graph = ComputationGraph::new();
//! let input = AnalysisInput {
//!     computation_graph: Some(graph),
//!     system_constraints: SystemConstraints::default(),
//!     user_preferences: UserPreferences::default(),
//!     benchmark_results: None,
//!     profiling_data: None,
//!     previous_optimizations: Vec::new(),
//!     abstract_analysis: None,
//!     symbolic_execution: None,
//! };
//!
//! let report = advisor.analyze_and_recommend(input)?;
//! println!("Found {} optimization opportunities", report.recommendations.len());
//! # Ok(())
//! # }
//! ```

// Re-export the entire advisor module for backward compatibility
pub use crate::advisor::*;

// Re-export the main advisor type specifically for clarity
pub use crate::advisor::OptimizationAdvisor;

// Convenience re-exports for common types
pub use crate::advisor::{
    AdvisorConfig, AnalysisInput, CostBenefitAnalysis, OptimizationRecommendation,
    OptimizationReport, OptimizationType, PatternAnalysis, PerformanceAnalysis, SystemConstraints,
    TargetPlatform, UserPreferences,
};

// Convenience functions for common workflows
pub use crate::advisor::{
    analyze_computation_graph, analyze_with_benchmarks, analyze_with_profiling, create_advisor,
    create_advisor_with_config, create_fast_config, create_production_config,
    create_thorough_config, quick_analyze,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advisor_creation() {
        let advisor = create_advisor();
        assert_eq!(advisor.get_version(), "1.0.0");
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that we can still create an advisor using the old interface
        let config = AdvisorConfig::default();
        let advisor = OptimizationAdvisor::new(config);
        assert_eq!(advisor.get_version(), "1.0.0");
    }

    #[test]
    fn test_configuration_options() {
        let fast_config = create_fast_config();
        let thorough_config = create_thorough_config();
        let production_config = create_production_config();

        assert_ne!(fast_config.analysis_depth, thorough_config.analysis_depth);
        assert_ne!(
            fast_config.max_recommendations,
            thorough_config.max_recommendations
        );
        assert_eq!(production_config.max_recommendations, 10);
    }

    #[test]
    fn test_minimal_input() {
        let input = create_minimal_analysis_input();
        assert!(input.computation_graph.is_none());
        assert!(input.benchmark_results.is_none());
        assert!(input.profiling_data.is_none());
    }

    #[test]
    fn test_advisor_info() {
        let info = get_advisor_info();
        assert_eq!(info.version, "1.0.0");
        assert!(!info.supported_optimizations.is_empty());
        assert!(!info.supported_patterns.is_empty());
        assert!(!info.supported_platforms.is_empty());
    }
}

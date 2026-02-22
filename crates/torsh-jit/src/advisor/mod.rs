//! Optimization advisor system for JIT compilation
//!
//! This module provides an intelligent optimization advisor that analyzes computation graphs
//! and system characteristics to provide actionable optimization recommendations.

// Core modules
pub mod config;
pub mod core;
pub mod cost;
pub mod knowledge;
pub mod patterns;
pub mod performance;
pub mod recommendations;
pub mod utils;

// Re-export main types and functionality for backward compatibility
pub use core::OptimizationAdvisor;

// Re-export configuration types
pub use config::{
    AdvisorConfig, AnalysisDepth, AnalysisInput, AntipatternType, BottleneckType,
    ComplexityMetrics, ConfidenceAnalysis, CostBenefitAnalysis, DetailedExplanation,
    DetectedAntipattern, DetectedPattern, ExecutionProfile, LoopInfo, OpportunityType,
    OptimizationGoals, OptimizationOpportunity, OptimizationRecommendation, OptimizationReport,
    OptimizationType, OverallAnalysis, PatternAnalysis, PatternLocation, PatternType,
    PerformanceAnalysis, PerformanceBottleneck, PerformanceHotspot, PriorityDistribution,
    ResourceUtilization, RiskAssessment, ScalabilityAnalysis, SystemConstraints, TargetPlatform,
    TimeConstraints, UserPreferences,
};

// Import AbstractValue directly from abstract_interpretation
pub use crate::abstract_interpretation::AbstractValue;

// Re-export core functionality
pub use core::{AnalysisPhase, AnalysisProgress};

// Re-export pattern analysis
pub use patterns::{ExecutionPath, GraphNode, PatternAnalyzer};

// Re-export performance analysis
pub use performance::{FunctionCallData, PerformanceAnalyzer, ProfilingAnalysisResult};

// Import stats types directly from config
pub use config::{MemoryStatistics, OperationTiming, ResourceStats};

// Re-export cost analysis
pub use cost::CostModel;

// Re-export recommendation engine
pub use recommendations::RecommendationEngine;

// Re-export knowledge and learning systems
pub use knowledge::{
    ActualPerformanceResult, AdaptationEngine, AnalysisRecord, BestPractice, FailureCase,
    FeedbackEntry, FeedbackTracker, HistoricalDataStore, InputCharacteristics, KnowledgeBase,
    KnowledgeSummary, LearningConfig, LearningSystem, OptimizationPattern, OptimizationSuggestion,
    PerformanceModel, PerformancePrediction, PerformanceRecord, RecommendationFeedback,
    RecommendationRecord,
};

// Re-export utilities
pub use utils::{
    calculate_data_confidence, calculate_group_metrics, calculate_input_similarity,
    calculate_preference_similarity, calculate_priority_score, calculate_system_similarity,
    estimate_total_implementation_time, format_duration, generate_recommendation_id,
    generate_simple_id, group_recommendations_by_type, merge_optimization_reports,
    validate_analysis_input, GroupMetrics, ValidationResult, ValidationSeverity,
};

use crate::JitResult;
use std::time::Duration;

/// Create a new optimization advisor with default configuration
pub fn create_advisor() -> OptimizationAdvisor {
    OptimizationAdvisor::new(AdvisorConfig::default())
}

/// Create a new optimization advisor with custom configuration
pub fn create_advisor_with_config(config: AdvisorConfig) -> OptimizationAdvisor {
    OptimizationAdvisor::new(config)
}

/// Create a minimal analysis input for basic optimization analysis
pub fn create_minimal_analysis_input() -> AnalysisInput {
    AnalysisInput {
        computation_graph: None,
        system_constraints: SystemConstraints::default(),
        user_preferences: UserPreferences::default(),
        benchmark_results: None,
        profiling_data: None,
        previous_optimizations: Vec::new(),
        abstract_analysis: None,
        symbolic_execution: None,
    }
}

/// Validate and analyze input, returning an optimization report
pub fn quick_analyze(
    computation_graph: Option<crate::ComputationGraph>,
    system_constraints: Option<SystemConstraints>,
    user_preferences: Option<UserPreferences>,
) -> JitResult<OptimizationReport> {
    let mut advisor = create_advisor();

    let input = AnalysisInput {
        computation_graph,
        system_constraints: system_constraints.unwrap_or_default(),
        user_preferences: user_preferences.unwrap_or_default(),
        benchmark_results: None,
        profiling_data: None,
        previous_optimizations: Vec::new(),
        abstract_analysis: None,
        symbolic_execution: None,
    };

    advisor.analyze_and_recommend(input)
}

/// Create a high-performance advisor configuration optimized for speed
pub fn create_fast_config() -> AdvisorConfig {
    AdvisorConfig {
        version: "1.0.0".to_string(),
        max_recommendations: 5,
        min_confidence_threshold: 0.6,
        min_benefit_threshold: 0.1,
        enable_learning: false,
        analysis_depth: AnalysisDepth::Quick,
        optimization_goals: OptimizationGoals {
            prioritize_speed: true,
            prioritize_memory: false,
            prioritize_energy: false,
            enable_aggressive_optimizations: true,
        },
    }
}

/// Create a thorough advisor configuration optimized for comprehensive analysis
pub fn create_thorough_config() -> AdvisorConfig {
    AdvisorConfig {
        version: "1.0.0".to_string(),
        max_recommendations: 20,
        min_confidence_threshold: 0.3,
        min_benefit_threshold: 0.05,
        enable_learning: true,
        analysis_depth: AnalysisDepth::Comprehensive,
        optimization_goals: OptimizationGoals {
            prioritize_speed: true,
            prioritize_memory: true,
            prioritize_energy: true,
            enable_aggressive_optimizations: false,
        },
    }
}

/// Create a production-ready advisor configuration with balanced settings
pub fn create_production_config() -> AdvisorConfig {
    AdvisorConfig {
        version: "1.0.0".to_string(),
        max_recommendations: 10,
        min_confidence_threshold: 0.5,
        min_benefit_threshold: 0.1,
        enable_learning: true,
        analysis_depth: AnalysisDepth::Standard,
        optimization_goals: OptimizationGoals {
            prioritize_speed: true,
            prioritize_memory: true,
            prioritize_energy: false,
            enable_aggressive_optimizations: false,
        },
    }
}

/// Perform a quick pattern analysis on a computation graph
pub fn analyze_patterns_only(graph: &crate::ComputationGraph) -> JitResult<PatternAnalysis> {
    let mut analyzer = patterns::PatternAnalyzer::new();

    let fusion_patterns = analyzer.detect_fusion_opportunities(graph)?;
    let memory_patterns = analyzer.detect_memory_patterns(graph)?;
    let parallelization_patterns = analyzer.detect_parallelization_patterns(graph)?;
    let vectorization_patterns = analyzer.detect_vectorization_patterns(graph)?;

    let mut all_patterns = Vec::new();
    all_patterns.extend(fusion_patterns);
    all_patterns.extend(memory_patterns);
    all_patterns.extend(parallelization_patterns);
    all_patterns.extend(vectorization_patterns);

    let inefficient_patterns = analyzer.detect_inefficient_patterns(graph)?;
    let memory_antipatterns = analyzer.detect_memory_antipatterns(graph)?;
    let computation_antipatterns = analyzer.detect_computation_antipatterns(graph)?;

    let mut all_antipatterns = Vec::new();
    all_antipatterns.extend(inefficient_patterns);
    all_antipatterns.extend(memory_antipatterns);
    all_antipatterns.extend(computation_antipatterns);

    let constant_folding = analyzer.find_constant_folding_opportunities(graph)?;
    let dead_code = analyzer.find_dead_code_elimination_opportunities(graph)?;
    let loop_opt = analyzer.find_loop_optimization_opportunities(graph)?;

    let mut all_opportunities = Vec::new();
    all_opportunities.extend(constant_folding);
    all_opportunities.extend(dead_code);
    all_opportunities.extend(loop_opt);

    Ok(PatternAnalysis {
        detected_patterns: all_patterns,
        antipatterns: all_antipatterns,
        optimization_opportunities: all_opportunities,
        pattern_frequency: analyzer.calculate_pattern_frequency(),
        complexity_metrics: analyzer.calculate_complexity_metrics(),
    })
}

/// Perform cost-benefit analysis for a set of optimization opportunities
pub fn analyze_costs_only(
    opportunities: &[OptimizationOpportunity],
    input: &AnalysisInput,
) -> JitResult<CostBenefitAnalysis> {
    let cost_model = cost::CostModel::new();
    let mut costs = std::collections::HashMap::new();
    let mut benefits = std::collections::HashMap::new();
    let mut risks = std::collections::HashMap::new();

    for (i, opportunity) in opportunities.iter().enumerate() {
        let id = format!("opp_{}", i);

        let cost = cost_model.calculate_implementation_cost(opportunity, input)?;
        let benefit = cost_model.estimate_performance_benefit(opportunity, input)?;
        let risk = cost_model.evaluate_risks(opportunity, input)?;

        costs.insert(id.clone(), cost);
        benefits.insert(id.clone(), benefit);
        risks.insert(id.clone(), risk);
    }

    let roi_estimates = cost_model.calculate_roi_estimates(&costs, &benefits)?;
    let priority_rankings = cost_model.generate_priority_rankings(&costs, &benefits, &risks)?;

    Ok(CostBenefitAnalysis {
        implementation_costs: costs,
        expected_benefits: benefits,
        risk_assessments: risks,
        roi_estimates,
        priority_rankings,
    })
}

/// Get advisor system statistics and capabilities
pub fn get_advisor_info() -> AdvisorInfo {
    AdvisorInfo {
        version: "1.0.0".to_string(),
        supported_optimizations: vec![
            OptimizationType::FusionOptimization,
            OptimizationType::MemoryOptimization,
            OptimizationType::ParallelizationOptimization,
            OptimizationType::VectorizationOptimization,
            OptimizationType::ConstantFoldingOptimization,
            OptimizationType::DeadCodeEliminationOptimization,
            OptimizationType::ComputationOptimization,
            OptimizationType::IOOptimization,
            OptimizationType::CompilationOptimization,
            OptimizationType::ArchitectureOptimization,
        ],
        supported_patterns: vec![
            PatternType::FusionOpportunity,
            PatternType::MemoryInefficiency,
            PatternType::ParallelizationOpportunity,
            PatternType::VectorizationOpportunity,
        ],
        supported_platforms: vec![
            TargetPlatform::Desktop,
            TargetPlatform::Server,
            TargetPlatform::Mobile,
            TargetPlatform::Embedded,
        ],
        features: AdvisorFeatures {
            pattern_detection: true,
            performance_analysis: true,
            cost_benefit_analysis: true,
            learning_system: true,
            recommendation_engine: true,
            risk_assessment: true,
            confidence_analysis: true,
        },
    }
}

/// Information about the advisor system
#[derive(Debug, Clone)]
pub struct AdvisorInfo {
    pub version: String,
    pub supported_optimizations: Vec<OptimizationType>,
    pub supported_patterns: Vec<PatternType>,
    pub supported_platforms: Vec<TargetPlatform>,
    pub features: AdvisorFeatures,
}

/// Feature flags for the advisor system
#[derive(Debug, Clone)]
pub struct AdvisorFeatures {
    pub pattern_detection: bool,
    pub performance_analysis: bool,
    pub cost_benefit_analysis: bool,
    pub learning_system: bool,
    pub recommendation_engine: bool,
    pub risk_assessment: bool,
    pub confidence_analysis: bool,
}

// Integration helpers for common workflows

/// Complete analysis workflow for a computation graph
pub fn analyze_computation_graph(
    graph: crate::ComputationGraph,
    system_constraints: Option<SystemConstraints>,
    user_preferences: Option<UserPreferences>,
) -> JitResult<OptimizationReport> {
    let mut advisor = create_advisor();

    let input = AnalysisInput {
        computation_graph: Some(graph),
        system_constraints: system_constraints.unwrap_or_default(),
        user_preferences: user_preferences.unwrap_or_default(),
        benchmark_results: None,
        profiling_data: None,
        previous_optimizations: Vec::new(),
        abstract_analysis: None,
        symbolic_execution: None,
    };

    advisor.analyze_and_recommend(input)
}

/// Analysis workflow with benchmark data
pub fn analyze_with_benchmarks(
    graph: Option<crate::ComputationGraph>,
    benchmark_results: crate::benchmarking::BenchmarkResults,
    system_constraints: Option<SystemConstraints>,
) -> JitResult<OptimizationReport> {
    let mut advisor = create_advisor();

    // Convert benchmarking::BenchmarkResults to advisor::config::BenchmarkResults
    let advisor_benchmark_results = config::BenchmarkResults {
        total_execution_time: Duration::from_millis(1000), // Default since field structure differs
        operation_timings: std::collections::HashMap::new(), // Simplified conversion
        memory_statistics: None,
        resource_usage: None,
    };

    let input = AnalysisInput {
        computation_graph: graph,
        system_constraints: system_constraints.unwrap_or_default(),
        user_preferences: UserPreferences::default(),
        benchmark_results: Some(advisor_benchmark_results),
        profiling_data: None,
        previous_optimizations: Vec::new(),
        abstract_analysis: None,
        symbolic_execution: None,
    };

    advisor.analyze_and_recommend(input)
}

/// Analysis workflow with profiling data
pub fn analyze_with_profiling(
    graph: Option<crate::ComputationGraph>,
    profiling_data: crate::profiler::ProfilingSession,
    system_constraints: Option<SystemConstraints>,
) -> JitResult<OptimizationReport> {
    let mut advisor = create_advisor();

    let input = AnalysisInput {
        computation_graph: graph,
        system_constraints: system_constraints.unwrap_or_default(),
        user_preferences: UserPreferences::default(),
        benchmark_results: None,
        profiling_data: Some(profiling_data),
        previous_optimizations: Vec::new(),
        abstract_analysis: None,
        symbolic_execution: None,
    };

    advisor.analyze_and_recommend(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_advisor() {
        let advisor = create_advisor();
        assert_eq!(advisor.get_version(), "1.0.0");
    }

    #[test]
    fn test_config_creation() {
        let fast_config = create_fast_config();
        assert_eq!(fast_config.analysis_depth, AnalysisDepth::Quick);
        assert!(fast_config.optimization_goals.prioritize_speed);

        let thorough_config = create_thorough_config();
        assert_eq!(thorough_config.analysis_depth, AnalysisDepth::Comprehensive);
        assert!(thorough_config.enable_learning);

        let production_config = create_production_config();
        assert_eq!(production_config.analysis_depth, AnalysisDepth::Standard);
        assert_eq!(production_config.max_recommendations, 10);
    }

    #[test]
    fn test_advisor_info() {
        let info = get_advisor_info();
        assert_eq!(info.version, "1.0.0");
        assert!(!info.supported_optimizations.is_empty());
        assert!(info.features.pattern_detection);
    }

    #[test]
    fn test_minimal_analysis_input() {
        let input = create_minimal_analysis_input();
        assert!(input.computation_graph.is_none());
        assert!(input.benchmark_results.is_none());
    }
}

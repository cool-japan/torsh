//! Configuration and core types for the optimization advisor

use crate::{profiler::ProfilingSession, ComputationGraph, NodeId};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Configuration for the optimization advisor
#[derive(Debug, Clone)]
pub struct AdvisorConfig {
    pub version: String,
    pub max_recommendations: usize,
    pub min_confidence_threshold: f64,
    pub min_benefit_threshold: f64,
    pub enable_learning: bool,
    pub analysis_depth: AnalysisDepth,
    pub optimization_goals: OptimizationGoals,
}

impl Default for AdvisorConfig {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            max_recommendations: 10,
            min_confidence_threshold: 0.5,
            min_benefit_threshold: 0.1,
            enable_learning: true,
            analysis_depth: AnalysisDepth::Standard,
            optimization_goals: OptimizationGoals::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisDepth {
    Quick,
    Standard,
    Comprehensive,
}

#[derive(Debug, Clone)]
pub struct OptimizationGoals {
    pub prioritize_speed: bool,
    pub prioritize_memory: bool,
    pub prioritize_energy: bool,
    pub enable_aggressive_optimizations: bool,
}

impl Default for OptimizationGoals {
    fn default() -> Self {
        Self {
            prioritize_speed: true,
            prioritize_memory: true,
            prioritize_energy: false,
            enable_aggressive_optimizations: false,
        }
    }
}

/// Input for optimization analysis
#[derive(Debug, Clone)]
pub struct AnalysisInput {
    pub computation_graph: Option<ComputationGraph>,
    pub benchmark_results: Option<BenchmarkResults>,
    pub profiling_data: Option<ProfilingSession>,
    pub system_constraints: SystemConstraints,
    pub user_preferences: UserPreferences,
    pub previous_optimizations: Vec<OptimizationRecommendation>,
    pub abstract_analysis: Option<crate::abstract_interpretation::AbstractAnalysisResult>,
    pub symbolic_execution: Option<crate::symbolic_execution::SymbolicExecutionResult>,
}

#[derive(Debug, Clone)]
pub struct SystemConstraints {
    pub memory_gb: usize,
    pub cpu_cores: usize,
    pub has_gpu: bool,
    pub target_platform: TargetPlatform,
}

impl Default for SystemConstraints {
    fn default() -> Self {
        Self {
            memory_gb: 8,
            cpu_cores: 4,
            has_gpu: false,
            target_platform: TargetPlatform::Desktop,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TargetPlatform {
    Desktop,
    Server,
    Mobile,
    Embedded,
}

#[derive(Debug, Clone)]
pub struct UserPreferences {
    pub optimization_aggressiveness: f64,
    pub risk_tolerance: f64,
    pub time_constraints: TimeConstraints,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            optimization_aggressiveness: 0.5,
            risk_tolerance: 0.5,
            time_constraints: TimeConstraints::Unlimited,
        }
    }
}

/// Complete optimization report
#[derive(Debug)]
pub struct OptimizationReport {
    pub recommendations: Vec<OptimizationRecommendation>,
    pub pattern_analysis: PatternAnalysis,
    pub performance_analysis: PerformanceAnalysis,
    pub cost_analysis: CostBenefitAnalysis,
    pub explanations: Vec<OptimizationExplanation>,
    pub confidence_scores: ConfidenceScores,
    pub implementation_complexity: ImplementationComplexity,
    pub expected_improvements: ExpectedImprovements,
    pub analysis_metadata: AnalysisMetadata,
}

/// Individual optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub id: String,
    pub optimization_type: OptimizationType,
    pub title: String,
    pub description: String,
    pub expected_speedup: f64,
    pub expected_memory_reduction: f64,
    pub implementation_complexity: f64,
    pub confidence: f64,
    pub risk_level: f64,
    pub priority_score: f64,
    pub affected_components: Vec<String>,
    pub prerequisites: Vec<String>,
    pub estimated_implementation_time: Duration,
}

/// Types of optimizations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizationType {
    FusionOptimization,
    MemoryOptimization,
    ParallelizationOptimization,
    VectorizationOptimization,
    ConstantFoldingOptimization,
    DeadCodeEliminationOptimization,
    ComputationOptimization,
    IOOptimization,
    CompilationOptimization,
    ArchitectureOptimization,
}

impl OptimizationType {
    pub fn description(&self) -> &'static str {
        match self {
            OptimizationType::FusionOptimization => "Fusion",
            OptimizationType::MemoryOptimization => "Memory",
            OptimizationType::ParallelizationOptimization => "Parallelization",
            OptimizationType::VectorizationOptimization => "Vectorization",
            OptimizationType::ConstantFoldingOptimization => "Constant Folding",
            OptimizationType::DeadCodeEliminationOptimization => "Dead Code Elimination",
            OptimizationType::ComputationOptimization => "Computation",
            OptimizationType::IOOptimization => "I/O",
            OptimizationType::CompilationOptimization => "Compilation",
            OptimizationType::ArchitectureOptimization => "Architecture",
        }
    }
}

/// Pattern analysis results
#[derive(Debug)]
pub struct PatternAnalysis {
    pub detected_patterns: Vec<DetectedPattern>,
    pub antipatterns: Vec<DetectedAntipattern>,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    pub pattern_frequency: HashMap<String, f64>,
    pub complexity_metrics: ComplexityMetrics,
}

/// Detected optimization pattern
#[derive(Debug)]
pub struct DetectedPattern {
    pub pattern_type: PatternType,
    pub location: PatternLocation,
    pub confidence: f64,
    pub description: String,
    pub estimated_benefit: f64,
}

#[derive(Debug, Clone)]
pub enum PatternType {
    FusionOpportunity,
    MemoryInefficiency,
    ParallelizationOpportunity,
    VectorizationOpportunity,
    ComputationPattern,
}

/// Detected antipattern
#[derive(Debug)]
pub struct DetectedAntipattern {
    pub antipattern_type: AntipatternType,
    pub location: PatternLocation,
    pub severity: f64,
    pub description: String,
    pub fix_suggestion: String,
}

#[derive(Debug, Clone)]
pub enum AntipatternType {
    RedundantComputation,
    PoorMemoryLocality,
    InefficientAlgorithm,
    ExcessiveAllocation,
}

impl AntipatternType {
    pub fn description(&self) -> &'static str {
        match self {
            AntipatternType::RedundantComputation => "Redundant Computation",
            AntipatternType::PoorMemoryLocality => "Poor Memory Locality",
            AntipatternType::InefficientAlgorithm => "Inefficient Algorithm",
            AntipatternType::ExcessiveAllocation => "Excessive Allocation",
        }
    }
}

/// Location of a pattern in the graph
#[derive(Debug)]
pub enum PatternLocation {
    Node(NodeId),
    Nodes(Vec<NodeId>),
    Subgraph(Vec<Vec<NodeId>>),
    Global,
}

/// Optimization opportunity
#[derive(Debug)]
pub struct OptimizationOpportunity {
    pub opportunity_type: OpportunityType,
    pub location: PatternLocation,
    pub estimated_benefit: f64,
    pub implementation_complexity: f64,
    pub prerequisites: Vec<String>,
    pub description: String,
}

impl OptimizationOpportunity {
    pub fn can_be_combined_with(&self, other: &Self) -> bool {
        use OptimizationType::*;

        matches!(
            (
                self.opportunity_type.to_optimization_type(),
                other.opportunity_type.to_optimization_type()
            ),
            (FusionOptimization, VectorizationOptimization)
                | (MemoryOptimization, ComputationOptimization)
                | (ConstantFoldingOptimization, FusionOptimization)
        )
    }
}

#[derive(Debug)]
pub enum OpportunityType {
    FusionOptimization,
    MemoryOptimization,
    ParallelizationOptimization,
    VectorizationOptimization,
    ConstantFolding,
    DeadCodeElimination,
    ComputationOptimization,
}

impl OpportunityType {
    pub fn to_optimization_type(&self) -> OptimizationType {
        match self {
            OpportunityType::FusionOptimization => OptimizationType::FusionOptimization,
            OpportunityType::MemoryOptimization => OptimizationType::MemoryOptimization,
            OpportunityType::ParallelizationOptimization => {
                OptimizationType::ParallelizationOptimization
            }
            OpportunityType::VectorizationOptimization => {
                OptimizationType::VectorizationOptimization
            }
            OpportunityType::ConstantFolding => OptimizationType::ConstantFoldingOptimization,
            OpportunityType::DeadCodeElimination => {
                OptimizationType::DeadCodeEliminationOptimization
            }
            OpportunityType::ComputationOptimization => OptimizationType::ComputationOptimization,
        }
    }
}

/// Performance analysis results
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub hotspots: Vec<PerformanceHotspot>,
    pub execution_profile: ExecutionProfile,
    pub resource_utilization: ResourceUtilization,
    pub scalability_analysis: ScalabilityAnalysis,
}

/// Performance bottleneck
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    pub bottleneck_type: BottleneckType,
    pub location: String,
    pub severity: f64,
    pub description: String,
    pub suggested_fixes: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum BottleneckType {
    Memory,
    Computation,
    IO,
    Synchronization,
}

impl BottleneckType {
    pub fn description(&self) -> &'static str {
        match self {
            BottleneckType::Memory => "Memory",
            BottleneckType::Computation => "Computation",
            BottleneckType::IO => "I/O",
            BottleneckType::Synchronization => "Synchronization",
        }
    }
}

/// Performance hotspot
#[derive(Debug, Clone)]
pub struct PerformanceHotspot {
    pub location: String,
    pub execution_time_percent: f64,
    pub memory_usage_percent: f64,
    pub frequency: usize,
    pub optimization_potential: f64,
}

/// Cost-benefit analysis
#[derive(Debug)]
pub struct CostBenefitAnalysis {
    pub implementation_costs: HashMap<String, f64>,
    pub expected_benefits: HashMap<String, f64>,
    pub risk_assessments: HashMap<String, f64>,
    pub roi_estimates: HashMap<String, f64>,
    pub priority_rankings: Vec<(String, f64)>,
}

/// Optimization explanation
#[derive(Debug, Clone)]
pub struct OptimizationExplanation {
    pub recommendation_id: String,
    pub why_beneficial: String,
    pub how_to_implement: String,
    pub potential_risks: Vec<String>,
    pub verification_steps: Vec<String>,
    pub expected_timeline: Duration,
}

/// Analysis metadata
#[derive(Debug)]
pub struct AnalysisMetadata {
    pub analysis_time: Duration,
    pub advisor_version: String,
    pub input_characteristics: String,
    pub recommendations_count: usize,
    pub timestamp: SystemTime,
}

/// Confidence scores for different aspects of analysis
#[derive(Debug)]
pub struct ConfidenceScores {
    pub overall_confidence: f64,
    pub pattern_detection_confidence: f64,
    pub performance_analysis_confidence: f64,
    pub cost_estimation_confidence: f64,
    pub implementation_assessment_confidence: f64,
}

/// Implementation complexity assessment
#[derive(Debug)]
pub struct ImplementationComplexity {
    pub overall_complexity: f64,
    pub technical_complexity: f64,
    pub coordination_complexity: f64,
    pub testing_complexity: f64,
    pub deployment_complexity: f64,
}

/// Expected improvements from optimizations
#[derive(Debug)]
pub struct ExpectedImprovements {
    pub performance_improvement: f64,
    pub memory_reduction: f64,
    pub energy_savings: f64,
    pub development_time_impact: Duration,
    pub maintenance_impact: f64,
}

/// Execution profile information
#[derive(Debug, Clone)]
pub struct ExecutionProfile {
    pub total_execution_time: Duration,
    pub memory_peak_usage: usize,
    pub cpu_utilization: f64,
    pub io_operations: usize,
    pub cache_miss_rate: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub io_bandwidth_usage: f64,
    pub network_usage: f64,
    pub gpu_usage: Option<f64>,
}

/// Scalability analysis results
#[derive(Debug, Clone)]
pub struct ScalabilityAnalysis {
    pub parallelization_potential: f64,
    pub memory_scalability: f64,
    pub io_scalability: f64,
    pub algorithmic_complexity: String,
    pub bottleneck_scalability: HashMap<String, f64>,
}

/// Complexity metrics
#[derive(Debug)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: usize,
    pub data_flow_complexity: usize,
    pub control_flow_complexity: usize,
    pub computational_complexity: String,
    pub memory_complexity: String,
}

/// Loop information for analysis
#[derive(Debug)]
pub struct LoopInfo {
    pub nodes: Vec<NodeId>,
    pub iteration_bound: Option<usize>,
    pub optimization_potential: f64,
}

/// Known optimization pattern in knowledge base
#[derive(Debug)]
pub struct KnownPattern {
    pub name: String,
    pub description: String,
    pub detection_criteria: String,
    pub optimization_potential: f64,
}

/// Known optimization technique
#[derive(Debug)]
pub struct KnownOptimization {
    pub name: String,
    pub description: String,
    pub applicability_conditions: Vec<String>,
    pub expected_benefit: f64,
    pub implementation_complexity: f64,
}

/// Optimization heuristic
#[derive(Debug)]
pub struct OptimizationHeuristic {
    pub name: String,
    pub condition: String,
    pub recommendation: String,
    pub confidence: f64,
}

/// Analysis record for learning system
#[derive(Debug)]
pub struct AnalysisRecord {
    pub timestamp: SystemTime,
    pub input_characteristics: String,
    pub recommendations_count: usize,
    pub avg_confidence: f64,
}

/// Time constraints for optimization work
#[derive(Debug, Clone, PartialEq)]
pub enum TimeConstraints {
    Unlimited,
    Hours(u32),
    Days(u32),
    Weeks(u32),
}

/// Overall analysis summary
#[derive(Debug, Clone)]
pub struct OverallAnalysis {
    pub total_opportunities: usize,
    pub estimated_speedup: f64,
    pub estimated_memory_reduction: f64,
    pub implementation_complexity: f64,
    pub confidence: f64,
    pub risk_assessment: RiskAssessment,
    pub priority_distribution: PriorityDistribution,
}

/// Detailed explanation for recommendations
#[derive(Debug, Clone)]
pub struct DetailedExplanation {
    pub recommendation_id: String,
    pub rationale: String,
    pub implementation_steps: Vec<String>,
    pub potential_issues: Vec<String>,
    pub success_indicators: Vec<String>,
}

/// Confidence analysis for the overall report
#[derive(Debug, Clone)]
pub struct ConfidenceAnalysis {
    pub overall_confidence: f64,
    pub data_quality_score: f64,
    pub analysis_completeness: f64,
    pub recommendation_reliability: f64,
}

/// Risk assessment for recommendations
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub overall_risk_level: f64,
    pub high_risk_recommendations: usize,
    pub risk_factors: Vec<String>,
    pub mitigation_strategies: Vec<String>,
}

/// Distribution of recommendation priorities
#[derive(Debug, Clone)]
pub struct PriorityDistribution {
    pub high_priority: usize,
    pub medium_priority: usize,
    pub low_priority: usize,
}

impl Default for RiskAssessment {
    fn default() -> Self {
        Self {
            overall_risk_level: 0.5,
            high_risk_recommendations: 0,
            risk_factors: Vec::new(),
            mitigation_strategies: Vec::new(),
        }
    }
}

/// Placeholder for benchmark results (to avoid external dependency)
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub total_execution_time: Duration,
    pub operation_timings: HashMap<String, OperationTiming>,
    pub memory_statistics: Option<MemoryStatistics>,
    pub resource_usage: Option<ResourceStats>,
}

/// Operation timing information
#[derive(Debug, Clone)]
pub struct OperationTiming {
    pub average_duration: Duration,
    pub sample_count: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    pub peak_usage: usize,
    pub allocated: usize,
}

/// Resource usage statistics
#[derive(Debug, Clone)]
pub struct ResourceStats {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub io_utilization: f64,
}

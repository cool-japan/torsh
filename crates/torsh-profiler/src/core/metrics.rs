//! Core profiling metrics and analysis types

/// Overhead statistics for profiling operations
#[derive(Debug, Clone, Default)]
pub struct OverheadStats {
    pub add_event_time_ns: u64,
    pub add_event_count: u64,
    pub stack_trace_time_ns: u64,
    pub stack_trace_count: u64,
    pub export_time_ns: u64,
    pub export_count: u64,
    pub total_overhead_ns: u64,
}

/// Bottleneck detection results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BottleneckAnalysis {
    pub slowest_operations: Vec<BottleneckEvent>,
    pub memory_hotspots: Vec<MemoryHotspot>,
    pub thread_contention: Vec<ThreadContentionEvent>,
    pub efficiency_issues: Vec<EfficiencyIssue>,
    pub recommendations: Vec<String>,
}

/// A performance bottleneck event
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BottleneckEvent {
    pub name: String,
    pub category: String,
    pub duration_us: u64,
    pub thread_id: usize,
    pub severity: BottleneckSeverity,
    pub impact_score: f64,
    pub recommendation: String,
}

/// Memory hotspot information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryHotspot {
    pub location: String,
    pub total_allocations: usize,
    pub total_bytes: usize,
    pub average_size: f64,
    pub peak_concurrent_allocations: usize,
    pub severity: BottleneckSeverity,
}

/// Thread contention event
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThreadContentionEvent {
    pub thread_id: usize,
    pub operation: String,
    pub wait_time_us: u64,
    pub contention_count: usize,
}

/// Efficiency issue
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EfficiencyIssue {
    pub issue_type: EfficiencyIssueType,
    pub description: String,
    pub affected_operations: Vec<String>,
    pub performance_impact: f64,
    pub recommendation: String,
}

/// Type of efficiency issue
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EfficiencyIssueType {
    LowThroughput,
    HighLatency,
    MemoryWaste,
    CpuUnderutilization,
    FrequentAllocation,
    LargeAllocation,
}

/// Severity of a bottleneck
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Efficiency metrics analysis
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EfficiencyMetrics {
    pub cpu_efficiency: CpuEfficiency,
    pub memory_efficiency: MemoryEfficiency,
    pub cache_efficiency: CacheEfficiency,
    pub throughput_metrics: ThroughputMetrics,
    pub resource_utilization: ResourceUtilization,
    pub overall_score: f64,
    pub recommendations: Vec<String>,
}

/// CPU efficiency metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CpuEfficiency {
    pub utilization_percentage: f64,
    pub instructions_per_cycle: f64,
    pub computational_intensity: f64,
    pub parallelism_efficiency: f64,
    pub idle_time_percentage: f64,
}

/// Memory efficiency metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryEfficiency {
    pub bandwidth_utilization: f64,
    pub cache_hit_ratio: f64,
    pub memory_access_pattern: MemoryAccessPattern,
    pub allocation_efficiency: f64,
    pub fragmentation_ratio: f64,
}

/// Cache efficiency metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheEfficiency {
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub l3_hit_rate: f64,
    pub memory_locality_score: f64,
    pub cache_miss_penalty: f64,
}

/// Throughput metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThroughputMetrics {
    pub operations_per_second: f64,
    pub flops_per_second: f64,
    pub bandwidth_gb_per_second: f64,
    pub latency_percentiles: LatencyPercentiles,
    pub throughput_efficiency: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceUtilization {
    pub cpu_cores_used: usize,
    pub memory_usage_mb: f64,
    pub peak_memory_mb: f64,
    pub thread_efficiency: f64,
    pub load_balance_score: f64,
}

/// Latency percentiles
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LatencyPercentiles {
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

/// Memory access pattern
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided,
    Mixed,
}

/// Correlation analysis results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CorrelationAnalysis {
    pub operation_correlations: Vec<OperationCorrelation>,
    pub performance_correlations: Vec<PerformanceCorrelation>,
    pub memory_correlations: Vec<MemoryCorrelation>,
    pub temporal_correlations: Vec<TemporalCorrelation>,
    pub correlation_summary: CorrelationSummary,
}

/// Correlation between two operations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OperationCorrelation {
    pub operation_a: String,
    pub operation_b: String,
    pub correlation_coefficient: f64,
    pub co_occurrence_frequency: f64,
    pub temporal_proximity: f64,
    pub correlation_strength: CorrelationStrength,
    pub correlation_type: CorrelationType,
    pub insights: Vec<String>,
}

/// Performance metric correlation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceCorrelation {
    pub metric_a: String,
    pub metric_b: String,
    pub correlation_coefficient: f64,
    pub significance_level: f64,
    pub sample_size: usize,
    pub correlation_strength: CorrelationStrength,
}

/// Memory usage correlation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryCorrelation {
    pub operation: String,
    pub memory_metric: String,
    pub duration_correlation: f64,
    pub bytes_correlation: f64,
    pub allocation_pattern: String,
    pub correlation_strength: CorrelationStrength,
}

/// Temporal relationship between operations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TemporalCorrelation {
    pub operation_sequence: Vec<String>,
    pub sequence_frequency: f64,
    pub average_interval: f64,
    pub sequence_efficiency: f64,
    pub optimization_potential: f64,
}

/// Correlation strength classification
#[derive(Debug, Clone, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub enum CorrelationStrength {
    VeryWeak,   // |r| < 0.2
    Weak,       // 0.2 <= |r| < 0.4
    Moderate,   // 0.4 <= |r| < 0.6
    Strong,     // 0.6 <= |r| < 0.8
    VeryStrong, // |r| >= 0.8
}

impl std::fmt::Display for CorrelationStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorrelationStrength::VeryWeak => write!(f, "Very Weak"),
            CorrelationStrength::Weak => write!(f, "Weak"),
            CorrelationStrength::Moderate => write!(f, "Moderate"),
            CorrelationStrength::Strong => write!(f, "Strong"),
            CorrelationStrength::VeryStrong => write!(f, "Very Strong"),
        }
    }
}

/// Type of correlation
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CorrelationType {
    Causal,        // One operation directly affects another
    Complementary, // Operations work together
    Competitive,   // Operations compete for resources
    Sequential,    // Operations follow in sequence
    Independent,   // No significant relationship
}

/// Summary of correlation analysis
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CorrelationSummary {
    pub total_correlations_analyzed: usize,
    pub strong_correlations_found: usize,
    pub causal_relationships: usize,
    pub bottleneck_correlations: usize,
    pub optimization_opportunities: Vec<String>,
    pub key_insights: Vec<String>,
}

/// Pattern detection analysis results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PatternAnalysis {
    pub performance_patterns: Vec<PerformancePattern>,
    pub bottleneck_patterns: Vec<BottleneckPattern>,
    pub resource_patterns: Vec<ResourcePattern>,
    pub temporal_patterns: Vec<TemporalPattern>,
    pub optimization_patterns: Vec<OptimizationPattern>,
    pub pattern_summary: PatternSummary,
}

/// Recurring performance pattern
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformancePattern {
    pub pattern_type: PerformancePatternType,
    pub operations: Vec<String>,
    pub frequency: f64,
    pub average_duration: f64,
    pub variance: f64,
    pub confidence_score: f64,
    pub impact_level: PatternImpact,
    pub description: String,
}

/// Type of performance pattern
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PerformancePatternType {
    RegularCycle,       // Recurring operations with regular intervals
    BurstActivity,      // Periods of high activity followed by low activity
    GradualDegradation, // Performance gradually getting worse
    SpikesAndDips,      // Irregular spikes and dips in performance
    ConstantLoad,       // Steady, consistent performance
    Oscillation,        // Performance oscillates between states
}

impl std::fmt::Display for PerformancePatternType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerformancePatternType::RegularCycle => write!(f, "Regular Cycle"),
            PerformancePatternType::BurstActivity => write!(f, "Burst Activity"),
            PerformancePatternType::GradualDegradation => write!(f, "Gradual Degradation"),
            PerformancePatternType::SpikesAndDips => write!(f, "Spikes and Dips"),
            PerformancePatternType::ConstantLoad => write!(f, "Constant Load"),
            PerformancePatternType::Oscillation => write!(f, "Oscillation"),
        }
    }
}

/// Bottleneck pattern
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BottleneckPattern {
    pub pattern_id: String,
    pub bottleneck_operations: Vec<String>,
    pub occurrence_frequency: f64,
    pub average_severity: f64,
    pub blocking_relationships: Vec<BlockingRelationship>,
    pub root_cause_analysis: RootCauseAnalysis,
    pub mitigation_strategies: Vec<String>,
}

/// Resource usage pattern
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourcePattern {
    pub resource_type: ResourceType,
    pub usage_pattern: UsagePatternType,
    pub peak_usage_times: Vec<u64>,
    pub utilization_efficiency: f64,
    pub waste_indicators: Vec<WasteIndicator>,
    pub optimization_potential: f64,
}

/// Temporal execution pattern
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TemporalPattern {
    pub pattern_name: String,
    pub operation_sequence: Vec<String>,
    pub sequence_probability: f64,
    pub execution_consistency: f64,
    pub timing_variance: f64,
    pub parallelization_opportunities: Vec<ParallelizationOpportunity>,
}

/// Optimization opportunity pattern
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizationPattern {
    pub optimization_type: OptimizationType,
    pub affected_operations: Vec<String>,
    pub potential_improvement: f64,
    pub implementation_complexity: ComplexityLevel,
    pub confidence_level: f64,
    pub prerequisites: Vec<String>,
    pub estimated_effort: EffortLevel,
}

/// Type of resource
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ResourceType {
    Cpu,
    Memory,
    Bandwidth,
    Storage,
    Network,
    Compute,
}

/// Usage pattern type
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum UsagePatternType {
    Steady,
    Bursty,
    Cyclical,
    Growing,
    Declining,
    Random,
}

/// Waste indicator
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WasteIndicator {
    pub indicator_type: WasteType,
    pub severity: f64,
    pub description: String,
    pub recommendations: Vec<String>,
}

/// Type of waste
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WasteType {
    IdleTime,
    MemoryLeaks,
    RedundantOperations,
    InefficientAlgorithm,
    PoorCacheUsage,
    ExcessiveAllocation,
}

/// Blocking relationship between operations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockingRelationship {
    pub blocker: String,
    pub blocked: String,
    pub blocking_duration: f64,
    pub frequency: f64,
    pub impact_score: f64,
}

/// Root cause analysis
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RootCauseAnalysis {
    pub primary_cause: String,
    pub contributing_factors: Vec<String>,
    pub evidence_strength: f64,
    pub analysis_confidence: f64,
}

/// Parallelization opportunity
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParallelizationOpportunity {
    pub operations: Vec<String>,
    pub parallel_potential: f64,
    pub data_dependencies: Vec<String>,
    pub expected_speedup: f64,
}

/// Type of optimization
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OptimizationType {
    Parallelization,
    Vectorization,
    CacheOptimization,
    MemoryPooling,
    AlgorithmChange,
    DataStructureOptimization,
    ConcurrencyImprovement,
}

impl std::fmt::Display for OptimizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationType::Parallelization => write!(f, "Parallelization"),
            OptimizationType::Vectorization => write!(f, "Vectorization"),
            OptimizationType::CacheOptimization => write!(f, "Cache Optimization"),
            OptimizationType::MemoryPooling => write!(f, "Memory Pooling"),
            OptimizationType::AlgorithmChange => write!(f, "Algorithm Change"),
            OptimizationType::DataStructureOptimization => write!(f, "Data Structure Optimization"),
            OptimizationType::ConcurrencyImprovement => write!(f, "Concurrency Improvement"),
        }
    }
}

/// Pattern impact level
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum PatternImpact {
    Negligible,
    Low,
    Medium,
    High,
    Critical,
}

/// Implementation complexity
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum ComplexityLevel {
    Trivial,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Implementation effort level
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum EffortLevel {
    Minimal,
    Low,
    Medium,
    High,
    Significant,
}

/// Summary of pattern detection
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PatternSummary {
    pub total_patterns_detected: usize,
    pub critical_patterns: usize,
    pub optimization_opportunities: usize,
    pub bottleneck_patterns: usize,
    pub key_recommendations: Vec<String>,
    pub pattern_confidence: f64,
    pub analysis_completeness: f64,
}

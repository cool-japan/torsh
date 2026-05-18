//! Transfer, migration, historical, and supporting statistics types
//!
//! Extracted from types.rs to maintain the 2000-line policy.

use super::*;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Transfer performance metrics
#[derive(Debug, Clone)]
pub struct TransferPerformanceMetrics {
    /// Average bandwidth achieved
    pub average_bandwidth: f64,
    /// Peak bandwidth achieved
    pub peak_bandwidth: f64,
    /// Bandwidth consistency
    pub bandwidth_consistency: f32,
    /// Latency statistics
    pub latency_stats: LatencyStatistics,
    /// Performance vs theoretical maximum
    pub efficiency_vs_max: f32,
}
/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum MemoryPressureLevel {
    #[default]
    None,
    Low,
    Medium,
    High,
    Critical,
}
/// Memory usage snapshot at a point in time
#[derive(Debug, Clone)]
pub struct MemoryUsageSnapshot {
    /// Timestamp of snapshot
    pub timestamp: Instant,
    /// Total allocated memory
    pub total_allocated: usize,
    /// Available memory
    pub available_memory: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Peak memory usage in this period
    pub peak_usage: usize,
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Memory utilization percentage
    pub utilization_percentage: f32,
    /// Memory pressure level
    pub pressure_level: MemoryPressureLevel,
}
/// Migration cost-benefit analysis
#[derive(Debug, Clone)]
pub struct MigrationCostBenefit {
    /// Total migration costs
    pub total_cost: f64,
    /// Total benefits gained
    pub total_benefit: f64,
    /// Net benefit
    pub net_benefit: f64,
    /// Return on investment
    pub return_on_investment: f32,
}
/// Performance characteristics of transfer patterns
#[derive(Debug, Clone)]
pub struct TransferPatternPerformance {
    /// Average bandwidth
    pub bandwidth: f64,
    /// Latency characteristics
    pub latency: Duration,
    /// Efficiency score
    pub efficiency: f32,
    /// Optimization potential
    pub optimization_potential: f32,
}
/// Memory usage breakdown across types
#[derive(Debug, Clone)]
pub struct MemoryUsageBreakdown {
    /// Device memory usage
    pub device_memory: usize,
    /// Unified memory usage
    pub unified_memory: usize,
    /// Pinned memory usage
    pub pinned_memory: usize,
    /// Total memory usage
    pub total_usage: usize,
    /// Usage percentages
    pub usage_percentages: MemoryUsagePercentages,
}
/// System scalability patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalabilityPattern {
    Linear,
    Sublinear,
    Superlinear,
    Logarithmic,
    Plateau,
    Declining,
}
/// Performance bottleneck analysis
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    /// Identified bottlenecks
    pub bottlenecks: Vec<SystemBottleneck>,
    /// Primary bottleneck
    pub primary_bottleneck: Option<SystemBottleneck>,
    /// Bottleneck impact assessment
    pub impact_assessment: BottleneckImpactAssessment,
    /// Resolution recommendations
    pub recommendations: Vec<BottleneckResolution>,
}
/// Unified memory migration statistics
#[derive(Debug, Clone)]
pub struct MigrationStatistics {
    /// Total migrations performed
    pub total_migrations: u64,
    /// Migration frequency per allocation
    pub migration_frequency: f64,
    /// Migration overhead percentage
    pub migration_overhead: f32,
    /// Migration effectiveness score
    pub effectiveness_score: f32,
    /// Migration patterns
    pub patterns: Vec<MigrationPattern>,
    /// Cost-benefit analysis
    pub cost_benefit: MigrationCostBenefit,
}
/// Historical performance record
#[derive(Debug, Clone)]
pub struct HistoricalPerformance {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Performance snapshot
    pub performance: PerformanceSnapshot,
    /// Operation counts
    pub operation_counts: HashMap<String, u64>,
    /// Latency measurements
    pub latencies: HashMap<String, Duration>,
}
/// Throughput metrics for memory operations
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    /// Operations per second
    pub operations_per_second: f64,
    /// Bytes per second throughput
    pub bytes_per_second: f64,
    /// Peak throughput achieved
    pub peak_throughput: f64,
    /// Throughput consistency score
    pub consistency_score: f32,
    /// Throughput efficiency vs theoretical maximum
    pub efficiency_vs_theoretical: f32,
}
/// Analysis results
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Analysis type
    pub analysis_type: String,
    /// Results data
    pub results: HashMap<String, f64>,
    /// Confidence level
    pub confidence: f32,
    /// Analysis timestamp
    pub timestamp: Instant,
}
/// Statistical analysis configuration
#[derive(Debug, Clone)]
pub struct StatisticalAnalysisConfig {
    /// Minimum sample size for analysis
    pub min_sample_size: usize,
    /// Confidence interval
    pub confidence_interval: f64,
    /// Enable outlier detection
    pub enable_outlier_detection: bool,
    /// Statistical significance threshold
    pub significance_threshold: f64,
}
/// Defragmentation complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefragmentationComplexity {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}
/// Data compression settings
#[derive(Debug, Clone)]
pub struct CompressionSettings {
    /// Enable compression
    pub enable_compression: bool,
    /// Compression ratio target
    pub compression_ratio: f32,
    /// Compression algorithm preference
    pub algorithm: CompressionAlgorithm,
}
/// Transfer pattern analysis
#[derive(Debug, Clone)]
pub struct TransferPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Transfer direction
    pub direction: TransferDirection,
    /// Pattern frequency
    pub frequency: f64,
    /// Average transfer size in pattern
    pub average_size: f64,
    /// Performance characteristics
    pub performance: TransferPatternPerformance,
}
/// Advice-based optimization opportunities
#[derive(Debug, Clone)]
pub struct AdviceOptimization {
    /// Advice type
    pub advice_type: String,
    /// Current effectiveness
    pub current_effectiveness: f32,
    /// Potential improvement
    pub potential_improvement: f32,
    /// Optimization strategy
    pub strategy: String,
}
/// System latency characteristics
#[derive(Debug, Clone)]
pub struct SystemLatencyCharacteristics {
    /// Average system latency
    pub average_latency: Duration,
    /// Latency percentiles
    pub percentiles: PercentileStats,
    /// Latency consistency
    pub consistency: f32,
    /// Latency under different loads
    pub load_latency: LoadLatencyCharacteristics,
}
/// Mapping overhead analysis
#[derive(Debug, Clone)]
pub struct MappingOverheadAnalysis {
    /// Setup overhead per mapping
    pub setup_overhead: Duration,
    /// Memory overhead per mapping
    pub memory_overhead: usize,
    /// Performance overhead
    pub performance_overhead: f32,
    /// Overall overhead score
    pub overall_overhead: f32,
}
/// User experience impact levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum UserImpactLevel {
    None,
    Minimal,
    Noticeable,
    Significant,
    Severe,
}
/// Pattern-based optimization opportunities
#[derive(Debug, Clone)]
pub struct PatternOptimization {
    /// Optimization description
    pub description: String,
    /// Expected performance improvement
    pub expected_improvement: f32,
    /// Implementation complexity
    pub complexity: OptimizationComplexity,
    /// Confidence in the optimization
    pub confidence: f32,
}
/// Access pattern identification
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Pattern type
    pub pattern_type: AccessPatternType,
    /// Pattern strength (0.0 to 1.0)
    pub strength: f32,
    /// Pattern duration
    pub duration: Duration,
    /// Devices involved
    pub devices: Vec<usize>,
    /// Performance characteristics
    pub performance: AccessPerformance,
}
/// Severity levels for latency spikes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpikeSeverity {
    Minor,
    Moderate,
    Major,
    Critical,
}
/// Bottleneck severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BottleneckSeverity {
    Minor,
    Moderate,
    Significant,
    Major,
    Critical,
}
/// Bottleneck impact assessment
#[derive(Debug, Clone)]
pub struct BottleneckImpactAssessment {
    /// Performance degradation
    pub performance_degradation: f32,
    /// Resource waste
    pub resource_waste: f32,
    /// User experience impact
    pub user_impact: UserImpactLevel,
    /// Business impact
    pub business_impact: BusinessImpactLevel,
}
/// Scalability bottlenecks
#[derive(Debug, Clone)]
pub struct ScalabilityBottleneck {
    /// Bottleneck component
    pub component: String,
    /// Impact on scalability
    pub impact: f32,
    /// Resolution complexity
    pub resolution_complexity: OptimizationComplexity,
}
/// System bottleneck identification
#[derive(Debug, Clone)]
pub struct SystemBottleneck {
    /// Bottleneck location
    pub location: String,
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity level
    pub severity: BottleneckSeverity,
    /// Performance impact
    pub performance_impact: f32,
    /// Frequency of occurrence
    pub frequency: f32,
}
/// Statistical model for analysis
#[derive(Debug, Clone)]
pub struct StatisticalModel {
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Model accuracy
    pub accuracy: f32,
    /// Last training time
    pub last_training: Instant,
}

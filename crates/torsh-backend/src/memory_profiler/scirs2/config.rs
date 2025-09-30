//! Configuration types and defaults for SciRS2 integration
//!
//! This module contains all configuration structures, enums, and default
//! values for controlling SciRS2 integration behavior and features.

use std::time::Duration;

/// SciRS2 integration configuration
///
/// Configuration options for SciRS2 integration behavior and features.
#[derive(Debug, Clone)]
pub struct ScirS2IntegrationConfig {
    /// Enable real-time statistics synchronization
    pub enable_realtime_sync: bool,

    /// Statistics synchronization interval
    pub sync_interval: Duration,

    /// Enable memory event callbacks
    pub enable_event_callbacks: bool,

    /// Track detailed allocation patterns
    pub track_allocation_patterns: bool,

    /// Enable performance optimization suggestions
    pub enable_optimization_suggestions: bool,

    /// Advanced configuration options
    pub advanced_config: AdvancedIntegrationConfig,
}

/// Advanced integration configuration
#[derive(Debug, Clone)]
pub struct AdvancedIntegrationConfig {
    /// Enable predictive modeling
    pub enable_predictive_modeling: bool,

    /// Model update frequency
    pub model_update_frequency: Duration,

    /// Enable automated optimization
    pub enable_automated_optimization: bool,

    /// Optimization aggressiveness (0.0 to 1.0)
    pub optimization_aggressiveness: f64,

    /// Enable pool health monitoring
    pub enable_health_monitoring: bool,

    /// Health check interval
    pub health_check_interval: Duration,

    /// Enable performance profiling
    pub enable_performance_profiling: bool,

    /// Profiling detail level
    pub profiling_detail_level: ProfilingDetailLevel,
}

/// Profiling detail levels
#[derive(Debug, Clone)]
pub enum ProfilingDetailLevel {
    Basic,
    Detailed,
    Comprehensive,
    Debug,
}

/// Performance tier classification
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerformanceTier {
    Critical,
    Low,
    Medium,
    High,
}

/// Optimization types
#[derive(Debug, Clone)]
pub enum OptimizationType {
    CacheOptimization,
    ContentionReduction,
    FragmentationReduction,
    LatencyOptimization,
    ThroughputOptimization,
    MemoryEfficiency,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Pool optimization types
#[derive(Debug, Clone)]
pub enum PoolOptimizationType {
    CapacityAdjustment,
    AllocationStrategyChange,
    PreallocationOptimization,
    GarbageCollectionTuning,
    AccessPatternOptimization,
    CacheOptimization,
}

/// Recommendation priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Health severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum HealthSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Risk types
#[derive(Debug, Clone)]
pub enum RiskType {
    MemoryLeak,
    FragmentationIncrease,
    PerformanceDegradation,
    CapacityOverflow,
    AccessPatternChange,
}

/// Health trend
#[derive(Debug, Clone)]
pub enum HealthTrend {
    Improving,
    Stable,
    Declining,
    Unstable,
}

/// Fragmentation trend
#[derive(Debug, Clone)]
pub enum FragmentationTrend {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Utilization change reasons
#[derive(Debug, Clone)]
pub enum UtilizationChangeReason {
    AllocationIncrease,
    AllocationDecrease,
    PoolExpansion,
    PoolShrinkage,
    WorkloadChange,
    ConfigurationChange,
}

/// Fragmentation types
#[derive(Debug, Clone)]
pub enum FragmentationType {
    Internal,
    External,
    Temporal,
    Spatial,
}

/// Pressure trend
#[derive(Debug, Clone)]
pub enum PressureTrend {
    Increasing,
    Stable,
    Decreasing,
    Fluctuating,
}

/// Comparison types for alerts
#[derive(Debug, Clone)]
pub enum ComparisonType {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Configuration recommendation
#[derive(Debug, Clone)]
pub struct ConfigRecommendation {
    /// Parameter name
    pub parameter: String,

    /// Current value
    pub current_value: String,

    /// Recommended value
    pub recommended_value: String,

    /// Expected impact
    pub expected_impact: f64,

    /// Justification
    pub justification: String,
}

/// Optimization opportunity
#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    /// Opportunity type
    pub opportunity_type: OptimizationType,

    /// Potential improvement
    pub potential_improvement: f64,

    /// Implementation difficulty
    pub implementation_difficulty: DifficultyLevel,

    /// Description
    pub description: String,
}

/// Resource requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Additional memory needed
    pub additional_memory: usize,

    /// CPU overhead
    pub cpu_overhead: f64,

    /// Implementation time
    pub implementation_time: Duration,

    /// Maintenance overhead
    pub maintenance_overhead: f64,
}

/// Pool optimization recommendation
#[derive(Debug, Clone)]
pub struct PoolOptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: PoolOptimizationType,

    /// Priority level
    pub priority: RecommendationPriority,

    /// Description
    pub description: String,

    /// Implementation steps
    pub implementation_steps: Vec<String>,

    /// Expected benefits
    pub expected_benefits: Vec<String>,

    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Health indicator
#[derive(Debug, Clone)]
pub struct HealthIndicator {
    /// Indicator name
    pub name: String,

    /// Current value
    pub value: f64,

    /// Healthy range
    pub healthy_range: (f64, f64),

    /// Severity if out of range
    pub severity: HealthSeverity,
}

/// Risk factor
#[derive(Debug, Clone)]
pub struct RiskFactor {
    /// Risk type
    pub risk_type: RiskType,

    /// Risk probability
    pub probability: f64,

    /// Potential impact
    pub impact: f64,

    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Cleanup status
#[derive(Debug, Clone)]
pub struct CleanupStatus {
    /// Successfully cleaned up
    pub success: bool,

    /// Memory leaked
    pub memory_leaked: usize,

    /// Cleanup duration
    pub cleanup_duration: Duration,

    /// Error messages
    pub errors: Vec<String>,
}

/// Integration status information
#[derive(Debug, Clone)]
pub struct IntegrationStatus {
    /// Whether integration is active
    pub active: bool,

    /// Last synchronization timestamp
    pub last_sync: Option<std::time::Instant>,

    /// Number of tracked allocators
    pub allocator_count: usize,

    /// Number of tracked pools
    pub pool_count: usize,

    /// Overall health score (0.0 to 1.0)
    pub health_score: f64,

    /// Synchronization interval
    pub sync_interval: Duration,

    /// List of enabled features
    pub features_enabled: Vec<String>,
}

impl Default for ScirS2IntegrationConfig {
    fn default() -> Self {
        Self {
            enable_realtime_sync: true,
            sync_interval: Duration::from_secs(5),
            enable_event_callbacks: true,
            track_allocation_patterns: true,
            enable_optimization_suggestions: true,
            advanced_config: AdvancedIntegrationConfig {
                enable_predictive_modeling: false,
                model_update_frequency: Duration::from_secs(60),
                enable_automated_optimization: false,
                optimization_aggressiveness: 0.5,
                enable_health_monitoring: true,
                health_check_interval: Duration::from_secs(30),
                enable_performance_profiling: true,
                profiling_detail_level: ProfilingDetailLevel::Detailed,
            },
        }
    }
}

impl AdvancedIntegrationConfig {
    /// Create a conservative advanced configuration
    pub fn conservative() -> Self {
        Self {
            enable_predictive_modeling: false,
            model_update_frequency: Duration::from_secs(120),
            enable_automated_optimization: false,
            optimization_aggressiveness: 0.3,
            enable_health_monitoring: true,
            health_check_interval: Duration::from_secs(60),
            enable_performance_profiling: false,
            profiling_detail_level: ProfilingDetailLevel::Basic,
        }
    }

    /// Create an aggressive advanced configuration
    pub fn aggressive() -> Self {
        Self {
            enable_predictive_modeling: true,
            model_update_frequency: Duration::from_secs(30),
            enable_automated_optimization: true,
            optimization_aggressiveness: 0.8,
            enable_health_monitoring: true,
            health_check_interval: Duration::from_secs(10),
            enable_performance_profiling: true,
            profiling_detail_level: ProfilingDetailLevel::Comprehensive,
        }
    }

    /// Create a debug configuration with maximum monitoring
    pub fn debug() -> Self {
        Self {
            enable_predictive_modeling: true,
            model_update_frequency: Duration::from_secs(10),
            enable_automated_optimization: false,
            optimization_aggressiveness: 0.0,
            enable_health_monitoring: true,
            health_check_interval: Duration::from_secs(5),
            enable_performance_profiling: true,
            profiling_detail_level: ProfilingDetailLevel::Debug,
        }
    }
}

impl Default for AdvancedIntegrationConfig {
    fn default() -> Self {
        Self::conservative()
    }
}

impl ScirS2IntegrationConfig {
    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            enable_realtime_sync: true,
            sync_interval: Duration::from_secs(1),
            enable_event_callbacks: true,
            track_allocation_patterns: true,
            enable_optimization_suggestions: true,
            advanced_config: AdvancedIntegrationConfig::aggressive(),
        }
    }

    /// Create a low-overhead configuration
    pub fn low_overhead() -> Self {
        Self {
            enable_realtime_sync: false,
            sync_interval: Duration::from_secs(30),
            enable_event_callbacks: false,
            track_allocation_patterns: false,
            enable_optimization_suggestions: false,
            advanced_config: AdvancedIntegrationConfig::conservative(),
        }
    }

    /// Create a debugging configuration
    pub fn debug() -> Self {
        Self {
            enable_realtime_sync: true,
            sync_interval: Duration::from_millis(100),
            enable_event_callbacks: true,
            track_allocation_patterns: true,
            enable_optimization_suggestions: true,
            advanced_config: AdvancedIntegrationConfig::debug(),
        }
    }
}

/// Validate configuration parameters
pub fn validate_config(config: &ScirS2IntegrationConfig) -> Result<(), String> {
    // Validate sync interval
    if config.sync_interval.as_millis() < 10 {
        return Err("Sync interval too small (minimum 10ms)".to_string());
    }

    if config.sync_interval.as_secs() > 3600 {
        return Err("Sync interval too large (maximum 1 hour)".to_string());
    }

    // Validate advanced configuration
    if config.advanced_config.optimization_aggressiveness < 0.0
        || config.advanced_config.optimization_aggressiveness > 1.0
    {
        return Err("Optimization aggressiveness must be between 0.0 and 1.0".to_string());
    }

    if config.advanced_config.health_check_interval.as_millis() < 1000 {
        return Err("Health check interval too small (minimum 1 second)".to_string());
    }

    // Validate feature combinations
    if config.advanced_config.enable_automated_optimization
        && !config.enable_optimization_suggestions
    {
        return Err(
            "Automated optimization requires optimization suggestions to be enabled".to_string(),
        );
    }

    if config.advanced_config.enable_predictive_modeling && !config.track_allocation_patterns {
        return Err("Predictive modeling requires allocation pattern tracking".to_string());
    }

    Ok(())
}

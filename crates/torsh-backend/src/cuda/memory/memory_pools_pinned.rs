//! Supporting types for CUDA memory pool management
//!
//! Contains `GlobalPoolStats`, `ResourceTracker`, `PoolOptimizationEngine`,
//! `MemoryPressureMonitor`, `CrossPoolAnalytics`, and all their supporting
//! structs, enums, impls, and `Default` implementations.

use super::*;

/// Global pool statistics across all memory types
#[derive(Debug, Clone)]
pub struct GlobalPoolStats {
    /// Total pools managed
    pub total_pools: usize,

    /// Device pools count
    pub device_pools: usize,

    /// Unified pools count
    pub unified_pools: usize,

    /// Pinned pools count
    pub pinned_pools: usize,

    /// Total memory managed
    pub total_memory_managed: usize,

    /// Global hit rate
    pub global_hit_rate: f32,

    /// Cross-pool efficiency
    pub cross_pool_efficiency: f32,

    /// Memory waste across all pools
    pub global_memory_waste: f32,

    /// Overall pool health score
    pub overall_health_score: f32,
}

/// Resource allocation tracker
#[derive(Debug)]
pub struct ResourceTracker {
    /// Resource usage by device
    pub(super) device_usage: HashMap<usize, DeviceResourceUsage>,

    /// Global resource limits
    pub(super) global_limits: ResourceLimits,

    /// Resource allocation history
    pub(super) allocation_history: Vec<ResourceAllocationEvent>,

    /// Resource pressure indicators
    pub(super) pressure_indicators: Vec<ResourcePressureIndicator>,
}

/// Device-specific resource usage
#[derive(Debug, Clone)]
pub struct DeviceResourceUsage {
    /// Device ID
    pub device_id: usize,

    /// Total device memory allocated
    pub device_memory_allocated: usize,

    /// Unified memory allocated on device
    pub unified_memory_allocated: usize,

    /// Pinned memory associated with device
    pub pinned_memory_allocated: usize,

    /// Memory utilization percentage
    pub utilization_percentage: f32,

    /// Resource pressure level
    pub pressure_level: ResourcePressureLevel,
}

/// Global resource limits
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum total device memory
    pub max_device_memory: Option<usize>,

    /// Maximum total unified memory
    pub max_unified_memory: Option<usize>,

    /// Maximum total pinned memory
    pub max_pinned_memory: Option<usize>,

    /// Maximum pools per device
    pub max_pools_per_device: Option<usize>,

    /// Global memory limit across all types
    pub global_memory_limit: Option<usize>,
}

/// Resource allocation event
#[derive(Debug, Clone)]
pub struct ResourceAllocationEvent {
    /// Event timestamp
    pub timestamp: Instant,

    /// Event type
    pub event_type: AllocationEventType,

    /// Device ID involved
    pub device_id: Option<usize>,

    /// Memory type
    pub memory_type: MemoryType,

    /// Size involved
    pub size: usize,

    /// Success status
    pub success: bool,
}

/// Types of allocation events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationEventType {
    Allocation,
    Deallocation,
    PoolCreation,
    PoolDestruction,
    Migration,
    Optimization,
}

/// Memory types for tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Device,
    Unified,
    Pinned,
}

/// Resource pressure indicators
#[derive(Debug, Clone)]
pub struct ResourcePressureIndicator {
    /// Indicator type
    pub indicator_type: PressureIndicatorType,

    /// Current value
    pub current_value: f32,

    /// Threshold value
    pub threshold_value: f32,

    /// Severity level
    pub severity: PressureSeverity,

    /// Recommended action
    pub recommended_action: String,
}

/// Types of pressure indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureIndicatorType {
    MemoryUtilization,
    AllocationFailureRate,
    PoolFragmentation,
    PerformanceDegradation,
}

/// Resource pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResourcePressureLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Pressure severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PressureSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Pool optimization engine
#[derive(Debug)]
pub struct PoolOptimizationEngine {
    /// Optimization strategies
    pub(super) strategies: Vec<OptimizationStrategy>,

    /// Optimization history
    pub(super) history: Vec<OptimizationResult>,

    /// Current optimization state
    pub(super) current_state: OptimizationState,

    /// Performance baseline
    pub(super) performance_baseline: PerformanceBaseline,

    /// Optimization rules
    pub(super) optimization_rules: Vec<OptimizationRule>,
}

/// Optimization strategies
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    /// Strategy name
    pub name: String,

    /// Target pool types
    pub target_pool_types: Vec<MemoryType>,

    /// Optimization conditions
    pub conditions: Vec<OptimizationCondition>,

    /// Expected improvement
    pub expected_improvement: f32,

    /// Confidence level
    pub confidence: f32,

    /// Implementation complexity
    pub complexity: OptimizationComplexity,
}

/// Optimization conditions
#[derive(Debug, Clone)]
pub enum OptimizationCondition {
    HighFragmentation { threshold: f32 },
    LowHitRate { threshold: f32 },
    MemoryPressure { level: ResourcePressureLevel },
    PerformanceDegradation { threshold: f32 },
    InefficiientGrowth,
}

/// Optimization complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationComplexity {
    Simple,
    Moderate,
    Complex,
    Advanced,
}

/// Optimization results
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimization timestamp
    pub timestamp: Instant,

    /// Applied strategy
    pub strategy: OptimizationStrategy,

    /// Target pools affected
    pub affected_pools: Vec<String>,

    /// Performance improvement achieved
    pub improvement_achieved: f32,

    /// Optimization duration
    pub duration: Duration,

    /// Success status
    pub success: bool,

    /// Side effects observed
    pub side_effects: Vec<String>,
}

/// Current optimization state
#[derive(Debug, Clone)]
pub struct OptimizationState {
    /// Currently running optimizations
    pub active_optimizations: Vec<ActiveOptimization>,

    /// Optimization queue
    pub optimization_queue: Vec<QueuedOptimization>,

    /// Last optimization time
    pub last_optimization: Option<Instant>,

    /// Optimization frequency
    pub optimization_frequency: Duration,

    /// Total optimizations performed
    pub total_optimizations: u64,
}

/// Active optimization tracking
#[derive(Debug, Clone)]
pub struct ActiveOptimization {
    /// Optimization ID
    pub id: u64,

    /// Strategy being applied
    pub strategy: OptimizationStrategy,

    /// Start time
    pub start_time: Instant,

    /// Expected completion time
    pub expected_completion: Instant,

    /// Current progress
    pub progress: f32,
}

/// Queued optimization
#[derive(Debug, Clone)]
pub struct QueuedOptimization {
    /// Optimization ID
    pub id: u64,

    /// Strategy to apply
    pub strategy: OptimizationStrategy,

    /// Priority level
    pub priority: OptimizationPriority,

    /// Queue timestamp
    pub queued_at: Instant,

    /// Prerequisites
    pub prerequisites: Vec<String>,
}

/// Optimization priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Performance baseline for comparison
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// Baseline allocation rate
    pub allocation_rate: f64,

    /// Baseline hit rate
    pub hit_rate: f32,

    /// Baseline memory efficiency
    pub memory_efficiency: f32,

    /// Baseline established time
    pub established_at: Instant,

    /// Baseline validity duration
    pub validity_duration: Duration,
}

/// Optimization rules for automatic optimization
#[derive(Debug, Clone)]
pub struct OptimizationRule {
    /// Rule identifier
    pub id: String,

    /// Rule description
    pub description: String,

    /// Condition to trigger rule
    pub trigger_condition: OptimizationCondition,

    /// Action to take
    pub action: OptimizationAction,

    /// Rule priority
    pub priority: OptimizationPriority,

    /// Success rate of this rule
    pub success_rate: f32,
}

/// Optimization actions
#[derive(Debug, Clone)]
pub enum OptimizationAction {
    DefragmentPools,
    AdjustPoolSizes { factor: f32 },
    RebalanceAllocations,
    ChangeGrowthStrategy(PoolGrowthStrategy),
    ForceCleanup,
    MigrateAllocations { target_location: Location },
}

/// Memory pressure monitor
#[derive(Debug)]
pub struct MemoryPressureMonitor {
    /// Current pressure levels by device
    pub(super) device_pressure: HashMap<usize, ResourcePressureLevel>,

    /// Global pressure level
    pub(super) global_pressure: ResourcePressureLevel,

    /// Pressure history
    pub(super) pressure_history: Vec<PressureEvent>,

    /// Pressure thresholds
    pub(super) pressure_thresholds: PressureThresholds,

    /// Alert conditions
    pub(super) alert_conditions: Vec<PressureAlertCondition>,
}

/// Pressure event recording
#[derive(Debug, Clone)]
pub struct PressureEvent {
    /// Event timestamp
    pub timestamp: Instant,

    /// Device ID (if device-specific)
    pub device_id: Option<usize>,

    /// Pressure level
    pub pressure_level: ResourcePressureLevel,

    /// Trigger cause
    pub trigger_cause: String,

    /// Mitigation actions taken
    pub actions_taken: Vec<String>,
}

/// Pressure monitoring thresholds
#[derive(Debug, Clone)]
pub struct PressureThresholds {
    /// Low pressure threshold
    pub low_threshold: f32,

    /// Medium pressure threshold
    pub medium_threshold: f32,

    /// High pressure threshold
    pub high_threshold: f32,

    /// Critical pressure threshold
    pub critical_threshold: f32,
}

/// Pressure alert conditions
#[derive(Debug, Clone)]
pub struct PressureAlertCondition {
    /// Condition name
    pub name: String,

    /// Pressure level trigger
    pub trigger_level: ResourcePressureLevel,

    /// Alert message
    pub alert_message: String,

    /// Recommended actions
    pub recommended_actions: Vec<String>,

    /// Alert cooldown duration
    pub cooldown: Duration,

    /// Last alert time
    pub last_alert: Option<Instant>,
}

/// Cross-pool analytics engine
#[derive(Debug)]
pub struct CrossPoolAnalytics {
    /// Pool correlation analysis
    pub(super) correlations: HashMap<String, PoolCorrelation>,

    /// Cross-pool optimization opportunities
    pub(super) optimization_opportunities: Vec<CrossPoolOptimization>,

    /// Resource sharing analysis
    pub(super) resource_sharing: ResourceSharingAnalysis,

    /// Performance impact analysis
    pub(super) impact_analysis: PerformanceImpactAnalysis,
}

/// Pool correlation data
#[derive(Debug, Clone)]
pub struct PoolCorrelation {
    /// Pool pair identifier
    pub pool_pair: String,

    /// Correlation coefficient
    pub correlation: f64,

    /// Correlation type
    pub correlation_type: CorrelationType,

    /// Statistical significance
    pub significance: f64,

    /// Potential for optimization
    pub optimization_potential: f32,
}

/// Types of correlations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationType {
    Positive,
    Negative,
    NoCorrelation,
}

/// Cross-pool optimization opportunities
#[derive(Debug, Clone)]
pub struct CrossPoolOptimization {
    /// Optimization identifier
    pub id: String,

    /// Involved pools
    pub involved_pools: Vec<String>,

    /// Optimization type
    pub optimization_type: CrossPoolOptimizationType,

    /// Expected benefit
    pub expected_benefit: f32,

    /// Implementation complexity
    pub complexity: OptimizationComplexity,

    /// Risk assessment
    pub risk_level: RiskLevel,
}

/// Types of cross-pool optimizations
#[derive(Debug, Clone)]
pub enum CrossPoolOptimizationType {
    /// Rebalance allocations between pools
    LoadBalancing,
    /// Share resources between pools
    ResourceSharing,
    /// Coordinate growth strategies
    CoordinatedGrowth,
    /// Unified cleanup strategy
    UnifiedCleanup,
    /// Cross-pool migration
    CrossPoolMigration,
}

/// Risk levels for optimizations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Resource sharing analysis
#[derive(Debug, Clone)]
pub struct ResourceSharingAnalysis {
    /// Sharing opportunities
    pub opportunities: Vec<SharingOpportunity>,

    /// Current sharing efficiency
    pub sharing_efficiency: f32,

    /// Potential savings
    pub potential_savings: usize,

    /// Sharing conflicts
    pub conflicts: Vec<SharingConflict>,
}

/// Resource sharing opportunity
#[derive(Debug, Clone)]
pub struct SharingOpportunity {
    /// Opportunity identifier
    pub id: String,

    /// Pools that can share
    pub candidate_pools: Vec<String>,

    /// Resource type to share
    pub resource_type: MemoryType,

    /// Estimated savings
    pub estimated_savings: usize,

    /// Feasibility score
    pub feasibility: f32,
}

/// Resource sharing conflict
#[derive(Debug, Clone)]
pub struct SharingConflict {
    /// Conflict identifier
    pub id: String,

    /// Conflicting pools
    pub conflicting_pools: Vec<String>,

    /// Conflict type
    pub conflict_type: ConflictType,

    /// Severity
    pub severity: ConflictSeverity,

    /// Resolution suggestions
    pub resolution_suggestions: Vec<String>,
}

/// Types of sharing conflicts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    ResourceContention,
    PerformanceInterference,
    SecurityConstraint,
    CompatibilityIssue,
}

/// Conflict severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConflictSeverity {
    Minor,
    Moderate,
    Major,
    Critical,
}

/// Performance impact analysis
#[derive(Debug, Clone)]
pub struct PerformanceImpactAnalysis {
    /// Impact measurements
    pub impact_measurements: Vec<ImpactMeasurement>,

    /// Overall impact score
    pub overall_impact: f32,

    /// Performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,

    /// Improvement recommendations
    pub recommendations: Vec<PerformanceRecommendation>,
}

/// Individual impact measurement
#[derive(Debug, Clone)]
pub struct ImpactMeasurement {
    /// Measurement identifier
    pub id: String,

    /// Pool or system affected
    pub target: String,

    /// Metric measured
    pub metric: String,

    /// Impact value
    pub impact_value: f32,

    /// Measurement confidence
    pub confidence: f32,

    /// Measurement timestamp
    pub timestamp: Instant,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    /// Bottleneck identifier
    pub id: String,

    /// Location of bottleneck
    pub location: String,

    /// Bottleneck type
    pub bottleneck_type: BottleneckType,

    /// Severity assessment
    pub severity: BottleneckSeverity,

    /// Performance impact
    pub performance_impact: f32,

    /// Resolution complexity
    pub resolution_complexity: OptimizationComplexity,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BottleneckType {
    MemoryBandwidth,
    AllocationLatency,
    PoolContention,
    FragmentationOverhead,
    MigrationCost,
}

/// Bottleneck severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BottleneckSeverity {
    Minor,
    Moderate,
    Significant,
    Critical,
}

/// Performance improvement recommendation
#[derive(Debug, Clone)]
pub struct PerformanceRecommendation {
    /// Recommendation identifier
    pub id: String,

    /// Recommendation description
    pub description: String,

    /// Target improvement
    pub target_improvement: f32,

    /// Implementation effort
    pub effort_required: EffortLevel,

    /// Risk assessment
    pub risk: RiskLevel,

    /// Priority level
    pub priority: RecommendationPriority,

    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Effort levels for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffortLevel {
    Minimal,
    Low,
    Moderate,
    High,
    Extensive,
}

/// Recommendation priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Optional,
    Recommended,
    Important,
    Critical,
}

// ──────────────────────────────────────────────────────────────
// Impls
// ──────────────────────────────────────────────────────────────

impl ResourceTracker {
    pub(super) fn new() -> Self {
        Self {
            device_usage: HashMap::new(),
            global_limits: ResourceLimits::default(),
            allocation_history: Vec::new(),
            pressure_indicators: Vec::new(),
        }
    }
}

impl PoolOptimizationEngine {
    pub(super) fn new() -> Self {
        Self {
            strategies: Vec::new(),
            history: Vec::new(),
            current_state: OptimizationState::default(),
            performance_baseline: PerformanceBaseline::default(),
            optimization_rules: Vec::new(),
        }
    }
}

impl MemoryPressureMonitor {
    pub(super) fn new() -> Self {
        Self {
            device_pressure: HashMap::new(),
            global_pressure: ResourcePressureLevel::Low,
            pressure_history: Vec::new(),
            pressure_thresholds: PressureThresholds::default(),
            alert_conditions: Vec::new(),
        }
    }
}

impl CrossPoolAnalytics {
    pub(super) fn new() -> Self {
        Self {
            correlations: HashMap::new(),
            optimization_opportunities: Vec::new(),
            resource_sharing: ResourceSharingAnalysis::default(),
            impact_analysis: PerformanceImpactAnalysis::default(),
        }
    }

    pub(super) fn identify_optimization_opportunities(&self) -> Vec<CrossPoolOptimization> {
        Vec::new()
    }
}

// ──────────────────────────────────────────────────────────────
// Default implementations
// ──────────────────────────────────────────────────────────────

impl Default for GlobalPoolStats {
    fn default() -> Self {
        Self {
            total_pools: 0,
            device_pools: 0,
            unified_pools: 0,
            pinned_pools: 0,
            total_memory_managed: 0,
            global_hit_rate: 0.0,
            cross_pool_efficiency: 0.0,
            global_memory_waste: 0.0,
            overall_health_score: 1.0,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_device_memory: None,
            max_unified_memory: None,
            max_pinned_memory: None,
            max_pools_per_device: None,
            global_memory_limit: None,
        }
    }
}

impl Default for OptimizationState {
    fn default() -> Self {
        Self {
            active_optimizations: Vec::new(),
            optimization_queue: Vec::new(),
            last_optimization: None,
            optimization_frequency: Duration::from_secs(300),
            total_optimizations: 0,
        }
    }
}

impl Default for PerformanceBaseline {
    fn default() -> Self {
        Self {
            allocation_rate: 0.0,
            hit_rate: 0.0,
            memory_efficiency: 1.0,
            established_at: Instant::now(),
            validity_duration: Duration::from_secs(3600),
        }
    }
}

impl Default for PressureThresholds {
    fn default() -> Self {
        Self {
            low_threshold: 0.25,
            medium_threshold: 0.5,
            high_threshold: 0.75,
            critical_threshold: 0.9,
        }
    }
}

impl Default for ResourceSharingAnalysis {
    fn default() -> Self {
        Self {
            opportunities: Vec::new(),
            sharing_efficiency: 0.0,
            potential_savings: 0,
            conflicts: Vec::new(),
        }
    }
}

impl Default for PerformanceImpactAnalysis {
    fn default() -> Self {
        Self {
            impact_measurements: Vec::new(),
            overall_impact: 0.0,
            bottlenecks: Vec::new(),
            recommendations: Vec::new(),
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Type aliases and compatibility types
// ──────────────────────────────────────────────────────────────

/// Cross-pool metrics
#[derive(Debug, Clone, Default)]
pub struct CrossPoolMetrics {
    /// Total allocations across all pools
    pub total_allocations: usize,
    /// Total memory used across pools
    pub total_memory_used: usize,
    /// Sharing efficiency between pools
    pub sharing_efficiency: f64,
}

/// Pool coordination strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolCoordinationStrategy {
    /// Pools operate independently
    Independent,
    /// Cooperative allocation across pools
    Cooperative,
    /// Centralized coordination
    Centralized,
}

impl Default for PoolCoordinationStrategy {
    fn default() -> Self {
        Self::Cooperative
    }
}

/// Resource sharing configuration
#[derive(Debug, Clone)]
pub struct ResourceSharingConfig {
    /// Enable cross-pool sharing
    pub enable_sharing: bool,
    /// Maximum percentage of memory that can be shared
    pub max_sharing_percentage: f64,
    /// Coordination strategy to use
    pub strategy: PoolCoordinationStrategy,
}

impl Default for ResourceSharingConfig {
    fn default() -> Self {
        Self {
            enable_sharing: true,
            max_sharing_percentage: 0.3,
            strategy: PoolCoordinationStrategy::default(),
        }
    }
}

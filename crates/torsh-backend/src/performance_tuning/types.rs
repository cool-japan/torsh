//! Performance Tuning Type Definitions
//!
//! This module contains all the type definitions used throughout the performance tuning
//! system, including core coordination types, backend strategy interfaces, workload
//! characteristics, system state monitoring, and specialized backend implementations.

use crate::backend::BackendType;
use crate::error::BackendResult;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

// ================================================================================================
// Core Coordination Types
// ================================================================================================

/// Performance tuning coordinator for all backends
#[derive(Debug)]
pub struct PerformanceTuningCoordinator {
    /// Backend-specific tuning strategies
    pub strategies: Arc<RwLock<HashMap<BackendType, Box<dyn BackendTuningStrategy + Send + Sync>>>>,
    /// Global performance monitor
    pub global_monitor: Arc<Mutex<GlobalPerformanceMonitor>>,
    /// Workload classifier
    pub workload_classifier: WorkloadClassifier,
    /// Adaptive tuning controller
    pub adaptive_controller: AdaptiveTuningController,
    /// Performance optimization cache
    pub optimization_cache: Arc<RwLock<OptimizationCache>>,
}

/// Trait for backend-specific tuning strategies
pub trait BackendTuningStrategy: std::fmt::Debug + Send + Sync {
    /// Get the backend type this strategy is for
    fn backend_type(&self) -> BackendType;

    /// Tune parameters for a specific workload
    fn tune_for_workload(
        &self,
        workload: &WorkloadCharacteristics,
        system_state: &SystemState,
        constraints: &TuningConstraints,
    ) -> BackendResult<TuningRecommendation>;

    /// Update strategy based on performance feedback
    fn update_from_feedback(&mut self, feedback: &PerformanceFeedback) -> BackendResult<()>;

    /// Get strategy-specific metrics
    fn get_strategy_metrics(&self) -> BackendResult<StrategyMetrics>;

    /// Predict performance for given parameters
    fn predict_performance(
        &self,
        workload: &WorkloadCharacteristics,
        parameters: &TuningParameters,
    ) -> BackendResult<PerformancePrediction>;
}

// ================================================================================================
// Workload and System Characteristics
// ================================================================================================

/// Workload characteristics for tuning decisions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct WorkloadCharacteristics {
    pub operation_type: OperationType,
    pub data_size: usize,
    pub data_shape: Vec<usize>,
    pub data_type: DataType,
    pub access_pattern: AccessPattern,
    pub compute_intensity: f64,
    pub memory_bandwidth_requirement: f64,
    pub parallelization_potential: f64,
    pub cache_locality: f64,
    pub branch_predictability: f64,
    pub vectorization_potential: f64,
}

/// Current system state for tuning decisions
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct SystemState {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub thermal_state: ThermalState,
    pub power_state: PowerState,
    pub concurrent_workloads: usize,
    pub available_memory_bandwidth: f64,
    pub cache_pressure: f64,
    pub numa_topology: NumaTopologyState,
}

/// Thermal state of the system
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ThermalState {
    pub cpu_temperature: f32,
    pub gpu_temperature: Option<f32>,
    pub thermal_throttling_active: bool,
    pub cooling_efficiency: f64,
}

/// Power state of the system
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct PowerState {
    pub power_limit: Option<f32>, // Watts
    pub current_power_draw: f32,
    pub battery_level: Option<f32>, // 0.0 to 1.0
    pub power_efficiency_mode: PowerEfficiencyMode,
}

/// NUMA topology state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct NumaTopologyState {
    pub node_count: usize,
    pub current_node: usize,
    pub memory_distribution: Vec<f64>, // Memory usage per node
    pub cross_node_traffic: f64,
}

/// Tuning constraints
#[derive(Debug, Clone)]
pub struct TuningConstraints {
    pub max_memory_usage: Option<usize>,
    pub max_power_draw: Option<f32>,
    pub max_temperature: Option<f32>,
    pub latency_requirement: Option<Duration>,
    pub throughput_requirement: Option<f64>,
    pub energy_budget: Option<f64>,
    pub real_time_constraints: bool,
}

// ================================================================================================
// Operation and Data Type Enums
// ================================================================================================

/// Types of operations for performance tuning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum OperationType {
    ElementWise,
    MatrixMultiply,
    Convolution2D,
    Convolution3D,
    Pooling,
    Normalization,
    Activation,
    Reduction,
    Sort,
    Scan,
    FFT,
    Gather,
    Scatter,
    Transpose,
    Reshape,
    Custom(u32),
}

/// Data types for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum DataType {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    Complex64,
    Complex128,
}

/// Memory access patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum AccessPattern {
    Sequential,
    Random,
    Strided { stride: usize },
    Blocked { block_size: usize },
    Hierarchical,
    Sparse,
}

/// Power efficiency modes
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum PowerEfficiencyMode {
    MaxPerformance,
    Balanced,
    PowerSaver,
    Custom { performance_ratio: f32 },
}

// ================================================================================================
// Tuning Configuration and Results
// ================================================================================================

/// Tuning recommendation from strategy
#[derive(Debug, Clone)]
pub struct TuningRecommendation {
    pub parameters: TuningParameters,
    pub expected_performance: PerformancePrediction,
    pub confidence_score: f64,
    pub alternative_configs: Vec<TuningParameters>,
    pub reasoning: String,
}

/// Tuning parameters for backend configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct TuningParameters {
    pub thread_count: usize,
    pub vector_width: usize,
    pub block_size: Option<usize>,
    pub tile_size: Option<(usize, usize)>,
    pub unroll_factor: usize,
    pub scheduling_strategy: SchedulingStrategy,
    pub memory_allocation_strategy: MemoryAllocationStrategy,
    pub optimization_level: OptimizationLevel,
    pub backend_specific: HashMap<String, TuningValue>,
}

/// Scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum SchedulingStrategy {
    Static,
    Dynamic,
    Guided,
    WorkStealing,
    NumaAware,
    ThermalAware,
    PowerAware,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum MemoryAllocationStrategy {
    Default,
    Pinned,
    Unified,
    NumaLocal,
    NumaInterleaved,
    LargePages,
}

/// Optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum OptimizationLevel {
    Debug,
    Default,
    Optimized,
    Aggressive,
}

/// Backend-specific tuning values
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum TuningValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Array(Vec<TuningValue>),
}

// ================================================================================================
// Performance Prediction and Feedback
// ================================================================================================

/// Performance prediction result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct PerformancePrediction {
    pub execution_time: Duration,
    pub throughput: f64,
    pub memory_usage: usize,
    pub power_consumption: f32,
    pub cache_efficiency: f64,
    pub thermal_impact: f32,
    pub confidence_interval: (f64, f64),
}

/// Performance feedback for learning
#[derive(Debug, Clone)]
pub struct PerformanceFeedback {
    pub workload: WorkloadCharacteristics,
    pub parameters_used: TuningParameters,
    pub actual_performance: ActualPerformance,
    pub system_state_during_execution: SystemState,
    pub timestamp: Instant,
}

/// Actual measured performance
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ActualPerformance {
    pub execution_time: Duration,
    pub throughput: f64,
    pub memory_usage_peak: usize,
    pub power_consumption_avg: f32,
    pub cache_hit_ratio: f64,
    pub thermal_increase: f32,
    pub cpu_utilization: f64,
}

/// Strategy-specific metrics
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    pub prediction_accuracy: f64,
    pub optimization_success_rate: f64,
    pub average_speedup: f64,
    pub energy_efficiency_improvement: f64,
    pub total_optimizations: usize,
    pub strategy_overhead: Duration,
}

/// Global performance statistics with enhanced analytics
#[derive(Debug, Clone)]
pub struct GlobalPerformanceStats {
    pub total_operations: usize,
    pub average_execution_time: Duration,
    pub overall_throughput: f64,
    pub energy_efficiency: f64,
    pub cache_hit_ratio: f64,
    pub thermal_efficiency: f64,
    pub backend_utilization: HashMap<BackendType, f64>,

    // Enhanced analytics fields
    pub average_efficiency: f64,
    pub memory_utilization: f64,
    pub fragmentation_ratio: f64,
    pub average_latency_ms: f64,
    pub throughput_ops_per_sec: f64,

    // Trend analysis
    pub efficiency_trend: TrendDirection,
    pub throughput_trend: TrendDirection,
    pub latency_trend: TrendDirection,

    // Predictions
    pub predicted_efficiency: f64,
    pub predicted_bottlenecks: Vec<String>,

    // Recommendations
    pub optimization_recommendations: Vec<String>,
}

/// Trend direction for performance metrics
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Declining,
    Stable,
}

/// Performance trend analysis results
#[derive(Debug, Clone)]
pub struct PerformanceTrendAnalysis {
    pub efficiency_trend: TrendDirection,
    pub throughput_trend: TrendDirection,
    pub latency_trend: TrendDirection,
    pub sample_size: usize,
    pub confidence_level: f64,
}

impl PerformanceTrendAnalysis {
    pub fn insufficient_data() -> Self {
        Self {
            efficiency_trend: TrendDirection::Stable,
            throughput_trend: TrendDirection::Stable,
            latency_trend: TrendDirection::Stable,
            sample_size: 0,
            confidence_level: 0.0,
        }
    }
}

/// Performance predictions
#[derive(Debug, Clone)]
pub struct PerformancePredictions {
    pub next_efficiency: f64,
    pub likely_bottlenecks: Vec<String>,
    pub prediction_confidence: f64,
}

/// Performance metric for trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    pub efficiency_score: f64,
    pub throughput_ops_per_sec: f64,
    pub average_latency_ms: f64,
    pub timestamp: Instant,
}

// ================================================================================================
// Backend Strategy Implementations
// ================================================================================================

/// CPU-specific tuning strategy
#[derive(Debug)]
pub struct CpuTuningStrategy {}

/// CUDA-specific tuning strategy
#[derive(Debug)]
pub struct CudaTuningStrategy {}

/// Metal-specific tuning strategy
#[derive(Debug)]
pub struct MetalTuningStrategy {}

/// WebGPU-specific tuning strategy
#[derive(Debug)]
pub struct WebGpuTuningStrategy {}

// ================================================================================================
// Internal Monitoring and Analysis Types
// ================================================================================================

/// Global performance monitor
#[derive(Debug)]
pub struct GlobalPerformanceMonitor {
    pub backend_performance: HashMap<BackendType, BackendPerformanceStats>,
    pub cross_backend_analysis: CrossBackendAnalysis,
    pub system_health_monitor: SystemHealthMonitor,
}

/// Performance statistics per backend
#[derive(Debug, Default)]
pub struct BackendPerformanceStats {
    pub total_operations: usize,
    pub total_execution_time: Duration,
    pub average_throughput: f64,
    pub peak_memory_usage: usize,
    pub thermal_events: usize,
    pub power_efficiency_score: f64,
}

/// Cross-backend performance analysis
#[derive(Debug)]
pub struct CrossBackendAnalysis {
    pub backend_selection_recommendations: HashMap<OperationType, BackendType>,
    pub workload_migration_opportunities: Vec<WorkloadMigrationOpportunity>,
    pub hybrid_execution_strategies: Vec<HybridExecutionStrategy>,
}

/// Workload migration opportunity
#[derive(Debug)]
pub struct WorkloadMigrationOpportunity {
    pub from_backend: BackendType,
    pub to_backend: BackendType,
    pub workload_pattern: WorkloadCharacteristics,
    pub expected_improvement: f64,
    pub migration_cost: f64,
}

/// Hybrid execution strategy
#[derive(Debug)]
pub struct HybridExecutionStrategy {
    pub backends: Vec<BackendType>,
    pub workload_distribution: Vec<f64>, // Percentage of work per backend
    pub coordination_overhead: f64,
    pub expected_performance_gain: f64,
}

/// System health monitoring
#[derive(Debug)]
pub struct SystemHealthMonitor {
    pub thermal_history: Vec<ThermalMeasurement>,
    pub power_history: Vec<PowerMeasurement>,
    pub performance_degradation_events: Vec<PerformanceDegradationEvent>,
}

/// Thermal measurement record
#[derive(Debug, Clone)]
pub struct ThermalMeasurement {
    pub timestamp: Instant,
    pub cpu_temp: f32,
    pub gpu_temp: Option<f32>,
    pub ambient_temp: Option<f32>,
    pub cooling_state: CoolingState,
}

/// Power measurement record
#[derive(Debug, Clone)]
pub struct PowerMeasurement {
    pub timestamp: Instant,
    pub total_power: f32,
    pub cpu_power: f32,
    pub gpu_power: Option<f32>,
    pub memory_power: f32,
    pub efficiency_score: f64,
}

/// Performance degradation event
#[derive(Debug, Clone)]
pub struct PerformanceDegradationEvent {
    pub timestamp: Instant,
    pub backend: BackendType,
    pub workload: WorkloadCharacteristics,
    pub performance_drop: f64,
    pub suspected_cause: DegradationCause,
}

/// Cooling system state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoolingState {
    Passive,
    Active { fan_speed: u8 },
    Liquid { pump_speed: u8 },
    Throttled,
}

/// Suspected causes of performance degradation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradationCause {
    ThermalThrottling,
    PowerLimiting,
    MemoryPressure,
    CacheContention,
    ResourceStarvation,
    SystemLoad,
    Unknown,
}

// ================================================================================================
// Workload Classification Types
// ================================================================================================

/// Workload classifier for automatic optimization
#[derive(Debug)]
pub struct WorkloadClassifier {
    pub classification_models: HashMap<BackendType, Box<dyn ClassificationModel>>,
    pub feature_extractors: Vec<Box<dyn FeatureExtractor + Send + Sync>>,
    pub classification_cache: HashMap<u64, WorkloadClass>,
}

/// Workload classification result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadClass {
    ComputeBound,
    MemoryBound,
    CacheBound,
    IOBound,
    LatencySensitive,
    ThroughputOptimized,
    PowerConstrained,
    ThermalConstrained,
}

/// Classification model interface
pub trait ClassificationModel: std::fmt::Debug + Send + Sync {
    fn classify(&self, features: &[f64]) -> BackendResult<WorkloadClass>;
    fn update(&mut self, features: &[f64], actual_class: WorkloadClass) -> BackendResult<()>;
    fn confidence(&self, features: &[f64]) -> BackendResult<f64>;
}

/// Feature extractor interface
pub trait FeatureExtractor: std::fmt::Debug + Send + Sync {
    fn extract_features(&self, workload: &WorkloadCharacteristics) -> BackendResult<Vec<f64>>;
    fn feature_names(&self) -> Vec<String>;
}

/// Simple linear classification model
#[derive(Debug)]
pub struct LinearClassificationModel {
    pub weights: HashMap<WorkloadClass, Vec<f64>>,
    pub bias: HashMap<WorkloadClass, f64>,
    pub learning_rate: f64,
}

// ================================================================================================
// Adaptive Learning and Control Types
// ================================================================================================

/// Adaptive tuning controller using reinforcement learning principles
#[derive(Debug)]
pub struct AdaptiveTuningController {
    pub exploration_rate: f64,
    pub learning_rate: f64,
    pub discount_factor: f64,
    pub state_action_values: HashMap<(WorkloadClass, TuningParameters), f64>,
    pub experience_replay: Vec<TuningExperience>,
    pub performance_baseline: HashMap<OperationType, f64>,
}

/// Tuning experience for learning
#[derive(Debug, Clone)]
pub struct TuningExperience {
    pub workload_class: WorkloadClass,
    pub parameters: TuningParameters,
    pub reward: f64,
    pub next_state: Option<WorkloadClass>,
}

// ================================================================================================
// Optimization Cache Types
// ================================================================================================

/// Optimization cache for quick lookups
#[derive(Debug)]
pub struct OptimizationCache {
    pub cache: HashMap<u64, CachedOptimization>,
    pub hit_count: usize,
    pub miss_count: usize,
    pub max_entries: usize,
}

/// Cached optimization result
#[derive(Debug, Clone)]
pub struct CachedOptimization {
    pub parameters: TuningParameters,
    pub prediction: PerformancePrediction,
    pub timestamp: Instant,
    pub hit_count: usize,
    pub confidence: f64,
}

// ================================================================================================
// Implementation Support for TuningParameters Hash/Eq
// ================================================================================================

impl PartialEq for TuningParameters {
    fn eq(&self, other: &Self) -> bool {
        self.thread_count == other.thread_count
            && self.vector_width == other.vector_width
            && self.block_size == other.block_size
            && self.tile_size == other.tile_size
            && self.unroll_factor == other.unroll_factor
            && self.scheduling_strategy == other.scheduling_strategy
            && self.memory_allocation_strategy == other.memory_allocation_strategy
            && self.optimization_level == other.optimization_level
        // Note: backend_specific HashMap comparison is simplified for hashing
    }
}

impl Eq for TuningParameters {}

impl std::hash::Hash for TuningParameters {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.thread_count.hash(state);
        self.vector_width.hash(state);
        self.block_size.hash(state);
        self.tile_size.hash(state);
        self.unroll_factor.hash(state);
        self.scheduling_strategy.hash(state);
        self.memory_allocation_strategy.hash(state);
        self.optimization_level.hash(state);
        // Note: backend_specific HashMap is not hashed for simplicity
    }
}

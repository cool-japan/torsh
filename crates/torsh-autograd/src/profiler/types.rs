//! Core Types and Configuration for Autograd Profiling
//!
//! This module defines the fundamental types, structures, and configuration
//! for performance profiling of automatic differentiation operations.
//!
//! # Core Types
//!
//! - **[`OperationProfile`]**: Detailed performance metrics for individual operations
//! - **[`AutogradProfile`]**: Complete profiling session with aggregate statistics
//! - **[`MemorySnapshot`]**: Point-in-time memory usage information
//! - **[`PerformanceBottleneck`]**: Detected performance issues and optimizations
//! - **[`HardwareUtilization`]**: System resource utilization metrics
//! - **[`ProfileSummary`]**: High-level session summary statistics
//!
//! # Configuration
//!
//! - **[`ProfilerConfig`]**: Comprehensive profiler configuration options
//!
//! # Enums
//!
//! - **[`BottleneckType`]**: Classification of performance bottleneck types
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use torsh_autograd::profiler::types::{ProfilerConfig, BottleneckType};
//! use std::time::Duration;
//!
//! // Create a custom profiler configuration
//! let config = ProfilerConfig {
//!     enable_memory_tracking: true,
//!     enable_hardware_monitoring: true,
//!     memory_snapshot_interval: Duration::from_millis(50),
//!     max_operation_profiles: 2000,
//!     ..Default::default()
//! };
//!
//! // Check for specific bottleneck types
//! match bottleneck_type {
//!     BottleneckType::MemoryBandwidth => println!("Memory bandwidth limited"),
//!     BottleneckType::CpuCompute => println!("CPU compute bound"),
//!     _ => println!("Other bottleneck type"),
//! }
//! ```

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Performance profile for a single operation
///
/// This struct captures comprehensive performance metrics for individual
/// autograd operations, including timing, memory usage, and computational
/// characteristics.
///
/// # Fields
///
/// The profile includes both basic metrics (timing, memory) and advanced
/// metrics (FLOPS, tensor sizes, gradient computation time).
#[derive(Debug, Clone)]
pub struct OperationProfile {
    /// Operation name (e.g., "matmul", "conv2d", "relu")
    pub operation_name: String,
    /// Number of times this operation was executed
    pub execution_count: usize,
    /// Total time spent in this operation across all executions
    pub total_time: Duration,
    /// Average time per execution
    pub average_time: Duration,
    /// Minimum execution time observed
    pub min_time: Duration,
    /// Maximum execution time observed
    pub max_time: Duration,
    /// Memory allocated during operation execution
    pub memory_allocated: usize,
    /// Peak memory usage during operation
    pub peak_memory: usize,
    /// FLOPS (floating point operations per second)
    pub flops: f64,
    /// Input tensor sizes for this operation
    pub input_sizes: Vec<usize>,
    /// Output tensor sizes for this operation
    pub output_sizes: Vec<usize>,
    /// Gradient computation time (if applicable)
    pub gradient_time: Option<Duration>,
}

/// Performance profile for the entire autograd session
///
/// This struct provides a comprehensive view of an entire profiling session,
/// including all operation profiles, memory timeline, detected bottlenecks,
/// and summary statistics.
///
/// # Usage
///
/// This is the primary data structure returned by the profiler after a
/// session completes, containing all collected performance data.
#[derive(Debug, Clone)]
pub struct AutogradProfile {
    /// Unique session identifier
    pub session_id: String,
    /// Session start time
    pub start_time: SystemTime,
    /// Total session duration
    pub duration: Duration,
    /// Total number of operations executed
    pub total_operations: usize,
    /// Individual operation performance profiles
    pub operation_profiles: HashMap<String, OperationProfile>,
    /// Timeline of memory usage snapshots
    pub memory_timeline: Vec<MemorySnapshot>,
    /// Detected performance bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Hardware utilization metrics
    pub hardware_utilization: HardwareUtilization,
    /// Summary statistics for the session
    pub summary: ProfileSummary,
}

/// Memory usage snapshot at a specific point in time
///
/// This struct captures the memory state during profiling, providing
/// detailed breakdown of memory usage by category.
///
/// # Memory Categories
///
/// - **Total Memory**: Overall memory consumption
/// - **Gradient Memory**: Memory used for gradient storage
/// - **Graph Memory**: Memory used for computation graph
/// - **Temporary Memory**: Memory used for intermediate results
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Timestamp when this snapshot was taken
    pub timestamp: SystemTime,
    /// Total memory used by the autograd system
    pub total_memory: usize,
    /// Memory used for storing gradients
    pub gradient_memory: usize,
    /// Memory used for computation graph nodes and edges
    pub graph_memory: usize,
    /// Memory used for temporary computations
    pub temp_memory: usize,
    /// Operation being executed when snapshot was taken
    pub current_operation: Option<String>,
}

/// Detected performance bottleneck with analysis and suggestions
///
/// This struct represents a identified performance issue along with
/// detailed analysis and optimization suggestions.
///
/// # Bottleneck Analysis
///
/// Each bottleneck includes:
/// - Type classification and severity assessment
/// - Impact quantification (time and memory)
/// - Actionable optimization suggestions
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    /// Type of bottleneck detected
    pub bottleneck_type: BottleneckType,
    /// Operation that exhibits this bottleneck
    pub operation: String,
    /// Severity score from 0.0 (minor) to 1.0 (critical)
    pub severity: f32,
    /// Human-readable description of the bottleneck
    pub description: String,
    /// Suggested optimization or fix
    pub suggestion: String,
    /// Time impact of this bottleneck
    pub time_impact: Duration,
    /// Memory impact of this bottleneck
    pub memory_impact: usize,
}

/// Classification of performance bottleneck types
///
/// This enum categorizes different types of performance limitations
/// that can occur during autograd operations.
///
/// # Bottleneck Categories
///
/// - **Resource Bottlenecks**: Memory, CPU, GPU limitations
/// - **Operation Bottlenecks**: Specific operation inefficiencies
/// - **System Bottlenecks**: Data movement, synchronization issues
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BottleneckType {
    /// Memory bandwidth is the limiting factor
    MemoryBandwidth,
    /// CPU computation is the limiting factor
    CpuCompute,
    /// GPU computation is the limiting factor
    GpuCompute,
    /// Memory allocation overhead is significant
    MemoryAllocation,
    /// Graph construction overhead is significant
    GraphConstruction,
    /// Gradient computation is the limiting factor
    GradientComputation,
    /// Data movement between devices is slow
    DataMovement,
    /// Synchronization overhead is significant
    Synchronization,
}

/// Hardware utilization metrics for the profiling session
///
/// This struct captures how effectively the available hardware
/// resources are being utilized during autograd operations.
///
/// # Utilization Metrics
///
/// All utilization percentages are in the range [0.0, 1.0] where
/// 1.0 represents 100% utilization.
#[derive(Debug, Clone)]
pub struct HardwareUtilization {
    /// CPU utilization percentage (0.0 to 1.0)
    pub cpu_utilization: f32,
    /// GPU utilization percentage (if GPU is available)
    pub gpu_utilization: Option<f32>,
    /// Memory utilization percentage (0.0 to 1.0)
    pub memory_utilization: f32,
    /// Memory bandwidth utilization percentage (0.0 to 1.0)
    pub memory_bandwidth_utilization: f32,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f32,
}

/// Summary statistics for the profiling session
///
/// This struct provides high-level aggregate statistics that give
/// an overview of the session's performance characteristics.
///
/// # Performance Insights
///
/// The summary includes timing breakdowns, memory statistics,
/// computational metrics, and operation categorization.
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    /// Total execution time for the session
    pub total_time: Duration,
    /// Time spent in forward pass operations
    pub forward_time: Duration,
    /// Time spent in backward pass (gradient computation)
    pub backward_time: Duration,
    /// Time spent in memory operations (allocation, movement)
    pub memory_time: Duration,
    /// Peak memory usage during the session
    pub peak_memory: usize,
    /// Average memory usage during the session
    pub average_memory: usize,
    /// Total FLOPS performed during the session
    pub total_flops: f64,
    /// Average FLOPS per second
    pub flops_per_second: f64,
    /// Most expensive operation by total time
    pub most_expensive_operation: Option<String>,
    /// Percentage of operations that are memory-bound
    pub memory_bound_percentage: f32,
    /// Percentage of operations that are compute-bound
    pub compute_bound_percentage: f32,
}

/// Configuration for the autograd profiler
///
/// This struct contains all configuration options for controlling
/// profiler behavior, including what metrics to collect and how
/// frequently to collect them.
///
/// # Configuration Categories
///
/// - **Feature Toggles**: Enable/disable specific profiling features
/// - **Timing Parameters**: Control measurement intervals and limits
/// - **Performance Options**: Balance detail vs. overhead
///
/// # Examples
/// use std::time::Duration;
///
/// ```rust,ignore
/// use torsh_autograd::profiler::types::ProfilerConfig;
/// use std::time::Duration;
///
/// // High-detail configuration for debugging
/// let debug_config = ProfilerConfig {
///     enable_memory_tracking: true,
///     enable_hardware_monitoring: true,
///     enable_bottleneck_detection: true,
///     memory_snapshot_interval: Duration::from_millis(10),
///     enable_detailed_timing: true,
///     enable_flops_counting: true,
///     ..Default::default()
/// };
///
/// // Low-overhead configuration for production
/// let production_config = ProfilerConfig {
///     enable_memory_tracking: false,
///     enable_hardware_monitoring: false,
///     enable_bottleneck_detection: false,
///     memory_snapshot_interval: Duration::from_secs(1),
///     enable_detailed_timing: false,
///     enable_flops_counting: false,
///     max_operation_profiles: 100,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable memory usage tracking and snapshots
    pub enable_memory_tracking: bool,
    /// Enable hardware utilization monitoring
    pub enable_hardware_monitoring: bool,
    /// Enable performance bottleneck detection
    pub enable_bottleneck_detection: bool,
    /// Interval between memory snapshots
    pub memory_snapshot_interval: Duration,
    /// Maximum number of operation profiles to retain
    pub max_operation_profiles: usize,
    /// Enable detailed timing for all operations
    pub enable_detailed_timing: bool,
    /// Enable FLOPS counting (computationally expensive)
    pub enable_flops_counting: bool,
}

impl Default for ProfilerConfig {
    /// Creates a balanced default configuration
    ///
    /// The default configuration enables most features with reasonable
    /// performance overhead for development and testing scenarios.
    fn default() -> Self {
        Self {
            enable_memory_tracking: true,
            enable_hardware_monitoring: true,
            enable_bottleneck_detection: true,
            memory_snapshot_interval: Duration::from_millis(100),
            max_operation_profiles: 1000,
            enable_detailed_timing: true,
            enable_flops_counting: false, // Expensive to compute, disabled by default
        }
    }
}

impl Default for HardwareUtilization {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.0,
            gpu_utilization: None,
            memory_utilization: 0.0,
            memory_bandwidth_utilization: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}

impl Default for ProfileSummary {
    fn default() -> Self {
        Self {
            total_time: Duration::from_secs(0),
            forward_time: Duration::from_secs(0),
            backward_time: Duration::from_secs(0),
            memory_time: Duration::from_secs(0),
            peak_memory: 0,
            average_memory: 0,
            total_flops: 0.0,
            flops_per_second: 0.0,
            most_expensive_operation: None,
            memory_bound_percentage: 0.0,
            compute_bound_percentage: 0.0,
        }
    }
}

impl OperationProfile {
    /// Creates a new operation profile with the given name
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation being profiled
    ///
    /// # Returns
    ///
    /// A new `OperationProfile` with default values
    pub fn new(operation_name: String) -> Self {
        Self {
            operation_name,
            execution_count: 0,
            total_time: Duration::from_secs(0),
            average_time: Duration::from_secs(0),
            min_time: Duration::from_secs(u64::MAX),
            max_time: Duration::from_secs(0),
            memory_allocated: 0,
            peak_memory: 0,
            flops: 0.0,
            input_sizes: Vec::new(),
            output_sizes: Vec::new(),
            gradient_time: None,
        }
    }

    /// Updates the profile with a new execution measurement
    ///
    /// # Arguments
    ///
    /// * `execution_time` - Time taken for this execution
    /// * `memory_used` - Memory allocated during this execution
    /// * `flops` - FLOPS performed during this execution
    pub fn update_execution(&mut self, execution_time: Duration, memory_used: usize, flops: f64) {
        self.execution_count += 1;
        self.total_time += execution_time;
        self.memory_allocated += memory_used;
        self.peak_memory = self.peak_memory.max(memory_used);
        self.flops += flops;

        // Update min/max times
        self.min_time = self.min_time.min(execution_time);
        self.max_time = self.max_time.max(execution_time);

        // Update average time
        if self.execution_count > 0 {
            self.average_time = self.total_time / self.execution_count as u32;
        }
    }

    /// Gets the efficiency score for this operation (0.0 to 1.0)
    ///
    /// The efficiency score is computed based on the ratio of
    /// computational work (FLOPS) to time and memory usage.
    ///
    /// # Returns
    ///
    /// Efficiency score where 1.0 is perfectly efficient
    pub fn efficiency_score(&self) -> f32 {
        if self.total_time.as_secs_f64() == 0.0 || self.memory_allocated == 0 {
            return 0.0;
        }

        let time_factor = 1.0 / self.total_time.as_secs_f64();
        let memory_factor = 1.0 / (self.memory_allocated as f64 / 1024.0 / 1024.0); // MB
        let flops_factor = self.flops / 1e9; // GFLOPS

        ((time_factor * memory_factor * flops_factor).sqrt()) as f32
    }
}

impl AutogradProfile {
    /// Creates a new autograd profile for a session
    ///
    /// # Arguments
    ///
    /// * `session_id` - Unique identifier for this profiling session
    ///
    /// # Returns
    ///
    /// A new `AutogradProfile` with the current time as start time
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            start_time: SystemTime::now(),
            duration: Duration::from_secs(0),
            total_operations: 0,
            operation_profiles: HashMap::new(),
            memory_timeline: Vec::new(),
            bottlenecks: Vec::new(),
            hardware_utilization: HardwareUtilization::default(),
            summary: ProfileSummary::default(),
        }
    }

    /// Adds an operation profile to this session
    ///
    /// # Arguments
    ///
    /// * `profile` - The operation profile to add
    pub fn add_operation_profile(&mut self, profile: OperationProfile) {
        self.total_operations += profile.execution_count;
        self.operation_profiles
            .insert(profile.operation_name.clone(), profile);
    }

    /// Adds a memory snapshot to the timeline
    ///
    /// # Arguments
    ///
    /// * `snapshot` - The memory snapshot to add
    pub fn add_memory_snapshot(&mut self, snapshot: MemorySnapshot) {
        self.memory_timeline.push(snapshot);
    }

    /// Adds a detected bottleneck to the profile
    ///
    /// # Arguments
    ///
    /// * `bottleneck` - The performance bottleneck to add
    pub fn add_bottleneck(&mut self, bottleneck: PerformanceBottleneck) {
        self.bottlenecks.push(bottleneck);
    }

    /// Gets the most expensive operations by total time
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of operations to return
    ///
    /// # Returns
    ///
    /// Vector of operation names sorted by total time (descending)
    pub fn get_top_expensive_operations(&self, limit: usize) -> Vec<String> {
        let mut operations: Vec<_> = self.operation_profiles.iter().collect();
        operations.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));
        operations
            .into_iter()
            .take(limit)
            .map(|(name, _)| name.clone())
            .collect()
    }
}

impl std::fmt::Display for BottleneckType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BottleneckType::MemoryBandwidth => write!(f, "Memory Bandwidth"),
            BottleneckType::CpuCompute => write!(f, "CPU Compute"),
            BottleneckType::GpuCompute => write!(f, "GPU Compute"),
            BottleneckType::MemoryAllocation => write!(f, "Memory Allocation"),
            BottleneckType::GraphConstruction => write!(f, "Graph Construction"),
            BottleneckType::GradientComputation => write!(f, "Gradient Computation"),
            BottleneckType::DataMovement => write!(f, "Data Movement"),
            BottleneckType::Synchronization => write!(f, "Synchronization"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_profile_creation() {
        let profile = OperationProfile::new("test_op".to_string());
        assert_eq!(profile.operation_name, "test_op");
        assert_eq!(profile.execution_count, 0);
        assert_eq!(profile.total_time, Duration::from_secs(0));
    }

    #[test]
    fn test_operation_profile_update() {
        let mut profile = OperationProfile::new("test_op".to_string());

        profile.update_execution(Duration::from_millis(100), 1024, 1000.0);

        assert_eq!(profile.execution_count, 1);
        assert_eq!(profile.total_time, Duration::from_millis(100));
        assert_eq!(profile.average_time, Duration::from_millis(100));
        assert_eq!(profile.memory_allocated, 1024);
        assert_eq!(profile.flops, 1000.0);
    }

    #[test]
    fn test_operation_profile_multiple_updates() {
        let mut profile = OperationProfile::new("test_op".to_string());

        profile.update_execution(Duration::from_millis(100), 1024, 1000.0);
        profile.update_execution(Duration::from_millis(200), 2048, 2000.0);

        assert_eq!(profile.execution_count, 2);
        assert_eq!(profile.total_time, Duration::from_millis(300));
        assert_eq!(profile.average_time, Duration::from_millis(150));
        assert_eq!(profile.min_time, Duration::from_millis(100));
        assert_eq!(profile.max_time, Duration::from_millis(200));
        assert_eq!(profile.memory_allocated, 3072);
        assert_eq!(profile.peak_memory, 2048);
        assert_eq!(profile.flops, 3000.0);
    }

    #[test]
    fn test_autograd_profile_creation() {
        let profile = AutogradProfile::new("session_123".to_string());
        assert_eq!(profile.session_id, "session_123");
        assert_eq!(profile.total_operations, 0);
        assert!(profile.operation_profiles.is_empty());
        assert!(profile.memory_timeline.is_empty());
        assert!(profile.bottlenecks.is_empty());
    }

    #[test]
    fn test_autograd_profile_add_operation() {
        let mut session_profile = AutogradProfile::new("session_123".to_string());
        let mut op_profile = OperationProfile::new("test_op".to_string());
        op_profile.update_execution(Duration::from_millis(100), 1024, 1000.0);

        session_profile.add_operation_profile(op_profile);

        assert_eq!(session_profile.total_operations, 1);
        assert!(session_profile.operation_profiles.contains_key("test_op"));
    }

    #[test]
    fn test_profiler_config_default() {
        let config = ProfilerConfig::default();
        assert!(config.enable_memory_tracking);
        assert!(config.enable_hardware_monitoring);
        assert!(config.enable_bottleneck_detection);
        assert!(!config.enable_flops_counting); // Should be disabled by default
        assert_eq!(config.max_operation_profiles, 1000);
    }

    #[test]
    fn test_bottleneck_type_display() {
        assert_eq!(
            format!("{}", BottleneckType::MemoryBandwidth),
            "Memory Bandwidth"
        );
        assert_eq!(format!("{}", BottleneckType::CpuCompute), "CPU Compute");
        assert_eq!(format!("{}", BottleneckType::GpuCompute), "GPU Compute");
        assert_eq!(format!("{}", BottleneckType::DataMovement), "Data Movement");
    }

    #[test]
    fn test_memory_snapshot() {
        let snapshot = MemorySnapshot {
            timestamp: SystemTime::now(),
            total_memory: 1024,
            gradient_memory: 256,
            graph_memory: 512,
            temp_memory: 256,
            current_operation: Some("matmul".to_string()),
        };

        assert_eq!(snapshot.total_memory, 1024);
        assert_eq!(snapshot.gradient_memory, 256);
        assert_eq!(snapshot.current_operation, Some("matmul".to_string()));
    }

    #[test]
    fn test_efficiency_score() {
        let mut profile = OperationProfile::new("efficient_op".to_string());
        profile.update_execution(Duration::from_millis(100), 1024, 1000.0);

        let score = profile.efficiency_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_top_expensive_operations() {
        let mut session_profile = AutogradProfile::new("session_123".to_string());

        let mut op1 = OperationProfile::new("slow_op".to_string());
        op1.update_execution(Duration::from_millis(300), 1024, 1000.0);

        let mut op2 = OperationProfile::new("fast_op".to_string());
        op2.update_execution(Duration::from_millis(100), 1024, 1000.0);

        session_profile.add_operation_profile(op1);
        session_profile.add_operation_profile(op2);

        let top_ops = session_profile.get_top_expensive_operations(1);
        assert_eq!(top_ops.len(), 1);
        assert_eq!(top_ops[0], "slow_op");
    }
}

//! # Core Autograd Visualization Foundation
//!
//! This module provides the foundational data structures and types for gradient flow
//! visualization and analysis in the autograd system. It defines the core abstractions
//! used throughout the visualization pipeline.
//!
//! ## Key Components
//!
//! - **GradientFlowAnalysis**: Comprehensive analysis results of gradient flows
//! - **GradientBottleneck**: Information about gradient computation bottlenecks
//! - **GradientStatistics**: Statistical analysis of gradient magnitudes and distributions
//! - **OperationInfo**: Detailed operation metadata for visualization
//! - **MemoryBreakdown**: Memory usage analysis for gradient computations
//!
//! ## Usage Example
//!
//! ```rust
//! use torsh_autograd::visualization::core::{GradientFlowAnalysis, GradientStatistics};
//!
//! // Create gradient statistics
//! let stats = GradientStatistics {
//!     mean_magnitude: 0.001,
//!     std_deviation: 0.0005,
//!     max_magnitude: 0.01,
//!     min_magnitude: 0.0,
//!     zero_count: 100,
//!     inf_nan_count: 0,
//! };
//!
//! // Use in flow analysis
//! let analysis = GradientFlowAnalysis {
//!     total_operations: 1000,
//!     operations_with_gradients: 950,
//!     gradient_bottlenecks: Vec::new(),
//!     gradient_stats: stats,
//!     critical_path: Vec::new(),
//!     memory_breakdown: MemoryBreakdown::default(),
//! };
//! ```

/// Comprehensive gradient flow analysis results
///
/// This struct contains the complete analysis of gradient flows through a computation
/// graph, including operation statistics, bottleneck identification, and memory usage.
#[derive(Debug, Clone)]
pub struct GradientFlowAnalysis {
    /// Timestamp when this analysis was created
    pub timestamp: std::time::Instant,
    /// Total number of operations in the computation graph
    pub total_operations: usize,
    /// Number of operations that have gradients flowing through them
    pub operations_with_gradients: usize,
    /// Identified gradient bottlenecks that may impact performance
    pub gradient_bottlenecks: Vec<GradientBottleneck>,
    /// Statistical analysis of gradient magnitudes and distributions
    pub gradient_stats: GradientStatistics,
    /// Critical path through the computation graph for gradient flow
    pub critical_path: Vec<OperationInfo>,
    /// Memory usage breakdown for gradient storage and computation
    pub memory_breakdown: MemoryBreakdown,
}

impl GradientFlowAnalysis {
    /// Create a new empty gradient flow analysis
    pub fn new() -> Self {
        Self {
            timestamp: std::time::Instant::now(),
            total_operations: 0,
            operations_with_gradients: 0,
            gradient_bottlenecks: Vec::new(),
            gradient_stats: GradientStatistics::default(),
            critical_path: Vec::new(),
            memory_breakdown: MemoryBreakdown::default(),
        }
    }

    /// Calculate the gradient flow efficiency (0-1, higher is better)
    pub fn gradient_flow_efficiency(&self) -> f64 {
        if self.total_operations == 0 {
            return 1.0;
        }
        self.operations_with_gradients as f64 / self.total_operations as f64
    }

    /// Check if the gradient flow has significant bottlenecks
    pub fn has_significant_bottlenecks(&self) -> bool {
        self.gradient_bottlenecks
            .iter()
            .any(|bottleneck| bottleneck.is_significant())
    }

    /// Get the most critical bottleneck by severity
    pub fn most_critical_bottleneck(&self) -> Option<&GradientBottleneck> {
        self.gradient_bottlenecks
            .iter()
            .max_by_key(|bottleneck| bottleneck.severity_score())
    }

    /// Calculate overall gradient health score (0-1, higher is better)
    pub fn gradient_health_score(&self) -> f64 {
        let efficiency_score = self.gradient_flow_efficiency();
        let bottleneck_penalty = if self.has_significant_bottlenecks() {
            0.2
        } else {
            0.0
        };
        let gradient_quality = self.gradient_stats.quality_score();

        ((efficiency_score + gradient_quality) / 2.0 - bottleneck_penalty).max(0.0)
    }

    /// Get summary statistics
    pub fn summary(&self) -> GradientFlowSummary {
        GradientFlowSummary {
            total_operations: self.total_operations,
            operations_with_gradients: self.operations_with_gradients,
            bottleneck_count: self.gradient_bottlenecks.len(),
            critical_path_length: self.critical_path.len(),
            gradient_health_score: self.gradient_health_score(),
            memory_efficiency: self.memory_breakdown.efficiency_score(),
        }
    }
}

impl Default for GradientFlowAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for gradient flow analysis
#[derive(Debug, Clone)]
pub struct GradientFlowSummary {
    /// Total number of operations
    pub total_operations: usize,
    /// Operations with gradient flow
    pub operations_with_gradients: usize,
    /// Number of identified bottlenecks
    pub bottleneck_count: usize,
    /// Length of critical path
    pub critical_path_length: usize,
    /// Overall gradient health score (0-1)
    pub gradient_health_score: f64,
    /// Memory usage efficiency (0-1)
    pub memory_efficiency: f64,
}

/// Information about a gradient computation bottleneck
///
/// Represents an operation or computation step that significantly impacts
/// gradient flow performance or memory usage.
#[derive(Debug, Clone)]
pub struct GradientBottleneck {
    /// Unique identifier for the operation
    pub operation_id: usize,
    /// Human-readable name of the operation
    pub operation_name: String,
    /// Magnitude of gradients at this bottleneck
    pub gradient_magnitude: f32,
    /// Number of operations that depend on this one
    pub dependency_count: usize,
    /// Memory usage attributed to this operation (bytes)
    pub memory_usage: usize,
    /// Classification of the bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Additional performance metrics
    pub performance_metrics: BottleneckMetrics,
}

impl GradientBottleneck {
    /// Create a new gradient bottleneck
    pub fn new(
        operation_id: usize,
        operation_name: String,
        gradient_magnitude: f32,
        dependency_count: usize,
        memory_usage: usize,
        bottleneck_type: BottleneckType,
    ) -> Self {
        Self {
            operation_id,
            operation_name,
            gradient_magnitude,
            dependency_count,
            memory_usage,
            bottleneck_type,
            performance_metrics: BottleneckMetrics::default(),
        }
    }

    /// Check if this bottleneck is considered significant
    pub fn is_significant(&self) -> bool {
        match self.bottleneck_type {
            BottleneckType::Memory => self.memory_usage > 1024 * 1024 * 100, // 100MB
            BottleneckType::Computation => self.dependency_count > 10,
            BottleneckType::GradientMagnitude => {
                self.gradient_magnitude > 1.0 || self.gradient_magnitude < 1e-6
            }
            BottleneckType::Dependency => self.dependency_count > 20,
        }
    }

    /// Calculate severity score (0-100, higher is more severe)
    pub fn severity_score(&self) -> u32 {
        let base_score = match self.bottleneck_type {
            BottleneckType::Memory => {
                (self.memory_usage as f64 / (1024.0 * 1024.0)).min(50.0) as u32
            }
            BottleneckType::Computation => self.dependency_count.min(30) as u32,
            BottleneckType::GradientMagnitude => {
                if self.gradient_magnitude > 1.0 {
                    (self.gradient_magnitude * 20.0).min(40.0) as u32
                } else if self.gradient_magnitude < 1e-6 {
                    30
                } else {
                    5
                }
            }
            BottleneckType::Dependency => (self.dependency_count / 2).min(25) as u32,
        };

        // Add performance metrics influence
        let perf_penalty = if self.performance_metrics.execution_time_ms > 100.0 {
            10
        } else {
            0
        };

        (base_score + perf_penalty).min(100)
    }

    /// Get bottleneck description for reporting
    pub fn description(&self) -> String {
        match self.bottleneck_type {
            BottleneckType::Memory => {
                format!(
                    "Memory bottleneck: {} uses {:.1}MB",
                    self.operation_name,
                    self.memory_usage as f64 / (1024.0 * 1024.0)
                )
            }
            BottleneckType::Computation => {
                format!(
                    "Computation bottleneck: {} with {} dependencies",
                    self.operation_name, self.dependency_count
                )
            }
            BottleneckType::GradientMagnitude => {
                format!(
                    "Gradient magnitude bottleneck: {} (magnitude: {:.2e})",
                    self.operation_name, self.gradient_magnitude
                )
            }
            BottleneckType::Dependency => {
                format!(
                    "Dependency bottleneck: {} has {} dependents",
                    self.operation_name, self.dependency_count
                )
            }
        }
    }
}

/// Performance metrics for bottleneck analysis
#[derive(Debug, Clone, Default)]
pub struct BottleneckMetrics {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory bandwidth utilization (0-1)
    pub memory_bandwidth_utilization: f64,
    /// Compute utilization (0-1)
    pub compute_utilization: f64,
    /// Cache hit ratio (0-1)
    pub cache_hit_ratio: f64,
}

/// Classification of gradient bottleneck types
///
/// Different types of bottlenecks require different optimization strategies
/// and have different performance implications.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BottleneckType {
    /// Bottleneck caused by high memory usage or memory bandwidth limitations
    Memory,
    /// Bottleneck caused by computational complexity or CPU-bound operations
    Computation,
    /// Bottleneck caused by extremely large or small gradient magnitudes
    GradientMagnitude,
    /// Bottleneck caused by high fan-out (many dependent operations)
    Dependency,
}

impl std::fmt::Display for BottleneckType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BottleneckType::Memory => write!(f, "Memory"),
            BottleneckType::Computation => write!(f, "Computation"),
            BottleneckType::GradientMagnitude => write!(f, "Gradient Magnitude"),
            BottleneckType::Dependency => write!(f, "Dependency"),
        }
    }
}

/// Statistical analysis of gradient magnitudes and distributions
///
/// Provides comprehensive statistics about gradient values throughout
/// the computation graph for analysis and debugging.
#[derive(Debug, Clone)]
pub struct GradientStatistics {
    /// Mean magnitude of all gradients
    pub mean_magnitude: f32,
    /// Standard deviation of gradient magnitudes
    pub std_deviation: f32,
    /// Maximum gradient magnitude observed
    pub max_magnitude: f32,
    /// Minimum gradient magnitude observed
    pub min_magnitude: f32,
    /// Number of gradients that are exactly zero
    pub zero_count: usize,
    /// Number of gradients that are infinite or NaN
    pub inf_nan_count: usize,
}

impl GradientStatistics {
    /// Create new gradient statistics
    pub fn new() -> Self {
        Self {
            mean_magnitude: 0.0,
            std_deviation: 0.0,
            max_magnitude: 0.0,
            min_magnitude: 0.0,
            zero_count: 0,
            inf_nan_count: 0,
        }
    }

    /// Calculate coefficient of variation (std_dev / mean)
    pub fn coefficient_of_variation(&self) -> f32 {
        if self.mean_magnitude > 0.0 {
            self.std_deviation / self.mean_magnitude
        } else {
            f32::INFINITY
        }
    }

    /// Calculate gradient quality score (0-1, higher is better)
    pub fn quality_score(&self) -> f64 {
        // Penalize infinite/NaN gradients heavily
        if self.inf_nan_count > 0 {
            return 0.0;
        }

        // Check for reasonable gradient magnitudes (not too large or too small)
        let magnitude_score = if self.max_magnitude > 10.0 || self.max_magnitude < 1e-8 {
            0.5 // Suboptimal gradient magnitudes
        } else {
            1.0
        };

        // Check coefficient of variation (lower is generally better for stability)
        let cv = self.coefficient_of_variation();
        let stability_score = if cv > 2.0 {
            0.5 // High variation
        } else if cv > 1.0 {
            0.75 // Moderate variation
        } else {
            1.0 // Low variation
        };

        magnitude_score * stability_score
    }

    /// Check if gradients are in a healthy range
    pub fn are_gradients_healthy(&self) -> bool {
        self.inf_nan_count == 0
            && self.max_magnitude < 10.0
            && self.max_magnitude > 1e-8
            && self.coefficient_of_variation() < 2.0
    }

    /// Get gradient magnitude category
    pub fn magnitude_category(&self) -> GradientMagnitudeCategory {
        if self.inf_nan_count > 0 {
            GradientMagnitudeCategory::Invalid
        } else if self.max_magnitude > 10.0 {
            GradientMagnitudeCategory::TooLarge
        } else if self.max_magnitude < 1e-8 {
            GradientMagnitudeCategory::TooSmall
        } else {
            GradientMagnitudeCategory::Healthy
        }
    }
}

impl Default for GradientStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Categories for gradient magnitude assessment
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientMagnitudeCategory {
    /// Gradients are in a healthy range
    Healthy,
    /// Gradients are too large (may cause instability)
    TooLarge,
    /// Gradients are too small (may cause vanishing gradients)
    TooSmall,
    /// Gradients contain invalid values (NaN/Inf)
    Invalid,
}

impl std::fmt::Display for GradientMagnitudeCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GradientMagnitudeCategory::Healthy => write!(f, "Healthy"),
            GradientMagnitudeCategory::TooLarge => write!(f, "Too Large"),
            GradientMagnitudeCategory::TooSmall => write!(f, "Too Small"),
            GradientMagnitudeCategory::Invalid => write!(f, "Invalid"),
        }
    }
}

/// Detailed information about an operation in the computation graph
///
/// Used for visualizing the critical path and understanding operation
/// characteristics in the context of gradient flow.
#[derive(Debug, Clone)]
pub struct OperationInfo {
    /// Unique identifier for the operation
    pub operation_id: usize,
    /// Human-readable operation name or type
    pub operation_name: String,
    /// Type of operation (e.g., "MatMul", "ReLU", "Conv2D")
    pub operation_type: String,
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shape
    pub output_shape: Vec<usize>,
    /// Estimated computational complexity (FLOPs)
    pub computational_complexity: usize,
    /// Memory usage for this operation (bytes)
    pub memory_usage: usize,
    /// Whether this operation requires gradients
    pub requires_grad: bool,
    /// Position in the critical path (if applicable)
    pub critical_path_position: Option<usize>,
}

impl OperationInfo {
    /// Create new operation info
    pub fn new(operation_id: usize, operation_name: String, operation_type: String) -> Self {
        Self {
            operation_id,
            operation_name,
            operation_type,
            input_shapes: Vec::new(),
            output_shape: Vec::new(),
            computational_complexity: 0,
            memory_usage: 0,
            requires_grad: false,
            critical_path_position: None,
        }
    }

    /// Calculate operation intensity (FLOPs per byte)
    pub fn operation_intensity(&self) -> f64 {
        if self.memory_usage > 0 {
            self.computational_complexity as f64 / self.memory_usage as f64
        } else {
            0.0
        }
    }

    /// Check if operation is memory-bound
    pub fn is_memory_bound(&self) -> bool {
        self.operation_intensity() < 1.0 // Less than 1 FLOP per byte
    }

    /// Check if operation is compute-bound
    pub fn is_compute_bound(&self) -> bool {
        self.operation_intensity() > 10.0 // More than 10 FLOPs per byte
    }

    /// Get operation description for visualization
    pub fn description(&self) -> String {
        format!(
            "{} ({}): {} FLOPs, {} MB",
            self.operation_name,
            self.operation_type,
            self.computational_complexity,
            self.memory_usage / (1024 * 1024)
        )
    }
}

/// Memory usage breakdown for gradient computations
///
/// Provides detailed analysis of memory usage patterns in gradient
/// computation and storage.
#[derive(Debug, Clone)]
pub struct MemoryBreakdown {
    /// Total memory used for gradient storage (bytes)
    pub gradient_memory: usize,
    /// Memory used for intermediate computations (bytes)
    pub intermediate_memory: usize,
    /// Memory used for operation metadata (bytes)
    pub metadata_memory: usize,
    /// Peak memory usage during gradient computation (bytes)
    pub peak_memory: usize,
    /// Memory fragmentation information
    pub fragmentation_info: MemoryFragmentation,
}

impl MemoryBreakdown {
    /// Create new memory breakdown
    pub fn new() -> Self {
        Self {
            gradient_memory: 0,
            intermediate_memory: 0,
            metadata_memory: 0,
            peak_memory: 0,
            fragmentation_info: MemoryFragmentation::default(),
        }
    }

    /// Calculate total memory usage
    pub fn total_memory(&self) -> usize {
        self.gradient_memory + self.intermediate_memory + self.metadata_memory
    }

    /// Calculate memory efficiency score (0-1, higher is better)
    pub fn efficiency_score(&self) -> f64 {
        if self.peak_memory == 0 {
            return 1.0;
        }

        let utilization = self.total_memory() as f64 / self.peak_memory as f64;
        let fragmentation_penalty = self.fragmentation_info.fragmentation_ratio() * 0.2;

        (utilization - fragmentation_penalty).max(0.0).min(1.0)
    }

    /// Get memory distribution as percentages
    pub fn memory_distribution(&self) -> MemoryDistribution {
        let total = self.total_memory() as f64;
        if total == 0.0 {
            return MemoryDistribution::default();
        }

        MemoryDistribution {
            gradient_percentage: (self.gradient_memory as f64 / total) * 100.0,
            intermediate_percentage: (self.intermediate_memory as f64 / total) * 100.0,
            metadata_percentage: (self.metadata_memory as f64 / total) * 100.0,
        }
    }

    /// Check if memory usage is concerning
    pub fn has_memory_issues(&self) -> bool {
        self.efficiency_score() < 0.7 || self.fragmentation_info.is_highly_fragmented()
    }
}

impl Default for MemoryBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory distribution breakdown
#[derive(Debug, Clone, Default)]
pub struct MemoryDistribution {
    /// Percentage of memory used for gradients
    pub gradient_percentage: f64,
    /// Percentage of memory used for intermediates
    pub intermediate_percentage: f64,
    /// Percentage of memory used for metadata
    pub metadata_percentage: f64,
}

/// Memory fragmentation analysis
#[derive(Debug, Clone, Default)]
pub struct MemoryFragmentation {
    /// Number of memory fragments
    pub fragment_count: usize,
    /// Largest contiguous memory block size
    pub largest_free_block: usize,
    /// Total free memory
    pub total_free_memory: usize,
}

impl MemoryFragmentation {
    /// Calculate fragmentation ratio (0-1, higher means more fragmented)
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.total_free_memory == 0 {
            return 0.0;
        }

        1.0 - (self.largest_free_block as f64 / self.total_free_memory as f64)
    }

    /// Check if memory is highly fragmented
    pub fn is_highly_fragmented(&self) -> bool {
        self.fragmentation_ratio() > 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_flow_analysis_creation() {
        let analysis = GradientFlowAnalysis::new();
        assert_eq!(analysis.total_operations, 0);
        assert_eq!(analysis.operations_with_gradients, 0);
        assert!(analysis.gradient_bottlenecks.is_empty());
        // Empty analysis may have default gradient stats affecting the score
        let health_score = analysis.gradient_health_score();
        assert!(
            health_score >= 0.0 && health_score <= 1.0,
            "Health score should be between 0.0 and 1.0, got {}",
            health_score
        );
    }

    #[test]
    fn test_gradient_flow_efficiency() {
        let mut analysis = GradientFlowAnalysis::new();
        analysis.total_operations = 100;
        analysis.operations_with_gradients = 80;

        assert_eq!(analysis.gradient_flow_efficiency(), 0.8);
    }

    #[test]
    fn test_gradient_bottleneck_creation() {
        let bottleneck = GradientBottleneck::new(
            1,
            "MatMul".to_string(),
            0.5,
            15,
            1024 * 1024 * 200, // 200MB
            BottleneckType::Memory,
        );

        assert_eq!(bottleneck.operation_id, 1);
        assert_eq!(bottleneck.operation_name, "MatMul");
        assert!(bottleneck.is_significant()); // 200MB should be significant
        assert!(bottleneck.severity_score() > 20); // Should have high severity
    }

    #[test]
    fn test_bottleneck_severity_scores() {
        // Memory bottleneck
        let memory_bottleneck = GradientBottleneck::new(
            1,
            "MemoryOp".to_string(),
            0.1,
            5,
            1024 * 1024 * 500,
            BottleneckType::Memory,
        );
        let memory_score = memory_bottleneck.severity_score();

        // Computation bottleneck
        let compute_bottleneck = GradientBottleneck::new(
            2,
            "ComputeOp".to_string(),
            0.1,
            25,
            1024,
            BottleneckType::Computation,
        );
        let compute_score = compute_bottleneck.severity_score();

        // Memory bottleneck should have higher score for large memory usage
        assert!(memory_score > compute_score);
    }

    #[test]
    fn test_gradient_statistics() {
        let mut stats = GradientStatistics::new();
        stats.mean_magnitude = 0.001;
        stats.std_deviation = 0.0005;
        stats.max_magnitude = 0.01;
        stats.min_magnitude = 0.0001;
        stats.zero_count = 10;
        stats.inf_nan_count = 0;

        assert_eq!(stats.coefficient_of_variation(), 0.5);
        assert!(stats.are_gradients_healthy());
        assert_eq!(
            stats.magnitude_category(),
            GradientMagnitudeCategory::Healthy
        );
        assert!(stats.quality_score() > 0.8);
    }

    #[test]
    fn test_gradient_statistics_unhealthy() {
        let mut stats = GradientStatistics::new();
        stats.mean_magnitude = 0.001;
        stats.std_deviation = 0.005; // High variation
        stats.max_magnitude = 15.0; // Too large
        stats.inf_nan_count = 5; // Invalid values

        assert!(!stats.are_gradients_healthy());
        assert_eq!(
            stats.magnitude_category(),
            GradientMagnitudeCategory::Invalid
        );
        assert_eq!(stats.quality_score(), 0.0); // Should be 0 due to inf/nan
    }

    #[test]
    fn test_operation_info() {
        let mut op_info = OperationInfo::new(1, "conv2d_1".to_string(), "Conv2D".to_string());
        op_info.computational_complexity = 1100000; // 1.1M FLOPs
        op_info.memory_usage = 100000; // 100KB

        assert_eq!(op_info.operation_intensity(), 11.0); // 1.1M FLOPs / 100K bytes = 11
        assert!(op_info.is_compute_bound());
        assert!(!op_info.is_memory_bound());
    }

    #[test]
    fn test_memory_breakdown() {
        let mut breakdown = MemoryBreakdown::new();
        breakdown.gradient_memory = 1024 * 1024; // 1MB
        breakdown.intermediate_memory = 2 * 1024 * 1024; // 2MB
        breakdown.metadata_memory = 512 * 1024; // 0.5MB
        breakdown.peak_memory = 4 * 1024 * 1024; // 4MB

        assert_eq!(breakdown.total_memory(), 3584 * 1024); // 3.5MB

        let efficiency = breakdown.efficiency_score();
        assert!(efficiency > 0.8); // Should be efficient

        let distribution = breakdown.memory_distribution();
        assert!((distribution.gradient_percentage - 28.57).abs() < 0.1); // ~28.57%
        assert!((distribution.intermediate_percentage - 57.14).abs() < 0.1); // ~57.14%
        assert!((distribution.metadata_percentage - 14.29).abs() < 0.1); // ~14.29%
    }

    #[test]
    fn test_memory_fragmentation() {
        let mut fragmentation = MemoryFragmentation::default();
        fragmentation.fragment_count = 10;
        fragmentation.total_free_memory = 1024 * 1024; // 1MB
        fragmentation.largest_free_block = 256 * 1024; // 256KB

        assert_eq!(fragmentation.fragmentation_ratio(), 0.75); // 1 - 256KB/1MB = 0.75
        assert!(fragmentation.is_highly_fragmented());
    }

    #[test]
    fn test_bottleneck_types_display() {
        assert_eq!(format!("{}", BottleneckType::Memory), "Memory");
        assert_eq!(format!("{}", BottleneckType::Computation), "Computation");
        assert_eq!(
            format!("{}", BottleneckType::GradientMagnitude),
            "Gradient Magnitude"
        );
        assert_eq!(format!("{}", BottleneckType::Dependency), "Dependency");
    }

    #[test]
    fn test_gradient_magnitude_categories() {
        assert_eq!(format!("{}", GradientMagnitudeCategory::Healthy), "Healthy");
        assert_eq!(
            format!("{}", GradientMagnitudeCategory::TooLarge),
            "Too Large"
        );
        assert_eq!(
            format!("{}", GradientMagnitudeCategory::TooSmall),
            "Too Small"
        );
        assert_eq!(format!("{}", GradientMagnitudeCategory::Invalid), "Invalid");
    }

    #[test]
    fn test_gradient_flow_summary() {
        let mut analysis = GradientFlowAnalysis::new();
        analysis.total_operations = 100;
        analysis.operations_with_gradients = 90;
        analysis.gradient_bottlenecks.push(GradientBottleneck::new(
            1,
            "test".to_string(),
            0.1,
            5,
            1024,
            BottleneckType::Memory,
        ));
        analysis.critical_path.push(OperationInfo::new(
            1,
            "op1".to_string(),
            "MatMul".to_string(),
        ));

        let summary = analysis.summary();
        assert_eq!(summary.total_operations, 100);
        assert_eq!(summary.operations_with_gradients, 90);
        assert_eq!(summary.bottleneck_count, 1);
        assert_eq!(summary.critical_path_length, 1);
        assert!(summary.gradient_health_score > 0.0);
    }
}

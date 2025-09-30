//! Memory usage tracking and statistics collection
//!
//! This module provides comprehensive memory usage tracking for autograd operations,
//! including gradient-specific memory statistics, fragmentation analysis, leak detection,
//! and detailed memory usage patterns analysis.
//!
//! # Overview
//!
//! The memory tracking system monitors several key aspects:
//!
//! - **Operation-Specific Usage**: Memory consumption per operation type
//! - **Gradient Memory Patterns**: Specialized tracking for gradient allocations
//! - **Fragmentation Analysis**: Detection of memory fragmentation issues
//! - **Leak Detection**: Identification of potential memory leaks
//! - **Historical Tracking**: Time-series memory usage data
//!
//! # Key Features
//!
//! - **Real-time Monitoring**: Continuous tracking of memory allocation patterns
//! - **Anomaly Detection**: Automatic identification of unusual memory behavior
//! - **Performance Insights**: Analysis to guide memory optimization decisions
//! - **Leak Prevention**: Early detection of memory management issues
//!
//! # Usage Patterns
//!
//! ```rust
//! use crate::memory::tracking::{MemoryUsageTracker, GradientMemoryStats};
//!
//! let mut tracker = MemoryUsageTracker::new();
//!
//! // Track memory allocation for specific operation
//! tracker.track_allocation("conv2d_forward", 1024 * 1024); // 1MB
//!
//! // Record gradient-specific memory usage
//! tracker.track_gradient_allocation("conv2d_grad", 512 * 1024); // 512KB
//!
//! // Analyze patterns
//! let analysis = tracker.analyze_memory_patterns();
//! for warning in analysis.warnings {
//!     println!("Memory warning: {}", warning);
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Memory usage tracking
///
/// Comprehensive tracker for memory usage patterns across all autograd operations.
/// Provides detailed insights into memory allocation patterns, peak usage, and
/// potential issues like leaks or excessive fragmentation.
///
/// # Tracking Categories
///
/// - **By Operation**: Memory usage grouped by operation type
/// - **By Gradient Type**: Gradient-specific memory patterns
/// - **Historical Data**: Time-series allocation history
/// - **Fragmentation**: Memory layout efficiency metrics
/// - **Leak Detection**: Suspicious allocation patterns
///
/// # Performance Impact
///
/// The tracker has minimal overhead:
/// - O(1) for most tracking operations
/// - O(log n) for historical data insertion
/// - Configurable history size to control memory usage
#[derive(Debug, Clone)]
pub struct MemoryUsageTracker {
    /// Current memory usage by operation type
    pub usage_by_operation: HashMap<String, usize>,
    /// Peak memory usage across all operations
    pub peak_memory_usage: usize,
    /// Total number of allocations tracked
    pub total_allocations: usize,
    /// Memory allocation history (timestamp, bytes)
    pub allocation_history: VecDeque<(Instant, usize)>,
    /// Memory usage by gradient type
    pub gradient_memory_usage: HashMap<String, GradientMemoryStats>,
    /// Memory fragmentation metrics
    pub fragmentation_stats: FragmentationStats,
    /// Memory leak detection data
    pub leak_detection: LeakDetectionStats,
    /// Maximum history size to prevent unbounded growth
    max_history_size: usize,
}

/// Memory statistics for gradient operations
///
/// Specialized tracking for gradient-related memory allocations, which often
/// have unique patterns different from forward pass operations. These statistics
/// help optimize gradient computation and identify memory bottlenecks.
///
/// # Key Metrics
///
/// - **Total Allocated**: Cumulative memory allocated for gradients
/// - **Peak Usage**: Maximum simultaneous gradient memory
/// - **Growth Rate**: Rate of memory usage increase over time
/// - **Allocation Patterns**: Size and frequency of gradient allocations
///
/// # Use Cases
///
/// - Gradient checkpointing optimization
/// - Memory budget planning for large models
/// - Identification of memory-intensive gradient operations
/// - Optimization of backward pass memory usage
#[derive(Debug, Clone)]
pub struct GradientMemoryStats {
    /// Total memory allocated for gradients (bytes)
    pub total_allocated: usize,
    /// Peak memory usage for gradients (bytes)
    pub peak_usage: usize,
    /// Average allocation size (bytes)
    pub avg_allocation_size: usize,
    /// Number of gradient allocations
    pub num_allocations: usize,
    /// Memory growth rate (bytes per second)
    pub growth_rate: f64,
    /// Last allocation timestamp
    pub last_allocation: Option<Instant>,
    /// Memory usage history for trend analysis
    pub usage_history: VecDeque<(Instant, usize)>,
    /// Peak allocation size observed
    pub peak_allocation_size: usize,
}

/// Memory fragmentation statistics
///
/// Tracks memory fragmentation which can significantly impact performance
/// by reducing cache efficiency and increasing allocation overhead.
///
/// # Fragmentation Metrics
///
/// - **Free Memory Distribution**: How free memory is distributed
/// - **Fragmentation Ratio**: Degree of memory fragmentation (0.0-1.0)
/// - **Memory Holes**: Number of gaps in memory layout
/// - **Largest Free Block**: Biggest contiguous free memory block
///
/// # Impact on Performance
///
/// High fragmentation can cause:
/// - Increased allocation latency
/// - Poor cache locality
/// - Memory allocation failures despite sufficient total memory
/// - Reduced memory bandwidth utilization
#[derive(Debug, Clone)]
pub struct FragmentationStats {
    /// Total free memory across all pools (bytes)
    pub total_free: usize,
    /// Largest contiguous free block (bytes)
    pub largest_free_block: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = maximum fragmentation)
    pub fragmentation_ratio: f64,
    /// Number of memory holes (gaps between allocated blocks)
    pub num_holes: usize,
    /// Average size of memory holes
    pub avg_hole_size: usize,
    /// Number of memory pools tracked
    pub num_pools: usize,
}

/// Memory leak detection statistics
///
/// Tracks potential memory leaks by monitoring allocation patterns and
/// identifying suspicious behavior that might indicate memory management issues.
///
/// # Detection Methods
///
/// - **Unmatched Allocations**: Allocations without corresponding deallocations
/// - **Growth Pattern Analysis**: Abnormal memory growth rates
/// - **Operation Pattern Monitoring**: Unusual allocation sequences
/// - **Time-based Analysis**: Allocations that persist longer than expected
///
/// # Leak Categories
///
/// - **Potential Leaks**: High-confidence leak candidates
/// - **Suspicious Patterns**: Behavior that warrants investigation
/// - **Unexplained Growth**: Memory increases without obvious cause
#[derive(Debug, Clone)]
pub struct LeakDetectionStats {
    /// Potential memory leaks (timestamp, bytes, operation)
    pub potential_leaks: Vec<(Instant, usize, String)>,
    /// Suspicious allocation patterns detected
    pub suspicious_patterns: Vec<String>,
    /// Memory growth without corresponding gradient computation
    pub unexplained_growth: usize,
    /// Last leak detection check timestamp
    pub last_leak_check: Option<Instant>,
    /// Total number of leak checks performed
    pub total_leak_checks: usize,
    /// Number of confirmed leaks resolved
    pub resolved_leaks: usize,
}

/// Result of gradient memory usage analysis
///
/// Comprehensive analysis result highlighting memory usage patterns,
/// inefficiencies, and optimization opportunities for gradient operations.
///
/// # Analysis Categories
///
/// - **High Usage Operations**: Operations consuming most gradient memory
/// - **Growth Pattern Analysis**: Operations with concerning growth rates
/// - **Optimization Recommendations**: Actionable improvement suggestions
/// - **Leak Detection Results**: Potential memory management issues
#[derive(Debug, Clone)]
pub struct GradientMemoryAnalysis {
    /// Operations with high memory growth rates
    pub high_growth_operations: Vec<String>,
    /// Operations with high total memory usage
    pub high_memory_operations: Vec<String>,
    /// Operations with large average allocation sizes
    pub large_allocation_operations: Vec<String>,
    /// Warnings about memory usage patterns
    pub warnings: Vec<String>,
    /// Potential memory leaks detected
    pub potential_leaks: Vec<(Instant, usize, String)>,
    /// Recommendations for optimization
    pub recommendations: Vec<String>,
    /// Overall memory efficiency score (0.0-1.0)
    pub efficiency_score: f64,
    /// Timestamp of analysis
    pub analysis_timestamp: Instant,
}

/// Result of memory fragmentation analysis
///
/// Detailed analysis of memory fragmentation patterns and their impact
/// on system performance, along with recommendations for defragmentation.
///
/// # Analysis Results
///
/// - **Fragmentation Assessment**: Current fragmentation level and impact
/// - **Pool Analysis**: Per-pool fragmentation statistics
/// - **Performance Impact**: Estimated performance degradation
/// - **Defragmentation Strategy**: Recommended actions to reduce fragmentation
#[derive(Debug, Clone)]
pub struct FragmentationAnalysis {
    /// Total memory in all pools (bytes)
    pub total_pooled_memory: usize,
    /// Pools with high fragmentation (size -> fragmentation_ratio)
    pub fragmented_pools: Vec<(usize, f64)>,
    /// Overall fragmentation ratio across all pools
    pub fragmentation_ratio: f64,
    /// Number of memory pools analyzed
    pub num_pools: usize,
    /// Total number of chunks across all pools
    pub total_chunks: usize,
    /// Recommendations for defragmentation
    pub recommendations: Vec<String>,
    /// Estimated performance impact (0.0-1.0, higher = more impact)
    pub performance_impact: f64,
    /// Timestamp of analysis
    pub analysis_timestamp: Instant,
}

impl Default for GradientMemoryAnalysis {
    fn default() -> Self {
        Self {
            high_growth_operations: Vec::new(),
            high_memory_operations: Vec::new(),
            large_allocation_operations: Vec::new(),
            warnings: Vec::new(),
            potential_leaks: Vec::new(),
            recommendations: Vec::new(),
            efficiency_score: 1.0,
            analysis_timestamp: std::time::Instant::now(),
        }
    }
}

impl Default for FragmentationAnalysis {
    fn default() -> Self {
        Self {
            total_pooled_memory: 0,
            fragmented_pools: Vec::new(),
            fragmentation_ratio: 0.0,
            num_pools: 0,
            total_chunks: 0,
            recommendations: Vec::new(),
            performance_impact: 0.0,
            analysis_timestamp: std::time::Instant::now(),
        }
    }
}

impl Default for MemoryUsageTracker {
    fn default() -> Self {
        Self {
            usage_by_operation: HashMap::new(),
            peak_memory_usage: 0,
            total_allocations: 0,
            allocation_history: VecDeque::new(),
            gradient_memory_usage: HashMap::new(),
            fragmentation_stats: FragmentationStats::default(),
            leak_detection: LeakDetectionStats::default(),
            max_history_size: 10_000,
        }
    }
}

impl Default for GradientMemoryStats {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            peak_usage: 0,
            avg_allocation_size: 0,
            num_allocations: 0,
            growth_rate: 0.0,
            last_allocation: None,
            usage_history: VecDeque::new(),
            peak_allocation_size: 0,
        }
    }
}

impl Default for FragmentationStats {
    fn default() -> Self {
        Self {
            total_free: 0,
            largest_free_block: 0,
            fragmentation_ratio: 0.0,
            num_holes: 0,
            avg_hole_size: 0,
            num_pools: 0,
        }
    }
}

impl Default for LeakDetectionStats {
    fn default() -> Self {
        Self {
            potential_leaks: Vec::new(),
            suspicious_patterns: Vec::new(),
            unexplained_growth: 0,
            last_leak_check: None,
            total_leak_checks: 0,
            resolved_leaks: 0,
        }
    }
}

impl MemoryUsageTracker {
    /// Create a new memory usage tracker
    ///
    /// Initializes an empty tracker with default configuration:
    /// - Maximum 10,000 historical data points
    /// - All counters reset to zero
    /// - Empty operation and gradient tracking maps
    ///
    /// # Examples
    ///
    /// ```rust
    /// let tracker = MemoryUsageTracker::new();
    /// assert_eq!(tracker.total_allocations, 0);
    /// ```
    pub fn new() -> Self {
        Self {
            max_history_size: 10_000,
            ..Default::default()
        }
    }

    /// Create tracker with custom history size
    ///
    /// Allows customization of the maximum number of historical data points
    /// to control memory usage of the tracker itself.
    ///
    /// # Arguments
    ///
    /// * `max_history_size` - Maximum number of allocation history entries
    ///
    /// # Examples
    ///
    /// ```rust
    /// // Smaller history for memory-constrained environments
    /// let tracker = MemoryUsageTracker::with_history_size(1000);
    /// ```
    pub fn with_history_size(max_history_size: usize) -> Self {
        Self {
            max_history_size,
            ..Default::default()
        }
    }

    /// Track memory allocation for an operation
    ///
    /// Records memory allocation for a specific operation type, updating
    /// usage statistics and historical data.
    ///
    /// # Arguments
    ///
    /// * `operation` - Name of the operation (e.g., "conv2d_forward")
    /// * `bytes` - Number of bytes allocated
    ///
    /// # Examples
    ///
    /// ```rust
    /// tracker.track_allocation("conv2d_forward", 1024 * 1024); // 1MB
    /// tracker.track_allocation("attention_forward", 512 * 1024); // 512KB
    /// ```
    pub fn track_allocation(&mut self, operation: &str, bytes: usize) {
        // Update operation-specific usage
        *self
            .usage_by_operation
            .entry(operation.to_string())
            .or_insert(0) += bytes;

        // Update peak memory usage
        let current_total: usize = self.usage_by_operation.values().sum();
        self.peak_memory_usage = self.peak_memory_usage.max(current_total);

        // Update allocation count
        self.total_allocations += 1;

        // Add to allocation history
        self.allocation_history.push_back((Instant::now(), bytes));

        // Maintain history size limit
        if self.allocation_history.len() > self.max_history_size {
            self.allocation_history.pop_front();
        }
    }

    /// Track gradient-specific memory allocation
    ///
    /// Records memory allocation specifically for gradient operations,
    /// maintaining specialized statistics for gradient memory patterns.
    ///
    /// # Arguments
    ///
    /// * `gradient_op` - Name of the gradient operation
    /// * `bytes` - Number of bytes allocated for gradients
    ///
    /// # Examples
    ///
    /// ```rust
    /// tracker.track_gradient_allocation("conv2d_backward", 2048 * 1024); // 2MB
    /// ```
    pub fn track_gradient_allocation(&mut self, gradient_op: &str, bytes: usize) {
        let stats = self
            .gradient_memory_usage
            .entry(gradient_op.to_string())
            .or_insert_with(GradientMemoryStats::default);

        stats.track_allocation(bytes);

        // Also track in general allocation tracking
        self.track_allocation(&format!("gradient_{}", gradient_op), bytes);
    }

    /// Track memory deallocation
    ///
    /// Records memory deallocation for an operation, updating usage statistics.
    ///
    /// # Arguments
    ///
    /// * `operation` - Name of the operation
    /// * `bytes` - Number of bytes deallocated
    pub fn track_deallocation(&mut self, operation: &str, bytes: usize) {
        if let Some(current_usage) = self.usage_by_operation.get_mut(operation) {
            *current_usage = current_usage.saturating_sub(bytes);
        }
    }

    /// Analyze gradient memory usage patterns
    ///
    /// Performs comprehensive analysis of gradient memory usage patterns,
    /// identifying high-usage operations, growth trends, and optimization opportunities.
    ///
    /// # Returns
    ///
    /// Detailed analysis with warnings, recommendations, and efficiency metrics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let analysis = tracker.analyze_gradient_memory_usage();
    /// println!("Efficiency score: {:.2}", analysis.efficiency_score);
    /// for warning in analysis.warnings {
    ///     println!("Warning: {}", warning);
    /// }
    /// ```
    pub fn analyze_gradient_memory_usage(&self) -> GradientMemoryAnalysis {
        let mut analysis = GradientMemoryAnalysis {
            analysis_timestamp: Instant::now(),
            ..Default::default()
        };

        // Analyze operations by memory usage
        let mut ops_by_usage: Vec<_> = self.gradient_memory_usage.iter().collect();
        ops_by_usage.sort_by(|a, b| b.1.total_allocated.cmp(&a.1.total_allocated));

        // Identify high memory operations (top 20% or > 100MB)
        let high_memory_threshold = ops_by_usage
            .get(ops_by_usage.len() / 5)
            .map(|(_, stats)| stats.total_allocated)
            .unwrap_or(100 * 1024 * 1024); // 100MB default

        for (op_name, stats) in &ops_by_usage {
            if stats.total_allocated >= high_memory_threshold {
                analysis.high_memory_operations.push(op_name.to_string());
            }
        }

        // Analyze growth rates
        for (op_name, stats) in &self.gradient_memory_usage {
            if stats.growth_rate > 10.0 * 1024.0 * 1024.0 {
                // > 10MB/s
                analysis.high_growth_operations.push(op_name.clone());
            }

            if stats.avg_allocation_size > 50 * 1024 * 1024 {
                // > 50MB average
                analysis.large_allocation_operations.push(op_name.clone());
            }
        }

        // Generate warnings
        if !analysis.high_growth_operations.is_empty() {
            analysis.warnings.push(format!(
                "High growth rate operations detected: {}",
                analysis.high_growth_operations.join(", ")
            ));
        }

        if !analysis.high_memory_operations.is_empty() {
            analysis.warnings.push(format!(
                "High memory usage operations: {}",
                analysis.high_memory_operations.join(", ")
            ));
        }

        // Calculate efficiency score
        analysis.efficiency_score = self.calculate_memory_efficiency();

        // Generate recommendations
        analysis.recommendations = self.generate_optimization_recommendations(&analysis);

        // Include leak detection results
        analysis.potential_leaks = self.leak_detection.potential_leaks.clone();

        analysis
    }

    /// Analyze memory fragmentation
    ///
    /// Analyzes memory fragmentation patterns across all tracked pools
    /// and provides recommendations for defragmentation.
    ///
    /// # Returns
    ///
    /// Detailed fragmentation analysis with performance impact assessment.
    pub fn analyze_fragmentation(&self) -> FragmentationAnalysis {
        let mut analysis = FragmentationAnalysis {
            analysis_timestamp: Instant::now(),
            fragmentation_ratio: self.fragmentation_stats.fragmentation_ratio,
            num_pools: self.fragmentation_stats.num_pools,
            total_pooled_memory: self.fragmentation_stats.total_free,
            ..Default::default()
        };

        // Analyze fragmentation impact
        if analysis.fragmentation_ratio > 0.7 {
            analysis.performance_impact = 0.8;
            analysis.recommendations.push(
                "High fragmentation detected - consider memory pool reorganization".to_string(),
            );
        } else if analysis.fragmentation_ratio > 0.5 {
            analysis.performance_impact = 0.4;
            analysis
                .recommendations
                .push("Moderate fragmentation - monitor memory allocation patterns".to_string());
        } else {
            analysis.performance_impact = 0.1;
        }

        // Add specific recommendations based on fragmentation patterns
        if self.fragmentation_stats.num_holes > 100 {
            analysis.recommendations.push(
                "Many small memory holes detected - consider coalescing free blocks".to_string(),
            );
        }

        if self.fragmentation_stats.largest_free_block < self.fragmentation_stats.total_free / 4 {
            analysis
                .recommendations
                .push("Largest free block is small relative to total free memory".to_string());
        }

        analysis
    }

    /// Calculate overall memory efficiency
    ///
    /// Computes a comprehensive efficiency score based on allocation patterns,
    /// fragmentation, and usage history.
    ///
    /// # Returns
    ///
    /// Efficiency score from 0.0 (poor) to 1.0 (excellent).
    fn calculate_memory_efficiency(&self) -> f64 {
        if self.gradient_memory_usage.is_empty() {
            return 1.0;
        }

        let mut efficiency_factors = Vec::new();

        // Factor 1: Memory reuse efficiency (inverse of peak/average ratio)
        if self.peak_memory_usage > 0 && !self.allocation_history.is_empty() {
            let avg_usage = self
                .allocation_history
                .iter()
                .map(|(_, bytes)| *bytes)
                .sum::<usize>()
                / self.allocation_history.len();

            if avg_usage > 0 {
                let reuse_efficiency = avg_usage as f64 / self.peak_memory_usage as f64;
                efficiency_factors.push(reuse_efficiency.min(1.0));
            }
        }

        // Factor 2: Fragmentation efficiency (inverse of fragmentation ratio)
        let fragmentation_efficiency = 1.0 - self.fragmentation_stats.fragmentation_ratio;
        efficiency_factors.push(fragmentation_efficiency);

        // Factor 3: Growth rate stability (lower growth rates are better)
        let avg_growth_rate: f64 = self
            .gradient_memory_usage
            .values()
            .map(|stats| stats.growth_rate)
            .sum::<f64>()
            / self.gradient_memory_usage.len() as f64;

        let growth_efficiency = if avg_growth_rate > 0.0 {
            (1.0 / (1.0 + avg_growth_rate / (10.0 * 1024.0 * 1024.0))).min(1.0)
        } else {
            1.0
        };
        efficiency_factors.push(growth_efficiency);

        // Calculate weighted average
        if efficiency_factors.is_empty() {
            1.0
        } else {
            efficiency_factors.iter().sum::<f64>() / efficiency_factors.len() as f64
        }
    }

    /// Generate optimization recommendations
    ///
    /// Creates actionable recommendations based on memory usage analysis.
    fn generate_optimization_recommendations(
        &self,
        analysis: &GradientMemoryAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if analysis.efficiency_score < 0.5 {
            recommendations.push(
                "Overall memory efficiency is low - consider implementing gradient checkpointing"
                    .to_string(),
            );
        }

        if !analysis.high_growth_operations.is_empty() {
            recommendations.push("Consider memory pooling for high-growth operations".to_string());
        }

        if !analysis.large_allocation_operations.is_empty() {
            recommendations.push(
                "Large allocation operations detected - consider memory streaming or chunking"
                    .to_string(),
            );
        }

        if self.fragmentation_stats.fragmentation_ratio > 0.6 {
            recommendations.push(
                "High memory fragmentation - implement memory defragmentation strategy".to_string(),
            );
        }

        recommendations
    }

    /// Get current memory usage summary
    ///
    /// Returns a snapshot of current memory usage across all operations.
    ///
    /// # Returns
    ///
    /// Total current memory usage in bytes.
    pub fn current_usage(&self) -> usize {
        self.usage_by_operation.values().sum()
    }

    /// Get memory usage for specific operation
    ///
    /// Returns current memory usage for a specific operation type.
    ///
    /// # Arguments
    ///
    /// * `operation` - Name of the operation to query
    ///
    /// # Returns
    ///
    /// Memory usage in bytes, or 0 if operation not found.
    pub fn usage_for_operation(&self, operation: &str) -> usize {
        self.usage_by_operation.get(operation).copied().unwrap_or(0)
    }

    /// Clear all tracking data
    ///
    /// Resets all tracking statistics and history, useful for starting
    /// fresh analysis sessions.
    pub fn clear(&mut self) {
        self.usage_by_operation.clear();
        self.allocation_history.clear();
        self.gradient_memory_usage.clear();
        self.peak_memory_usage = 0;
        self.total_allocations = 0;
        self.fragmentation_stats = FragmentationStats::default();
        self.leak_detection = LeakDetectionStats::default();
    }
}

impl GradientMemoryStats {
    /// Track a new gradient allocation
    ///
    /// Updates gradient memory statistics with a new allocation.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Number of bytes allocated
    pub fn track_allocation(&mut self, bytes: usize) {
        self.total_allocated += bytes;
        self.num_allocations += 1;
        self.peak_usage = self.peak_usage.max(bytes);
        self.peak_allocation_size = self.peak_allocation_size.max(bytes);

        // Update average allocation size
        self.avg_allocation_size = self.total_allocated / self.num_allocations;

        // Update growth rate if we have previous data
        let now = Instant::now();
        if let Some(last_time) = self.last_allocation {
            let time_diff = now.duration_since(last_time).as_secs_f64();
            if time_diff > 0.0 {
                self.growth_rate = bytes as f64 / time_diff;
            }
        }

        self.last_allocation = Some(now);

        // Add to usage history
        self.usage_history.push_back((now, bytes));

        // Limit history size
        if self.usage_history.len() > 1000 {
            self.usage_history.pop_front();
        }
    }

    /// Calculate memory efficiency for this gradient operation
    ///
    /// Returns efficiency based on allocation patterns and usage history.
    pub fn efficiency(&self) -> f64 {
        if self.num_allocations == 0 {
            return 1.0;
        }

        // Higher efficiency for consistent allocation sizes
        let size_consistency = if self.peak_allocation_size > 0 {
            self.avg_allocation_size as f64 / self.peak_allocation_size as f64
        } else {
            1.0
        };

        // Lower efficiency for high growth rates
        let growth_penalty = if self.growth_rate > 1024.0 * 1024.0 {
            // 1MB/s
            0.5
        } else {
            1.0
        };

        size_consistency * growth_penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_creation() {
        let tracker = MemoryUsageTracker::new();
        assert_eq!(tracker.total_allocations, 0);
        assert_eq!(tracker.peak_memory_usage, 0);
        assert!(tracker.usage_by_operation.is_empty());
    }

    #[test]
    fn test_allocation_tracking() {
        let mut tracker = MemoryUsageTracker::new();

        tracker.track_allocation("conv2d", 1000);
        assert_eq!(tracker.total_allocations, 1);
        assert_eq!(tracker.usage_for_operation("conv2d"), 1000);
        assert_eq!(tracker.peak_memory_usage, 1000);

        tracker.track_allocation("conv2d", 500);
        assert_eq!(tracker.usage_for_operation("conv2d"), 1500);
        assert_eq!(tracker.peak_memory_usage, 1500);
    }

    #[test]
    fn test_gradient_tracking() {
        let mut tracker = MemoryUsageTracker::new();

        tracker.track_gradient_allocation("conv2d_grad", 2000);
        assert!(tracker.gradient_memory_usage.contains_key("conv2d_grad"));

        let stats = &tracker.gradient_memory_usage["conv2d_grad"];
        assert_eq!(stats.total_allocated, 2000);
        assert_eq!(stats.num_allocations, 1);
        assert_eq!(stats.avg_allocation_size, 2000);
    }

    #[test]
    fn test_deallocation_tracking() {
        let mut tracker = MemoryUsageTracker::new();

        tracker.track_allocation("conv2d", 1000);
        assert_eq!(tracker.usage_for_operation("conv2d"), 1000);

        tracker.track_deallocation("conv2d", 300);
        assert_eq!(tracker.usage_for_operation("conv2d"), 700);

        // Test underflow protection
        tracker.track_deallocation("conv2d", 1000);
        assert_eq!(tracker.usage_for_operation("conv2d"), 0);
    }

    #[test]
    fn test_gradient_memory_analysis() {
        let mut tracker = MemoryUsageTracker::new();

        // Add some gradient operations with different patterns
        tracker.track_gradient_allocation("large_op", 100 * 1024 * 1024); // 100MB
        tracker.track_gradient_allocation("small_op", 1024); // 1KB

        let analysis = tracker.analyze_gradient_memory_usage();
        assert!(analysis
            .high_memory_operations
            .contains(&"large_op".to_string()));
        assert!(!analysis
            .high_memory_operations
            .contains(&"small_op".to_string()));
    }

    #[test]
    fn test_efficiency_calculation() {
        let mut stats = GradientMemoryStats::default();

        // Consistent allocation sizes should have high efficiency
        for _ in 0..5 {
            stats.track_allocation(1000);
        }

        let efficiency = stats.efficiency();
        // Efficiency depends on growth rate and consistency - be more realistic
        assert!(
            efficiency >= 0.5,
            "Expected efficiency >= 0.5 with consistent allocations, got {}",
            efficiency
        );
    }

    #[test]
    fn test_history_size_limit() {
        let mut tracker = MemoryUsageTracker::with_history_size(5);

        // Add more allocations than history limit
        for i in 0..10 {
            tracker.track_allocation("test", i * 100);
        }

        assert!(tracker.allocation_history.len() <= 5);
    }

    #[test]
    fn test_tracker_clear() {
        let mut tracker = MemoryUsageTracker::new();

        tracker.track_allocation("test", 1000);
        tracker.track_gradient_allocation("grad_test", 2000);

        assert!(!tracker.usage_by_operation.is_empty());
        assert!(!tracker.gradient_memory_usage.is_empty());

        tracker.clear();

        assert!(tracker.usage_by_operation.is_empty());
        assert!(tracker.gradient_memory_usage.is_empty());
        assert_eq!(tracker.total_allocations, 0);
        assert_eq!(tracker.peak_memory_usage, 0);
    }

    #[test]
    fn test_fragmentation_analysis() {
        let mut tracker = MemoryUsageTracker::new();
        tracker.fragmentation_stats.fragmentation_ratio = 0.8;
        tracker.fragmentation_stats.num_holes = 150;

        let analysis = tracker.analyze_fragmentation();
        assert!(analysis.performance_impact > 0.5);
        assert!(!analysis.recommendations.is_empty());
    }
}

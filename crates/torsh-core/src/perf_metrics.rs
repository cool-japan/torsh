//! Advanced Performance Metrics for ToRSh Operations
//!
//! This module provides enhanced performance tracking beyond basic profiling,
//! including SIMD utilization, cache efficiency, memory bandwidth analysis,
//! and performance regression detection.
//!
//! # SciRS2 POLICY COMPLIANCE
//! This module integrates with scirs2-core performance monitoring when available
//! and provides fallback implementations for comprehensive metrics tracking.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// Global advanced metrics tracker
static METRICS_TRACKER: OnceLock<Arc<Mutex<AdvancedMetricsTracker>>> = OnceLock::new();

/// Advanced performance metrics configuration
#[derive(Debug, Clone)]
pub struct AdvancedMetricsConfig {
    /// Track SIMD utilization
    pub track_simd_utilization: bool,
    /// Track cache efficiency
    pub track_cache_efficiency: bool,
    /// Track memory bandwidth
    pub track_memory_bandwidth: bool,
    /// Track parallel efficiency
    pub track_parallel_efficiency: bool,
    /// Enable regression detection
    pub enable_regression_detection: bool,
    /// Regression threshold (percentage slowdown)
    pub regression_threshold: f64,
    /// Maximum history size for regression detection
    pub max_history_size: usize,
}

impl Default for AdvancedMetricsConfig {
    fn default() -> Self {
        Self {
            track_simd_utilization: true,
            track_cache_efficiency: true,
            track_memory_bandwidth: true,
            track_parallel_efficiency: true,
            enable_regression_detection: true,
            regression_threshold: 10.0, // 10% slowdown
            max_history_size: 1000,
        }
    }
}

/// SIMD utilization metrics
#[derive(Debug, Clone, Default)]
pub struct SimdUtilizationMetrics {
    /// Number of SIMD operations executed
    pub simd_ops: u64,
    /// Number of scalar fallback operations
    pub scalar_ops: u64,
    /// Total elements processed via SIMD
    pub simd_elements: u64,
    /// Total elements processed via scalar
    pub scalar_elements: u64,
    /// SIMD width used (4 for NEON, 8 for AVX2, etc.)
    pub simd_width: usize,
    /// Estimated SIMD speedup achieved
    pub estimated_speedup: f64,
}

impl SimdUtilizationMetrics {
    /// Calculate SIMD utilization percentage
    pub fn utilization_percentage(&self) -> f64 {
        let total_ops = self.simd_ops + self.scalar_ops;
        if total_ops == 0 {
            0.0
        } else {
            (self.simd_ops as f64 / total_ops as f64) * 100.0
        }
    }

    /// Calculate element coverage percentage
    pub fn element_coverage(&self) -> f64 {
        let total_elements = self.simd_elements + self.scalar_elements;
        if total_elements == 0 {
            0.0
        } else {
            (self.simd_elements as f64 / total_elements as f64) * 100.0
        }
    }

    /// Get performance recommendation
    pub fn recommendation(&self) -> String {
        let utilization = self.utilization_percentage();
        let coverage = self.element_coverage();

        if utilization < 50.0 {
            format!(
                "Low SIMD utilization ({:.1}%). Consider using aligned memory \
                 and ensuring array sizes are multiples of {}.",
                utilization, self.simd_width
            )
        } else if coverage < 80.0 {
            format!(
                "Good SIMD utilization ({:.1}%), but only {:.1}% element coverage. \
                 Increase batch sizes for better performance.",
                utilization, coverage
            )
        } else {
            format!(
                "Excellent SIMD utilization ({:.1}%, {:.1}% coverage). \
                 Estimated {:.2}x speedup achieved.",
                utilization, coverage, self.estimated_speedup
            )
        }
    }
}

/// Cache efficiency metrics
#[derive(Debug, Clone, Default)]
pub struct CacheEfficiencyMetrics {
    /// Estimated L1 cache hit rate
    pub l1_hit_rate: f64,
    /// Estimated L2 cache hit rate
    pub l2_hit_rate: f64,
    /// Estimated L3 cache hit rate
    pub l3_hit_rate: f64,
    /// Number of cache-friendly operations
    pub cache_friendly_ops: u64,
    /// Number of cache-unfriendly operations
    pub cache_unfriendly_ops: u64,
    /// Total memory accesses tracked
    pub total_accesses: u64,
    /// Average access stride
    pub avg_stride: f64,
}

impl CacheEfficiencyMetrics {
    /// Calculate overall cache efficiency score (0-100)
    pub fn efficiency_score(&self) -> f64 {
        // Weighted average favoring L1 > L2 > L3
        (self.l1_hit_rate * 0.5 + self.l2_hit_rate * 0.3 + self.l3_hit_rate * 0.2) * 100.0
    }

    /// Get performance recommendation
    pub fn recommendation(&self) -> String {
        let score = self.efficiency_score();

        if score < 60.0 {
            format!(
                "Poor cache efficiency ({:.1}/100). High cache miss rate detected. \
                 Consider using cache-blocking techniques and reducing working set size.",
                score
            )
        } else if score < 80.0 {
            format!(
                "Moderate cache efficiency ({:.1}/100). L1 hit rate: {:.1}%, L2: {:.1}%, L3: {:.1}%. \
                 Optimize data locality for better performance.",
                score, self.l1_hit_rate * 100.0, self.l2_hit_rate * 100.0, self.l3_hit_rate * 100.0
            )
        } else {
            format!(
                "Excellent cache efficiency ({:.1}/100). Good data locality achieved.",
                score
            )
        }
    }
}

/// Memory bandwidth metrics
#[derive(Debug, Clone, Default)]
pub struct MemoryBandwidthMetrics {
    /// Total bytes read
    pub bytes_read: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Peak bandwidth achieved (GB/s)
    pub peak_bandwidth_gbs: f64,
    /// Average bandwidth (GB/s)
    pub avg_bandwidth_gbs: f64,
    /// Theoretical maximum bandwidth (GB/s)
    pub theoretical_max_gbs: f64,
    /// Duration measured
    pub duration: Duration,
}

impl MemoryBandwidthMetrics {
    /// Calculate bandwidth utilization percentage
    pub fn bandwidth_utilization(&self) -> f64 {
        if self.theoretical_max_gbs == 0.0 {
            0.0
        } else {
            (self.avg_bandwidth_gbs / self.theoretical_max_gbs) * 100.0
        }
    }

    /// Get performance recommendation
    pub fn recommendation(&self) -> String {
        let utilization = self.bandwidth_utilization();

        if utilization < 30.0 {
            format!(
                "Low memory bandwidth utilization ({:.1}%). Avg: {:.2} GB/s, Peak: {:.2} GB/s. \
                 Consider using larger batch sizes and prefetching.",
                utilization, self.avg_bandwidth_gbs, self.peak_bandwidth_gbs
            )
        } else if utilization < 70.0 {
            format!(
                "Moderate memory bandwidth utilization ({:.1}%). Avg: {:.2} GB/s. \
                 Some room for optimization through better memory access patterns.",
                utilization, self.avg_bandwidth_gbs
            )
        } else {
            format!(
                "Excellent memory bandwidth utilization ({:.1}%). Avg: {:.2} GB/s, Peak: {:.2} GB/s.",
                utilization, self.avg_bandwidth_gbs, self.peak_bandwidth_gbs
            )
        }
    }
}

/// Parallel efficiency metrics
#[derive(Debug, Clone, Default)]
pub struct ParallelEfficiencyMetrics {
    /// Number of threads used
    pub num_threads: usize,
    /// Parallel execution time
    pub parallel_time: Duration,
    /// Estimated sequential time
    pub sequential_time: Duration,
    /// Actual speedup achieved
    pub actual_speedup: f64,
    /// Theoretical speedup (num_threads)
    pub theoretical_speedup: f64,
    /// Parallel efficiency (actual/theoretical)
    pub efficiency: f64,
    /// Load imbalance percentage
    pub load_imbalance: f64,
}

impl ParallelEfficiencyMetrics {
    /// Calculate from measurements
    pub fn calculate(
        num_threads: usize,
        parallel_time: Duration,
        sequential_time: Duration,
    ) -> Self {
        let theoretical_speedup = num_threads as f64;
        let actual_speedup = if parallel_time.as_secs_f64() > 0.0 {
            sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
        } else {
            1.0
        };
        let efficiency = if theoretical_speedup > 0.0 {
            actual_speedup / theoretical_speedup
        } else {
            0.0
        };

        // Estimate load imbalance from efficiency loss
        // Perfect efficiency (1.0) means zero imbalance
        // Lower efficiency suggests higher imbalance
        let load_imbalance = if efficiency < 1.0 {
            (1.0 - efficiency) * 100.0
        } else {
            0.0
        };

        Self {
            num_threads,
            parallel_time,
            sequential_time,
            actual_speedup,
            theoretical_speedup,
            efficiency,
            load_imbalance,
        }
    }

    /// Calculate from measurements with per-thread timing data
    ///
    /// This method provides more accurate load imbalance measurement by analyzing
    /// the variance in per-thread execution times.
    ///
    /// # Arguments
    /// * `num_threads` - Number of threads used
    /// * `parallel_time` - Total parallel execution time
    /// * `sequential_time` - Estimated sequential execution time
    /// * `thread_times` - Execution time for each thread
    ///
    /// # Load Imbalance Calculation
    /// Load imbalance is measured as the coefficient of variation (CV) of thread times:
    /// `CV = (std_dev / mean) * 100%`
    ///
    /// - 0-10%: Excellent load balance
    /// - 10-25%: Good load balance
    /// - 25-50%: Moderate imbalance
    /// - 50%+: Significant imbalance (needs optimization)
    ///
    /// # Example
    /// ```rust
    /// use torsh_core::perf_metrics::ParallelEfficiencyMetrics;
    /// use std::time::Duration;
    ///
    /// let thread_times = vec![
    ///     Duration::from_millis(100),
    ///     Duration::from_millis(105),
    ///     Duration::from_millis(98),
    ///     Duration::from_millis(102),
    /// ];
    ///
    /// let metrics = ParallelEfficiencyMetrics::calculate_with_thread_times(
    ///     4,
    ///     Duration::from_millis(105), // Max thread time
    ///     Duration::from_millis(400), // Sequential time
    ///     &thread_times,
    /// );
    ///
    /// println!("Load imbalance: {:.2}%", metrics.load_imbalance);
    /// println!("Efficiency: {:.2}%", metrics.efficiency * 100.0);
    /// ```
    pub fn calculate_with_thread_times(
        num_threads: usize,
        parallel_time: Duration,
        sequential_time: Duration,
        thread_times: &[Duration],
    ) -> Self {
        let theoretical_speedup = num_threads as f64;
        let actual_speedup = if parallel_time.as_secs_f64() > 0.0 {
            sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
        } else {
            1.0
        };
        let efficiency = if theoretical_speedup > 0.0 {
            actual_speedup / theoretical_speedup
        } else {
            0.0
        };

        // Calculate load imbalance from thread time variance
        let load_imbalance = if thread_times.len() > 1 {
            // Convert to f64 seconds for calculation
            let times_f64: Vec<f64> = thread_times.iter().map(|d| d.as_secs_f64()).collect();

            // Calculate mean
            let mean = times_f64.iter().sum::<f64>() / times_f64.len() as f64;

            if mean > 0.0 {
                // Calculate standard deviation
                let variance = times_f64
                    .iter()
                    .map(|&t| {
                        let diff = t - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / times_f64.len() as f64;

                let std_dev = variance.sqrt();

                // Coefficient of variation as percentage
                (std_dev / mean) * 100.0
            } else {
                0.0
            }
        } else {
            // Fallback to efficiency-based estimate
            if efficiency < 1.0 {
                (1.0 - efficiency) * 100.0
            } else {
                0.0
            }
        };

        Self {
            num_threads,
            parallel_time,
            sequential_time,
            actual_speedup,
            theoretical_speedup,
            efficiency,
            load_imbalance,
        }
    }

    /// Get performance recommendation
    ///
    /// Provides actionable recommendations based on parallel efficiency and load imbalance metrics.
    pub fn recommendation(&self) -> String {
        let load_balance_status = if self.load_imbalance < 10.0 {
            "excellent load balance"
        } else if self.load_imbalance < 25.0 {
            "good load balance"
        } else if self.load_imbalance < 50.0 {
            "moderate load imbalance"
        } else {
            "significant load imbalance"
        };

        if self.efficiency < 0.5 {
            format!(
                "Poor parallel efficiency ({:.1}%). Speedup: {:.2}x with {} threads (expected {:.1}x). \
                 Load imbalance: {:.1}% ({}). Recommendations: \
                 1) Use intelligent chunking (ChunkConfig::compute_intensive()), \
                 2) Profile per-thread workload distribution, \
                 3) Consider work-stealing scheduler.",
                self.efficiency * 100.0,
                self.actual_speedup,
                self.num_threads,
                self.theoretical_speedup,
                self.load_imbalance,
                load_balance_status
            )
        } else if self.efficiency < 0.8 {
            format!(
                "Moderate parallel efficiency ({:.1}%). Speedup: {:.2}x with {} threads. \
                 Load imbalance: {:.1}% ({}). Consider optimizing chunk sizes and reducing synchronization. \
                 Try scirs2_core::chunking strategies for 15-30% improvement.",
                self.efficiency * 100.0,
                self.actual_speedup,
                self.num_threads,
                self.load_imbalance,
                load_balance_status
            )
        } else {
            format!(
                "Excellent parallel efficiency ({:.1}%). Achieving {:.2}x speedup with {} threads. \
                 Load imbalance: {:.1}% ({}). Well-balanced parallel execution.",
                self.efficiency * 100.0,
                self.actual_speedup,
                self.num_threads,
                self.load_imbalance,
                load_balance_status
            )
        }
    }
}

/// Performance regression detection result
#[derive(Debug, Clone)]
pub struct RegressionDetection {
    /// Operation name
    pub operation: String,
    /// Current average duration
    pub current_avg: Duration,
    /// Historical average duration
    pub historical_avg: Duration,
    /// Percentage change (positive = slowdown)
    pub percentage_change: f64,
    /// Whether this is a regression
    pub is_regression: bool,
    /// Confidence level (0-1)
    pub confidence: f64,
}

impl RegressionDetection {
    /// Create a regression detection result
    pub fn new(
        operation: String,
        current_avg: Duration,
        historical_avg: Duration,
        threshold: f64,
    ) -> Self {
        let percentage_change = if historical_avg.as_secs_f64() > 0.0 {
            ((current_avg.as_secs_f64() - historical_avg.as_secs_f64())
                / historical_avg.as_secs_f64())
                * 100.0
        } else {
            0.0
        };

        let is_regression = percentage_change > threshold;

        // Simple confidence based on magnitude of change
        let confidence = if is_regression {
            (percentage_change / (threshold * 2.0)).min(1.0)
        } else {
            1.0 - (percentage_change.abs() / threshold).min(1.0)
        };

        Self {
            operation,
            current_avg,
            historical_avg,
            percentage_change,
            is_regression,
            confidence,
        }
    }

    /// Format as string for display
    pub fn format(&self) -> String {
        if self.is_regression {
            format!(
                "⚠️  REGRESSION: {} is {:.1}% slower ({:.2}ms vs {:.2}ms historical, confidence: {:.1}%)",
                self.operation,
                self.percentage_change,
                self.current_avg.as_secs_f64() * 1000.0,
                self.historical_avg.as_secs_f64() * 1000.0,
                self.confidence * 100.0
            )
        } else {
            format!(
                "✓ {} performance stable ({:.1}% change, confidence: {:.1}%)",
                self.operation,
                self.percentage_change,
                self.confidence * 100.0
            )
        }
    }
}

/// Historical performance data for regression detection
#[derive(Debug, Clone)]
struct HistoricalPerformance {
    /// Operation name
    #[allow(dead_code)]
    operation: String,
    /// Recent durations
    durations: VecDeque<Duration>,
    /// Maximum history size
    max_size: usize,
}

impl HistoricalPerformance {
    fn new(operation: String, max_size: usize) -> Self {
        Self {
            operation,
            durations: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn add_measurement(&mut self, duration: Duration) {
        if self.durations.len() >= self.max_size {
            self.durations.pop_front();
        }
        self.durations.push_back(duration);
    }

    fn average(&self) -> Duration {
        if self.durations.is_empty() {
            Duration::ZERO
        } else {
            let total: Duration = self.durations.iter().sum();
            total / self.durations.len() as u32
        }
    }
}

/// Advanced metrics tracker
pub struct AdvancedMetricsTracker {
    /// Configuration
    config: AdvancedMetricsConfig,
    /// SIMD utilization metrics
    simd_metrics: SimdUtilizationMetrics,
    /// Cache efficiency metrics
    cache_metrics: CacheEfficiencyMetrics,
    /// Memory bandwidth metrics
    memory_metrics: MemoryBandwidthMetrics,
    /// Parallel efficiency metrics per operation
    parallel_metrics: HashMap<String, ParallelEfficiencyMetrics>,
    /// Historical performance data
    historical_data: HashMap<String, HistoricalPerformance>,
}

impl AdvancedMetricsTracker {
    /// Create a new advanced metrics tracker
    pub fn new(config: AdvancedMetricsConfig) -> Self {
        Self {
            config,
            simd_metrics: SimdUtilizationMetrics::default(),
            cache_metrics: CacheEfficiencyMetrics::default(),
            memory_metrics: MemoryBandwidthMetrics::default(),
            parallel_metrics: HashMap::new(),
            historical_data: HashMap::new(),
        }
    }

    /// Record SIMD operation
    pub fn record_simd_op(&mut self, elements: u64, width: usize, speedup: f64) {
        self.simd_metrics.simd_ops += 1;
        self.simd_metrics.simd_elements += elements;
        self.simd_metrics.simd_width = width;
        self.simd_metrics.estimated_speedup = speedup;
    }

    /// Record scalar operation
    pub fn record_scalar_op(&mut self, elements: u64) {
        self.simd_metrics.scalar_ops += 1;
        self.simd_metrics.scalar_elements += elements;
    }

    /// Record cache access pattern
    pub fn record_cache_access(&mut self, stride: usize, is_friendly: bool) {
        self.cache_metrics.total_accesses += 1;

        // Update average stride
        let total = self.cache_metrics.total_accesses;
        self.cache_metrics.avg_stride =
            (self.cache_metrics.avg_stride * (total - 1) as f64 + stride as f64) / total as f64;

        if is_friendly {
            self.cache_metrics.cache_friendly_ops += 1;
        } else {
            self.cache_metrics.cache_unfriendly_ops += 1;
        }

        // Estimate cache hit rates based on stride
        self.update_cache_estimates(stride);
    }

    fn update_cache_estimates(&mut self, stride: usize) {
        // Simple heuristic: smaller strides = better cache locality
        if stride <= 64 {
            // Sequential or near-sequential: excellent L1
            self.cache_metrics.l1_hit_rate = 0.95;
            self.cache_metrics.l2_hit_rate = 0.98;
            self.cache_metrics.l3_hit_rate = 0.99;
        } else if stride <= 4096 {
            // Moderate stride: good L2/L3
            self.cache_metrics.l1_hit_rate = 0.70;
            self.cache_metrics.l2_hit_rate = 0.85;
            self.cache_metrics.l3_hit_rate = 0.95;
        } else {
            // Large stride: poor cache performance
            self.cache_metrics.l1_hit_rate = 0.30;
            self.cache_metrics.l2_hit_rate = 0.50;
            self.cache_metrics.l3_hit_rate = 0.70;
        }
    }

    /// Record memory bandwidth measurement
    pub fn record_bandwidth(
        &mut self,
        bytes_read: u64,
        bytes_written: u64,
        duration: Duration,
        theoretical_max_gbs: f64,
    ) {
        let total_bytes = bytes_read + bytes_written;
        let duration_secs = duration.as_secs_f64();
        let bandwidth_gbs = if duration_secs > 0.0 {
            (total_bytes as f64 / duration_secs) / 1_000_000_000.0
        } else {
            0.0
        };

        self.memory_metrics.bytes_read += bytes_read;
        self.memory_metrics.bytes_written += bytes_written;
        self.memory_metrics.duration += duration;
        self.memory_metrics.theoretical_max_gbs = theoretical_max_gbs;

        // Update peak
        if bandwidth_gbs > self.memory_metrics.peak_bandwidth_gbs {
            self.memory_metrics.peak_bandwidth_gbs = bandwidth_gbs;
        }

        // Update average
        let total_duration = self.memory_metrics.duration.as_secs_f64();
        if total_duration > 0.0 {
            let total_bytes = self.memory_metrics.bytes_read + self.memory_metrics.bytes_written;
            self.memory_metrics.avg_bandwidth_gbs =
                (total_bytes as f64 / total_duration) / 1_000_000_000.0;
        }
    }

    /// Record parallel operation
    pub fn record_parallel_op(
        &mut self,
        operation: String,
        num_threads: usize,
        parallel_time: Duration,
        sequential_time: Duration,
    ) {
        let metrics =
            ParallelEfficiencyMetrics::calculate(num_threads, parallel_time, sequential_time);
        self.parallel_metrics.insert(operation, metrics);
    }

    /// Record operation for regression detection
    pub fn record_operation(&mut self, operation: String, duration: Duration) {
        if !self.config.enable_regression_detection {
            return;
        }

        let history = self
            .historical_data
            .entry(operation.clone())
            .or_insert_with(|| HistoricalPerformance::new(operation, self.config.max_history_size));

        history.add_measurement(duration);
    }

    /// Check for performance regressions
    pub fn check_regressions(
        &self,
        operation: &str,
        current_avg: Duration,
    ) -> Option<RegressionDetection> {
        if !self.config.enable_regression_detection {
            return None;
        }

        let history = self.historical_data.get(operation)?;

        // Need at least 10 measurements for reliable regression detection
        if history.durations.len() < 10 {
            return None;
        }

        let historical_avg = history.average();
        Some(RegressionDetection::new(
            operation.to_string(),
            current_avg,
            historical_avg,
            self.config.regression_threshold,
        ))
    }

    /// Get SIMD metrics
    pub fn simd_metrics(&self) -> &SimdUtilizationMetrics {
        &self.simd_metrics
    }

    /// Get cache metrics
    pub fn cache_metrics(&self) -> &CacheEfficiencyMetrics {
        &self.cache_metrics
    }

    /// Get memory metrics
    pub fn memory_metrics(&self) -> &MemoryBandwidthMetrics {
        &self.memory_metrics
    }

    /// Get parallel metrics for an operation
    pub fn parallel_metrics(&self, operation: &str) -> Option<&ParallelEfficiencyMetrics> {
        self.parallel_metrics.get(operation)
    }

    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Advanced Performance Metrics Report ===\n\n");

        // SIMD metrics
        report.push_str("SIMD Utilization:\n");
        report.push_str(&format!("  {}\n\n", self.simd_metrics.recommendation()));

        // Cache metrics
        report.push_str("Cache Efficiency:\n");
        report.push_str(&format!("  {}\n\n", self.cache_metrics.recommendation()));

        // Memory bandwidth
        report.push_str("Memory Bandwidth:\n");
        report.push_str(&format!("  {}\n\n", self.memory_metrics.recommendation()));

        // Parallel efficiency
        if !self.parallel_metrics.is_empty() {
            report.push_str("Parallel Efficiency:\n");
            for (op, metrics) in &self.parallel_metrics {
                report.push_str(&format!("  {}: {}\n", op, metrics.recommendation()));
            }
            report.push('\n');
        }

        report
    }
}

/// Initialize the global metrics tracker
pub fn init_metrics_tracker(config: AdvancedMetricsConfig) {
    let _ = METRICS_TRACKER.set(Arc::new(Mutex::new(AdvancedMetricsTracker::new(config))));
}

/// Get the global metrics tracker
pub fn get_metrics_tracker() -> Option<Arc<Mutex<AdvancedMetricsTracker>>> {
    METRICS_TRACKER.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_metrics() {
        let mut metrics = SimdUtilizationMetrics::default();
        metrics.simd_ops = 80;
        metrics.scalar_ops = 20;
        metrics.simd_elements = 8000;
        metrics.scalar_elements = 200;
        metrics.simd_width = 8;
        metrics.estimated_speedup = 3.5;

        assert_eq!(metrics.utilization_percentage(), 80.0);
        assert!((metrics.element_coverage() - 97.56).abs() < 0.1);
        assert!(metrics.recommendation().contains("Excellent"));
    }

    #[test]
    fn test_cache_metrics() {
        let mut metrics = CacheEfficiencyMetrics::default();
        metrics.l1_hit_rate = 0.95;
        metrics.l2_hit_rate = 0.85;
        metrics.l3_hit_rate = 0.75;

        let score = metrics.efficiency_score();
        assert!(score > 85.0 && score < 95.0);
    }

    #[test]
    fn test_memory_bandwidth_metrics() {
        let mut metrics = MemoryBandwidthMetrics::default();
        metrics.avg_bandwidth_gbs = 20.0; // Less than 30% threshold
        metrics.peak_bandwidth_gbs = 35.0;
        metrics.theoretical_max_gbs = 100.0;

        assert_eq!(metrics.bandwidth_utilization(), 20.0);
        assert!(metrics.recommendation().contains("Low"));
    }

    #[test]
    fn test_parallel_efficiency() {
        let metrics = ParallelEfficiencyMetrics::calculate(
            8,
            Duration::from_millis(125),
            Duration::from_secs(1),
        );

        assert_eq!(metrics.num_threads, 8);
        assert_eq!(metrics.theoretical_speedup, 8.0);
        assert!(metrics.actual_speedup > 7.0 && metrics.actual_speedup < 9.0);
        assert!(metrics.efficiency > 0.9);
    }

    #[test]
    fn test_regression_detection() {
        let regression = RegressionDetection::new(
            "matmul".to_string(),
            Duration::from_millis(150),
            Duration::from_millis(100),
            10.0,
        );

        assert!(regression.is_regression);
        // Use approximate equality for floating point
        assert!((regression.percentage_change - 50.0).abs() < 0.01);
        assert!(regression.confidence > 0.5);
    }

    #[test]
    fn test_metrics_tracker() {
        let config = AdvancedMetricsConfig::default();
        let mut tracker = AdvancedMetricsTracker::new(config);

        tracker.record_simd_op(1000, 8, 3.5);
        tracker.record_scalar_op(50);
        tracker.record_cache_access(64, true);
        tracker.record_bandwidth(1_000_000, 500_000, Duration::from_millis(10), 50.0);

        assert!(tracker.simd_metrics().utilization_percentage() > 0.0);
        assert!(tracker.cache_metrics().efficiency_score() > 0.0);
        assert!(tracker.memory_metrics().avg_bandwidth_gbs > 0.0);
    }
}

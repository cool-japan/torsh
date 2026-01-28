//! # Advanced Quantization Profiler
//!
//! Comprehensive profiling and monitoring system for quantization operations,
//! providing detailed performance analytics, bottleneck detection, and optimization suggestions.

use crate::TorshResult;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use torsh_tensor::Tensor;

/// Advanced profiler for quantization operations
#[derive(Debug, Clone)]
pub struct QuantizationProfiler {
    /// Performance metrics for different operations
    operation_metrics: HashMap<String, OperationMetrics>,
    /// Memory usage tracking
    memory_tracker: MemoryTracker,
    /// Configuration for profiling
    config: ProfilerConfig,
    /// Global profiling session data
    session: ProfilerSession,
}

/// Configuration for the quantization profiler
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Maximum number of samples to keep per operation
    pub max_samples: usize,
    /// Enable detailed memory tracking
    pub track_memory: bool,
    /// Enable performance regression detection
    pub detect_regressions: bool,
    /// Threshold for performance regression detection (percentage)
    pub regression_threshold: f64,
    /// Enable real-time performance alerts
    pub enable_alerts: bool,
    /// Minimum execution time to profile (microseconds)
    pub min_profile_time_us: u64,
}

/// Performance metrics for a specific operation
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    /// Operation name
    pub name: String,
    /// Execution times (in microseconds)
    pub execution_times: VecDeque<u64>,
    /// Throughput measurements (elements/second)
    pub throughput: VecDeque<f64>,
    /// Memory usage measurements (bytes)
    pub memory_usage: VecDeque<usize>,
    /// Number of elements processed per operation
    pub element_counts: VecDeque<usize>,
    /// Tensor shapes processed
    pub tensor_shapes: VecDeque<Vec<usize>>,
    /// Performance statistics
    pub stats: PerformanceStats,
}

/// Statistical analysis of performance data
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Average execution time (microseconds)
    pub avg_time_us: f64,
    /// Median execution time (microseconds)  
    pub median_time_us: f64,
    /// Standard deviation of execution times
    pub std_dev_time_us: f64,
    /// 95th percentile execution time
    pub p95_time_us: f64,
    /// 99th percentile execution time
    pub p99_time_us: f64,
    /// Average throughput (elements/second)
    pub avg_throughput: f64,
    /// Peak throughput achieved
    pub peak_throughput: f64,
    /// Average memory usage (bytes)
    pub avg_memory_usage: f64,
    /// Total number of samples
    pub sample_count: usize,
}

/// Memory usage tracking
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    /// Current memory usage by operation
    current_usage: HashMap<String, usize>,
    /// Peak memory usage ever recorded
    peak_usage: usize,
    /// Memory usage history
    usage_history: VecDeque<MemorySnapshot>,
}

/// Memory usage snapshot at a point in time
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Timestamp of the snapshot
    pub timestamp: Instant,
    /// Total memory usage (bytes)
    pub total_usage: usize,
    /// Memory usage breakdown by operation
    pub breakdown: HashMap<String, usize>,
}

/// Profiling session data
#[derive(Debug, Clone)]
pub struct ProfilerSession {
    /// Session start time
    pub start_time: Instant,
    /// Total operations profiled
    pub total_operations: usize,
    /// Session configuration snapshot
    pub config_snapshot: String,
    /// Performance alerts generated
    pub alerts: Vec<PerformanceAlert>,
}

/// Performance alert types
#[derive(Debug, Clone)]
pub enum PerformanceAlert {
    /// Performance regression detected
    Regression {
        operation: String,
        previous_avg: f64,
        current_avg: f64,
        regression_percent: f64,
    },
    /// Memory usage spike detected
    MemorySpike {
        operation: String,
        current_usage: usize,
        previous_avg: usize,
        spike_percent: f64,
    },
    /// Unusually slow operation detected
    SlowOperation {
        operation: String,
        execution_time_us: u64,
        expected_time_us: u64,
    },
}

/// Profiling result for a single operation
#[derive(Debug, Clone)]
pub struct ProfilingResult {
    /// Operation name
    pub operation: String,
    /// Execution time (microseconds)
    pub execution_time_us: u64,
    /// Throughput (elements/second)
    pub throughput: f64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Number of elements processed
    pub element_count: usize,
    /// Tensor shape
    pub tensor_shape: Vec<usize>,
}

/// Comprehensive performance report
#[derive(Debug)]
pub struct PerformanceReport {
    /// Report generation timestamp
    pub timestamp: Instant,
    /// Session duration
    pub session_duration: Duration,
    /// Performance metrics for each operation
    pub operation_metrics: HashMap<String, OperationMetrics>,
    /// Memory usage analysis
    pub memory_analysis: MemoryAnalysis,
    /// Performance bottlenecks identified
    pub bottlenecks: Vec<Bottleneck>,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
    /// Performance alerts
    pub alerts: Vec<PerformanceAlert>,
}

/// Memory usage analysis
#[derive(Debug)]
pub struct MemoryAnalysis {
    /// Peak memory usage during session
    pub peak_usage: usize,
    /// Average memory usage
    pub avg_usage: f64,
    /// Memory efficiency score (0-100)
    pub efficiency_score: f64,
    /// Memory usage trends
    pub trends: Vec<String>,
}

/// Performance bottleneck identification
#[derive(Debug)]
pub struct Bottleneck {
    /// Operation causing the bottleneck
    pub operation: String,
    /// Severity score (0-100)
    pub severity: f64,
    /// Description of the bottleneck
    pub description: String,
    /// Suggested optimization
    pub suggestion: String,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            max_samples: 1000,
            track_memory: true,
            detect_regressions: true,
            regression_threshold: 10.0, // 10% regression threshold
            enable_alerts: true,
            min_profile_time_us: 100, // Only profile operations > 100μs
        }
    }
}

impl QuantizationProfiler {
    /// Create a new quantization profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            operation_metrics: HashMap::new(),
            memory_tracker: MemoryTracker::new(),
            config,
            session: ProfilerSession {
                start_time: Instant::now(),
                total_operations: 0,
                config_snapshot: "default".to_string(),
                alerts: Vec::new(),
            },
        }
    }

    /// Profile a quantization operation
    pub fn profile_operation<F, R>(
        &mut self,
        operation_name: &str,
        tensor: &Tensor,
        operation: F,
    ) -> TorshResult<(R, ProfilingResult)>
    where
        F: FnOnce(&Tensor) -> TorshResult<R>,
    {
        let start_memory = if self.config.track_memory {
            self.estimate_memory_usage(tensor)
        } else {
            0
        };

        let start_time = Instant::now();
        let result = operation(tensor)?;
        let execution_time = start_time.elapsed();

        let execution_time_us = execution_time.as_micros() as u64;

        // Only profile operations that meet minimum time threshold
        if execution_time_us < self.config.min_profile_time_us {
            return Ok((
                result,
                ProfilingResult {
                    operation: operation_name.to_string(),
                    execution_time_us,
                    throughput: 0.0,
                    memory_usage: start_memory,
                    element_count: tensor.numel(),
                    tensor_shape: tensor.shape().dims().to_vec(),
                },
            ));
        }

        let element_count = tensor.numel();
        let throughput = if execution_time_us > 0 {
            element_count as f64 / (execution_time_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        let profiling_result = ProfilingResult {
            operation: operation_name.to_string(),
            execution_time_us,
            throughput,
            memory_usage: start_memory,
            element_count,
            tensor_shape: tensor.shape().dims().to_vec(),
        };

        // Record metrics
        self.record_metrics(operation_name, &profiling_result);

        // Check for performance alerts
        if self.config.enable_alerts {
            self.check_performance_alerts(operation_name, &profiling_result);
        }

        self.session.total_operations += 1;

        Ok((result, profiling_result))
    }

    /// Record performance metrics for an operation
    fn record_metrics(&mut self, operation_name: &str, result: &ProfilingResult) {
        let metrics = self
            .operation_metrics
            .entry(operation_name.to_string())
            .or_insert_with(|| OperationMetrics::new(operation_name));

        // Add new measurements
        metrics.execution_times.push_back(result.execution_time_us);
        metrics.throughput.push_back(result.throughput);
        metrics.memory_usage.push_back(result.memory_usage);
        metrics.element_counts.push_back(result.element_count);
        metrics.tensor_shapes.push_back(result.tensor_shape.clone());

        // Maintain maximum sample size
        if metrics.execution_times.len() > self.config.max_samples {
            metrics.execution_times.pop_front();
            metrics.throughput.pop_front();
            metrics.memory_usage.pop_front();
            metrics.element_counts.pop_front();
            metrics.tensor_shapes.pop_front();
        }

        // Update statistics
        metrics.update_statistics();

        // Update memory tracker
        if self.config.track_memory {
            self.memory_tracker
                .record_usage(operation_name, result.memory_usage);
        }
    }

    /// Check for performance alerts and regressions
    fn check_performance_alerts(&mut self, operation_name: &str, result: &ProfilingResult) {
        if let Some(metrics) = self.operation_metrics.get(operation_name) {
            // Check for performance regression
            if self.config.detect_regressions && metrics.execution_times.len() > 10 {
                let recent_avg =
                    metrics.execution_times.iter().rev().take(5).sum::<u64>() as f64 / 5.0;
                let historical_avg = metrics.stats.avg_time_us;

                if recent_avg > historical_avg * (1.0 + self.config.regression_threshold / 100.0) {
                    let regression_percent =
                        ((recent_avg - historical_avg) / historical_avg) * 100.0;
                    self.session.alerts.push(PerformanceAlert::Regression {
                        operation: operation_name.to_string(),
                        previous_avg: historical_avg,
                        current_avg: recent_avg,
                        regression_percent,
                    });
                }
            }

            // Check for memory spikes
            if result.memory_usage > metrics.stats.avg_memory_usage as usize * 2 {
                let spike_percent = ((result.memory_usage as f64 - metrics.stats.avg_memory_usage)
                    / metrics.stats.avg_memory_usage)
                    * 100.0;
                self.session.alerts.push(PerformanceAlert::MemorySpike {
                    operation: operation_name.to_string(),
                    current_usage: result.memory_usage,
                    previous_avg: metrics.stats.avg_memory_usage as usize,
                    spike_percent,
                });
            }

            // Check for unusually slow operations
            if result.execution_time_us > metrics.stats.p95_time_us as u64 * 2 {
                self.session.alerts.push(PerformanceAlert::SlowOperation {
                    operation: operation_name.to_string(),
                    execution_time_us: result.execution_time_us,
                    expected_time_us: metrics.stats.avg_time_us as u64,
                });
            }
        }
    }

    /// Estimate memory usage for a tensor
    fn estimate_memory_usage(&self, tensor: &Tensor) -> usize {
        let element_size = match tensor.dtype() {
            torsh_core::DType::F32 => 4,
            torsh_core::DType::F64 => 8,
            torsh_core::DType::I32 => 4,
            torsh_core::DType::I64 => 8,
            _ => 4, // Default fallback
        };
        tensor.numel() * element_size
    }

    /// Generate comprehensive performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let session_duration = self.session.start_time.elapsed();

        // Analyze memory usage
        let memory_analysis = self.analyze_memory_usage();

        // Identify bottlenecks
        let bottlenecks = self.identify_bottlenecks();

        // Generate optimization recommendations
        let recommendations = self.generate_recommendations(&bottlenecks);

        PerformanceReport {
            timestamp: Instant::now(),
            session_duration,
            operation_metrics: self.operation_metrics.clone(),
            memory_analysis,
            bottlenecks,
            recommendations,
            alerts: self.session.alerts.clone(),
        }
    }

    /// Analyze memory usage patterns
    fn analyze_memory_usage(&self) -> MemoryAnalysis {
        let total_usage: usize = self.memory_tracker.current_usage.values().sum();
        let avg_usage = if self.memory_tracker.usage_history.is_empty() {
            total_usage as f64
        } else {
            self.memory_tracker
                .usage_history
                .iter()
                .map(|s| s.total_usage as f64)
                .sum::<f64>()
                / self.memory_tracker.usage_history.len() as f64
        };

        let efficiency_score = if self.memory_tracker.peak_usage > 0 {
            ((self.memory_tracker.peak_usage as f64 - avg_usage)
                / self.memory_tracker.peak_usage as f64)
                * 100.0
        } else {
            100.0
        };

        MemoryAnalysis {
            peak_usage: self.memory_tracker.peak_usage,
            avg_usage,
            efficiency_score,
            trends: vec!["Memory usage is stable".to_string()], // Simplified for now
        }
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(&self) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        for (op_name, metrics) in &self.operation_metrics {
            // Check for high variance in execution times
            if metrics.stats.std_dev_time_us > metrics.stats.avg_time_us * 0.5 {
                bottlenecks.push(Bottleneck {
                    operation: op_name.clone(),
                    severity: 75.0,
                    description: "High variance in execution times detected".to_string(),
                    suggestion: "Consider optimizing for consistent performance".to_string(),
                });
            }

            // Check for low throughput
            if metrics.stats.avg_throughput < 1_000_000.0 {
                // Less than 1M elements/second
                bottlenecks.push(Bottleneck {
                    operation: op_name.clone(),
                    severity: 60.0,
                    description: "Low throughput detected".to_string(),
                    suggestion: "Consider hardware acceleration or algorithm optimization"
                        .to_string(),
                });
            }
        }

        bottlenecks
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self, bottlenecks: &[Bottleneck]) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !bottlenecks.is_empty() {
            recommendations
                .push("Consider enabling hardware acceleration for better performance".to_string());
        }

        if self.session.total_operations > 1000 {
            recommendations.push(
                "Large number of operations detected - consider batch processing".to_string(),
            );
        }

        if self.memory_tracker.peak_usage > 1_000_000_000 {
            // > 1GB
            recommendations.push(
                "High memory usage detected - consider streaming or chunked processing".to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations
                .push("Performance looks optimal - no specific recommendations".to_string());
        }

        recommendations
    }

    /// Get current session statistics
    pub fn get_session_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        stats.insert(
            "total_operations".to_string(),
            self.session.total_operations as f64,
        );
        stats.insert(
            "session_duration_seconds".to_string(),
            self.session.start_time.elapsed().as_secs_f64(),
        );
        stats.insert(
            "operations_per_second".to_string(),
            self.session.total_operations as f64 / self.session.start_time.elapsed().as_secs_f64(),
        );
        stats.insert(
            "alerts_generated".to_string(),
            self.session.alerts.len() as f64,
        );

        if let Some(fastest_op) = self.operation_metrics.values().min_by(|a, b| {
            a.stats
                .avg_time_us
                .partial_cmp(&b.stats.avg_time_us)
                .expect("avg_time_us values should be comparable")
        }) {
            stats.insert(
                "fastest_avg_time_us".to_string(),
                fastest_op.stats.avg_time_us,
            );
        }

        if let Some(slowest_op) = self.operation_metrics.values().max_by(|a, b| {
            a.stats
                .avg_time_us
                .partial_cmp(&b.stats.avg_time_us)
                .expect("avg_time_us values should be comparable")
        }) {
            stats.insert(
                "slowest_avg_time_us".to_string(),
                slowest_op.stats.avg_time_us,
            );
        }

        stats
    }

    /// Reset profiler statistics
    pub fn reset(&mut self) {
        self.operation_metrics.clear();
        self.memory_tracker = MemoryTracker::new();
        self.session = ProfilerSession {
            start_time: Instant::now(),
            total_operations: 0,
            config_snapshot: format!("{:?}", self.config),
            alerts: Vec::new(),
        };
    }
}

impl OperationMetrics {
    /// Create new operation metrics
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            execution_times: VecDeque::new(),
            throughput: VecDeque::new(),
            memory_usage: VecDeque::new(),
            element_counts: VecDeque::new(),
            tensor_shapes: VecDeque::new(),
            stats: PerformanceStats::default(),
        }
    }

    /// Update performance statistics
    fn update_statistics(&mut self) {
        if self.execution_times.is_empty() {
            return;
        }

        let times: Vec<f64> = self.execution_times.iter().map(|&t| t as f64).collect();
        let throughputs: Vec<f64> = self.throughput.iter().copied().collect();
        let memory_usages: Vec<f64> = self.memory_usage.iter().map(|&m| m as f64).collect();

        // Calculate statistics
        let avg_time = times.iter().sum::<f64>() / times.len() as f64;

        let mut sorted_times = times.clone();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).expect("time values should be comparable"));
        let median_time = sorted_times[sorted_times.len() / 2];

        let mean = avg_time;
        let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
        let std_dev_time = variance.sqrt();

        let p95_time = Self::percentile(&times, 95.0);
        let p99_time = Self::percentile(&times, 99.0);

        self.stats = PerformanceStats {
            avg_time_us: avg_time,
            median_time_us: median_time,
            std_dev_time_us: std_dev_time,
            p95_time_us: p95_time,
            p99_time_us: p99_time,
            avg_throughput: throughputs.iter().sum::<f64>() / throughputs.len() as f64,
            peak_throughput: throughputs.iter().copied().fold(0.0, f64::max),
            avg_memory_usage: memory_usages.iter().sum::<f64>() / memory_usages.len() as f64,
            sample_count: times.len(),
        };
    }

    /// Calculate percentile value
    fn percentile(values: &[f64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("values should be comparable for percentile calculation")
        });

        let index = (percentile / 100.0) * (sorted.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted[lower]
        } else {
            let weight = index - lower as f64;
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            avg_time_us: 0.0,
            median_time_us: 0.0,
            std_dev_time_us: 0.0,
            p95_time_us: 0.0,
            p99_time_us: 0.0,
            avg_throughput: 0.0,
            peak_throughput: 0.0,
            avg_memory_usage: 0.0,
            sample_count: 0,
        }
    }
}

impl MemoryTracker {
    /// Create new memory tracker
    fn new() -> Self {
        Self {
            current_usage: HashMap::new(),
            peak_usage: 0,
            usage_history: VecDeque::new(),
        }
    }

    /// Record memory usage for an operation
    fn record_usage(&mut self, operation: &str, usage: usize) {
        self.current_usage.insert(operation.to_string(), usage);

        let total_usage: usize = self.current_usage.values().sum();
        if total_usage > self.peak_usage {
            self.peak_usage = total_usage;
        }

        // Record snapshot
        self.usage_history.push_back(MemorySnapshot {
            timestamp: Instant::now(),
            total_usage,
            breakdown: self.current_usage.clone(),
        });

        // Maintain history size
        if self.usage_history.len() > 1000 {
            self.usage_history.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_profiler_creation() {
        let config = ProfilerConfig::default();
        let profiler = QuantizationProfiler::new(config);
        assert_eq!(profiler.operation_metrics.len(), 0);
        assert_eq!(profiler.session.total_operations, 0);
    }

    #[test]
    fn test_profile_operation() {
        let mut profiler = QuantizationProfiler::new(ProfilerConfig::default());
        let tensor = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = profiler.profile_operation("test_op", &tensor, |t| {
            // Simulate some work
            std::thread::sleep(std::time::Duration::from_micros(200));
            Ok(t.clone())
        });

        assert!(result.is_ok());
        let (_, profiling_result) = result.unwrap();
        assert_eq!(profiling_result.operation, "test_op");
        assert!(profiling_result.execution_time_us >= 200);
        assert_eq!(profiling_result.element_count, 4);
    }

    #[test]
    fn test_performance_stats_calculation() {
        let mut metrics = OperationMetrics::new("test");
        metrics.execution_times.extend([100, 200, 150, 180, 120]);
        metrics
            .throughput
            .extend([1000.0, 2000.0, 1500.0, 1800.0, 1200.0]);
        metrics.memory_usage.extend([1000, 2000, 1500, 1800, 1200]);

        metrics.update_statistics();

        assert_eq!(metrics.stats.avg_time_us, 150.0);
        assert_eq!(metrics.stats.sample_count, 5);
        assert!(metrics.stats.std_dev_time_us > 0.0);
    }

    #[test]
    fn test_memory_tracking() {
        let mut tracker = MemoryTracker::new();
        tracker.record_usage("op1", 1000);
        tracker.record_usage("op2", 2000);

        assert_eq!(tracker.peak_usage, 3000);
        assert_eq!(tracker.usage_history.len(), 2);
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut profiler = QuantizationProfiler::new(ProfilerConfig::default());

        // Add metrics with high variance
        let metrics = OperationMetrics {
            name: "slow_op".to_string(),
            execution_times: VecDeque::from([100, 500, 200, 600, 150]),
            throughput: VecDeque::from([500.0, 100.0, 400.0, 90.0, 450.0]),
            memory_usage: VecDeque::new(),
            element_counts: VecDeque::new(),
            tensor_shapes: VecDeque::new(),
            stats: PerformanceStats {
                avg_time_us: 310.0,
                std_dev_time_us: 220.0, // High variance
                avg_throughput: 308.0,  // Low throughput
                ..PerformanceStats::default()
            },
        };

        profiler
            .operation_metrics
            .insert("slow_op".to_string(), metrics);

        let bottlenecks = profiler.identify_bottlenecks();
        assert!(!bottlenecks.is_empty());
    }

    #[test]
    fn test_session_stats() {
        let mut profiler = QuantizationProfiler::new(ProfilerConfig::default());
        profiler.session.total_operations = 100;

        let stats = profiler.get_session_stats();
        assert_eq!(stats.get("total_operations"), Some(&100.0));
        assert!(stats.contains_key("session_duration_seconds"));
    }

    #[test]
    fn test_profiler_reset() {
        let mut profiler = QuantizationProfiler::new(ProfilerConfig::default());
        profiler.session.total_operations = 100;
        profiler
            .operation_metrics
            .insert("test".to_string(), OperationMetrics::new("test"));

        profiler.reset();

        assert_eq!(profiler.session.total_operations, 0);
        assert!(profiler.operation_metrics.is_empty());
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert_eq!(OperationMetrics::percentile(&values, 50.0), 5.5);
        assert!((OperationMetrics::percentile(&values, 95.0) - 9.55).abs() < 0.01);
        assert_eq!(OperationMetrics::percentile(&values, 0.0), 1.0);
        assert_eq!(OperationMetrics::percentile(&values, 100.0), 10.0);
    }

    #[test]
    fn test_performance_alert_regression() {
        let mut profiler = QuantizationProfiler::new(ProfilerConfig {
            regression_threshold: 20.0,
            ..ProfilerConfig::default()
        });

        // Add historical data (need >10 samples for regression detection)
        let mut metrics = OperationMetrics::new("test_op");
        // Add older good performance
        metrics
            .execution_times
            .extend([100, 105, 95, 110, 100, 98, 102, 108, 96, 104, 101, 99]);
        // Add recent slow performance to trigger regression
        metrics.execution_times.extend([150, 145, 155, 160, 148]); // Recent slow samples
        metrics.update_statistics();
        profiler
            .operation_metrics
            .insert("test_op".to_string(), metrics);

        // Simulate current slow result (>20% regression)
        let slow_result = ProfilingResult {
            operation: "test_op".to_string(),
            execution_time_us: 150, // Much slower
            throughput: 1000.0,
            memory_usage: 1000,
            element_count: 100,
            tensor_shape: vec![100],
        };

        profiler.check_performance_alerts("test_op", &slow_result);

        // Should detect regression
        assert!(!profiler.session.alerts.is_empty());
        match &profiler.session.alerts[0] {
            PerformanceAlert::Regression {
                regression_percent, ..
            } => {
                assert!(*regression_percent > 20.0);
            }
            _ => {
                assert!(
                    false,
                    "Expected regression alert, but got different alert type"
                );
            }
        }
    }
}

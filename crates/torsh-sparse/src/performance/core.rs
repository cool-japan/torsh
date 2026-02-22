//! Core types and utilities for sparse tensor performance analysis
//!
//! This module provides fundamental data structures and configuration types
//! used throughout the performance analysis system.

use std::collections::HashMap;
use std::time::Duration;

/// Performance measurement result
///
/// Captures comprehensive performance metrics for a single operation,
/// including timing, memory usage, and custom metrics.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::PerformanceMeasurement;
/// use std::time::Duration;
/// use std::collections::HashMap;
///
/// let mut metrics = HashMap::new();
/// metrics.insert("flops".to_string(), 1000000.0);
/// metrics.insert("efficiency".to_string(), 0.85);
///
/// let measurement = PerformanceMeasurement {
///     operation: "sparse_matmul".to_string(),
///     duration: Duration::from_millis(50),
///     memory_before: 1024 * 1024,
///     memory_after: 2048 * 1024,
///     peak_memory: 3072 * 1024,
///     metrics,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    /// Operation name
    pub operation: String,
    /// Execution time
    pub duration: Duration,
    /// Memory usage before operation (bytes)
    pub memory_before: usize,
    /// Memory usage after operation (bytes)
    pub memory_after: usize,
    /// Peak memory usage during operation (bytes)
    pub peak_memory: usize,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

impl PerformanceMeasurement {
    /// Create a new performance measurement
    pub fn new(operation: String) -> Self {
        Self {
            operation,
            duration: Duration::default(),
            memory_before: 0,
            memory_after: 0,
            peak_memory: 0,
            metrics: HashMap::new(),
        }
    }

    /// Add a custom metric to this measurement
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
    }

    /// Get memory delta (difference between after and before)
    pub fn memory_delta(&self) -> i64 {
        self.memory_after as i64 - self.memory_before as i64
    }

    /// Get memory efficiency (how much of peak memory was actually used)
    pub fn memory_efficiency(&self) -> f64 {
        if self.peak_memory == 0 {
            1.0
        } else {
            self.memory_after as f64 / self.peak_memory as f64
        }
    }

    /// Get operations per second (if flops metric is available)
    pub fn ops_per_second(&self) -> Option<f64> {
        self.metrics.get("flops").map(|&flops| {
            flops / self.duration.as_secs_f64()
        })
    }
}

/// Benchmarking configuration
///
/// Defines parameters for running performance benchmarks, including
/// warm-up iterations, measurement iterations, and various options
/// for collecting detailed performance data.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::BenchmarkConfig;
/// use std::time::Duration;
///
/// let config = BenchmarkConfig {
///     warmup_iterations: 5,
///     measured_iterations: 20,
///     collect_memory: true,
///     gc_between_iterations: false,
///     max_iteration_time: Duration::from_secs(60),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warm-up iterations
    pub warmup_iterations: usize,
    /// Number of measured iterations
    pub measured_iterations: usize,
    /// Whether to collect memory statistics
    pub collect_memory: bool,
    /// Whether to run garbage collection between iterations
    pub gc_between_iterations: bool,
    /// Maximum allowed execution time per iteration
    pub max_iteration_time: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            measured_iterations: 10,
            collect_memory: true,
            gc_between_iterations: true,
            max_iteration_time: Duration::from_secs(30),
        }
    }
}

impl BenchmarkConfig {
    /// Create a fast benchmark configuration with minimal overhead
    pub fn fast() -> Self {
        Self {
            warmup_iterations: 1,
            measured_iterations: 3,
            collect_memory: false,
            gc_between_iterations: false,
            max_iteration_time: Duration::from_secs(5),
        }
    }

    /// Create a comprehensive benchmark configuration for detailed analysis
    pub fn comprehensive() -> Self {
        Self {
            warmup_iterations: 10,
            measured_iterations: 50,
            collect_memory: true,
            gc_between_iterations: true,
            max_iteration_time: Duration::from_secs(120),
        }
    }

    /// Create a memory-focused benchmark configuration
    pub fn memory_focused() -> Self {
        Self {
            warmup_iterations: 2,
            measured_iterations: 5,
            collect_memory: true,
            gc_between_iterations: true,
            max_iteration_time: Duration::from_secs(60),
        }
    }

    /// Get total benchmark time estimate
    pub fn estimated_total_time(&self) -> Duration {
        let total_iterations = self.warmup_iterations + self.measured_iterations;
        self.max_iteration_time * total_iterations as u32
    }
}

/// Memory measurement utilities
pub mod memory {
    use std::sync::{Arc, Mutex};

    /// Simple memory tracker for monitoring memory usage
    #[derive(Debug, Clone, Default)]
    pub struct MemoryTracker {
        current_usage: Arc<Mutex<usize>>,
        peak_usage: Arc<Mutex<usize>>,
    }

    impl MemoryTracker {
        /// Create a new memory tracker
        pub fn new() -> Self {
            Self::default()
        }

        /// Record current memory usage
        pub fn record_usage(&self, usage: usize) {
            *self.current_usage.lock().expect("lock should not be poisoned") = usage;
            let mut peak = self.peak_usage.lock().expect("lock should not be poisoned");
            if usage > *peak {
                *peak = usage;
            }
        }

        /// Get current memory usage
        pub fn current_usage(&self) -> usize {
            *self.current_usage.lock().expect("lock should not be poisoned")
        }

        /// Get peak memory usage
        pub fn peak_usage(&self) -> usize {
            *self.peak_usage.lock().expect("lock should not be poisoned")
        }

        /// Reset memory tracking
        pub fn reset(&self) {
            *self.current_usage.lock().expect("lock should not be poisoned") = 0;
            *self.peak_usage.lock().expect("lock should not be poisoned") = 0;
        }
    }
}

/// Time measurement utilities
pub mod timing {
    use std::time::{Duration, Instant};

    /// High-precision timer for performance measurements
    pub struct Timer {
        start: Instant,
    }

    impl Timer {
        /// Create and start a new timer
        pub fn start() -> Self {
            Self {
                start: Instant::now(),
            }
        }

        /// Get elapsed time since timer was started
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }

        /// Stop timer and return elapsed time
        pub fn stop(self) -> Duration {
            self.elapsed()
        }
    }

    /// Measure execution time of a closure
    pub fn measure_time<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Measure execution time of an async closure
    pub async fn measure_time_async<F, Fut, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let start = Instant::now();
        let result = f().await;
        let duration = start.elapsed();
        (result, duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_measurement_creation() {
        let measurement = PerformanceMeasurement::new("test_op".to_string());
        assert_eq!(measurement.operation, "test_op");
        assert_eq!(measurement.duration, Duration::default());
        assert!(measurement.metrics.is_empty());
    }

    #[test]
    fn test_performance_measurement_metrics() {
        let mut measurement = PerformanceMeasurement::new("test".to_string());
        measurement.add_metric("flops".to_string(), 1000.0);
        measurement.duration = Duration::from_secs(1);

        assert_eq!(measurement.metrics.get("flops"), Some(&1000.0));
        assert_eq!(measurement.ops_per_second(), Some(1000.0));
    }

    #[test]
    fn test_performance_measurement_memory() {
        let measurement = PerformanceMeasurement {
            operation: "test".to_string(),
            duration: Duration::default(),
            memory_before: 1000,
            memory_after: 1500,
            peak_memory: 2000,
            metrics: HashMap::new(),
        };

        assert_eq!(measurement.memory_delta(), 500);
        assert_eq!(measurement.memory_efficiency(), 0.75);
    }

    #[test]
    fn test_benchmark_config_defaults() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 3);
        assert_eq!(config.measured_iterations, 10);
        assert!(config.collect_memory);
        assert!(config.gc_between_iterations);
    }

    #[test]
    fn test_benchmark_config_presets() {
        let fast = BenchmarkConfig::fast();
        assert_eq!(fast.warmup_iterations, 1);
        assert_eq!(fast.measured_iterations, 3);
        assert!(!fast.collect_memory);

        let comprehensive = BenchmarkConfig::comprehensive();
        assert_eq!(comprehensive.warmup_iterations, 10);
        assert_eq!(comprehensive.measured_iterations, 50);
        assert!(comprehensive.collect_memory);

        let memory_focused = BenchmarkConfig::memory_focused();
        assert!(memory_focused.collect_memory);
        assert!(memory_focused.gc_between_iterations);
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = memory::MemoryTracker::new();

        tracker.record_usage(1000);
        assert_eq!(tracker.current_usage(), 1000);
        assert_eq!(tracker.peak_usage(), 1000);

        tracker.record_usage(1500);
        assert_eq!(tracker.current_usage(), 1500);
        assert_eq!(tracker.peak_usage(), 1500);

        tracker.record_usage(1200);
        assert_eq!(tracker.current_usage(), 1200);
        assert_eq!(tracker.peak_usage(), 1500);

        tracker.reset();
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.peak_usage(), 0);
    }

    #[test]
    fn test_timer() {
        use std::thread;

        let timer = timing::Timer::start();
        thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();

        assert!(elapsed >= Duration::from_millis(10));
        assert!(elapsed <= Duration::from_millis(50)); // Allow some tolerance
    }

    #[test]
    fn test_measure_time() {
        let (result, duration) = timing::measure_time(|| {
            std::thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(10));
        assert!(duration <= Duration::from_millis(50)); // Allow some tolerance
    }
}
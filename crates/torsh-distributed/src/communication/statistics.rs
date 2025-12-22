//! Communication statistics and metrics collection
//!
//! This module provides unified statistics collection across all
//! communication modules to eliminate duplication.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// Individual operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    /// Total number of operations
    pub count: u64,
    /// Total duration of all operations
    pub total_duration: Duration,
    /// Number of successful operations
    pub success_count: u64,
    /// Number of failed operations
    pub error_count: u64,
    /// Average duration per operation
    pub avg_duration: Duration,
    /// Minimum operation duration
    pub min_duration: Duration,
    /// Maximum operation duration
    pub max_duration: Duration,
    /// Total bytes transferred (if applicable)
    pub total_bytes: u64,
    /// Last operation timestamp
    pub last_operation: Option<SystemTime>,
}

impl Default for OperationStats {
    fn default() -> Self {
        Self {
            count: 0,
            total_duration: Duration::ZERO,
            success_count: 0,
            error_count: 0,
            avg_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            total_bytes: 0,
            last_operation: None,
        }
    }
}

impl OperationStats {
    /// Record a successful operation
    pub fn record_success(&mut self, duration: Duration, bytes: Option<u64>) {
        self.count += 1;
        self.success_count += 1;
        self.total_duration += duration;
        self.avg_duration = self.total_duration / self.count as u32;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.last_operation = Some(SystemTime::now());

        if let Some(bytes) = bytes {
            self.total_bytes += bytes;
        }
    }

    /// Record a failed operation
    pub fn record_failure(&mut self, duration: Duration) {
        self.count += 1;
        self.error_count += 1;
        self.total_duration += duration;
        self.avg_duration = self.total_duration / self.count as u32;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        self.last_operation = Some(SystemTime::now());
    }

    /// Get success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.success_count as f64 / self.count as f64) * 100.0
        }
    }

    /// Get average throughput in bytes per second
    pub fn avg_throughput(&self) -> f64 {
        if self.total_duration.is_zero() {
            0.0
        } else {
            self.total_bytes as f64 / self.total_duration.as_secs_f64()
        }
    }
}

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: SystemTime,
    pub value: f64,
}

/// Time series for tracking metrics over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    points: Vec<TimeSeriesPoint>,
    max_points: usize,
}

impl Default for TimeSeries {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            max_points: 1000,
        }
    }
}

impl TimeSeries {
    pub fn new(max_points: usize) -> Self {
        Self {
            points: Vec::new(),
            max_points,
        }
    }

    pub fn add_point(&mut self, value: f64) {
        let point = TimeSeriesPoint {
            timestamp: SystemTime::now(),
            value,
        };

        self.points.push(point);

        // Keep only the most recent points
        if self.points.len() > self.max_points {
            self.points.remove(0);
        }
    }

    pub fn get_points(&self) -> &[TimeSeriesPoint] {
        &self.points
    }

    pub fn latest_value(&self) -> Option<f64> {
        self.points.last().map(|p| p.value)
    }

    pub fn average(&self) -> f64 {
        if self.points.is_empty() {
            0.0
        } else {
            let sum: f64 = self.points.iter().map(|p| p.value).sum();
            sum / self.points.len() as f64
        }
    }

    pub fn max(&self) -> Option<f64> {
        self.points
            .iter()
            .map(|p| p.value)
            .fold(None, |acc, val| Some(acc.map_or(val, |a| a.max(val))))
    }

    pub fn min(&self) -> Option<f64> {
        self.points
            .iter()
            .map(|p| p.value)
            .fold(None, |acc, val| Some(acc.map_or(val, |a| a.min(val))))
    }
}

/// Comprehensive communication statistics
#[derive(Debug, Default)]
pub struct CommunicationStats {
    /// Per-operation statistics
    pub operations: HashMap<String, OperationStats>,
    /// Bandwidth usage over time (bytes per second)
    pub bandwidth_usage: TimeSeries,
    /// Error counts by error type
    pub error_counts: HashMap<String, u64>,
    /// Total communication time
    pub total_communication_time: Duration,
    /// Start time for statistics collection
    pub start_time: Option<SystemTime>,
}

impl CommunicationStats {
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            bandwidth_usage: TimeSeries::new(1000), // Keep last 1000 points
            error_counts: HashMap::new(),
            total_communication_time: Duration::ZERO,
            start_time: Some(SystemTime::now()),
        }
    }

    /// Record a communication operation
    pub fn record_operation(
        &mut self,
        op_type: &str,
        duration: Duration,
        success: bool,
        bytes: Option<u64>,
    ) {
        let stats = self.operations.entry(op_type.to_string()).or_default();

        if success {
            stats.record_success(duration, bytes);
        } else {
            stats.record_failure(duration);
        }

        self.total_communication_time += duration;

        // Record bandwidth if bytes were transferred
        if let Some(bytes) = bytes {
            let bytes_per_sec = if duration.as_secs_f64() > 0.0 {
                bytes as f64 / duration.as_secs_f64()
            } else {
                0.0
            };
            self.bandwidth_usage.add_point(bytes_per_sec);
        }
    }

    /// Record an error
    pub fn record_error(&mut self, error_type: &str) {
        *self.error_counts.entry(error_type.to_string()).or_insert(0) += 1;
    }

    /// Get total number of operations
    pub fn total_operations(&self) -> u64 {
        self.operations.values().map(|stats| stats.count).sum()
    }

    /// Get total successful operations
    pub fn total_successful_operations(&self) -> u64 {
        self.operations
            .values()
            .map(|stats| stats.success_count)
            .sum()
    }

    /// Get overall success rate
    pub fn overall_success_rate(&self) -> f64 {
        let total = self.total_operations();
        if total == 0 {
            0.0
        } else {
            (self.total_successful_operations() as f64 / total as f64) * 100.0
        }
    }

    /// Get total bytes transferred
    pub fn total_bytes_transferred(&self) -> u64 {
        self.operations
            .values()
            .map(|stats| stats.total_bytes)
            .sum()
    }

    /// Get average bandwidth utilization
    pub fn average_bandwidth(&self) -> f64 {
        self.bandwidth_usage.average()
    }

    /// Get peak bandwidth utilization
    pub fn peak_bandwidth(&self) -> f64 {
        self.bandwidth_usage.max().unwrap_or(0.0)
    }

    /// Get uptime since statistics collection started
    pub fn uptime(&self) -> Duration {
        self.start_time
            .map(|start| start.elapsed().unwrap_or(Duration::ZERO))
            .unwrap_or(Duration::ZERO)
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.operations.clear();
        self.bandwidth_usage = TimeSeries::new(1000);
        self.error_counts.clear();
        self.total_communication_time = Duration::ZERO;
        self.start_time = Some(SystemTime::now());
    }

    /// Export statistics as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        #[derive(Serialize)]
        struct ExportStats {
            operations: HashMap<String, OperationStats>,
            bandwidth_usage: Vec<TimeSeriesPoint>,
            error_counts: HashMap<String, u64>,
            total_communication_time_secs: f64,
            uptime_secs: f64,
            total_operations: u64,
            total_successful_operations: u64,
            overall_success_rate: f64,
            total_bytes_transferred: u64,
            average_bandwidth: f64,
            peak_bandwidth: f64,
        }

        let export = ExportStats {
            operations: self.operations.clone(),
            bandwidth_usage: self.bandwidth_usage.get_points().to_vec(),
            error_counts: self.error_counts.clone(),
            total_communication_time_secs: self.total_communication_time.as_secs_f64(),
            uptime_secs: self.uptime().as_secs_f64(),
            total_operations: self.total_operations(),
            total_successful_operations: self.total_successful_operations(),
            overall_success_rate: self.overall_success_rate(),
            total_bytes_transferred: self.total_bytes_transferred(),
            average_bandwidth: self.average_bandwidth(),
            peak_bandwidth: self.peak_bandwidth(),
        };

        serde_json::to_string_pretty(&export)
    }
}

/// Trait for collecting statistics in communication modules
pub trait StatsCollector {
    /// Record an operation with timing and success information
    fn record_operation(&mut self, op_type: &str, duration: Duration, success: bool);

    /// Record bandwidth usage
    fn record_bandwidth(&mut self, bytes: u64, duration: Duration);

    /// Record an error
    fn record_error(&mut self, error_type: &str);

    /// Get current statistics snapshot
    fn get_stats(&self) -> &CommunicationStats;
}

/// Operation timer for measuring operation duration
pub struct OperationTimer {
    start_time: Instant,
    operation_name: String,
}

impl OperationTimer {
    pub fn new(operation_name: impl Into<String>) -> Self {
        Self {
            start_time: Instant::now(),
            operation_name: operation_name.into(),
        }
    }

    pub fn finish<T>(
        self,
        stats: &mut dyn StatsCollector,
        result: &Result<T, impl std::error::Error>,
    ) -> Duration {
        let duration = self.start_time.elapsed();
        let success = result.is_ok();
        stats.record_operation(&self.operation_name, duration, success);
        duration
    }

    pub fn finish_with_bytes<T>(
        self,
        stats: &mut dyn StatsCollector,
        result: &Result<T, impl std::error::Error>,
        bytes: u64,
    ) -> Duration {
        let duration = self.start_time.elapsed();
        let success = result.is_ok();
        stats.record_operation(&self.operation_name, duration, success);
        if success {
            stats.record_bandwidth(bytes, duration);
        }
        duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;
    use std::thread::sleep;

    #[test]
    fn test_operation_stats() {
        let mut stats = OperationStats::default();

        // Record some operations
        stats.record_success(Duration::from_millis(100), Some(1024));
        stats.record_success(Duration::from_millis(200), Some(2048));
        stats.record_failure(Duration::from_millis(50));

        assert_eq!(stats.count, 3);
        assert_eq!(stats.success_count, 2);
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.total_bytes, 3072);
        // Use approximate equality for floating-point comparison
        let expected = 200.0 / 3.0;
        let actual = stats.success_rate();
        assert!(
            (actual - expected).abs() < 1e-10,
            "Expected ~{}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_time_series() {
        let mut ts = TimeSeries::new(3);

        ts.add_point(10.0);
        ts.add_point(20.0);
        ts.add_point(30.0);
        ts.add_point(40.0); // Should remove the first point

        assert_eq!(ts.get_points().len(), 3);
        assert_eq!(ts.latest_value(), Some(40.0));
        assert_eq!(ts.average(), 30.0);
    }

    #[test]
    fn test_communication_stats() {
        let mut stats = CommunicationStats::new();

        stats.record_operation("all_reduce", Duration::from_millis(100), true, Some(1024));
        stats.record_operation("all_reduce", Duration::from_millis(200), true, Some(2048));
        stats.record_operation("broadcast", Duration::from_millis(50), false, None);
        stats.record_error("timeout");

        assert_eq!(stats.total_operations(), 3);
        assert_eq!(stats.total_successful_operations(), 2);
        assert_eq!(stats.total_bytes_transferred(), 3072);
        assert!(stats.overall_success_rate() > 65.0 && stats.overall_success_rate() < 67.0);
        assert_eq!(stats.error_counts.get("timeout"), Some(&1));
    }

    #[test]
    fn test_operation_timer() {
        struct MockCollector {
            recorded_ops: Vec<(String, Duration, bool)>,
        }

        impl StatsCollector for MockCollector {
            fn record_operation(&mut self, op_type: &str, duration: Duration, success: bool) {
                self.recorded_ops
                    .push((op_type.to_string(), duration, success));
            }

            fn record_bandwidth(&mut self, _bytes: u64, _duration: Duration) {}
            fn record_error(&mut self, _error_type: &str) {}
            fn get_stats(&self) -> &CommunicationStats {
                // For testing purposes, return a static default
                static DEFAULT_STATS: LazyLock<CommunicationStats> =
                    LazyLock::new(|| CommunicationStats {
                        operations: HashMap::new(),
                        bandwidth_usage: TimeSeries {
                            points: Vec::new(),
                            max_points: 1000,
                        },
                        error_counts: HashMap::new(),
                        total_communication_time: Duration::ZERO,
                        start_time: None,
                    });
                &DEFAULT_STATS
            }
        }

        let mut collector = MockCollector {
            recorded_ops: Vec::new(),
        };
        let timer = OperationTimer::new("test_op");

        sleep(Duration::from_millis(10));
        let result: Result<(), std::io::Error> = Ok(());
        let duration = timer.finish(&mut collector, &result);

        assert_eq!(collector.recorded_ops.len(), 1);
        assert_eq!(collector.recorded_ops[0].0, "test_op");
        assert!(collector.recorded_ops[0].1 >= Duration::from_millis(10));
        assert!(collector.recorded_ops[0].2); // success
        assert!(duration >= Duration::from_millis(10));
    }
}

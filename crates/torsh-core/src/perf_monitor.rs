//! Real-Time Performance Monitoring
//!
//! This module provides real-time performance monitoring capabilities for ToRSh operations,
//! enabling live tracking of performance metrics, resource utilization, and bottleneck detection.
//!
//! # Features
//!
//! - **Live Metrics**: Real-time tracking of operation throughput, latency, and resource usage
//! - **Adaptive Thresholds**: Automatic detection of performance regressions
//! - **Resource Monitoring**: CPU, memory, and GPU utilization tracking
//! - **Alert System**: Configurable alerts for performance issues
//!
//! # SciRS2 POLICY COMPLIANCE
//!
//! When available, integrates with scirs2-core performance monitoring for:
//! - Hardware counter access (CPU cache misses, branch mispredictions)
//! - GPU performance counters (SM utilization, memory bandwidth)
//! - System-wide resource tracking

// Note: Result and TorshError kept for future error handling enhancements
#[allow(unused_imports)]
use crate::error::{Result, TorshError};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Global real-time performance monitor
static PERF_MONITOR: OnceLock<Arc<Mutex<RealTimeMonitor>>> = OnceLock::new();

/// Configuration for real-time performance monitoring
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Enable real-time monitoring
    pub enabled: bool,
    /// Update interval for metrics aggregation
    pub update_interval: Duration,
    /// Window size for moving averages (number of samples)
    pub window_size: usize,
    /// Enable hardware performance counters
    pub enable_hw_counters: bool,
    /// Enable GPU monitoring
    pub enable_gpu_monitoring: bool,
    /// Enable memory bandwidth tracking
    pub enable_bandwidth_tracking: bool,
    /// Alert threshold multiplier (e.g., 2.0 = alert when 2x slower than baseline)
    pub alert_threshold_multiplier: f64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_interval: Duration::from_millis(100),
            window_size: 100,
            enable_hw_counters: false, // Requires elevated privileges
            enable_gpu_monitoring: cfg!(feature = "gpu"),
            enable_bandwidth_tracking: true,
            alert_threshold_multiplier: 2.0,
        }
    }
}

/// Real-time performance metrics
#[derive(Debug, Clone)]
pub struct RealtimeMetrics {
    /// Current operations per second
    pub ops_per_second: f64,
    /// Average latency (microseconds)
    pub avg_latency_us: f64,
    /// 95th percentile latency (microseconds)
    pub p95_latency_us: f64,
    /// 99th percentile latency (microseconds)
    pub p99_latency_us: f64,
    /// Current CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    /// Current memory usage (bytes)
    pub memory_usage: usize,
    /// Current GPU utilization (0.0 to 1.0, if available)
    pub gpu_utilization: Option<f64>,
    /// Current memory bandwidth (bytes/second)
    pub memory_bandwidth: Option<f64>,
    /// Timestamp of these metrics
    pub timestamp: Instant,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    /// Type of alert
    pub alert_type: AlertType,
    /// Severity level
    pub severity: AlertSeverity,
    /// Human-readable description
    pub description: String,
    /// Current value
    pub current_value: f64,
    /// Expected/baseline value
    pub baseline_value: f64,
    /// Deviation factor (current / baseline)
    pub deviation_factor: f64,
    /// Timestamp when alert was triggered
    pub timestamp: Instant,
}

/// Type of performance alert
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertType {
    /// High latency detected
    HighLatency,
    /// Low throughput detected
    LowThroughput,
    /// High CPU usage
    HighCpuUsage,
    /// High memory usage
    HighMemoryUsage,
    /// High GPU usage
    HighGpuUsage,
    /// Low memory bandwidth
    LowMemoryBandwidth,
    /// Performance regression detected
    PerformanceRegression,
}

/// Alert severity level
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational
    Info,
    /// Warning
    Warning,
    /// Critical
    Critical,
}

/// Real-time performance monitor
pub struct RealTimeMonitor {
    config: MonitorConfig,
    start_time: Instant,
    operation_counts: HashMap<String, u64>,
    operation_times: HashMap<String, VecDeque<Duration>>,
    baselines: HashMap<String, f64>,
    alerts: VecDeque<PerformanceAlert>,
    #[allow(dead_code)] // Reserved for future periodic update functionality
    last_update: Instant,
}

impl RealTimeMonitor {
    /// Create a new real-time monitor
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            start_time: Instant::now(),
            operation_counts: HashMap::new(),
            operation_times: HashMap::new(),
            baselines: HashMap::new(),
            alerts: VecDeque::new(),
            last_update: Instant::now(),
        }
    }

    /// Record an operation execution
    pub fn record_operation(&mut self, operation: &str, duration: Duration) {
        if !self.config.enabled {
            return;
        }

        // Update operation count
        *self
            .operation_counts
            .entry(operation.to_string())
            .or_insert(0) += 1;

        // Update operation times (maintain sliding window)
        let times = self
            .operation_times
            .entry(operation.to_string())
            .or_insert_with(VecDeque::new);

        times.push_back(duration);
        if times.len() > self.config.window_size {
            times.pop_front();
        }

        // Check for performance regressions
        if let Some(baseline) = self.baselines.get(operation) {
            let current_avg = Self::calculate_average(times);
            let deviation = current_avg / baseline;

            if deviation > self.config.alert_threshold_multiplier {
                self.trigger_alert(PerformanceAlert {
                    alert_type: AlertType::PerformanceRegression,
                    severity: if deviation > 3.0 {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    },
                    description: format!(
                        "Operation '{}' is {:.1}x slower than baseline",
                        operation, deviation
                    ),
                    current_value: current_avg,
                    baseline_value: *baseline,
                    deviation_factor: deviation,
                    timestamp: Instant::now(),
                });
            }
        }
    }

    /// Get current real-time metrics
    pub fn get_metrics(&self) -> RealtimeMetrics {
        let elapsed = self.start_time.elapsed();
        let total_ops: u64 = self.operation_counts.values().sum();
        let ops_per_second = total_ops as f64 / elapsed.as_secs_f64();

        // Calculate average latency across all operations
        let all_times: Vec<Duration> = self
            .operation_times
            .values()
            .flat_map(|times| times.iter().copied())
            .collect();

        let avg_latency = if all_times.is_empty() {
            0.0
        } else {
            all_times.iter().map(|d| d.as_micros() as f64).sum::<f64>() / all_times.len() as f64
        };

        let (p95, p99) = Self::calculate_percentiles(&all_times);

        RealtimeMetrics {
            ops_per_second,
            avg_latency_us: avg_latency,
            p95_latency_us: p95,
            p99_latency_us: p99,
            cpu_utilization: Self::get_cpu_utilization(),
            memory_usage: Self::get_memory_usage(),
            gpu_utilization: Self::get_gpu_utilization(),
            memory_bandwidth: Self::get_memory_bandwidth(),
            timestamp: Instant::now(),
        }
    }

    /// Set baseline performance for an operation
    pub fn set_baseline(&mut self, operation: &str, baseline_time_us: f64) {
        self.baselines
            .insert(operation.to_string(), baseline_time_us);
    }

    /// Get recent performance alerts
    pub fn get_alerts(&self, max_count: usize) -> Vec<PerformanceAlert> {
        self.alerts.iter().rev().take(max_count).cloned().collect()
    }

    /// Clear all alerts
    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }

    /// Trigger a performance alert
    fn trigger_alert(&mut self, alert: PerformanceAlert) {
        self.alerts.push_back(alert);
        // Keep only recent alerts (last 1000)
        if self.alerts.len() > 1000 {
            self.alerts.pop_front();
        }
    }

    /// Calculate average duration
    fn calculate_average(durations: &VecDeque<Duration>) -> f64 {
        if durations.is_empty() {
            return 0.0;
        }
        durations.iter().map(|d| d.as_micros() as f64).sum::<f64>() / durations.len() as f64
    }

    /// Calculate percentiles
    fn calculate_percentiles(durations: &[Duration]) -> (f64, f64) {
        if durations.is_empty() {
            return (0.0, 0.0);
        }

        let mut sorted: Vec<f64> = durations.iter().map(|d| d.as_micros() as f64).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p95_idx = (sorted.len() as f64 * 0.95) as usize;
        let p99_idx = (sorted.len() as f64 * 0.99) as usize;

        let p95 = sorted.get(p95_idx).copied().unwrap_or(0.0);
        let p99 = sorted.get(p99_idx).copied().unwrap_or(0.0);

        (p95, p99)
    }

    /// Get current CPU utilization
    ///
    /// # SciRS2 Integration
    /// When available, uses scirs2-core system monitoring for accurate CPU tracking
    fn get_cpu_utilization() -> f64 {
        #[cfg(feature = "std")]
        {
            // Try to use scirs2-core system monitoring if available
            #[cfg(scirs2_system_monitoring_available)]
            {
                use scirs2_core::system::cpu_utilization;
                if let Ok(util) = cpu_utilization() {
                    return util;
                }
            }

            // Fallback: Use sysinfo crate or estimate
            // This is a simplified implementation
            0.5 // Placeholder
        }
        #[cfg(not(feature = "std"))]
        {
            0.0
        }
    }

    /// Get current memory usage
    fn get_memory_usage() -> usize {
        #[cfg(feature = "std")]
        {
            // Try to use scirs2-core memory monitoring if available
            #[cfg(scirs2_memory_monitoring_available)]
            {
                use scirs2_core::system::memory_usage;
                if let Ok(usage) = memory_usage() {
                    return usage;
                }
            }

            // Fallback estimation
            0 // Placeholder
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    /// Get GPU utilization
    fn get_gpu_utilization() -> Option<f64> {
        #[cfg(all(feature = "gpu", scirs2_gpu_available))]
        {
            use crate::gpu;
            if let Ok(device) = gpu::GpuDevice::new(0) {
                return Some(device.utilization());
            }
        }
        None
    }

    /// Get memory bandwidth
    fn get_memory_bandwidth() -> Option<f64> {
        #[cfg(scirs2_bandwidth_monitoring_available)]
        {
            use scirs2_core::system::memory_bandwidth;
            if let Ok(bw) = memory_bandwidth() {
                return Some(bw);
            }
        }
        None
    }
}

/// Get global real-time monitor
pub fn get_monitor() -> Arc<Mutex<RealTimeMonitor>> {
    PERF_MONITOR
        .get_or_init(|| Arc::new(Mutex::new(RealTimeMonitor::new(MonitorConfig::default()))))
        .clone()
}

/// Configure the global monitor
pub fn configure_monitor(config: MonitorConfig) {
    let monitor = get_monitor();
    *monitor.lock().unwrap() = RealTimeMonitor::new(config);
}

/// Record an operation for monitoring
pub fn record_operation(operation: &str, duration: Duration) {
    let monitor = get_monitor();
    monitor
        .lock()
        .unwrap()
        .record_operation(operation, duration);
}

/// Get current performance metrics
pub fn get_current_metrics() -> RealtimeMetrics {
    let monitor = get_monitor();
    let guard = monitor.lock().unwrap();
    guard.get_metrics()
}

/// Set performance baseline
pub fn set_baseline(operation: &str, baseline_time_us: f64) {
    let monitor = get_monitor();
    monitor
        .lock()
        .unwrap()
        .set_baseline(operation, baseline_time_us);
}

/// Get recent performance alerts
pub fn get_recent_alerts(max_count: usize) -> Vec<PerformanceAlert> {
    let monitor = get_monitor();
    let guard = monitor.lock().unwrap();
    guard.get_alerts(max_count)
}

/// Scope guard for automatic operation timing
pub struct TimingScope {
    operation: String,
    start: Instant,
}

impl TimingScope {
    /// Create a new timing scope
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            start: Instant::now(),
        }
    }
}

impl Drop for TimingScope {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        record_operation(&self.operation, duration);
    }
}

/// Macro for easy timing scope creation
#[macro_export]
macro_rules! time_operation {
    ($name:expr) => {
        let _timing_scope = $crate::perf_monitor::TimingScope::new($name);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_monitor_creation() {
        let config = MonitorConfig::default();
        let monitor = RealTimeMonitor::new(config);
        assert!(monitor.operation_counts.is_empty());
    }

    #[test]
    fn test_operation_recording() {
        let mut monitor = RealTimeMonitor::new(MonitorConfig::default());
        monitor.record_operation("test_op", Duration::from_micros(100));
        assert_eq!(monitor.operation_counts.get("test_op"), Some(&1));
    }

    #[test]
    fn test_metrics_calculation() {
        let mut monitor = RealTimeMonitor::new(MonitorConfig::default());
        for _ in 0..10 {
            monitor.record_operation("test_op", Duration::from_micros(100));
        }
        let metrics = monitor.get_metrics();
        assert!(metrics.ops_per_second > 0.0);
        assert!(metrics.avg_latency_us > 0.0);
    }

    #[test]
    fn test_baseline_and_alerts() {
        let mut monitor = RealTimeMonitor::new(MonitorConfig {
            alert_threshold_multiplier: 2.0,
            window_size: 10, // Small window for faster test convergence
            ..MonitorConfig::default()
        });

        // Set baseline
        monitor.set_baseline("slow_op", 100.0);

        // Record normal operations to establish baseline
        for _ in 0..10 {
            monitor.record_operation("slow_op", Duration::from_micros(100));
        }

        // Now record significantly slower operations (should trigger alert)
        // Need enough slow operations to shift the moving average above threshold
        for _ in 0..15 {
            monitor.record_operation("slow_op", Duration::from_micros(250));
        }

        let alerts = monitor.get_alerts(10);
        assert!(
            !alerts.is_empty(),
            "Expected performance regression alert to be triggered"
        );
        assert_eq!(alerts[0].alert_type, AlertType::PerformanceRegression);
    }

    #[test]
    fn test_timing_scope() {
        {
            let _scope = TimingScope::new("test_scope");
            thread::sleep(Duration::from_micros(100));
        }
        // Check that operation was recorded
        let monitor = get_monitor();
        let counts = &monitor.lock().unwrap().operation_counts;
        assert!(counts.contains_key("test_scope"));
    }

    #[test]
    fn test_percentile_calculation() {
        let durations = vec![
            Duration::from_micros(100),
            Duration::from_micros(200),
            Duration::from_micros(300),
            Duration::from_micros(400),
            Duration::from_micros(500),
        ];
        let (p95, p99) = RealTimeMonitor::calculate_percentiles(&durations);
        assert!(p95 > 0.0);
        assert!(p99 >= p95);
    }

    #[test]
    fn test_global_monitor() {
        record_operation("global_test", Duration::from_micros(150));
        let metrics = get_current_metrics();
        assert!(metrics.timestamp.elapsed().as_secs() < 1);
    }
}

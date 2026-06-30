//! Performance metrics collection for distributed training
//!
//! This module provides comprehensive performance metrics collection including
//! system resources, communication efficiency, training progress, and real-time monitoring.
//!
//! Enhanced with SciRS2 profiling and benchmarking capabilities for production-ready
//! performance analysis and optimization.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::profiling::get_global_profiler;
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

// Enhanced SciRS2 integration for advanced profiling
// TODO: These features are not yet available in scirs2_core
// Uncomment when scirs2_core provides these modules
// #[cfg(feature = "scirs2-profiling")]
// use scirs2_core::benchmarking::{BenchmarkRunner, BenchmarkSuite};
// #[cfg(feature = "scirs2-profiling")]
// Metrics types available when needed
// use scirs2_core::metrics::{Counter, Gauge, Histogram, MetricsRegistry, Timer};
// #[cfg(feature = "scirs2-profiling")]
// use scirs2_core::observability::{audit, tracing as scirs2_tracing};
// #[cfg(feature = "scirs2-profiling")]
// use scirs2_core::profiling::{profiling_memory_tracker, Profiler};

/// Enhanced system resource metrics with SciRS2 profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU utilization percentage (0.0 to 100.0)
    pub cpu_usage_pct: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Available memory in bytes
    pub memory_available_bytes: u64,
    /// Memory utilization percentage (0.0 to 100.0)
    pub memory_usage_pct: f64,
    /// GPU memory usage in bytes (if available)
    pub gpu_memory_usage_bytes: Option<u64>,
    /// GPU memory total in bytes (if available)
    pub gpu_memory_total_bytes: Option<u64>,
    /// GPU utilization percentage (0.0 to 100.0, if available)
    pub gpu_usage_pct: Option<f64>,
    /// Network bytes received
    pub network_bytes_rx: u64,
    /// Network bytes transmitted
    pub network_bytes_tx: u64,
    /// Disk read bytes
    pub disk_bytes_read: u64,
    /// Disk write bytes
    pub disk_bytes_write: u64,
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
    /// SciRS2 profiling data
    #[cfg(feature = "scirs2-profiling")]
    pub scirs2_profile: Option<HashMap<String, f64>>,
    /// Memory profiling data
    #[cfg(feature = "scirs2-profiling")]
    pub memory_profile: Option<HashMap<String, u64>>,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            cpu_usage_pct: 0.0,
            memory_usage_bytes: 0,
            memory_available_bytes: 0,
            memory_usage_pct: 0.0,
            gpu_memory_usage_bytes: None,
            gpu_memory_total_bytes: None,
            gpu_usage_pct: None,
            network_bytes_rx: 0,
            network_bytes_tx: 0,
            disk_bytes_read: 0,
            disk_bytes_write: 0,
            timestamp: SystemTime::now(),
            #[cfg(feature = "scirs2-profiling")]
            scirs2_profile: None,
            #[cfg(feature = "scirs2-profiling")]
            memory_profile: None,
        }
    }
}

/// Communication performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationMetrics {
    /// Total communication operations performed
    pub total_operations: u64,
    /// Total bytes communicated
    pub total_bytes: u64,
    /// Average communication latency in milliseconds
    pub avg_latency_ms: f64,
    /// Average bandwidth in MB/s
    pub avg_bandwidth_mbps: f64,
    /// Communication efficiency (0.0 to 1.0)
    pub efficiency_ratio: f64,
    /// Time spent in communication vs computation
    pub communication_compute_ratio: f64,
    /// Number of failed operations
    pub failed_operations: u64,
    /// Operations per second
    pub ops_per_second: f64,
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
}

impl Default for CommunicationMetrics {
    fn default() -> Self {
        Self {
            total_operations: 0,
            total_bytes: 0,
            avg_latency_ms: 0.0,
            avg_bandwidth_mbps: 0.0,
            efficiency_ratio: 1.0,
            communication_compute_ratio: 0.0,
            failed_operations: 0,
            ops_per_second: 0.0,
            timestamp: SystemTime::now(),
        }
    }
}

/// Training progress metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Current epoch
    pub current_epoch: u32,
    /// Current step/iteration
    pub current_step: u64,
    /// Training loss
    pub training_loss: Option<f64>,
    /// Validation loss
    pub validation_loss: Option<f64>,
    /// Learning rate
    pub learning_rate: Option<f64>,
    /// Gradient norm
    pub gradient_norm: Option<f64>,
    /// Samples processed per second
    pub samples_per_second: f64,
    /// Model parameters count
    pub model_parameters: u64,
    /// Time per step in milliseconds
    pub time_per_step_ms: f64,
    /// Estimated time remaining
    pub eta_seconds: Option<u64>,
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            current_epoch: 0,
            current_step: 0,
            training_loss: None,
            validation_loss: None,
            learning_rate: None,
            gradient_norm: None,
            samples_per_second: 0.0,
            model_parameters: 0,
            time_per_step_ms: 0.0,
            eta_seconds: None,
            timestamp: SystemTime::now(),
        }
    }
}

/// Comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// System resource metrics
    pub system: SystemMetrics,
    /// Communication performance metrics
    pub communication: CommunicationMetrics,
    /// Training progress metrics
    pub training: TrainingMetrics,
    /// Overall timestamp
    pub timestamp: SystemTime,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            system: SystemMetrics::default(),
            communication: CommunicationMetrics::default(),
            training: TrainingMetrics::default(),
            timestamp: SystemTime::now(),
        }
    }
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Whether metrics collection is enabled
    pub enabled: bool,
    /// Collection interval in seconds
    pub collection_interval_secs: u64,
    /// Maximum number of metric snapshots to keep
    pub max_snapshots: usize,
    /// Whether to collect system metrics
    pub collect_system_metrics: bool,
    /// Whether to collect communication metrics
    pub collect_communication_metrics: bool,
    /// Whether to collect training metrics
    pub collect_training_metrics: bool,
    /// Whether to export metrics to external systems
    pub enable_export: bool,
    /// Export interval in seconds
    pub export_interval_secs: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_secs: 5,
            max_snapshots: 1000,
            collect_system_metrics: true,
            collect_communication_metrics: true,
            collect_training_metrics: true,
            enable_export: false,
            export_interval_secs: 60,
        }
    }
}

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint<T> {
    pub timestamp: SystemTime,
    pub value: T,
}

impl<T> TimeSeriesPoint<T> {
    pub fn new(value: T) -> Self {
        Self {
            timestamp: SystemTime::now(),
            value,
        }
    }
}

/// Time series data collection
#[derive(Debug)]
pub struct TimeSeries<T> {
    points: VecDeque<TimeSeriesPoint<T>>,
    max_size: usize,
}

impl<T> TimeSeries<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            points: VecDeque::new(),
            max_size,
        }
    }

    pub fn add_point(&mut self, value: T) {
        self.points.push_back(TimeSeriesPoint::new(value));
        if self.points.len() > self.max_size {
            self.points.pop_front();
        }
    }

    pub fn get_points(&self) -> &VecDeque<TimeSeriesPoint<T>> {
        &self.points
    }

    pub fn latest(&self) -> Option<&TimeSeriesPoint<T>> {
        self.points.back()
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl<T: Clone> TimeSeries<T> {
    pub fn get_range(&self, start: SystemTime, end: SystemTime) -> Vec<TimeSeriesPoint<T>> {
        self.points
            .iter()
            .filter(|point| point.timestamp >= start && point.timestamp <= end)
            .cloned()
            .collect()
    }

    pub fn get_last_n(&self, n: usize) -> Vec<TimeSeriesPoint<T>> {
        self.points
            .iter()
            .rev()
            .take(n)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }
}

/// Performance metrics collector
pub struct MetricsCollector {
    /// Configuration
    config: RwLock<MetricsConfig>,
    /// Time series of performance metrics
    metrics_history: Mutex<TimeSeries<PerformanceMetrics>>,
    /// System metrics history
    system_history: Mutex<TimeSeries<SystemMetrics>>,
    /// Communication metrics history
    communication_history: Mutex<TimeSeries<CommunicationMetrics>>,
    /// Training metrics history
    training_history: Mutex<TimeSeries<TrainingMetrics>>,
    /// Last collection timestamp
    last_collection: Mutex<Option<Instant>>,
    /// Collection thread handle
    collection_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Shutdown flag
    shutdown: Arc<Mutex<bool>>,
    /// Custom metrics
    custom_metrics: RwLock<HashMap<String, f64>>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self::with_config(MetricsConfig::default())
    }

    /// Create a new metrics collector with custom configuration
    pub fn with_config(config: MetricsConfig) -> Self {
        let max_snapshots = config.max_snapshots;
        Self {
            config: RwLock::new(config),
            metrics_history: Mutex::new(TimeSeries::new(max_snapshots)),
            system_history: Mutex::new(TimeSeries::new(max_snapshots)),
            communication_history: Mutex::new(TimeSeries::new(max_snapshots)),
            training_history: Mutex::new(TimeSeries::new(max_snapshots)),
            last_collection: Mutex::new(None),
            collection_thread: Mutex::new(None),
            shutdown: Arc::new(Mutex::new(false)),
            custom_metrics: RwLock::new(HashMap::new()),
        }
    }

    /// Start automatic metrics collection
    pub fn start_collection(&self) -> TorshResult<()> {
        let config = self
            .config
            .read()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;

        if !config.enabled {
            return Ok(());
        }

        let interval = Duration::from_secs(config.collection_interval_secs);
        let collect_system = config.collect_system_metrics;
        let collect_communication = config.collect_communication_metrics;
        let collect_training = config.collect_training_metrics;

        drop(config);

        let shutdown_flag = Arc::clone(&self.shutdown);
        let metrics_history = Arc::new(Mutex::new(TimeSeries::new(1000)));
        let system_history = Arc::new(Mutex::new(TimeSeries::new(1000)));
        let communication_history = Arc::new(Mutex::new(TimeSeries::new(1000)));
        let training_history = Arc::new(Mutex::new(TimeSeries::new(1000)));

        let handle = std::thread::spawn(move || {
            loop {
                // Check shutdown flag
                {
                    let shutdown = shutdown_flag.lock().expect("lock should not be poisoned");
                    if *shutdown {
                        break;
                    }
                }

                // Collect metrics
                let mut performance_metrics = PerformanceMetrics::default();

                if collect_system {
                    let system_metrics = Self::collect_system_metrics();
                    performance_metrics.system = system_metrics.clone();

                    if let Ok(mut history) = system_history.lock() {
                        history.add_point(system_metrics);
                    }
                }

                if collect_communication {
                    let communication_metrics = Self::collect_communication_metrics();
                    performance_metrics.communication = communication_metrics.clone();

                    if let Ok(mut history) = communication_history.lock() {
                        history.add_point(communication_metrics);
                    }
                }

                if collect_training {
                    let training_metrics = Self::collect_training_metrics();
                    performance_metrics.training = training_metrics.clone();

                    if let Ok(mut history) = training_history.lock() {
                        history.add_point(training_metrics);
                    }
                }

                // Store combined metrics
                if let Ok(mut history) = metrics_history.lock() {
                    history.add_point(performance_metrics);
                }

                std::thread::sleep(interval);
            }
        });

        let mut thread_handle = self
            .collection_thread
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
        *thread_handle = Some(handle);

        Ok(())
    }

    /// Stop metrics collection
    pub fn stop_collection(&self) -> TorshResult<()> {
        // Set shutdown flag
        {
            let mut shutdown = self
                .shutdown
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
            *shutdown = true;
        }

        // Wait for thread to finish
        let mut thread_handle = self
            .collection_thread
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
        if let Some(handle) = thread_handle.take() {
            handle.join().map_err(|_| {
                TorshDistributedError::backend_error("system", "Failed to join collection thread")
            })?;
        }

        // Reset shutdown flag
        {
            let mut shutdown = self
                .shutdown
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
            *shutdown = false;
        }

        Ok(())
    }

    /// Collect real system metrics from /proc on Linux; returns zeros on other platforms.
    fn collect_system_metrics() -> SystemMetrics {
        #[cfg(target_os = "linux")]
        {
            // --- CPU utilisation via two /proc/stat samples 100 ms apart ---
            fn read_cpu_jiffies() -> Option<(u64, u64)> {
                let content = std::fs::read_to_string("/proc/stat").ok()?;
                let first_line = content.lines().next()?;
                // "cpu  user nice system idle iowait irq softirq steal guest guest_nice"
                let mut parts = first_line.split_whitespace();
                parts.next(); // skip "cpu"
                let values: Vec<u64> = parts.filter_map(|s| s.parse().ok()).collect();
                if values.len() < 4 {
                    return None;
                }
                let idle = values[3];
                let total: u64 = values.iter().sum();
                Some((idle, total))
            }

            let cpu_usage_pct = if let (Some((idle0, total0)), _) = (read_cpu_jiffies(), ()) {
                std::thread::sleep(Duration::from_millis(100));
                if let Some((idle1, total1)) = read_cpu_jiffies() {
                    let d_total = total1.saturating_sub(total0) as f64;
                    let d_idle = idle1.saturating_sub(idle0) as f64;
                    if d_total > 0.0 {
                        (1.0 - d_idle / d_total) * 100.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // --- Memory via /proc/meminfo ---
            let (memory_usage_bytes, memory_available_bytes, memory_usage_pct) = {
                let mut mem_total: u64 = 0;
                let mut mem_available: u64 = 0;
                if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                    for line in content.lines() {
                        let mut parts = line.split_whitespace();
                        match parts.next() {
                            Some("MemTotal:") => {
                                mem_total = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                                mem_total *= 1024; // kB -> bytes
                            }
                            Some("MemAvailable:") => {
                                mem_available =
                                    parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                                mem_available *= 1024;
                            }
                            _ => {}
                        }
                    }
                }
                let usage = mem_total.saturating_sub(mem_available);
                let pct = if mem_total > 0 {
                    usage as f64 / mem_total as f64 * 100.0
                } else {
                    0.0
                };
                (usage, mem_available, pct)
            };

            // --- Network via /proc/net/dev (cumulative counters) ---
            let (network_bytes_rx, network_bytes_tx) = {
                let mut rx: u64 = 0;
                let mut tx: u64 = 0;
                if let Ok(content) = std::fs::read_to_string("/proc/net/dev") {
                    for line in content.lines().skip(2) {
                        // skip header lines
                        // Format: iface: rx_bytes rx_pkts ... tx_bytes ...
                        if let Some(colon_pos) = line.find(':') {
                            let iface = line[..colon_pos].trim();
                            // Skip loopback
                            if iface == "lo" {
                                continue;
                            }
                            let fields: Vec<u64> = line[colon_pos + 1..]
                                .split_whitespace()
                                .filter_map(|s| s.parse().ok())
                                .collect();
                            // columns: rx_bytes(0) rx_pkts(1)...tx_bytes(8) tx_pkts(9)...
                            if fields.len() >= 9 {
                                rx = rx.saturating_add(fields[0]);
                                tx = tx.saturating_add(fields[8]);
                            }
                        }
                    }
                }
                (rx, tx)
            };

            // --- Disk via /proc/diskstats (sectors read/written × 512) ---
            let (disk_bytes_read, disk_bytes_write) = {
                let mut reads: u64 = 0;
                let mut writes: u64 = 0;
                if let Ok(content) = std::fs::read_to_string("/proc/diskstats") {
                    for line in content.lines() {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        // Only count whole-disk devices (no partition numbers in name)
                        if parts.len() >= 14 {
                            let name = parts[2];
                            // Skip loop, ram, partition entries (ends with digit after letter)
                            let is_partition = name
                                .chars()
                                .last()
                                .map(|c| c.is_ascii_digit())
                                .unwrap_or(false)
                                && name
                                    .chars()
                                    .rev()
                                    .skip(1)
                                    .next()
                                    .map(|c| !c.is_ascii_digit())
                                    .unwrap_or(false);
                            if !is_partition
                                && !name.starts_with("loop")
                                && !name.starts_with("ram")
                            {
                                let sectors_read: u64 = parts[5].parse().unwrap_or(0);
                                let sectors_written: u64 = parts[9].parse().unwrap_or(0);
                                reads = reads.saturating_add(sectors_read * 512);
                                writes = writes.saturating_add(sectors_written * 512);
                            }
                        }
                    }
                }
                (reads, writes)
            };

            SystemMetrics {
                cpu_usage_pct,
                memory_usage_bytes,
                memory_available_bytes,
                memory_usage_pct,
                // GPU metrics require NVML/cust bindings; report None without CUDA feature.
                gpu_memory_usage_bytes: None,
                gpu_memory_total_bytes: None,
                gpu_usage_pct: None,
                network_bytes_rx,
                network_bytes_tx,
                disk_bytes_read,
                disk_bytes_write,
                timestamp: SystemTime::now(),
                #[cfg(feature = "scirs2-profiling")]
                scirs2_profile: None,
                #[cfg(feature = "scirs2-profiling")]
                memory_profile: None,
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            // Non-Linux platforms: return zeros rather than fabricate values.
            SystemMetrics::default()
        }
    }

    /// Collect communication metrics from profiler
    fn collect_communication_metrics() -> CommunicationMetrics {
        let profiler = get_global_profiler();

        // Get all operation stats and aggregate
        if let Ok(all_stats) = profiler.get_all_operation_stats() {
            let mut total_ops = 0u64;
            let mut total_bytes = 0u64;
            let mut total_duration = Duration::ZERO;
            let mut total_bandwidth = 0.0;

            for stats in all_stats.values() {
                total_ops += stats.count;
                total_bytes += stats.total_bytes;
                total_duration += stats.total_duration;
                total_bandwidth += stats.avg_bandwidth_bps;
            }

            let avg_latency_ms = if total_ops > 0 {
                total_duration.as_secs_f64() * 1000.0 / total_ops as f64
            } else {
                0.0
            };

            let avg_bandwidth_mbps = if !all_stats.is_empty() {
                total_bandwidth / (all_stats.len() as f64 * 1024.0 * 1024.0)
            } else {
                0.0
            };

            CommunicationMetrics {
                total_operations: total_ops,
                total_bytes,
                avg_latency_ms,
                avg_bandwidth_mbps,
                efficiency_ratio: 0.85,           // Placeholder calculation
                communication_compute_ratio: 0.2, // Placeholder
                failed_operations: get_global_profiler().get_failed_operations_count(), // Track actual failures
                ops_per_second: total_ops as f64 / total_duration.as_secs_f64().max(1.0),
                timestamp: SystemTime::now(),
            }
        } else {
            CommunicationMetrics::default()
        }
    }

    /// Collect training metrics (placeholder implementation)
    fn collect_training_metrics() -> TrainingMetrics {
        // This would typically be updated by the training loop
        TrainingMetrics::default()
    }

    /// Update training metrics manually
    pub fn update_training_metrics(&self, metrics: TrainingMetrics) -> TorshResult<()> {
        let mut history = self
            .training_history
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
        history.add_point(metrics);
        Ok(())
    }

    /// Add custom metric
    pub fn add_custom_metric(&self, name: String, value: f64) -> TorshResult<()> {
        let mut custom_metrics = self
            .custom_metrics
            .write()
            .map_err(|_| TorshDistributedError::backend_error("metrics", "Lock poisoned"))?;
        custom_metrics.insert(name, value);
        Ok(())
    }

    /// Get custom metric
    pub fn get_custom_metric(&self, name: &str) -> TorshResult<Option<f64>> {
        let custom_metrics = self
            .custom_metrics
            .read()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
        Ok(custom_metrics.get(name).copied())
    }

    /// Get latest performance metrics
    pub fn get_latest_metrics(&self) -> TorshResult<Option<PerformanceMetrics>> {
        let history = self
            .metrics_history
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
        Ok(history.latest().map(|point| point.value.clone()))
    }

    /// Get metrics history
    pub fn get_metrics_history(&self) -> TorshResult<Vec<TimeSeriesPoint<PerformanceMetrics>>> {
        let history = self
            .metrics_history
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
        Ok(history.get_points().iter().cloned().collect())
    }

    /// Get system metrics history
    pub fn get_system_history(&self) -> TorshResult<Vec<TimeSeriesPoint<SystemMetrics>>> {
        let history = self
            .system_history
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
        Ok(history.get_points().iter().cloned().collect())
    }

    /// Get communication metrics history
    pub fn get_communication_history(
        &self,
    ) -> TorshResult<Vec<TimeSeriesPoint<CommunicationMetrics>>> {
        let history = self
            .communication_history
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
        Ok(history.get_points().iter().cloned().collect())
    }

    /// Get training metrics history
    pub fn get_training_history(&self) -> TorshResult<Vec<TimeSeriesPoint<TrainingMetrics>>> {
        let history = self
            .training_history
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
        Ok(history.get_points().iter().cloned().collect())
    }

    /// Export metrics to JSON
    pub fn export_metrics_json(&self) -> TorshResult<String> {
        #[derive(Serialize)]
        struct MetricsExport {
            config: MetricsConfig,
            latest_metrics: Option<PerformanceMetrics>,
            system_history: Vec<TimeSeriesPoint<SystemMetrics>>,
            communication_history: Vec<TimeSeriesPoint<CommunicationMetrics>>,
            training_history: Vec<TimeSeriesPoint<TrainingMetrics>>,
            custom_metrics: HashMap<String, f64>,
        }

        let config = self
            .config
            .read()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?
            .clone();
        let latest_metrics = self.get_latest_metrics()?;
        let system_history = self.get_system_history()?;
        let communication_history = self.get_communication_history()?;
        let training_history = self.get_training_history()?;
        let custom_metrics = self
            .custom_metrics
            .read()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?
            .clone();

        let export = MetricsExport {
            config,
            latest_metrics,
            system_history,
            communication_history,
            training_history,
            custom_metrics,
        };

        serde_json::to_string_pretty(&export).map_err(|e| {
            TorshDistributedError::backend_error(
                "metrics",
                format!("JSON serialization failed: {}", e),
            )
        })
    }

    /// Generate metrics summary report
    pub fn generate_summary(&self) -> TorshResult<String> {
        let mut report = String::new();
        report.push_str("=== Performance Metrics Summary ===\n\n");

        if let Some(latest) = self.get_latest_metrics()? {
            report.push_str("=== Latest System Metrics ===\n");
            report.push_str(&format!("CPU Usage: {:.1}%\n", latest.system.cpu_usage_pct));
            report.push_str(&format!(
                "Memory Usage: {:.1}% ({:.1} GB / {:.1} GB)\n",
                latest.system.memory_usage_pct,
                latest.system.memory_usage_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                latest.system.memory_available_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            ));

            if let Some(gpu_usage) = latest.system.gpu_usage_pct {
                report.push_str(&format!("GPU Usage: {:.1}%\n", gpu_usage));
            }

            report.push_str("\n=== Latest Communication Metrics ===\n");
            report.push_str(&format!(
                "Total Operations: {}\n",
                latest.communication.total_operations
            ));
            report.push_str(&format!(
                "Total Data: {:.1} MB\n",
                latest.communication.total_bytes as f64 / (1024.0 * 1024.0)
            ));
            report.push_str(&format!(
                "Average Latency: {:.2} ms\n",
                latest.communication.avg_latency_ms
            ));
            report.push_str(&format!(
                "Average Bandwidth: {:.2} MB/s\n",
                latest.communication.avg_bandwidth_mbps
            ));
            report.push_str(&format!(
                "Operations/sec: {:.1}\n",
                latest.communication.ops_per_second
            ));

            report.push_str("\n=== Latest Training Metrics ===\n");
            report.push_str(&format!(
                "Current Epoch: {}\n",
                latest.training.current_epoch
            ));
            report.push_str(&format!("Current Step: {}\n", latest.training.current_step));

            if let Some(loss) = latest.training.training_loss {
                report.push_str(&format!("Training Loss: {:.6}\n", loss));
            }
            if let Some(lr) = latest.training.learning_rate {
                report.push_str(&format!("Learning Rate: {:.2e}\n", lr));
            }

            report.push_str(&format!(
                "Samples/sec: {:.1}\n",
                latest.training.samples_per_second
            ));
        }

        let custom_metrics = self
            .custom_metrics
            .read()
            .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
        if !custom_metrics.is_empty() {
            report.push_str("\n=== Custom Metrics ===\n");
            for (name, value) in custom_metrics.iter() {
                report.push_str(&format!("{}: {:.6}\n", name, value));
            }
        }

        Ok(report)
    }

    /// Clear all metrics data
    pub fn clear(&self) -> TorshResult<()> {
        {
            let mut history = self
                .metrics_history
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
            *history = TimeSeries::new(history.max_size);
        }

        {
            let mut history = self
                .system_history
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
            *history = TimeSeries::new(history.max_size);
        }

        {
            let mut history = self
                .communication_history
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
            *history = TimeSeries::new(history.max_size);
        }

        {
            let mut history = self
                .training_history
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;
            *history = TimeSeries::new(history.max_size);
        }

        {
            let mut custom_metrics = self
                .custom_metrics
                .write()
                .map_err(|_| TorshDistributedError::backend_error("metrics", "Lock poisoned"))?;
            custom_metrics.clear();
        }

        Ok(())
    }

    // Enhanced SciRS2 integration methods

    /// Collect advanced system metrics using SciRS2 profiling
    #[cfg(feature = "scirs2-profiling")]
    pub fn collect_scirs2_system_metrics(&self) -> TorshResult<SystemMetrics> {
        // TODO: profiling module not yet available in scirs2_core
        // use scirs2_core::metrics::{Counter, Gauge, MetricsRegistry};
        // use scirs2_core::profiling::profiling_memory_tracker;

        let mut metrics = MetricsCollector::collect_system_metrics();

        // TODO: Enhanced memory profiling using SciRS2 - disabled until profiling module is available
        // Enhanced memory profiling using SciRS2
        // if let Ok(memory_tracker) = profiling_memory_tracker() {
        //     let mut memory_profile = HashMap::new();
        //     memory_profile.insert(
        //         "peak_memory_usage".to_string(),
        //         memory_tracker.peak_usage_bytes(),
        //     );
        //     memory_profile.insert(
        //         "current_allocations".to_string(),
        //         memory_tracker.current_allocations() as u64,
        //     );
        //     memory_profile.insert(
        //         "total_allocations".to_string(),
        //         memory_tracker.total_allocations() as u64,
        //     );
        //     memory_profile.insert(
        //         "fragmentation_ratio".to_string(),
        //         (memory_tracker.fragmentation_ratio() * 1000.0) as u64,
        //     );
        //     metrics.memory_profile = Some(memory_profile);
        // }

        // SciRS2 profiling metrics
        let mut scirs2_profile = HashMap::new();

        // TODO: CPU profiling with enhanced precision - disabled until profiling module is available
        // CPU profiling with enhanced precision
        // if let Some(profiler) = Profiler::global() {
        //     scirs2_profile.insert(
        //         "cpu_efficiency".to_string(),
        //         profiler.cpu_efficiency_ratio(),
        //     );
        //     scirs2_profile.insert("cache_hit_ratio".to_string(), profiler.cache_hit_ratio());
        //     scirs2_profile.insert(
        //         "simd_utilization".to_string(),
        //         profiler.simd_utilization_ratio(),
        //     );
        //     scirs2_profile.insert(
        //         "vectorization_efficiency".to_string(),
        //         profiler.vectorization_efficiency(),
        //     );
        // }

        // Memory bandwidth and latency metrics
        scirs2_profile.insert(
            "memory_bandwidth_gbps".to_string(),
            self.estimate_memory_bandwidth(),
        );
        scirs2_profile.insert(
            "memory_latency_ns".to_string(),
            self.estimate_memory_latency(),
        );

        metrics.scirs2_profile = Some(scirs2_profile);
        Ok(metrics)
    }

    /// Run comprehensive benchmarks using SciRS2 benchmarking suite
    /// TODO: Disabled until benchmarking module is available in scirs2_core
    #[cfg(feature = "scirs2-profiling")]
    pub fn run_performance_benchmarks(&self) -> TorshResult<HashMap<String, f64>> {
        // TODO: benchmarking module not yet available in scirs2_core
        // use scirs2_core::benchmarking::{BenchmarkRunner, BenchmarkSuite};

        // Return empty results until benchmarking module is available
        let results = HashMap::new();

        // TODO: Implement when scirs2_core benchmarking module is available
        // let mut benchmark_suite = BenchmarkSuite::new("distributed_training_benchmarks");
        // ...

        Ok(results)
    }

    /// Enhanced metrics collection with SciRS2 observability
    /// TODO: Disabled until observability module is available in scirs2_core
    #[cfg(feature = "scirs2-profiling")]
    pub fn collect_enhanced_metrics(&self) -> TorshResult<PerformanceMetrics> {
        // TODO: observability module not yet available in scirs2_core
        // use scirs2_core::observability::{audit, tracing};

        // TODO: Start enhanced tracing when available
        // let _trace = tracing::span!("enhanced_metrics_collection");

        let system_metrics = self.collect_scirs2_system_metrics()?;
        let comm_metrics = MetricsCollector::collect_communication_metrics();
        let training_metrics = MetricsCollector::collect_training_metrics();

        // TODO: Create audit trail when observability module is available
        // audit::log_event(
        //     "metrics_collected",
        //     &format!(
        //         "system_cpu={:.1}%, memory={:.1}%, comm_ops={}, training_epoch={}",
        //         system_metrics.cpu_usage_pct,
        //         system_metrics.memory_usage_pct,
        //         comm_metrics.total_operations,
        //         training_metrics.current_epoch
        //     ),
        // );

        Ok(PerformanceMetrics {
            system: system_metrics,
            communication: comm_metrics,
            training: training_metrics,
            timestamp: SystemTime::now(),
        })
    }

    // Helper methods for benchmarking

    #[cfg(feature = "scirs2-profiling")]
    fn benchmark_memory_throughput_sequential(&self) -> Duration {
        let start = Instant::now();
        // Simulate sequential memory access pattern
        let mut data = vec![0u8; 1024 * 1024]; // 1MB
        for (i, item) in data.iter_mut().enumerate() {
            *item = (i % 256) as u8;
        }
        let _ = data.iter().sum::<u8>();
        start.elapsed()
    }

    #[cfg(feature = "scirs2-profiling")]
    fn benchmark_memory_throughput_random(&self) -> Duration {
        let start = Instant::now();
        // Simulate random memory access pattern
        let mut data = vec![0u8; 1024 * 1024]; // 1MB
        for i in (0..data.len()).step_by(4096) {
            // Random access pattern
            data[i] = ((i * 7) % 256) as u8;
        }
        let _ = data.iter().sum::<u8>();
        start.elapsed()
    }

    #[cfg(feature = "scirs2-profiling")]
    fn benchmark_network_latency(&self) -> Duration {
        // This would normally ping network endpoints
        // For now, simulate network latency
        let start = Instant::now();
        std::thread::sleep(Duration::from_micros(100)); // Simulate 100μs latency
        start.elapsed()
    }

    #[cfg(feature = "scirs2-profiling")]
    fn benchmark_tensor_operations(&self) -> Duration {
        let start = Instant::now();
        // Simulate basic tensor operations
        let a = vec![1.0f32; 1000];
        let b = vec![2.0f32; 1000];
        let _c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
        start.elapsed()
    }

    /// Estimate memory bandwidth by measuring a large sequential copy.
    ///
    /// The result is a crude user-space approximation; hardware performance
    /// counters (e.g. via `perf_event_open`) would give a more accurate figure
    /// but require kernel integration not yet available here.
    #[cfg(feature = "scirs2-profiling")]
    fn estimate_memory_bandwidth(&self) -> f64 {
        // Copy 64 MiB and measure wall-clock time to approximate bandwidth.
        const BYTES: usize = 64 * 1024 * 1024;
        let src = vec![0u8; BYTES];
        let mut dst = vec![0u8; BYTES];
        let start = Instant::now();
        dst.copy_from_slice(&src);
        let elapsed = start.elapsed();
        // Prevent the compiler from eliding the copy.
        let _ = dst[0];
        if elapsed.as_secs_f64() > 0.0 {
            // Two transfers (read + write) per byte.
            BYTES as f64 * 2.0 / elapsed.as_secs_f64() / 1e9
        } else {
            // Timing granularity too coarse — report 0.0 (not yet measured).
            0.0
        }
    }

    /// Estimate memory access latency by pointer-chasing through a small array.
    ///
    /// Returns 0.0 if the timing resolution is insufficient — the value is
    /// "not yet measured", not a fabricated estimate.
    #[cfg(feature = "scirs2-profiling")]
    fn estimate_memory_latency(&self) -> f64 {
        // Build a circular pointer-chase list to defeat prefetchers.
        const N: usize = 256; // 256 × 8 bytes = 2 KiB — fits in L1 to measure cache
        let mut chain = vec![0usize; N];
        // Simple shuffle: link[i] = (i * 7 + 1) % N
        for i in 0..N {
            chain[i] = (i * 7 + 1) % N;
        }
        let iterations = 10_000usize;
        let start = Instant::now();
        let mut idx = 0usize;
        for _ in 0..iterations {
            idx = chain[idx];
        }
        let elapsed = start.elapsed();
        // Prevent dead-code elimination.
        let _ = idx;
        if elapsed.as_nanos() > 0 {
            elapsed.as_nanos() as f64 / iterations as f64
        } else {
            0.0 // not yet measured
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MetricsCollector {
    fn drop(&mut self) {
        let _ = self.stop_collection();
    }
}

/// Global metrics collector instance
static GLOBAL_METRICS_COLLECTOR: std::sync::OnceLock<Arc<MetricsCollector>> =
    std::sync::OnceLock::new();

/// Get the global metrics collector instance
pub fn get_global_metrics_collector() -> &'static Arc<MetricsCollector> {
    GLOBAL_METRICS_COLLECTOR.get_or_init(|| Arc::new(MetricsCollector::new()))
}

/// Initialize the global metrics collector with custom configuration
pub fn init_global_metrics_collector(config: MetricsConfig) -> TorshResult<()> {
    let collector = Arc::new(MetricsCollector::with_config(config));
    GLOBAL_METRICS_COLLECTOR.set(collector).map_err(|_| {
        TorshDistributedError::backend_error(
            "metrics",
            "Global metrics collector already initialized",
        )
    })?;
    Ok(())
}

/// Start global metrics collection
pub fn start_global_metrics_collection() -> TorshResult<()> {
    get_global_metrics_collector().start_collection()
}

/// Stop global metrics collection
pub fn stop_global_metrics_collection() -> TorshResult<()> {
    get_global_metrics_collector().stop_collection()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        let latest = collector.get_latest_metrics().unwrap();
        assert!(latest.is_none());
    }

    #[test]
    fn test_time_series() {
        let mut ts = TimeSeries::new(3);
        ts.add_point(1.0);
        ts.add_point(2.0);
        ts.add_point(3.0);
        ts.add_point(4.0); // Should evict first point

        assert_eq!(ts.len(), 3);
        assert_eq!(ts.latest().unwrap().value, 4.0);
    }

    #[test]
    fn test_custom_metrics() {
        let collector = MetricsCollector::new();

        collector
            .add_custom_metric("test_metric".to_string(), 42.0)
            .unwrap();
        let value = collector.get_custom_metric("test_metric").unwrap();
        assert_eq!(value, Some(42.0));
    }

    #[test]
    fn test_training_metrics_update() {
        let collector = MetricsCollector::new();

        let training_metrics = TrainingMetrics {
            current_epoch: 5,
            training_loss: Some(0.123),
            ..Default::default()
        };

        collector
            .update_training_metrics(training_metrics.clone())
            .unwrap();

        let history = collector.get_training_history().unwrap();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].value.current_epoch, 5);
        assert_eq!(history[0].value.training_loss, Some(0.123));
    }

    #[test]
    fn test_metrics_export() {
        let collector = MetricsCollector::new();
        collector
            .add_custom_metric("export_test".to_string(), std::f64::consts::PI)
            .unwrap();

        let json = collector.export_metrics_json().unwrap();
        assert!(json.contains("export_test"));
        assert!(json.contains("3.14"));
    }

    #[test]
    fn test_global_metrics_collector() {
        let collector = get_global_metrics_collector();
        collector
            .add_custom_metric("global_test".to_string(), 7.5)
            .unwrap();

        let value = collector.get_custom_metric("global_test").unwrap();
        assert_eq!(value, Some(7.5));
    }

    /// On Linux, `collect_system_metrics` must read real values from /proc.
    /// The test verifies that memory fields are non-zero (a machine with zero
    /// total memory cannot exist) and that CPU/network counters come from the
    /// kernel, not from the old hardcoded 50 %/ 75 % placeholders.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_system_metrics_not_hardcoded_on_linux() {
        let m = MetricsCollector::collect_system_metrics();

        // Memory must be non-zero on any real machine.
        assert!(
            m.memory_usage_bytes > 0 || m.memory_available_bytes > 0,
            "at least one memory field must be non-zero (got usage={}, avail={})",
            m.memory_usage_bytes,
            m.memory_available_bytes
        );

        // `cpu_usage_pct` must be a valid percentage; the retired placeholder
        // ignored the system entirely and always returned exactly 50.0. Note that
        // 50.0 is itself a legitimate real reading (idle == half of total over the
        // sample window, common on a 2-vCPU box), so rejecting it outright is
        // flaky. Instead, when the passive sample lands exactly on the old
        // constant, saturate every core and re-sample: a real /proc/stat reading
        // climbs well above 50 % under full load, whereas a hardcoded constant
        // does not move.
        assert!(
            (0.0..=100.0).contains(&m.cpu_usage_pct),
            "cpu_usage_pct must be a valid percentage, got {}",
            m.cpu_usage_pct
        );
        if (m.cpu_usage_pct - 50.0).abs() < f64::EPSILON {
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::Arc;

            let stop = Arc::new(AtomicBool::new(false));
            let n_workers = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(2);
            let workers: Vec<_> = (0..n_workers)
                .map(|_| {
                    let stop = Arc::clone(&stop);
                    std::thread::spawn(move || {
                        // Pure ALU spin so the cores stay busy (no syscalls/idle).
                        let mut acc: u64 = 0;
                        while !stop.load(Ordering::Relaxed) {
                            acc = acc.wrapping_mul(2654435761).wrapping_add(1);
                        }
                        acc
                    })
                })
                .collect();

            let loaded = MetricsCollector::collect_system_metrics();
            stop.store(true, Ordering::Relaxed);
            for w in workers {
                let _ = w.join();
            }

            assert!(
                (loaded.cpu_usage_pct - 50.0).abs() >= f64::EPSILON,
                "cpu_usage_pct stayed pinned at exactly 50.0 even under full CPU \
                 saturation — looks like a hardcoded placeholder, not /proc/stat sampling"
            );
        }
        assert!(
            m.gpu_usage_pct != Some(75.0),
            "gpu_usage_pct must not be the hardcoded 75.0 placeholder"
        );

        // Timestamp must be recent (within 10 seconds).
        let age = m
            .timestamp
            .elapsed()
            .unwrap_or(std::time::Duration::from_secs(999));
        assert!(
            age.as_secs() < 10,
            "SystemMetrics timestamp is too old: {:?}",
            age
        );
    }
}

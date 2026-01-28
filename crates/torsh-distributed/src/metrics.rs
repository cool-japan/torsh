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

    /// Collect system metrics (simplified implementation)
    fn collect_system_metrics() -> SystemMetrics {
        // This is a simplified implementation. In a real system, you would
        // use platform-specific APIs to get actual system metrics.
        SystemMetrics {
            cpu_usage_pct: 50.0,                                  // Placeholder
            memory_usage_bytes: 1024 * 1024 * 1024,               // 1GB placeholder
            memory_available_bytes: 8 * 1024 * 1024 * 1024,       // 8GB placeholder
            memory_usage_pct: 12.5,                               // Calculated from above
            gpu_memory_usage_bytes: Some(512 * 1024 * 1024),      // 512MB placeholder
            gpu_memory_total_bytes: Some(4 * 1024 * 1024 * 1024), // 4GB placeholder
            gpu_usage_pct: Some(75.0),                            // Placeholder
            network_bytes_rx: 1024 * 1024,                        // 1MB placeholder
            network_bytes_tx: 2 * 1024 * 1024,                    // 2MB placeholder
            disk_bytes_read: 10 * 1024 * 1024,                    // 10MB placeholder
            disk_bytes_write: 5 * 1024 * 1024,                    // 5MB placeholder
            timestamp: SystemTime::now(),
            #[cfg(feature = "scirs2-profiling")]
            scirs2_profile: None,
            #[cfg(feature = "scirs2-profiling")]
            memory_profile: None,
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

    #[cfg(feature = "scirs2-profiling")]
    fn estimate_memory_bandwidth(&self) -> f64 {
        // This would use actual hardware counters in a real implementation
        // For now, return a reasonable estimate
        4.0 // GB/s
    }

    #[cfg(feature = "scirs2-profiling")]
    fn estimate_memory_latency(&self) -> f64 {
        // This would use actual hardware counters in a real implementation
        // For now, return a reasonable estimate
        50.0 // nanoseconds
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
}

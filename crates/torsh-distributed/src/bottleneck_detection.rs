//! Bottleneck detection algorithms for distributed training
//!
//! This module provides algorithms to automatically detect performance bottlenecks
//! in distributed training systems by analyzing communication patterns, resource usage,
//! and training metrics.

use crate::metrics::get_global_metrics_collector;
use crate::profiling::{get_global_profiler, CommunicationOpType};
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Types of bottlenecks that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BottleneckType {
    /// Communication bottleneck (slow collective operations)
    Communication,
    /// Memory bottleneck (high memory usage, frequent allocation/deallocation)
    Memory,
    /// Compute bottleneck (CPU or GPU utilization issues)
    Compute,
    /// Network bottleneck (network bandwidth limitations)
    Network,
    /// Load imbalance between processes/devices
    LoadImbalance,
    /// Synchronization bottleneck (excessive barrier waiting)
    Synchronization,
    /// I/O bottleneck (disk read/write operations)
    IO,
    /// Custom bottleneck type
    Custom(String),
}

impl std::fmt::Display for BottleneckType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BottleneckType::Communication => write!(f, "Communication"),
            BottleneckType::Memory => write!(f, "Memory"),
            BottleneckType::Compute => write!(f, "Compute"),
            BottleneckType::Network => write!(f, "Network"),
            BottleneckType::LoadImbalance => write!(f, "LoadImbalance"),
            BottleneckType::Synchronization => write!(f, "Synchronization"),
            BottleneckType::IO => write!(f, "IO"),
            BottleneckType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Severity level of a detected bottleneck
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    /// Low impact on performance
    Low,
    /// Moderate impact on performance
    Medium,
    /// High impact on performance
    High,
    /// Critical performance issue
    Critical,
}

impl std::fmt::Display for BottleneckSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BottleneckSeverity::Low => write!(f, "Low"),
            BottleneckSeverity::Medium => write!(f, "Medium"),
            BottleneckSeverity::High => write!(f, "High"),
            BottleneckSeverity::Critical => write!(f, "Critical"),
        }
    }
}

/// A detected bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,
    /// Severity level
    pub severity: BottleneckSeverity,
    /// Human-readable description
    pub description: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Metrics that support this detection
    pub supporting_metrics: HashMap<String, f64>,
    /// Suggested remediation actions
    pub remediation: Vec<String>,
    /// Timestamp when detected
    pub detected_at: SystemTime,
    /// Duration of the bottleneck (if applicable)
    pub duration: Option<Duration>,
    /// Affected ranks (if applicable)
    pub affected_ranks: Vec<u32>,
}

impl Bottleneck {
    /// Create a new bottleneck
    pub fn new(
        bottleneck_type: BottleneckType,
        severity: BottleneckSeverity,
        description: String,
        confidence: f64,
    ) -> Self {
        Self {
            bottleneck_type,
            severity,
            description,
            confidence,
            supporting_metrics: HashMap::new(),
            remediation: Vec::new(),
            detected_at: SystemTime::now(),
            duration: None,
            affected_ranks: Vec::new(),
        }
    }

    /// Add supporting metric
    pub fn with_metric(mut self, name: String, value: f64) -> Self {
        self.supporting_metrics.insert(name, value);
        self
    }

    /// Add remediation suggestion
    pub fn with_remediation(mut self, suggestion: String) -> Self {
        self.remediation.push(suggestion);
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set affected ranks
    pub fn with_affected_ranks(mut self, ranks: Vec<u32>) -> Self {
        self.affected_ranks = ranks;
        self
    }
}

/// Configuration for bottleneck detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckDetectionConfig {
    /// Whether detection is enabled
    pub enabled: bool,
    /// Minimum confidence threshold for reporting bottlenecks
    pub min_confidence: f64,
    /// Analysis window size (number of recent samples to analyze)
    pub analysis_window_size: usize,
    /// Detection interval in seconds
    pub detection_interval_secs: u64,
    /// Thresholds for different metrics
    pub thresholds: BottleneckThresholds,
}

impl Default for BottleneckDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: 0.7,
            analysis_window_size: 100,
            detection_interval_secs: 30,
            thresholds: BottleneckThresholds::default(),
        }
    }
}

/// Thresholds for bottleneck detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckThresholds {
    /// High CPU usage threshold (percentage)
    pub high_cpu_usage_pct: f64,
    /// High memory usage threshold (percentage)
    pub high_memory_usage_pct: f64,
    /// High GPU usage threshold (percentage)
    pub high_gpu_usage_pct: f64,
    /// Slow communication latency threshold (milliseconds)
    pub slow_communication_latency_ms: f64,
    /// Low bandwidth threshold (MB/s)
    pub low_bandwidth_mbps: f64,
    /// Load imbalance threshold (coefficient of variation)
    pub load_imbalance_cv_threshold: f64,
    /// Long synchronization wait threshold (milliseconds)
    pub long_sync_wait_ms: f64,
    /// High I/O wait threshold (percentage)
    pub high_io_wait_pct: f64,
}

impl Default for BottleneckThresholds {
    fn default() -> Self {
        Self {
            high_cpu_usage_pct: 90.0,
            high_memory_usage_pct: 85.0,
            high_gpu_usage_pct: 95.0,
            slow_communication_latency_ms: 100.0,
            low_bandwidth_mbps: 10.0,
            load_imbalance_cv_threshold: 0.3,
            long_sync_wait_ms: 1000.0,
            high_io_wait_pct: 20.0,
        }
    }
}

/// Bottleneck detection engine
pub struct BottleneckDetector {
    /// Configuration
    config: BottleneckDetectionConfig,
    /// History of detected bottlenecks
    bottleneck_history: Vec<Bottleneck>,
    /// Detection statistics
    detection_stats: HashMap<BottleneckType, usize>,
}

impl BottleneckDetector {
    /// Create a new bottleneck detector
    pub fn new() -> Self {
        Self::with_config(BottleneckDetectionConfig::default())
    }

    /// Create a new bottleneck detector with custom configuration
    pub fn with_config(config: BottleneckDetectionConfig) -> Self {
        Self {
            config,
            bottleneck_history: Vec::new(),
            detection_stats: HashMap::new(),
        }
    }

    /// Run bottleneck detection analysis
    pub fn detect_bottlenecks(&mut self) -> TorshResult<Vec<Bottleneck>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let mut detected_bottlenecks = Vec::new();

        // Analyze communication bottlenecks
        if let Ok(communication_bottlenecks) = self.detect_communication_bottlenecks() {
            detected_bottlenecks.extend(communication_bottlenecks);
        }

        // Analyze resource bottlenecks
        if let Ok(resource_bottlenecks) = self.detect_resource_bottlenecks() {
            detected_bottlenecks.extend(resource_bottlenecks);
        }

        // Analyze load imbalance
        if let Ok(load_imbalance_bottlenecks) = self.detect_load_imbalance() {
            detected_bottlenecks.extend(load_imbalance_bottlenecks);
        }

        // Analyze synchronization bottlenecks
        if let Ok(sync_bottlenecks) = self.detect_synchronization_bottlenecks() {
            detected_bottlenecks.extend(sync_bottlenecks);
        }

        // Filter by confidence threshold
        detected_bottlenecks.retain(|b| b.confidence >= self.config.min_confidence);

        // Update statistics
        for bottleneck in &detected_bottlenecks {
            *self
                .detection_stats
                .entry(bottleneck.bottleneck_type.clone())
                .or_insert(0) += 1;
        }

        // Add to history
        self.bottleneck_history.extend(detected_bottlenecks.clone());

        Ok(detected_bottlenecks)
    }

    /// Detect communication bottlenecks
    fn detect_communication_bottlenecks(&self) -> TorshResult<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();
        let profiler = get_global_profiler();

        // Get recent communication statistics
        let all_stats = profiler.get_all_operation_stats()?;

        for (op_type, stats) in all_stats.iter() {
            // Check for slow communication operations
            let avg_latency_ms = stats.avg_latency.as_secs_f64() * 1000.0;
            if avg_latency_ms > self.config.thresholds.slow_communication_latency_ms {
                let confidence = (avg_latency_ms
                    / self.config.thresholds.slow_communication_latency_ms)
                    .min(1.0);

                let bottleneck = Bottleneck::new(
                    BottleneckType::Communication,
                    if avg_latency_ms > self.config.thresholds.slow_communication_latency_ms * 2.0 {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                    format!("{} operations are running slower than expected", op_type),
                    confidence,
                )
                .with_metric("avg_latency_ms".to_string(), avg_latency_ms)
                .with_metric(
                    "threshold_ms".to_string(),
                    self.config.thresholds.slow_communication_latency_ms,
                )
                .with_remediation("Consider optimizing communication patterns".to_string())
                .with_remediation("Check network bandwidth and latency".to_string())
                .with_remediation("Evaluate collective algorithm choice".to_string());

                bottlenecks.push(bottleneck);
            }

            // Check for low bandwidth
            let bandwidth_mbps = stats.avg_bandwidth_bps / (1024.0 * 1024.0);
            if bandwidth_mbps < self.config.thresholds.low_bandwidth_mbps && stats.count > 10 {
                let confidence =
                    (self.config.thresholds.low_bandwidth_mbps / bandwidth_mbps).min(1.0);

                let bottleneck = Bottleneck::new(
                    BottleneckType::Network,
                    if bandwidth_mbps < self.config.thresholds.low_bandwidth_mbps * 0.5 {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                    format!("{} operations have low bandwidth utilization", op_type),
                    confidence,
                )
                .with_metric("bandwidth_mbps".to_string(), bandwidth_mbps)
                .with_metric(
                    "threshold_mbps".to_string(),
                    self.config.thresholds.low_bandwidth_mbps,
                )
                .with_remediation("Check network infrastructure".to_string())
                .with_remediation("Consider message size optimization".to_string())
                .with_remediation("Evaluate network topology".to_string());

                bottlenecks.push(bottleneck);
            }
        }

        Ok(bottlenecks)
    }

    /// Detect resource bottlenecks (CPU, memory, GPU)
    fn detect_resource_bottlenecks(&self) -> TorshResult<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();
        let metrics_collector = get_global_metrics_collector();

        // Get recent system metrics
        let system_history = metrics_collector.get_system_history()?;
        if system_history.is_empty() {
            return Ok(bottlenecks);
        }

        // Analyze recent metrics (last N samples)
        let recent_samples = system_history
            .iter()
            .rev()
            .take(self.config.analysis_window_size.min(system_history.len()))
            .collect::<Vec<_>>();

        // Calculate average resource usage
        let avg_cpu = recent_samples
            .iter()
            .map(|s| s.value.cpu_usage_pct)
            .sum::<f64>()
            / recent_samples.len() as f64;
        let avg_memory = recent_samples
            .iter()
            .map(|s| s.value.memory_usage_pct)
            .sum::<f64>()
            / recent_samples.len() as f64;
        let avg_gpu = recent_samples
            .iter()
            .filter_map(|s| s.value.gpu_usage_pct)
            .collect::<Vec<_>>();
        let avg_gpu_usage = if !avg_gpu.is_empty() {
            Some(avg_gpu.iter().sum::<f64>() / avg_gpu.len() as f64)
        } else {
            None
        };

        // Detect CPU bottleneck
        if avg_cpu > self.config.thresholds.high_cpu_usage_pct {
            let confidence = (avg_cpu / 100.0).min(1.0);
            let bottleneck = Bottleneck::new(
                BottleneckType::Compute,
                if avg_cpu > 95.0 {
                    BottleneckSeverity::Critical
                } else if avg_cpu > 90.0 {
                    BottleneckSeverity::High
                } else {
                    BottleneckSeverity::Medium
                },
                format!("High CPU usage detected ({:.1}%)", avg_cpu),
                confidence,
            )
            .with_metric("cpu_usage_pct".to_string(), avg_cpu)
            .with_metric(
                "threshold_pct".to_string(),
                self.config.thresholds.high_cpu_usage_pct,
            )
            .with_remediation("Consider reducing computational load".to_string())
            .with_remediation("Optimize algorithms for better CPU efficiency".to_string())
            .with_remediation("Scale to more CPU cores if available".to_string());

            bottlenecks.push(bottleneck);
        }

        // Detect memory bottleneck
        if avg_memory > self.config.thresholds.high_memory_usage_pct {
            let confidence = (avg_memory / 100.0).min(1.0);
            let bottleneck = Bottleneck::new(
                BottleneckType::Memory,
                if avg_memory > 95.0 {
                    BottleneckSeverity::Critical
                } else if avg_memory > 90.0 {
                    BottleneckSeverity::High
                } else {
                    BottleneckSeverity::Medium
                },
                format!("High memory usage detected ({:.1}%)", avg_memory),
                confidence,
            )
            .with_metric("memory_usage_pct".to_string(), avg_memory)
            .with_metric(
                "threshold_pct".to_string(),
                self.config.thresholds.high_memory_usage_pct,
            )
            .with_remediation("Reduce memory footprint".to_string())
            .with_remediation("Enable gradient checkpointing".to_string())
            .with_remediation("Consider model sharding strategies".to_string());

            bottlenecks.push(bottleneck);
        }

        // Detect GPU bottleneck
        if let Some(gpu_usage) = avg_gpu_usage {
            if gpu_usage > self.config.thresholds.high_gpu_usage_pct {
                let confidence = (gpu_usage / 100.0_f64).min(1.0_f64);
                let bottleneck = Bottleneck::new(
                    BottleneckType::Compute,
                    if gpu_usage > 98.0 {
                        BottleneckSeverity::Critical
                    } else if gpu_usage > 95.0 {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                    format!("High GPU usage detected ({:.1}%)", gpu_usage),
                    confidence,
                )
                .with_metric("gpu_usage_pct".to_string(), gpu_usage)
                .with_metric(
                    "threshold_pct".to_string(),
                    self.config.thresholds.high_gpu_usage_pct,
                )
                .with_remediation("Optimize GPU kernel efficiency".to_string())
                .with_remediation("Consider mixed precision training".to_string())
                .with_remediation("Scale to more GPUs if available".to_string());

                bottlenecks.push(bottleneck);
            }
        }

        Ok(bottlenecks)
    }

    /// Detect load imbalance between processes
    fn detect_load_imbalance(&self) -> TorshResult<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();
        let profiler = get_global_profiler();

        // Get rank-specific statistics
        if let Ok(all_events) = profiler.get_all_events() {
            if all_events.is_empty() {
                return Ok(bottlenecks);
            }

            // Group events by rank and operation type
            let mut rank_op_times: HashMap<u32, HashMap<CommunicationOpType, Duration>> =
                HashMap::new();

            for event in &all_events {
                rank_op_times
                    .entry(event.rank)
                    .or_default()
                    .entry(event.op_type)
                    .and_modify(|d| *d += event.duration)
                    .or_insert(event.duration);
            }

            // Analyze load imbalance for each operation type
            let mut all_op_types: std::collections::HashSet<CommunicationOpType> =
                std::collections::HashSet::new();
            for rank_ops in rank_op_times.values() {
                all_op_types.extend(rank_ops.keys());
            }

            for op_type in all_op_types {
                let times: Vec<f64> = rank_op_times
                    .values()
                    .filter_map(|ops| ops.get(&op_type))
                    .map(|d| d.as_secs_f64())
                    .collect();

                if times.len() >= 2 {
                    let mean = times.iter().sum::<f64>() / times.len() as f64;
                    let variance =
                        times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
                    let std_dev = variance.sqrt();
                    let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

                    if cv > self.config.thresholds.load_imbalance_cv_threshold {
                        let confidence =
                            (cv / self.config.thresholds.load_imbalance_cv_threshold).min(1.0);

                        let affected_ranks: Vec<u32> = rank_op_times.keys().cloned().collect();

                        let bottleneck = Bottleneck::new(
                            BottleneckType::LoadImbalance,
                            if cv > 0.5 {
                                BottleneckSeverity::High
                            } else {
                                BottleneckSeverity::Medium
                            },
                            format!(
                                "Load imbalance detected for {} operations (CV: {:.3})",
                                op_type, cv
                            ),
                            confidence,
                        )
                        .with_metric("coefficient_of_variation".to_string(), cv)
                        .with_metric(
                            "threshold".to_string(),
                            self.config.thresholds.load_imbalance_cv_threshold,
                        )
                        .with_metric("mean_time_s".to_string(), mean)
                        .with_metric("std_dev_s".to_string(), std_dev)
                        .with_affected_ranks(affected_ranks)
                        .with_remediation("Balance workload across processes".to_string())
                        .with_remediation("Check for process-specific bottlenecks".to_string())
                        .with_remediation("Consider dynamic load balancing".to_string());

                        bottlenecks.push(bottleneck);
                    }
                }
            }
        }

        Ok(bottlenecks)
    }

    /// Detect synchronization bottlenecks
    fn detect_synchronization_bottlenecks(&self) -> TorshResult<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();
        let profiler = get_global_profiler();

        // Check for long barrier operations
        if let Ok(Some(stats)) = profiler.get_operation_stats(CommunicationOpType::Barrier) {
            let avg_barrier_time_ms = stats.avg_latency.as_secs_f64() * 1000.0;

            if avg_barrier_time_ms > self.config.thresholds.long_sync_wait_ms {
                let confidence =
                    (avg_barrier_time_ms / self.config.thresholds.long_sync_wait_ms).min(1.0);

                let bottleneck = Bottleneck::new(
                    BottleneckType::Synchronization,
                    if avg_barrier_time_ms > self.config.thresholds.long_sync_wait_ms * 3.0 {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                    format!(
                        "Long synchronization delays detected ({:.1} ms average)",
                        avg_barrier_time_ms
                    ),
                    confidence,
                )
                .with_metric("avg_barrier_time_ms".to_string(), avg_barrier_time_ms)
                .with_metric(
                    "threshold_ms".to_string(),
                    self.config.thresholds.long_sync_wait_ms,
                )
                .with_remediation("Reduce frequency of synchronization points".to_string())
                .with_remediation("Optimize computation-communication overlap".to_string())
                .with_remediation("Check for process-specific delays".to_string());

                bottlenecks.push(bottleneck);
            }
        }

        Ok(bottlenecks)
    }

    /// Get bottleneck detection statistics
    pub fn get_detection_stats(&self) -> &HashMap<BottleneckType, usize> {
        &self.detection_stats
    }

    /// Get bottleneck history
    pub fn get_bottleneck_history(&self) -> &[Bottleneck] {
        &self.bottleneck_history
    }

    /// Clear bottleneck history
    pub fn clear_history(&mut self) {
        self.bottleneck_history.clear();
        self.detection_stats.clear();
    }

    /// Generate bottleneck report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Bottleneck Detection Report ===\n\n");

        // Recent bottlenecks (last 10)
        let recent_bottlenecks: Vec<_> = self.bottleneck_history.iter().rev().take(10).collect();

        if recent_bottlenecks.is_empty() {
            report.push_str("No bottlenecks detected recently.\n");
        } else {
            report.push_str("=== Recent Bottlenecks ===\n");
            for (i, bottleneck) in recent_bottlenecks.iter().enumerate() {
                report.push_str(&format!(
                    "\n{}. {} - {} (Confidence: {:.1}%)\n",
                    i + 1,
                    bottleneck.bottleneck_type,
                    bottleneck.severity,
                    bottleneck.confidence * 100.0
                ));
                report.push_str(&format!("   Description: {}\n", bottleneck.description));

                if !bottleneck.remediation.is_empty() {
                    report.push_str("   Suggested Actions:\n");
                    for action in &bottleneck.remediation {
                        report.push_str(&format!("   - {}\n", action));
                    }
                }
            }
        }

        // Statistics summary
        report.push_str("\n=== Detection Statistics ===\n");
        for (bottleneck_type, count) in &self.detection_stats {
            report.push_str(&format!("{}: {} detections\n", bottleneck_type, count));
        }

        report
    }

    /// Export bottleneck data to JSON
    pub fn export_json(&self) -> TorshResult<String> {
        #[derive(Serialize)]
        struct BottleneckExport {
            config: BottleneckDetectionConfig,
            bottleneck_history: Vec<Bottleneck>,
            detection_stats: HashMap<BottleneckType, usize>,
        }

        let export = BottleneckExport {
            config: self.config.clone(),
            bottleneck_history: self.bottleneck_history.clone(),
            detection_stats: self.detection_stats.clone(),
        };

        serde_json::to_string_pretty(&export).map_err(|e| {
            TorshDistributedError::backend_error(
                "bottleneck_detection",
                format!("JSON serialization failed: {}", e),
            )
        })
    }
}

impl Default for BottleneckDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Global bottleneck detector instance
static GLOBAL_BOTTLENECK_DETECTOR: std::sync::Mutex<Option<BottleneckDetector>> =
    std::sync::Mutex::new(None);

/// Get or initialize the global bottleneck detector
pub fn with_global_bottleneck_detector<F, R>(f: F) -> TorshResult<R>
where
    F: FnOnce(&mut BottleneckDetector) -> TorshResult<R>,
{
    let mut guard = GLOBAL_BOTTLENECK_DETECTOR.lock().map_err(|_| {
        TorshDistributedError::backend_error("bottleneck_detection", "Lock poisoned")
    })?;

    if guard.is_none() {
        *guard = Some(BottleneckDetector::new());
    }

    f(guard
        .as_mut()
        .expect("global bottleneck detector should be initialized"))
}

/// Initialize the global bottleneck detector with custom configuration
pub fn init_global_bottleneck_detector(config: BottleneckDetectionConfig) -> TorshResult<()> {
    let mut guard = GLOBAL_BOTTLENECK_DETECTOR
        .lock()
        .map_err(|_| TorshDistributedError::backend_error("system", "Lock poisoned"))?;

    *guard = Some(BottleneckDetector::with_config(config));
    Ok(())
}

/// Run global bottleneck detection
pub fn run_global_bottleneck_detection() -> TorshResult<Vec<Bottleneck>> {
    with_global_bottleneck_detector(|detector| detector.detect_bottlenecks())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bottleneck_creation() {
        let bottleneck = Bottleneck::new(
            BottleneckType::Communication,
            BottleneckSeverity::High,
            "Test bottleneck".to_string(),
            0.8,
        )
        .with_metric("test_metric".to_string(), 42.0)
        .with_remediation("Fix the issue".to_string());

        assert_eq!(bottleneck.bottleneck_type, BottleneckType::Communication);
        assert_eq!(bottleneck.severity, BottleneckSeverity::High);
        assert_eq!(bottleneck.confidence, 0.8);
        assert_eq!(
            bottleneck.supporting_metrics.get("test_metric"),
            Some(&42.0)
        );
        assert_eq!(bottleneck.remediation[0], "Fix the issue");
    }

    #[test]
    fn test_bottleneck_detector_creation() {
        let detector = BottleneckDetector::new();
        assert!(detector.get_bottleneck_history().is_empty());
        assert!(detector.get_detection_stats().is_empty());
    }

    #[test]
    fn test_custom_config() {
        let config = BottleneckDetectionConfig {
            min_confidence: 0.9,
            analysis_window_size: 50,
            ..Default::default()
        };

        let detector = BottleneckDetector::with_config(config.clone());
        assert_eq!(detector.config.min_confidence, 0.9);
        assert_eq!(detector.config.analysis_window_size, 50);
    }

    #[test]
    fn test_bottleneck_severity_ordering() {
        assert!(BottleneckSeverity::Critical > BottleneckSeverity::High);
        assert!(BottleneckSeverity::High > BottleneckSeverity::Medium);
        assert!(BottleneckSeverity::Medium > BottleneckSeverity::Low);
    }

    #[test]
    fn test_report_generation() {
        let detector = BottleneckDetector::new();
        let report = detector.generate_report();
        assert!(report.contains("Bottleneck Detection Report"));
        assert!(report.contains("No bottlenecks detected recently"));
    }

    #[test]
    fn test_json_export() {
        let detector = BottleneckDetector::new();
        let json = detector.export_json().unwrap();
        assert!(json.contains("config"));
        assert!(json.contains("bottleneck_history"));
        assert!(json.contains("detection_stats"));
    }
}

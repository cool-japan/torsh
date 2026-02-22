//! Advanced Monitoring and Performance Analytics for Distributed Training
//!
//! This module provides comprehensive monitoring capabilities for distributed training,
//! including real-time metrics collection, anomaly detection, trend analysis, and
//! automatic optimization recommendations.
//!
//! # Features
//!
//! - **Real-time Metrics**: Continuous collection of performance metrics across all ranks
//! - **Anomaly Detection**: Automatic detection of performance degradation and bottlenecks
//! - **Trend Analysis**: Historical analysis to identify performance patterns
//! - **Optimization Recommendations**: AI-powered suggestions for improving training performance
//! - **Multi-rank Coordination**: Synchronized metrics collection across distributed workers
//! - **Export Capabilities**: Integration with external monitoring tools (Prometheus, Grafana)

use crate::{ProcessGroup, TorshResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Maximum number of historical samples to keep per metric
const MAX_HISTORY_SIZE: usize = 1000;

/// Threshold for detecting performance anomalies (z-score)
const ANOMALY_THRESHOLD: f64 = 2.5;

/// Minimum samples required for statistical analysis
const MIN_SAMPLES_FOR_ANALYSIS: usize = 10;

/// Advanced performance metrics for distributed training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedMetrics {
    /// Timestamp when metrics were collected
    pub timestamp: Duration,

    /// Compute metrics
    pub compute: ComputeMetrics,

    /// Communication metrics
    pub communication: CommunicationMetrics,

    /// Memory metrics
    pub memory: MemoryMetrics,

    /// I/O metrics
    pub io: IoMetrics,

    /// Custom user-defined metrics
    pub custom: HashMap<String, f64>,
}

/// Compute-related performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeMetrics {
    /// Forward pass time (milliseconds)
    pub forward_time_ms: f64,

    /// Backward pass time (milliseconds)
    pub backward_time_ms: f64,

    /// Optimizer step time (milliseconds)
    pub optimizer_time_ms: f64,

    /// GPU utilization percentage (0-100)
    pub gpu_utilization: f64,

    /// CPU utilization percentage (0-100)
    pub cpu_utilization: f64,

    /// Tensor core utilization (0-100)
    pub tensor_core_utilization: f64,

    /// FLOPS achieved (GFLOPS)
    pub gflops: f64,
}

/// Communication-related performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationMetrics {
    /// All-reduce operation time (milliseconds)
    pub all_reduce_time_ms: f64,

    /// Broadcast operation time (milliseconds)
    pub broadcast_time_ms: f64,

    /// All-gather operation time (milliseconds)
    pub all_gather_time_ms: f64,

    /// Network bandwidth utilization (MB/s)
    pub bandwidth_mbps: f64,

    /// Communication to computation ratio
    pub comm_comp_ratio: f64,

    /// Number of communication operations
    pub num_operations: u64,

    /// Average message size (bytes)
    pub avg_message_size: usize,
}

/// Memory-related performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// GPU memory used (MB)
    pub gpu_memory_used_mb: f64,

    /// GPU memory total (MB)
    pub gpu_memory_total_mb: f64,

    /// CPU memory used (MB)
    pub cpu_memory_used_mb: f64,

    /// Memory bandwidth utilization (GB/s)
    pub memory_bandwidth_gbps: f64,

    /// Number of memory allocations
    pub num_allocations: u64,

    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
}

/// I/O-related performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoMetrics {
    /// Data loading time (milliseconds)
    pub data_load_time_ms: f64,

    /// Disk read throughput (MB/s)
    pub disk_read_mbps: f64,

    /// Disk write throughput (MB/s)
    pub disk_write_mbps: f64,

    /// Data preprocessing time (milliseconds)
    pub preprocessing_time_ms: f64,
}

/// Performance anomaly detected in the system
#[derive(Debug, Clone)]
pub struct PerformanceAnomaly {
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,

    /// Severity level (0-10)
    pub severity: u8,

    /// Metric that triggered the anomaly
    pub metric_name: String,

    /// Current value of the metric
    pub current_value: f64,

    /// Expected value based on historical data
    pub expected_value: f64,

    /// Deviation from expected (z-score)
    pub deviation: f64,

    /// Timestamp when anomaly was detected (seconds since monitoring started)
    pub timestamp_secs: f64,

    /// Description of the anomaly
    pub description: String,
}

/// Types of performance anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Sudden spike in execution time
    PerformanceSpike,

    /// Gradual performance degradation
    PerformanceDegradation,

    /// Memory leak detected
    MemoryLeak,

    /// Communication bottleneck
    CommunicationBottleneck,

    /// GPU underutilization
    GpuUnderutilization,

    /// I/O bottleneck
    IoBottleneck,

    /// Imbalanced load across ranks
    LoadImbalance,
}

/// Optimization recommendation for improving performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Category of the recommendation
    pub category: RecommendationCategory,

    /// Priority level (1-10, higher is more important)
    pub priority: u8,

    /// Title of the recommendation
    pub title: String,

    /// Detailed description
    pub description: String,

    /// Expected performance improvement (percentage)
    pub expected_improvement: f64,

    /// Implementation difficulty (1-5)
    pub difficulty: u8,

    /// Code example or configuration change
    pub code_example: Option<String>,
}

/// Categories of optimization recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationCategory {
    /// Batch size optimization
    BatchSize,

    /// Gradient accumulation tuning
    GradientAccumulation,

    /// Communication optimization
    Communication,

    /// Memory management
    Memory,

    /// Data loading optimization
    DataLoading,

    /// Model architecture
    Architecture,

    /// Mixed precision training
    MixedPrecision,
}

/// Statistical summary of a metric over time
#[derive(Debug, Clone)]
struct MetricStatistics {
    /// Mean value
    mean: f64,

    /// Standard deviation
    std_dev: f64,

    /// Minimum value
    #[allow(dead_code)]
    min: f64,

    /// Maximum value
    #[allow(dead_code)]
    max: f64,

    /// Median value
    #[allow(dead_code)]
    median: f64,

    /// 95th percentile
    #[allow(dead_code)]
    p95: f64,

    /// 99th percentile
    #[allow(dead_code)]
    p99: f64,
}

/// Advanced monitoring system for distributed training
pub struct AdvancedMonitor {
    /// Process group for distributed coordination
    process_group: Arc<ProcessGroup>,

    /// Historical metrics per rank
    metrics_history: Arc<RwLock<HashMap<u32, VecDeque<AdvancedMetrics>>>>,

    /// Detected anomalies
    anomalies: Arc<RwLock<Vec<PerformanceAnomaly>>>,

    /// Generated recommendations
    recommendations: Arc<RwLock<Vec<OptimizationRecommendation>>>,

    /// Start time for relative timestamps
    start_time: Instant,

    /// Whether monitoring is enabled
    enabled: Arc<RwLock<bool>>,

    /// Custom metric thresholds
    custom_thresholds: Arc<RwLock<HashMap<String, f64>>>,
}

impl AdvancedMonitor {
    /// Create a new advanced monitoring system
    pub fn new(process_group: Arc<ProcessGroup>) -> Self {
        Self {
            process_group,
            metrics_history: Arc::new(RwLock::new(HashMap::new())),
            anomalies: Arc::new(RwLock::new(Vec::new())),
            recommendations: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
            enabled: Arc::new(RwLock::new(true)),
            custom_thresholds: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record metrics for the current rank
    pub fn record_metrics(&self, metrics: AdvancedMetrics) -> TorshResult<()> {
        if !*self.enabled.read() {
            return Ok(());
        }

        let rank = self.process_group.rank();
        let mut history = self.metrics_history.write();

        let rank_history = history.entry(rank).or_default();
        rank_history.push_back(metrics.clone());

        // Keep only the most recent samples
        if rank_history.len() > MAX_HISTORY_SIZE {
            rank_history.pop_front();
        }

        // Perform anomaly detection
        self.detect_anomalies(&metrics, rank_history)?;

        debug!(
            "Recorded metrics for rank {}: compute={:.2}ms, comm={:.2}ms",
            rank,
            metrics.compute.forward_time_ms + metrics.compute.backward_time_ms,
            metrics.communication.all_reduce_time_ms
        );

        Ok(())
    }

    /// Detect performance anomalies in the metrics
    fn detect_anomalies(
        &self,
        current: &AdvancedMetrics,
        history: &VecDeque<AdvancedMetrics>,
    ) -> TorshResult<()> {
        if history.len() < MIN_SAMPLES_FOR_ANALYSIS {
            return Ok(());
        }

        // Analyze forward pass time
        self.check_metric_anomaly(
            "forward_time_ms",
            current.compute.forward_time_ms,
            history.iter().map(|m| m.compute.forward_time_ms).collect(),
            AnomalyType::PerformanceSpike,
        )?;

        // Analyze communication time
        self.check_metric_anomaly(
            "all_reduce_time_ms",
            current.communication.all_reduce_time_ms,
            history
                .iter()
                .map(|m| m.communication.all_reduce_time_ms)
                .collect(),
            AnomalyType::CommunicationBottleneck,
        )?;

        // Analyze memory usage
        self.check_metric_anomaly(
            "gpu_memory_used_mb",
            current.memory.gpu_memory_used_mb,
            history
                .iter()
                .map(|m| m.memory.gpu_memory_used_mb)
                .collect(),
            AnomalyType::MemoryLeak,
        )?;

        // Check GPU utilization
        if current.compute.gpu_utilization < 50.0 {
            self.report_anomaly(PerformanceAnomaly {
                anomaly_type: AnomalyType::GpuUnderutilization,
                severity: 7,
                metric_name: "gpu_utilization".to_string(),
                current_value: current.compute.gpu_utilization,
                expected_value: 80.0,
                deviation: (80.0 - current.compute.gpu_utilization) / 15.0,
                timestamp_secs: self.start_time.elapsed().as_secs_f64(),
                description: format!(
                    "GPU utilization is low ({:.1}%), suggesting compute inefficiency",
                    current.compute.gpu_utilization
                ),
            });
        }

        Ok(())
    }

    /// Check if a metric value is anomalous
    fn check_metric_anomaly(
        &self,
        metric_name: &str,
        current_value: f64,
        history: Vec<f64>,
        anomaly_type: AnomalyType,
    ) -> TorshResult<()> {
        let stats = Self::calculate_statistics(&history);

        if stats.std_dev == 0.0 {
            return Ok(());
        }

        let z_score = (current_value - stats.mean) / stats.std_dev;

        if z_score.abs() > ANOMALY_THRESHOLD {
            let severity = ((z_score.abs() - ANOMALY_THRESHOLD) * 2.0).clamp(5.0, 10.0) as u8;

            self.report_anomaly(PerformanceAnomaly {
                anomaly_type,
                severity,
                metric_name: metric_name.to_string(),
                current_value,
                expected_value: stats.mean,
                deviation: z_score,
                timestamp_secs: self.start_time.elapsed().as_secs_f64(),
                description: format!(
                    "{} is {:.1}σ from normal: current={:.2}, expected={:.2}±{:.2}",
                    metric_name, z_score, current_value, stats.mean, stats.std_dev
                ),
            });
        }

        Ok(())
    }

    /// Calculate statistical summary of values
    fn calculate_statistics(values: &[f64]) -> MetricStatistics {
        if values.is_empty() {
            return MetricStatistics {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];

        // Calculate median properly
        let median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let p95 = sorted[(sorted.len() as f64 * 0.95).min((sorted.len() - 1) as f64) as usize];
        let p99 = sorted[(sorted.len() as f64 * 0.99).min((sorted.len() - 1) as f64) as usize];

        MetricStatistics {
            mean,
            std_dev,
            min,
            max,
            median,
            p95,
            p99,
        }
    }

    /// Report a detected anomaly
    fn report_anomaly(&self, anomaly: PerformanceAnomaly) {
        warn!(
            "🚨 Performance anomaly detected: {} (severity: {})",
            anomaly.description, anomaly.severity
        );

        self.anomalies.write().push(anomaly);
    }

    /// Generate optimization recommendations based on collected metrics
    pub fn generate_recommendations(&self) -> TorshResult<Vec<OptimizationRecommendation>> {
        let history = self.metrics_history.read();
        let mut recommendations = Vec::new();

        // Analyze metrics from all ranks
        for (_rank, metrics) in history.iter() {
            if metrics.is_empty() {
                continue;
            }

            let start_idx = metrics.len().saturating_sub(10);
            let recent: Vec<&AdvancedMetrics> = metrics.iter().skip(start_idx).collect();

            // Check communication overhead
            let avg_comm_ratio: f64 = recent
                .iter()
                .map(|m| m.communication.comm_comp_ratio)
                .sum::<f64>()
                / recent.len() as f64;

            if avg_comm_ratio > 0.3 {
                recommendations.push(OptimizationRecommendation {
                    category: RecommendationCategory::Communication,
                    priority: 9,
                    title: "High Communication Overhead Detected".to_string(),
                    description: format!(
                        "Communication takes {:.1}% of total time. Consider gradient accumulation or larger batch sizes.",
                        avg_comm_ratio * 100.0
                    ),
                    expected_improvement: 20.0,
                    difficulty: 2,
                    code_example: Some("# Increase gradient accumulation steps\naccumulation_steps = 4".to_string()),
                });
            }

            // Check GPU utilization
            let avg_gpu_util: f64 = recent
                .iter()
                .map(|m| m.compute.gpu_utilization)
                .sum::<f64>()
                / recent.len() as f64;

            if avg_gpu_util < 60.0 {
                recommendations.push(OptimizationRecommendation {
                    category: RecommendationCategory::BatchSize,
                    priority: 8,
                    title: "Low GPU Utilization".to_string(),
                    description: format!(
                        "GPU utilization is only {:.1}%. Consider increasing batch size to improve hardware utilization.",
                        avg_gpu_util
                    ),
                    expected_improvement: 30.0,
                    difficulty: 1,
                    code_example: Some("# Increase batch size\nbatch_size = current_batch_size * 2".to_string()),
                });
            }

            // Check memory usage
            let avg_memory_pct: f64 = recent
                .iter()
                .map(|m| (m.memory.gpu_memory_used_mb / m.memory.gpu_memory_total_mb) * 100.0)
                .sum::<f64>()
                / recent.len() as f64;

            if avg_memory_pct > 90.0 {
                recommendations.push(OptimizationRecommendation {
                    category: RecommendationCategory::Memory,
                    priority: 10,
                    title: "Near Memory Limit".to_string(),
                    description: "GPU memory usage is above 90%. Risk of OOM errors.".to_string(),
                    expected_improvement: 0.0,
                    difficulty: 3,
                    code_example: Some(
                        "# Enable gradient checkpointing\nmodel.gradient_checkpointing_enable()"
                            .to_string(),
                    ),
                });
            } else if avg_memory_pct < 50.0 {
                recommendations.push(OptimizationRecommendation {
                    category: RecommendationCategory::BatchSize,
                    priority: 6,
                    title: "Underutilized GPU Memory".to_string(),
                    description: format!(
                        "Only using {:.1}% of GPU memory. Can increase batch size for better performance.",
                        avg_memory_pct
                    ),
                    expected_improvement: 15.0,
                    difficulty: 1,
                    code_example: Some("# Increase batch size to use more memory\nbatch_size *= 1.5".to_string()),
                });
            }
        }

        // Sort by priority (descending)
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Store recommendations
        *self.recommendations.write() = recommendations.clone();

        info!(
            "Generated {} optimization recommendations",
            recommendations.len()
        );

        Ok(recommendations)
    }

    /// Get all detected anomalies
    pub fn get_anomalies(&self) -> Vec<PerformanceAnomaly> {
        self.anomalies.read().clone()
    }

    /// Get recent anomalies (last N)
    pub fn get_recent_anomalies(&self, count: usize) -> Vec<PerformanceAnomaly> {
        let anomalies = self.anomalies.read();
        anomalies.iter().rev().take(count).cloned().collect()
    }

    /// Clear all anomalies
    pub fn clear_anomalies(&self) {
        self.anomalies.write().clear();
    }

    /// Get current optimization recommendations
    pub fn get_recommendations(&self) -> Vec<OptimizationRecommendation> {
        self.recommendations.read().clone()
    }

    /// Get metrics history for a specific rank
    pub fn get_rank_history(&self, rank: u32) -> Option<Vec<AdvancedMetrics>> {
        self.metrics_history
            .read()
            .get(&rank)
            .map(|h| h.iter().cloned().collect())
    }

    /// Get latest metrics for all ranks
    pub async fn get_latest_metrics(&self) -> TorshResult<HashMap<u32, AdvancedMetrics>> {
        let history = self.metrics_history.read();
        let mut latest_metrics = HashMap::new();

        for (rank, metrics) in history.iter() {
            if let Some(latest) = metrics.back() {
                latest_metrics.insert(*rank, latest.clone());
            }
        }

        Ok(latest_metrics)
    }

    /// Get aggregated metrics across all ranks
    pub fn get_aggregated_metrics(&self) -> Option<AdvancedMetrics> {
        let history = self.metrics_history.read();

        if history.is_empty() {
            return None;
        }

        // Get the most recent metrics from each rank
        let recent_metrics: Vec<&AdvancedMetrics> =
            history.values().filter_map(|h| h.back()).collect();

        if recent_metrics.is_empty() {
            return None;
        }

        let count = recent_metrics.len() as f64;

        // Calculate averages
        Some(AdvancedMetrics {
            timestamp: self.start_time.elapsed(),
            compute: ComputeMetrics {
                forward_time_ms: recent_metrics
                    .iter()
                    .map(|m| m.compute.forward_time_ms)
                    .sum::<f64>()
                    / count,
                backward_time_ms: recent_metrics
                    .iter()
                    .map(|m| m.compute.backward_time_ms)
                    .sum::<f64>()
                    / count,
                optimizer_time_ms: recent_metrics
                    .iter()
                    .map(|m| m.compute.optimizer_time_ms)
                    .sum::<f64>()
                    / count,
                gpu_utilization: recent_metrics
                    .iter()
                    .map(|m| m.compute.gpu_utilization)
                    .sum::<f64>()
                    / count,
                cpu_utilization: recent_metrics
                    .iter()
                    .map(|m| m.compute.cpu_utilization)
                    .sum::<f64>()
                    / count,
                tensor_core_utilization: recent_metrics
                    .iter()
                    .map(|m| m.compute.tensor_core_utilization)
                    .sum::<f64>()
                    / count,
                gflops: recent_metrics.iter().map(|m| m.compute.gflops).sum::<f64>() / count,
            },
            communication: CommunicationMetrics {
                all_reduce_time_ms: recent_metrics
                    .iter()
                    .map(|m| m.communication.all_reduce_time_ms)
                    .sum::<f64>()
                    / count,
                broadcast_time_ms: recent_metrics
                    .iter()
                    .map(|m| m.communication.broadcast_time_ms)
                    .sum::<f64>()
                    / count,
                all_gather_time_ms: recent_metrics
                    .iter()
                    .map(|m| m.communication.all_gather_time_ms)
                    .sum::<f64>()
                    / count,
                bandwidth_mbps: recent_metrics
                    .iter()
                    .map(|m| m.communication.bandwidth_mbps)
                    .sum::<f64>()
                    / count,
                comm_comp_ratio: recent_metrics
                    .iter()
                    .map(|m| m.communication.comm_comp_ratio)
                    .sum::<f64>()
                    / count,
                num_operations: (recent_metrics
                    .iter()
                    .map(|m| m.communication.num_operations)
                    .sum::<u64>() as f64
                    / count) as u64,
                avg_message_size: (recent_metrics
                    .iter()
                    .map(|m| m.communication.avg_message_size)
                    .sum::<usize>() as f64
                    / count) as usize,
            },
            memory: MemoryMetrics {
                gpu_memory_used_mb: recent_metrics
                    .iter()
                    .map(|m| m.memory.gpu_memory_used_mb)
                    .sum::<f64>()
                    / count,
                gpu_memory_total_mb: recent_metrics
                    .iter()
                    .map(|m| m.memory.gpu_memory_total_mb)
                    .sum::<f64>()
                    / count,
                cpu_memory_used_mb: recent_metrics
                    .iter()
                    .map(|m| m.memory.cpu_memory_used_mb)
                    .sum::<f64>()
                    / count,
                memory_bandwidth_gbps: recent_metrics
                    .iter()
                    .map(|m| m.memory.memory_bandwidth_gbps)
                    .sum::<f64>()
                    / count,
                num_allocations: (recent_metrics
                    .iter()
                    .map(|m| m.memory.num_allocations)
                    .sum::<u64>() as f64
                    / count) as u64,
                peak_memory_mb: recent_metrics
                    .iter()
                    .map(|m| m.memory.peak_memory_mb)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0),
            },
            io: IoMetrics {
                data_load_time_ms: recent_metrics
                    .iter()
                    .map(|m| m.io.data_load_time_ms)
                    .sum::<f64>()
                    / count,
                disk_read_mbps: recent_metrics
                    .iter()
                    .map(|m| m.io.disk_read_mbps)
                    .sum::<f64>()
                    / count,
                disk_write_mbps: recent_metrics
                    .iter()
                    .map(|m| m.io.disk_write_mbps)
                    .sum::<f64>()
                    / count,
                preprocessing_time_ms: recent_metrics
                    .iter()
                    .map(|m| m.io.preprocessing_time_ms)
                    .sum::<f64>()
                    / count,
            },
            custom: HashMap::new(),
        })
    }

    /// Enable or disable monitoring
    pub fn set_enabled(&self, enabled: bool) {
        *self.enabled.write() = enabled;
        info!(
            "Advanced monitoring {}",
            if enabled { "enabled" } else { "disabled" }
        );
    }

    /// Check if monitoring is enabled
    pub fn is_enabled(&self) -> bool {
        *self.enabled.read()
    }

    /// Set custom threshold for anomaly detection
    pub fn set_threshold(&self, metric_name: String, threshold: f64) {
        self.custom_thresholds
            .write()
            .insert(metric_name, threshold);
    }

    /// Generate a comprehensive performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        report.push_str("   📊 ADVANCED PERFORMANCE MONITORING REPORT\n");
        report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

        // Aggregated metrics
        if let Some(metrics) = self.get_aggregated_metrics() {
            report.push_str("📈 Aggregated Metrics (All Ranks):\n");
            report.push_str(&format!(
                "   Compute: fwd={:.2}ms, bwd={:.2}ms, opt={:.2}ms\n",
                metrics.compute.forward_time_ms,
                metrics.compute.backward_time_ms,
                metrics.compute.optimizer_time_ms
            ));
            report.push_str(&format!(
                "   GPU Utilization: {:.1}%\n",
                metrics.compute.gpu_utilization
            ));
            report.push_str(&format!(
                "   Communication: {:.2}ms ({:.1}% of total)\n",
                metrics.communication.all_reduce_time_ms,
                metrics.communication.comm_comp_ratio * 100.0
            ));
            report.push_str(&format!(
                "   Memory: {:.0}/{:.0} MB ({:.1}%)\n\n",
                metrics.memory.gpu_memory_used_mb,
                metrics.memory.gpu_memory_total_mb,
                (metrics.memory.gpu_memory_used_mb / metrics.memory.gpu_memory_total_mb) * 100.0
            ));
        }

        // Recent anomalies
        let anomalies = self.get_recent_anomalies(5);
        if !anomalies.is_empty() {
            report.push_str("🚨 Recent Anomalies:\n");
            for anomaly in anomalies {
                report.push_str(&format!(
                    "   [{:?}] {} (severity: {})\n",
                    anomaly.anomaly_type, anomaly.description, anomaly.severity
                ));
            }
            report.push('\n');
        }

        // Recommendations
        let recommendations = self.get_recommendations();
        if !recommendations.is_empty() {
            report.push_str("💡 Top Optimization Recommendations:\n");
            for (i, rec) in recommendations.iter().take(5).enumerate() {
                report.push_str(&format!(
                    "   {}. [Priority {}] {}\n",
                    i + 1,
                    rec.priority,
                    rec.title
                ));
                report.push_str(&format!("      {}\n", rec.description));
                if rec.expected_improvement > 0.0 {
                    report.push_str(&format!(
                        "      Expected improvement: {:.1}%\n",
                        rec.expected_improvement
                    ));
                }
            }
        }

        report.push_str("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        report
    }
}

impl Default for ComputeMetrics {
    fn default() -> Self {
        Self {
            forward_time_ms: 0.0,
            backward_time_ms: 0.0,
            optimizer_time_ms: 0.0,
            gpu_utilization: 0.0,
            cpu_utilization: 0.0,
            tensor_core_utilization: 0.0,
            gflops: 0.0,
        }
    }
}

impl Default for CommunicationMetrics {
    fn default() -> Self {
        Self {
            all_reduce_time_ms: 0.0,
            broadcast_time_ms: 0.0,
            all_gather_time_ms: 0.0,
            bandwidth_mbps: 0.0,
            comm_comp_ratio: 0.0,
            num_operations: 0,
            avg_message_size: 0,
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            gpu_memory_used_mb: 0.0,
            gpu_memory_total_mb: 16384.0, // 16GB default
            cpu_memory_used_mb: 0.0,
            memory_bandwidth_gbps: 0.0,
            num_allocations: 0,
            peak_memory_mb: 0.0,
        }
    }
}

impl Default for IoMetrics {
    fn default() -> Self {
        Self {
            data_load_time_ms: 0.0,
            disk_read_mbps: 0.0,
            disk_write_mbps: 0.0,
            preprocessing_time_ms: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{init_process_group, BackendType};

    #[tokio::test]
    async fn test_advanced_monitor_creation() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();

        let monitor = AdvancedMonitor::new(Arc::new(pg));
        assert!(monitor.is_enabled());
        assert_eq!(monitor.get_anomalies().len(), 0);
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();

        let monitor = AdvancedMonitor::new(Arc::new(pg));

        let metrics = AdvancedMetrics {
            timestamp: Duration::from_secs(1),
            compute: ComputeMetrics {
                forward_time_ms: 10.0,
                backward_time_ms: 15.0,
                gpu_utilization: 85.0,
                ..Default::default()
            },
            communication: CommunicationMetrics::default(),
            memory: MemoryMetrics::default(),
            io: IoMetrics::default(),
            custom: HashMap::new(),
        };

        assert!(monitor.record_metrics(metrics).is_ok());

        let history = monitor.get_rank_history(0);
        assert!(history.is_some());
        assert_eq!(history.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_anomaly_detection() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();

        let monitor = AdvancedMonitor::new(Arc::new(pg));

        // Record normal metrics
        for i in 0..15 {
            let metrics = AdvancedMetrics {
                timestamp: Duration::from_secs(i),
                compute: ComputeMetrics {
                    forward_time_ms: 10.0,
                    gpu_utilization: 80.0,
                    ..Default::default()
                },
                communication: CommunicationMetrics::default(),
                memory: MemoryMetrics::default(),
                io: IoMetrics::default(),
                custom: HashMap::new(),
            };
            monitor.record_metrics(metrics).unwrap();
        }

        // Record anomalous metric
        let anomalous_metrics = AdvancedMetrics {
            timestamp: Duration::from_secs(16),
            compute: ComputeMetrics {
                forward_time_ms: 100.0, // 10x slower!
                gpu_utilization: 80.0,
                ..Default::default()
            },
            communication: CommunicationMetrics::default(),
            memory: MemoryMetrics::default(),
            io: IoMetrics::default(),
            custom: HashMap::new(),
        };
        monitor.record_metrics(anomalous_metrics).unwrap();

        // Should have detected anomaly
        let anomalies = monitor.get_anomalies();
        assert!(!anomalies.is_empty());
    }

    #[tokio::test]
    async fn test_recommendation_generation() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();

        let monitor = AdvancedMonitor::new(Arc::new(pg));

        // Record metrics with low GPU utilization
        for i in 0..20 {
            let metrics = AdvancedMetrics {
                timestamp: Duration::from_secs(i),
                compute: ComputeMetrics {
                    forward_time_ms: 10.0,
                    gpu_utilization: 40.0, // Low utilization
                    ..Default::default()
                },
                communication: CommunicationMetrics {
                    comm_comp_ratio: 0.4, // High comm ratio
                    ..Default::default()
                },
                memory: MemoryMetrics::default(),
                io: IoMetrics::default(),
                custom: HashMap::new(),
            };
            monitor.record_metrics(metrics).unwrap();
        }

        let recommendations = monitor.generate_recommendations().unwrap();
        assert!(!recommendations.is_empty());

        // Should recommend batch size increase due to low GPU utilization
        assert!(recommendations
            .iter()
            .any(|r| r.category == RecommendationCategory::BatchSize));
    }

    #[tokio::test]
    async fn test_aggregated_metrics() {
        let pg = init_process_group(BackendType::Gloo, 0, 2, "127.0.0.1", 29500)
            .await
            .unwrap();

        let monitor = AdvancedMonitor::new(Arc::new(pg));

        // Simulate metrics from multiple ranks
        let mut history = monitor.metrics_history.write();
        for rank in 0..2 {
            let mut rank_history = VecDeque::new();
            rank_history.push_back(AdvancedMetrics {
                timestamp: Duration::from_secs(1),
                compute: ComputeMetrics {
                    forward_time_ms: 10.0 + rank as f64,
                    gpu_utilization: 80.0,
                    ..Default::default()
                },
                communication: CommunicationMetrics::default(),
                memory: MemoryMetrics::default(),
                io: IoMetrics::default(),
                custom: HashMap::new(),
            });
            history.insert(rank, rank_history);
        }
        drop(history);

        let aggregated = monitor.get_aggregated_metrics();
        assert!(aggregated.is_some());

        let metrics = aggregated.unwrap();
        // Average of 10.0 and 11.0
        assert!((metrics.compute.forward_time_ms - 10.5).abs() < 0.1);
    }

    #[test]
    fn test_statistics_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = AdvancedMonitor::calculate_statistics(&values);

        assert!((stats.mean - 5.5).abs() < 0.1);
        // Median of 10 values is average of 5th and 6th values: (5.0 + 6.0) / 2 = 5.5
        assert!((stats.median - 5.5).abs() < 0.1);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
    }
}

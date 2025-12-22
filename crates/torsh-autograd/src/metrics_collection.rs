//! Comprehensive metrics collection for gradient statistics and monitoring
//!
//! This module provides detailed metrics collection, aggregation, and reporting
//! for autograd operations, enabling performance monitoring and analysis.

use crate::{AutogradTensor, Result};
use num_traits::{Float, ToPrimitive};
use scirs2_core::random::{thread_rng, Rng}; // SciRS2 POLICY compliant
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use torsh_core::dtype::TensorElement;
use torsh_core::error::TorshError;

/// Configuration for metrics collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable gradient statistics collection
    pub collect_gradient_stats: bool,
    /// Enable performance metrics collection
    pub collect_performance_metrics: bool,
    /// Enable memory usage tracking
    pub collect_memory_metrics: bool,
    /// Enable operation timing
    pub collect_timing_metrics: bool,
    /// Maximum number of historical samples to keep
    pub max_history_size: usize,
    /// Sampling rate (1.0 = collect all, 0.1 = collect 10%)
    pub sampling_rate: f64,
    /// Metrics aggregation window size (in samples)
    pub aggregation_window: usize,
    /// Enable real-time monitoring
    pub real_time_monitoring: bool,
    /// Export format for metrics
    pub export_format: ExportFormat,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collect_gradient_stats: true,
            collect_performance_metrics: true,
            collect_memory_metrics: true,
            collect_timing_metrics: true,
            max_history_size: 10000,
            sampling_rate: 1.0,
            aggregation_window: 100,
            real_time_monitoring: false,
            export_format: ExportFormat::Json,
        }
    }
}

/// Export format for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Prometheus,
    InfluxDB,
}

/// Comprehensive metrics collector
pub struct MetricsCollector {
    config: MetricsConfig,
    gradient_metrics: Arc<RwLock<GradientMetricsStore>>,
    performance_metrics: Arc<RwLock<PerformanceMetricsStore>>,
    memory_metrics: Arc<RwLock<MemoryMetricsStore>>,
    timing_metrics: Arc<RwLock<TimingMetricsStore>>,
    current_session: Arc<RwLock<MetricsSession>>,
    event_handlers: Arc<Mutex<Vec<Box<dyn MetricsEventHandler + Send + Sync>>>>,
}

/// Gradient-specific metrics storage
#[derive(Debug, Default)]
struct GradientMetricsStore {
    /// Historical gradient statistics by parameter name
    gradient_history: HashMap<String, VecDeque<GradientSnapshot>>,
    /// Aggregated statistics
    aggregated_stats: HashMap<String, AggregatedGradientStats>,
    /// Current training step
    current_step: usize,
}

/// Performance metrics storage
#[allow(dead_code)]
#[derive(Debug, Default)]
struct PerformanceMetricsStore {
    /// Operation throughput metrics
    throughput_history: VecDeque<ThroughputSnapshot>,
    /// Bottleneck analysis
    bottleneck_analysis: BottleneckAnalysis,
    /// Performance trends
    performance_trends: PerformanceTrends,
}

/// Memory usage metrics storage
#[derive(Debug, Default)]
struct MemoryMetricsStore {
    /// Memory usage snapshots
    memory_history: VecDeque<MemorySnapshot>,
    /// Peak memory usage
    peak_memory: usize,
    /// Memory efficiency metrics
    efficiency_metrics: MemoryEfficiencyMetrics,
}

/// Timing metrics storage
#[allow(dead_code)]
#[derive(Debug, Default)]
struct TimingMetricsStore {
    /// Operation timing history
    operation_timings: HashMap<String, VecDeque<Duration>>,
    /// Cumulative timing statistics
    cumulative_stats: HashMap<String, TimingStatistics>,
    /// Performance baselines
    baselines: HashMap<String, Duration>,
}

/// Current metrics session information
#[derive(Debug)]
struct MetricsSession {
    /// Session start time
    start_time: SystemTime,
    /// Session ID
    session_id: String,
    /// Total operations recorded
    total_operations: usize,
    /// Current status
    status: SessionStatus,
}

/// Session status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionStatus {
    Active,
    Paused,
    Stopped,
}

/// Snapshot of gradient metrics at a specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientSnapshot {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Training step
    pub step: usize,
    /// Parameter name
    pub parameter_name: String,
    /// Gradient statistics
    pub stats: GradientStatistics,
    /// Gradient norm
    pub gradient_norm: f64,
    /// Gradient sparsity (fraction of zero elements)
    pub sparsity: f64,
}

/// Detailed gradient statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStatistics {
    /// Mean gradient value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min_value: f64,
    /// Maximum value
    pub max_value: f64,
    /// L1 norm
    pub l1_norm: f64,
    /// L2 norm
    pub l2_norm: f64,
    /// Number of elements
    pub num_elements: usize,
    /// Number of zero elements
    pub zero_elements: usize,
    /// Number of NaN/Inf elements
    pub invalid_elements: usize,
}

/// Aggregated statistics over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedGradientStats {
    /// Parameter name
    pub parameter_name: String,
    /// Number of samples
    pub sample_count: usize,
    /// Mean of gradient norms
    pub mean_norm: f64,
    /// Standard deviation of gradient norms
    pub std_norm: f64,
    /// Trend analysis
    pub trend: GradientTrend,
    /// Anomaly detection results
    pub anomalies: Vec<GradientAnomaly>,
}

/// Gradient trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientTrend {
    /// Overall trend direction
    pub direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Recent change rate
    pub change_rate: f64,
    /// Stability indicator
    pub stability: f64,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
}

impl Default for TrendDirection {
    fn default() -> Self {
        TrendDirection::Stable
    }
}

/// Gradient anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientAnomaly {
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Detection timestamp
    pub timestamp: SystemTime,
    /// Step when detected
    pub step: usize,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Description
    pub description: String,
    /// Suggested action
    pub suggested_action: String,
}

/// Types of anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    GradientExplosion,
    GradientVanishing,
    UnusualSparsity,
    NonFiniteValues,
    SuddenChange,
    Stagnation,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Throughput measurement snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputSnapshot {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Operations per second
    pub ops_per_second: f64,
    /// Elements processed per second
    pub elements_per_second: f64,
    /// Memory bandwidth (MB/s)
    pub memory_bandwidth: f64,
}

/// Bottleneck analysis results
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    /// Identified bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
    /// Performance recommendations
    pub recommendations: Vec<PerformanceRecommendation>,
    /// Overall performance score (0.0 to 1.0)
    pub performance_score: f64,
}

/// Identified performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Affected operations
    pub affected_operations: Vec<String>,
    /// Impact score (0.0 to 1.0)
    pub impact: f64,
    /// Description
    pub description: String,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    Computation,
    Memory,
    Communication,
    IO,
    Synchronization,
}

/// Performance improvement recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Description
    pub description: String,
    /// Expected improvement
    pub expected_improvement: String,
    /// Implementation difficulty
    pub difficulty: ImplementationDifficulty,
}

/// Types of performance recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    OptimizeAlgorithm,
    IncreaseParallelism,
    ReduceMemoryUsage,
    ImproveDataLocality,
    OptimizeCommunication,
    TuneHyperparameters,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationDifficulty {
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Performance trends analysis
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Throughput trend
    pub throughput_trend: TrendDirection,
    /// Memory usage trend
    pub memory_trend: TrendDirection,
    /// Timing trend
    pub timing_trend: TrendDirection,
    /// Overall efficiency trend
    pub efficiency_trend: TrendDirection,
}

/// Memory usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Total memory used (bytes)
    pub total_memory: usize,
    /// Peak memory used (bytes)
    pub peak_memory: usize,
    /// Memory by category
    pub memory_breakdown: MemoryBreakdown,
    /// Memory efficiency score
    pub efficiency_score: f64,
}

/// Memory usage breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBreakdown {
    /// Tensor storage
    pub tensor_memory: usize,
    /// Gradient storage
    pub gradient_memory: usize,
    /// Graph storage
    pub graph_memory: usize,
    /// Temporary buffers
    pub buffer_memory: usize,
    /// Other overhead
    pub overhead_memory: usize,
}

/// Memory efficiency metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryEfficiencyMetrics {
    /// Average memory utilization
    pub avg_utilization: f64,
    /// Peak utilization
    pub peak_utilization: f64,
    /// Memory fragmentation score
    pub fragmentation_score: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Timing statistics for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStatistics {
    /// Operation name
    pub operation_name: String,
    /// Number of samples
    pub sample_count: usize,
    /// Mean execution time
    pub mean_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
    /// 95th percentile
    pub p95_time: Duration,
    /// 99th percentile
    pub p99_time: Duration,
}

/// Event handler trait for metrics events
pub trait MetricsEventHandler {
    /// Handle a new gradient metric
    fn on_gradient_metric(&self, snapshot: &GradientSnapshot);

    /// Handle a performance metric
    fn on_performance_metric(&self, snapshot: &ThroughputSnapshot);

    /// Handle a memory metric
    fn on_memory_metric(&self, snapshot: &MemorySnapshot);

    /// Handle an anomaly detection
    fn on_anomaly_detected(&self, anomaly: &GradientAnomaly);

    /// Handle session events
    fn on_session_event(&self, event: SessionEvent);
}

/// Session events
#[derive(Debug, Clone)]
pub enum SessionEvent {
    SessionStarted(String),
    SessionPaused(String),
    SessionResumed(String),
    SessionStopped(String),
    MilestoneReached(String, usize),
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self::with_config(MetricsConfig::default())
    }

    /// Create a new metrics collector with custom configuration
    pub fn with_config(config: MetricsConfig) -> Self {
        let session_id = format!(
            "session_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        let session = MetricsSession {
            start_time: SystemTime::now(),
            session_id,
            total_operations: 0,
            status: SessionStatus::Active,
        };

        Self {
            config,
            gradient_metrics: Arc::new(RwLock::new(GradientMetricsStore::default())),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetricsStore::default())),
            memory_metrics: Arc::new(RwLock::new(MemoryMetricsStore::default())),
            timing_metrics: Arc::new(RwLock::new(TimingMetricsStore::default())),
            current_session: Arc::new(RwLock::new(session)),
            event_handlers: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Record gradient metrics for a tensor
    pub fn record_gradient_metrics<T>(
        &self,
        parameter_name: &str,
        _tensor: &dyn AutogradTensor<T>,
        gradient: &[T],
    ) -> Result<()>
    where
        T: TensorElement + Float + ToPrimitive,
    {
        if !self.config.collect_gradient_stats {
            return Ok(());
        }

        // Apply sampling - SciRS2 POLICY compliant
        let random_value = thread_rng().random::<f64>();
        let sampling_rate_f64 =
            num_traits::ToPrimitive::to_f64(&self.config.sampling_rate).unwrap_or(1.0);
        if random_value > sampling_rate_f64 {
            return Ok(());
        }

        let stats = self.compute_gradient_statistics(gradient)?;
        let gradient_norm = stats.l2_norm;
        let sparsity = stats.zero_elements as f64 / stats.num_elements as f64;

        let mut metrics_store = self.gradient_metrics.write().unwrap();
        metrics_store.current_step += 1;

        let snapshot = GradientSnapshot {
            timestamp: SystemTime::now(),
            step: metrics_store.current_step,
            parameter_name: parameter_name.to_string(),
            stats,
            gradient_norm,
            sparsity,
        };

        // Store in history
        let history = metrics_store
            .gradient_history
            .entry(parameter_name.to_string())
            .or_insert_with(VecDeque::new);

        history.push_back(snapshot.clone());

        // Maintain history size limit
        if history.len() > self.config.max_history_size {
            history.pop_front();
        }

        // Update aggregated statistics
        self.update_aggregated_stats(&mut metrics_store, parameter_name, &snapshot)?;

        // Detect anomalies
        self.detect_gradient_anomalies(&metrics_store, parameter_name, &snapshot)?;

        // Notify event handlers
        self.notify_gradient_event(&snapshot);

        Ok(())
    }

    /// Record performance metrics
    pub fn record_performance_metrics(
        &self,
        ops_per_second: f64,
        elements_per_second: f64,
        memory_bandwidth: f64,
    ) -> Result<()> {
        if !self.config.collect_performance_metrics {
            return Ok(());
        }

        let snapshot = ThroughputSnapshot {
            timestamp: SystemTime::now(),
            ops_per_second,
            elements_per_second,
            memory_bandwidth,
        };

        let mut perf_store = self.performance_metrics.write().unwrap();
        perf_store.throughput_history.push_back(snapshot.clone());

        // Maintain history size
        if perf_store.throughput_history.len() > self.config.max_history_size {
            perf_store.throughput_history.pop_front();
        }

        // Update performance analysis
        self.update_performance_analysis(&mut perf_store)?;

        // Notify event handlers
        self.notify_performance_event(&snapshot);

        Ok(())
    }

    /// Record memory usage metrics
    pub fn record_memory_metrics(&self, memory_breakdown: MemoryBreakdown) -> Result<()> {
        if !self.config.collect_memory_metrics {
            return Ok(());
        }

        let total_memory = memory_breakdown.tensor_memory
            + memory_breakdown.gradient_memory
            + memory_breakdown.graph_memory
            + memory_breakdown.buffer_memory
            + memory_breakdown.overhead_memory;

        let mut memory_store = self.memory_metrics.write().unwrap();
        memory_store.peak_memory = memory_store.peak_memory.max(total_memory);

        let efficiency_score = self.compute_memory_efficiency(&memory_breakdown);

        let snapshot = MemorySnapshot {
            timestamp: SystemTime::now(),
            total_memory,
            peak_memory: memory_store.peak_memory,
            memory_breakdown,
            efficiency_score,
        };

        memory_store.memory_history.push_back(snapshot.clone());

        // Maintain history size
        if memory_store.memory_history.len() > self.config.max_history_size {
            memory_store.memory_history.pop_front();
        }

        // Update efficiency metrics
        self.update_memory_efficiency(&mut memory_store)?;

        // Notify event handlers
        self.notify_memory_event(&snapshot);

        Ok(())
    }

    /// Record operation timing
    pub fn record_timing(&self, operation_name: &str, duration: Duration) -> Result<()> {
        if !self.config.collect_timing_metrics {
            return Ok(());
        }

        let mut timing_store = self.timing_metrics.write().unwrap();

        // Store timing sample
        let timings = timing_store
            .operation_timings
            .entry(operation_name.to_string())
            .or_insert_with(VecDeque::new);

        timings.push_back(duration);

        // Maintain history size
        if timings.len() > self.config.max_history_size {
            timings.pop_front();
        }

        // Update cumulative statistics
        self.update_timing_statistics(&mut timing_store, operation_name)?;

        Ok(())
    }

    /// Get gradient metrics for a parameter
    pub fn get_gradient_metrics(&self, parameter_name: &str) -> Option<AggregatedGradientStats> {
        let metrics_store = self.gradient_metrics.read().unwrap();
        metrics_store.aggregated_stats.get(parameter_name).cloned()
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> BottleneckAnalysis {
        let perf_store = self.performance_metrics.read().unwrap();
        perf_store.bottleneck_analysis.clone()
    }

    /// Get memory efficiency metrics
    pub fn get_memory_metrics(&self) -> MemoryEfficiencyMetrics {
        let memory_store = self.memory_metrics.read().unwrap();
        memory_store.efficiency_metrics.clone()
    }

    /// Get timing statistics for an operation
    pub fn get_timing_statistics(&self, operation_name: &str) -> Option<TimingStatistics> {
        let timing_store = self.timing_metrics.read().unwrap();
        timing_store.cumulative_stats.get(operation_name).cloned()
    }

    /// Export metrics in the configured format
    pub fn export_metrics(&self) -> Result<String> {
        match self.config.export_format {
            ExportFormat::Json => self.export_json(),
            ExportFormat::Csv => self.export_csv(),
            ExportFormat::Prometheus => self.export_prometheus(),
            ExportFormat::InfluxDB => self.export_influxdb(),
        }
    }

    /// Add an event handler
    pub fn add_event_handler(&self, handler: Box<dyn MetricsEventHandler + Send + Sync>) {
        let mut handlers = self.event_handlers.lock().unwrap();
        handlers.push(handler);
    }

    /// Start a new metrics session
    pub fn start_session(&self, session_id: Option<String>) -> Result<String> {
        let mut session = self.current_session.write().unwrap();

        let id = session_id.unwrap_or_else(|| {
            format!(
                "session_{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            )
        });

        session.session_id = id.clone();
        session.start_time = SystemTime::now();
        session.status = SessionStatus::Active;
        session.total_operations = 0;

        self.notify_session_event(SessionEvent::SessionStarted(id.clone()));

        Ok(id)
    }

    /// Pause the current session
    pub fn pause_session(&self) -> Result<()> {
        let mut session = self.current_session.write().unwrap();
        if session.status == SessionStatus::Active {
            session.status = SessionStatus::Paused;
            self.notify_session_event(SessionEvent::SessionPaused(session.session_id.clone()));
        }
        Ok(())
    }

    /// Resume the current session
    pub fn resume_session(&self) -> Result<()> {
        let mut session = self.current_session.write().unwrap();
        if session.status == SessionStatus::Paused {
            session.status = SessionStatus::Active;
            self.notify_session_event(SessionEvent::SessionResumed(session.session_id.clone()));
        }
        Ok(())
    }

    /// Stop the current session
    pub fn stop_session(&self) -> Result<()> {
        let mut session = self.current_session.write().unwrap();
        session.status = SessionStatus::Stopped;
        self.notify_session_event(SessionEvent::SessionStopped(session.session_id.clone()));
        Ok(())
    }

    /// Get current session information
    pub fn get_session_info(&self) -> String {
        let session = self.current_session.read().unwrap();
        session.session_id.clone()
    }

    // Private helper methods

    fn compute_gradient_statistics<T>(&self, gradient: &[T]) -> Result<GradientStatistics>
    where
        T: TensorElement + Float + ToPrimitive,
    {
        if gradient.is_empty() {
            return Ok(GradientStatistics {
                mean: 0.0,
                std_dev: 0.0,
                min_value: 0.0,
                max_value: 0.0,
                l1_norm: 0.0,
                l2_norm: 0.0,
                num_elements: 0,
                zero_elements: 0,
                invalid_elements: 0,
            });
        }

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut l1_norm = 0.0;
        let mut l2_norm = 0.0;
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let mut zero_count = 0;
        let mut invalid_count = 0;
        let mut valid_count = 0;

        for &val in gradient {
            let val_f64 = num_traits::ToPrimitive::to_f64(&val).unwrap_or(0.0);

            if val_f64.is_nan() || val_f64.is_infinite() {
                invalid_count += 1;
                continue;
            }

            if val_f64.abs() < f64::EPSILON {
                zero_count += 1;
            }

            valid_count += 1;
            sum += val_f64;
            sum_sq += val_f64 * val_f64;
            l1_norm += val_f64.abs();
            l2_norm += val_f64 * val_f64;
            min_val = min_val.min(val_f64);
            max_val = max_val.max(val_f64);
        }

        let mean = if valid_count > 0 {
            sum / valid_count as f64
        } else {
            0.0
        };
        let variance = if valid_count > 1 {
            (sum_sq / valid_count as f64) - mean * mean
        } else {
            0.0
        };
        let std_dev = variance.max(0.0).sqrt();

        Ok(GradientStatistics {
            mean,
            std_dev,
            min_value: if min_val.is_finite() { min_val } else { 0.0 },
            max_value: if max_val.is_finite() { max_val } else { 0.0 },
            l1_norm,
            l2_norm: l2_norm.sqrt(),
            num_elements: gradient.len(),
            zero_elements: zero_count,
            invalid_elements: invalid_count,
        })
    }

    fn update_aggregated_stats(
        &self,
        metrics_store: &mut GradientMetricsStore,
        parameter_name: &str,
        snapshot: &GradientSnapshot,
    ) -> Result<()> {
        let needs_trend_update = {
            let stats = metrics_store
                .aggregated_stats
                .entry(parameter_name.to_string())
                .or_insert_with(|| AggregatedGradientStats {
                    parameter_name: parameter_name.to_string(),
                    sample_count: 0,
                    mean_norm: 0.0,
                    std_norm: 0.0,
                    trend: GradientTrend {
                        direction: TrendDirection::Stable,
                        strength: 0.0,
                        change_rate: 0.0,
                        stability: 1.0,
                    },
                    anomalies: Vec::new(),
                });

            // Update running statistics
            stats.sample_count += 1;
            let alpha = 1.0 / stats.sample_count as f64;
            stats.mean_norm = (1.0 - alpha) * stats.mean_norm + alpha * snapshot.gradient_norm;

            stats.sample_count > 1 // Need at least 2 samples for trend analysis
        };

        // Update trend analysis if we have enough data
        if needs_trend_update {
            self.update_gradient_trend_simple(parameter_name, snapshot)?;
        }

        Ok(())
    }

    fn update_gradient_trend_simple(
        &self,
        _parameter_name: &str,
        _snapshot: &GradientSnapshot,
    ) -> Result<()> {
        // Simplified trend analysis - placeholder implementation
        Ok(())
    }

    #[allow(dead_code)]
    fn update_gradient_trend(
        &self,
        _metrics_store: &GradientMetricsStore,
        _parameter_name: &str,
        _stats: &mut AggregatedGradientStats,
    ) -> Result<()> {
        // Simplified implementation - placeholder
        Ok(())
    }

    fn detect_gradient_anomalies(
        &self,
        _metrics_store: &GradientMetricsStore,
        _parameter_name: &str,
        snapshot: &GradientSnapshot,
    ) -> Result<()> {
        let mut anomalies = Vec::new();

        // Check for gradient explosion
        if snapshot.gradient_norm > 10.0 {
            anomalies.push(GradientAnomaly {
                anomaly_type: AnomalyType::GradientExplosion,
                timestamp: snapshot.timestamp,
                step: snapshot.step,
                severity: if snapshot.gradient_norm > 100.0 {
                    AnomalySeverity::Critical
                } else {
                    AnomalySeverity::High
                },
                description: format!("Large gradient norm: {:.6}", snapshot.gradient_norm),
                suggested_action: "Consider gradient clipping".to_string(),
            });
        }

        // Check for vanishing gradients
        if snapshot.gradient_norm < 1e-8 && snapshot.stats.num_elements > 0 {
            anomalies.push(GradientAnomaly {
                anomaly_type: AnomalyType::GradientVanishing,
                timestamp: snapshot.timestamp,
                step: snapshot.step,
                severity: AnomalySeverity::High,
                description: format!("Very small gradient norm: {:.12}", snapshot.gradient_norm),
                suggested_action: "Check for vanishing gradient problem".to_string(),
            });
        }

        // Check for invalid values
        if snapshot.stats.invalid_elements > 0 {
            anomalies.push(GradientAnomaly {
                anomaly_type: AnomalyType::NonFiniteValues,
                timestamp: snapshot.timestamp,
                step: snapshot.step,
                severity: AnomalySeverity::Critical,
                description: format!(
                    "{} invalid (NaN/Inf) gradient values",
                    snapshot.stats.invalid_elements
                ),
                suggested_action: "Check for numerical instability".to_string(),
            });
        }

        // Notify event handlers about anomalies
        for anomaly in &anomalies {
            self.notify_anomaly_event(anomaly);
        }

        Ok(())
    }

    fn compute_memory_efficiency(&self, breakdown: &MemoryBreakdown) -> f64 {
        let total = breakdown.tensor_memory
            + breakdown.gradient_memory
            + breakdown.graph_memory
            + breakdown.buffer_memory
            + breakdown.overhead_memory;

        if total == 0 {
            return 1.0;
        }

        let useful_memory = breakdown.tensor_memory + breakdown.gradient_memory;
        useful_memory as f64 / total as f64
    }

    fn update_performance_analysis(&self, _perf_store: &mut PerformanceMetricsStore) -> Result<()> {
        // Placeholder for performance analysis
        // In a real implementation, this would analyze throughput trends,
        // detect bottlenecks, and generate recommendations
        Ok(())
    }

    fn update_memory_efficiency(&self, _memory_store: &mut MemoryMetricsStore) -> Result<()> {
        // Placeholder for memory efficiency analysis
        // In a real implementation, this would compute efficiency metrics
        Ok(())
    }

    fn update_timing_statistics(
        &self,
        timing_store: &mut TimingMetricsStore,
        operation_name: &str,
    ) -> Result<()> {
        if let Some(timings) = timing_store.operation_timings.get(operation_name) {
            if timings.is_empty() {
                return Ok(());
            }

            let mut sorted_timings: Vec<Duration> = timings.iter().copied().collect();
            sorted_timings.sort();

            let count = sorted_timings.len();
            let mean_nanos = sorted_timings
                .iter()
                .map(|d| d.as_nanos() as f64)
                .sum::<f64>()
                / count as f64;
            let mean_time = Duration::from_nanos(mean_nanos as u64);

            let variance = sorted_timings
                .iter()
                .map(|d| (d.as_nanos() as f64 - mean_nanos).powi(2))
                .sum::<f64>()
                / count as f64;
            let std_dev = Duration::from_nanos(variance.sqrt() as u64);

            let p95_index = (count as f64 * 0.95) as usize;
            let p99_index = (count as f64 * 0.99) as usize;

            let stats = TimingStatistics {
                operation_name: operation_name.to_string(),
                sample_count: count,
                mean_time,
                std_dev,
                min_time: sorted_timings[0],
                max_time: sorted_timings[count - 1],
                p95_time: sorted_timings[p95_index.min(count - 1)],
                p99_time: sorted_timings[p99_index.min(count - 1)],
            };

            timing_store
                .cumulative_stats
                .insert(operation_name.to_string(), stats);
        }

        Ok(())
    }

    // Export methods

    fn export_json(&self) -> Result<String> {
        let gradient_metrics = self.gradient_metrics.read().unwrap();
        let performance_metrics = self.performance_metrics.read().unwrap();
        let memory_metrics = self.memory_metrics.read().unwrap();
        let timing_metrics = self.timing_metrics.read().unwrap();

        let export_data = serde_json::json!({
            "gradient_metrics": gradient_metrics.aggregated_stats,
            "performance_metrics": performance_metrics.bottleneck_analysis,
            "memory_metrics": memory_metrics.efficiency_metrics,
            "timing_metrics": timing_metrics.cumulative_stats
        });

        serde_json::to_string_pretty(&export_data)
            .map_err(|e| TorshError::AutogradError(format!("JSON export failed: {}", e)))
    }

    fn export_csv(&self) -> Result<String> {
        // Placeholder CSV export
        Ok("timestamp,parameter,gradient_norm,sparsity\n".to_string())
    }

    fn export_prometheus(&self) -> Result<String> {
        // Placeholder Prometheus export
        Ok("# TYPE gradient_norm gauge\n".to_string())
    }

    fn export_influxdb(&self) -> Result<String> {
        // Placeholder InfluxDB export
        Ok("gradient_metrics,parameter=layer1 norm=0.123 1234567890\n".to_string())
    }

    // Event notification methods

    fn notify_gradient_event(&self, snapshot: &GradientSnapshot) {
        if let Ok(handlers) = self.event_handlers.lock() {
            for handler in handlers.iter() {
                handler.on_gradient_metric(snapshot);
            }
        }
    }

    fn notify_performance_event(&self, snapshot: &ThroughputSnapshot) {
        if let Ok(handlers) = self.event_handlers.lock() {
            for handler in handlers.iter() {
                handler.on_performance_metric(snapshot);
            }
        }
    }

    fn notify_memory_event(&self, snapshot: &MemorySnapshot) {
        if let Ok(handlers) = self.event_handlers.lock() {
            for handler in handlers.iter() {
                handler.on_memory_metric(snapshot);
            }
        }
    }

    fn notify_anomaly_event(&self, anomaly: &GradientAnomaly) {
        if let Ok(handlers) = self.event_handlers.lock() {
            for handler in handlers.iter() {
                handler.on_anomaly_detected(anomaly);
            }
        }
    }

    fn notify_session_event(&self, event: SessionEvent) {
        if let Ok(handlers) = self.event_handlers.lock() {
            for handler in handlers.iter() {
                handler.on_session_event(event.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::CpuDevice;
    use torsh_core::shape::Shape;

    // Mock tensor for testing
    struct MockTensor {
        data: Vec<f32>,
        shape: Shape,
        requires_grad: bool,
        device: CpuDevice,
    }

    impl MockTensor {
        fn new(data: Vec<f32>, shape: Shape, requires_grad: bool) -> Self {
            Self {
                data,
                shape,
                requires_grad,
                device: CpuDevice::new(),
            }
        }
    }

    impl AutogradTensor<f32> for MockTensor {
        fn shape(&self) -> Shape {
            self.shape.clone()
        }

        fn requires_grad(&self) -> bool {
            self.requires_grad
        }

        fn data(&self) -> Box<dyn std::ops::Deref<Target = [f32]> + '_> {
            Box::new(self.data.as_slice())
        }

        fn clone_tensor(&self) -> Box<dyn AutogradTensor<f32>> {
            Box::new(MockTensor::new(
                self.data.clone(),
                self.shape.clone(),
                self.requires_grad,
            ))
        }

        fn to_vec(&self) -> Vec<f32> {
            self.data.clone()
        }

        fn device(&self) -> &dyn torsh_core::Device {
            &self.device
        }

        fn ones_like(&self) -> Box<dyn AutogradTensor<f32>> {
            let ones_data = vec![1.0; self.data.len()];
            Box::new(MockTensor::new(
                ones_data,
                self.shape.clone(),
                self.requires_grad,
            ))
        }

        fn zeros_like(&self) -> Box<dyn AutogradTensor<f32>> {
            let zeros_data = vec![0.0; self.data.len()];
            Box::new(MockTensor::new(
                zeros_data,
                self.shape.clone(),
                self.requires_grad,
            ))
        }
    }

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        assert!(collector.config.collect_gradient_stats);
        assert!(collector.config.collect_performance_metrics);
    }

    #[test]
    fn test_gradient_metrics_recording() {
        let collector = MetricsCollector::new();
        let tensor = MockTensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3]), true);
        let gradient = vec![0.1, 0.2, 0.3];

        let result = collector.record_gradient_metrics("test_param", &tensor, &gradient);
        assert!(result.is_ok());

        let metrics = collector.get_gradient_metrics("test_param");
        assert!(metrics.is_some());

        let stats = metrics.unwrap();
        assert_eq!(stats.parameter_name, "test_param");
        assert_eq!(stats.sample_count, 1);
    }

    #[test]
    fn test_performance_metrics_recording() {
        let collector = MetricsCollector::new();

        let result = collector.record_performance_metrics(100.0, 1000.0, 50.0);
        assert!(result.is_ok());

        let metrics = collector.get_performance_metrics();
        // Basic test - in a real implementation, this would have more detailed assertions
        assert_eq!(metrics.bottlenecks.len(), 0); // No bottlenecks with single sample
    }

    #[test]
    fn test_timing_metrics_recording() {
        let collector = MetricsCollector::new();

        let duration = Duration::from_millis(10);
        let result = collector.record_timing("test_op", duration);
        assert!(result.is_ok());

        let stats = collector.get_timing_statistics("test_op");
        assert!(stats.is_some());

        let timing_stats = stats.unwrap();
        assert_eq!(timing_stats.operation_name, "test_op");
        assert_eq!(timing_stats.sample_count, 1);
        assert_eq!(timing_stats.mean_time, duration);
    }

    #[test]
    fn test_session_management() {
        let collector = MetricsCollector::new();

        let session_id = collector
            .start_session(Some("test_session".to_string()))
            .unwrap();
        assert_eq!(session_id, "test_session");

        assert!(collector.pause_session().is_ok());
        assert!(collector.resume_session().is_ok());
        assert!(collector.stop_session().is_ok());
    }

    #[test]
    fn test_metrics_export() {
        let collector = MetricsCollector::new();

        // Record some sample data
        let tensor = MockTensor::new(vec![1.0, 2.0], Shape::new(vec![2]), true);
        let gradient = vec![0.1, 0.2];
        let _ = collector.record_gradient_metrics("test", &tensor, &gradient);

        let export_result = collector.export_metrics();
        assert!(export_result.is_ok());

        let exported = export_result.unwrap();
        assert!(exported.contains("gradient_metrics"));
    }

    #[test]
    fn test_gradient_statistics_computation() {
        let collector = MetricsCollector::new();
        let gradient = vec![1.0, 2.0, 0.0, -1.0, f32::NAN];

        let stats = collector.compute_gradient_statistics(&gradient).unwrap();
        assert_eq!(stats.num_elements, 5);
        assert_eq!(stats.zero_elements, 1);
        assert_eq!(stats.invalid_elements, 1);
        assert_eq!(stats.mean, 0.5); // (1 + 2 + 0 + (-1)) / 4 = 0.5
    }

    #[test]
    fn test_anomaly_detection() {
        let collector = MetricsCollector::new();
        let tensor = MockTensor::new(vec![1.0], Shape::new(vec![1]), true);

        // Test gradient explosion detection
        let large_gradient = vec![1000.0];
        let result = collector.record_gradient_metrics("exploding", &tensor, &large_gradient);
        assert!(result.is_ok());

        // Test vanishing gradient detection
        let small_gradient = vec![1e-10];
        let result = collector.record_gradient_metrics("vanishing", &tensor, &small_gradient);
        assert!(result.is_ok());

        // Test invalid values detection
        let invalid_gradient = vec![f32::NAN];
        let result = collector.record_gradient_metrics("invalid", &tensor, &invalid_gradient);
        assert!(result.is_ok());
    }
}

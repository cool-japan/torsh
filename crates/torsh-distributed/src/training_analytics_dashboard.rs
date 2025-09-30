//! Distributed Training Analytics Dashboard
//!
//! This module provides a comprehensive analytics dashboard for distributed training,
//! offering real-time insights, performance visualization, and intelligent analysis
//! of training progress, resource utilization, and system health.

use crate::distributed_memory_optimization::{
    DistributedMemoryOptimizer, MemoryOptimizationStatus,
};
use crate::distributed_monitoring::{ClusterSummary, DistributedMonitor, NodeMetrics};
use crate::enhanced_fault_tolerance::{EnhancedFaultTolerance, FaultToleranceStatus};
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};

/// Training analytics data aggregated across the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingAnalytics {
    /// Training performance metrics
    pub performance: TrainingPerformanceAnalytics,
    /// Resource utilization analytics
    pub resource_utilization: ResourceUtilizationAnalytics,
    /// Communication efficiency analytics
    pub communication: CommunicationAnalytics,
    /// System health analytics
    pub system_health: SystemHealthAnalytics,
    /// Training convergence analytics
    pub convergence: ConvergenceAnalytics,
    /// Efficiency and optimization analytics
    pub efficiency: EfficiencyAnalytics,
    /// Timestamp of analytics generation
    pub timestamp_ms: u64,
}

/// Training performance analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPerformanceAnalytics {
    /// Current epoch across all nodes
    pub current_epoch: u32,
    /// Current average loss across nodes
    pub avg_loss: f32,
    /// Loss trend (positive = increasing, negative = decreasing)
    pub loss_trend: f32,
    /// Training throughput (samples/second across cluster)
    pub cluster_throughput: f32,
    /// Throughput efficiency compared to theoretical maximum
    pub throughput_efficiency: f32,
    /// Average batch time across nodes (milliseconds)
    pub avg_batch_time_ms: u64,
    /// Batch time variance (indicator of load balance)
    pub batch_time_variance: f32,
    /// Training stability score (0.0 to 1.0)
    pub training_stability: f32,
    /// Estimated time to completion
    pub estimated_completion_time: Option<Duration>,
}

/// Resource utilization analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationAnalytics {
    /// Average CPU utilization across cluster
    pub avg_cpu_utilization: f32,
    /// Average GPU utilization across cluster
    pub avg_gpu_utilization: f32,
    /// Average memory utilization across cluster
    pub avg_memory_utilization: f32,
    /// Resource utilization balance score
    pub utilization_balance: f32,
    /// Peak resource usage
    pub peak_cpu: f32,
    pub peak_gpu: f32,
    pub peak_memory: f32,
    /// Resource efficiency score
    pub resource_efficiency: f32,
    /// Bottleneck identification
    pub primary_bottleneck: ResourceBottleneck,
}

/// Identified resource bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResourceBottleneck {
    CPU,
    GPU,
    Memory,
    Network,
    Storage,
    None,
}

impl std::fmt::Display for ResourceBottleneck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceBottleneck::CPU => write!(f, "CPU"),
            ResourceBottleneck::GPU => write!(f, "GPU"),
            ResourceBottleneck::Memory => write!(f, "Memory"),
            ResourceBottleneck::Network => write!(f, "Network"),
            ResourceBottleneck::Storage => write!(f, "Storage"),
            ResourceBottleneck::None => write!(f, "None"),
        }
    }
}

/// Communication efficiency analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationAnalytics {
    /// Average communication latency (microseconds)
    pub avg_latency_us: u64,
    /// Communication bandwidth utilization
    pub bandwidth_utilization: f32,
    /// Communication efficiency score
    pub efficiency_score: f32,
    /// Failed operations rate
    pub failed_operations_rate: f32,
    /// Communication patterns analysis
    pub communication_patterns: CommunicationPatterns,
    /// Network congestion indicator
    pub congestion_level: f32,
}

/// Communication patterns analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPatterns {
    /// All-reduce operation frequency
    pub allreduce_frequency: f32,
    /// All-gather operation frequency
    pub allgather_frequency: f32,
    /// Point-to-point communication frequency
    pub p2p_frequency: f32,
    /// Gradient synchronization frequency
    pub gradient_sync_frequency: f32,
    /// Communication hotspots (node pairs with high traffic)
    pub hotspots: Vec<CommunicationHotspot>,
}

/// Communication hotspot identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationHotspot {
    /// Source node
    pub source_node: String,
    /// Target node
    pub target_node: String,
    /// Traffic volume (MB/s)
    pub traffic_volume: f32,
    /// Congestion score (0.0 to 1.0)
    pub congestion_score: f32,
}

/// System health analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthAnalytics {
    /// Overall cluster health score
    pub cluster_health_score: f32,
    /// Number of healthy nodes
    pub healthy_nodes: usize,
    /// Number of degraded nodes
    pub degraded_nodes: usize,
    /// Number of critical nodes
    pub critical_nodes: usize,
    /// Number of failed nodes
    pub failed_nodes: usize,
    /// Active incidents count
    pub active_incidents: usize,
    /// System stability trend
    pub stability_trend: f32,
    /// Predicted failure probability (0.0 to 1.0)
    pub failure_probability: f32,
}

/// Training convergence analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalytics {
    /// Convergence rate (improvement per epoch)
    pub convergence_rate: f32,
    /// Convergence confidence (0.0 to 1.0)
    pub convergence_confidence: f32,
    /// Training progress percentage
    pub training_progress: f32,
    /// Loss smoothness indicator
    pub loss_smoothness: f32,
    /// Gradient norm statistics
    pub gradient_norm_stats: GradientNormStats,
    /// Learning rate effectiveness
    pub lr_effectiveness: f32,
    /// Overfitting risk score
    pub overfitting_risk: f32,
}

/// Gradient norm statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientNormStats {
    /// Average gradient norm
    pub avg_norm: f32,
    /// Gradient norm variance
    pub norm_variance: f32,
    /// Gradient norm trend
    pub norm_trend: f32,
    /// Gradient explosion risk
    pub explosion_risk: f32,
}

/// Training efficiency analytics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyAnalytics {
    /// Overall training efficiency score
    pub overall_efficiency: f32,
    /// Computational efficiency
    pub compute_efficiency: f32,
    /// Communication efficiency
    pub communication_efficiency: f32,
    /// Memory efficiency
    pub memory_efficiency: f32,
    /// Energy efficiency (performance per watt)
    pub energy_efficiency: f32,
    /// Cost efficiency (performance per dollar)
    pub cost_efficiency: f32,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation category
    pub category: RecommendationCategory,
    /// Recommendation title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Expected impact (0.0 to 1.0)
    pub expected_impact: f32,
    /// Implementation difficulty (0.0 to 1.0)
    pub difficulty: f32,
    /// Priority score
    pub priority: u32,
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecommendationCategory {
    Performance,
    Efficiency,
    Reliability,
    Cost,
    Scalability,
}

impl std::fmt::Display for RecommendationCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecommendationCategory::Performance => write!(f, "Performance"),
            RecommendationCategory::Efficiency => write!(f, "Efficiency"),
            RecommendationCategory::Reliability => write!(f, "Reliability"),
            RecommendationCategory::Cost => write!(f, "Cost"),
            RecommendationCategory::Scalability => write!(f, "Scalability"),
        }
    }
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Analytics update interval
    pub update_interval: Duration,
    /// Historical data retention period
    pub retention_period: Duration,
    /// Enable predictive analytics
    pub enable_predictions: bool,
    /// Enable optimization recommendations
    pub enable_recommendations: bool,
    /// Metrics aggregation window
    pub aggregation_window: Duration,
    /// Alert thresholds
    pub alert_thresholds: DashboardAlertThresholds,
}

/// Alert thresholds for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardAlertThresholds {
    /// Training efficiency threshold
    pub efficiency_threshold: f32,
    /// Resource utilization threshold
    pub utilization_threshold: f32,
    /// Communication latency threshold (microseconds)
    pub latency_threshold: u64,
    /// Convergence rate threshold
    pub convergence_threshold: f32,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(10),
            retention_period: Duration::from_secs(24 * 3600), // 24 hours
            enable_predictions: true,
            enable_recommendations: true,
            aggregation_window: Duration::from_secs(60),
            alert_thresholds: DashboardAlertThresholds {
                efficiency_threshold: 0.7,
                utilization_threshold: 0.9,
                latency_threshold: 10000,
                convergence_threshold: 0.1,
            },
        }
    }
}

/// Distributed training analytics dashboard
pub struct TrainingAnalyticsDashboard {
    /// Configuration
    config: DashboardConfig,
    /// Distributed monitoring system
    monitor: Arc<DistributedMonitor>,
    /// Enhanced fault tolerance system
    fault_tolerance: Arc<EnhancedFaultTolerance>,
    /// Memory optimization system
    memory_optimizer: Arc<DistributedMemoryOptimizer>,
    /// Current analytics
    current_analytics: Arc<RwLock<Option<TrainingAnalytics>>>,
    /// Analytics history
    analytics_history: Arc<Mutex<VecDeque<TrainingAnalytics>>>,
    /// Performance trend analyzer
    trend_analyzer: Arc<Mutex<TrendAnalyzer>>,
    /// Recommendation engine
    recommendation_engine: Arc<Mutex<RecommendationEngine>>,
    /// Last update time
    last_update: Arc<Mutex<Instant>>,
}

/// Trend analysis system
#[derive(Debug)]
struct TrendAnalyzer {
    /// Loss history for trend analysis
    loss_history: VecDeque<(u64, f32)>, // (timestamp, loss)
    /// Throughput history
    throughput_history: VecDeque<(u64, f32)>,
    /// Resource utilization history
    resource_history: VecDeque<(u64, ResourceSnapshot)>,
    /// Trend calculation window
    trend_window: Duration,
}

/// Resource utilization snapshot
#[derive(Debug, Clone)]
struct ResourceSnapshot {
    cpu: f32,
    gpu: f32,
    memory: f32,
}

impl TrendAnalyzer {
    fn new(trend_window: Duration) -> Self {
        Self {
            loss_history: VecDeque::with_capacity(1000),
            throughput_history: VecDeque::with_capacity(1000),
            resource_history: VecDeque::with_capacity(1000),
            trend_window,
        }
    }

    fn update_loss(&mut self, timestamp: u64, loss: f32) {
        self.loss_history.push_back((timestamp, loss));
        self.cleanup_old_data(timestamp);
    }

    fn update_throughput(&mut self, timestamp: u64, throughput: f32) {
        self.throughput_history.push_back((timestamp, throughput));
        self.cleanup_old_data(timestamp);
    }

    fn update_resources(&mut self, timestamp: u64, cpu: f32, gpu: f32, memory: f32) {
        self.resource_history
            .push_back((timestamp, ResourceSnapshot { cpu, gpu, memory }));
        self.cleanup_old_data(timestamp);
    }

    fn cleanup_old_data(&mut self, current_timestamp: u64) {
        let cutoff = current_timestamp.saturating_sub(self.trend_window.as_millis() as u64);

        self.loss_history.retain(|(ts, _)| *ts >= cutoff);
        self.throughput_history.retain(|(ts, _)| *ts >= cutoff);
        self.resource_history.retain(|(ts, _)| *ts >= cutoff);
    }

    fn calculate_loss_trend(&self) -> f32 {
        if self.loss_history.len() < 10 {
            return 0.0;
        }

        let recent_data: Vec<f32> = self
            .loss_history
            .iter()
            .rev()
            .take(10)
            .map(|(_, loss)| *loss)
            .collect();
        let early_avg = recent_data[5..].iter().sum::<f32>() / (recent_data.len() - 5) as f32;
        let late_avg = recent_data[..5].iter().sum::<f32>() / 5.0;

        (early_avg - late_avg) / early_avg.max(0.001) // Negative means improvement
    }

    fn calculate_throughput_trend(&self) -> f32 {
        if self.throughput_history.len() < 10 {
            return 0.0;
        }

        let recent_data: Vec<f32> = self
            .throughput_history
            .iter()
            .rev()
            .take(10)
            .map(|(_, tput)| *tput)
            .collect();
        let early_avg = recent_data[5..].iter().sum::<f32>() / (recent_data.len() - 5) as f32;
        let late_avg = recent_data[..5].iter().sum::<f32>() / 5.0;

        (late_avg - early_avg) / early_avg.max(0.001) // Positive means improvement
    }

    fn calculate_stability(&self) -> f32 {
        if self.loss_history.len() < 20 {
            return 0.5; // Neutral stability
        }

        let recent_losses: Vec<f32> = self
            .loss_history
            .iter()
            .rev()
            .take(20)
            .map(|(_, loss)| *loss)
            .collect();
        let mean = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let variance = recent_losses
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / recent_losses.len() as f32;
        let std_dev = variance.sqrt();

        // Lower coefficient of variation = higher stability
        let cv = if mean > 0.001 { std_dev / mean } else { 1.0 };
        (1.0 - cv.min(1.0)).max(0.0)
    }
}

/// Intelligent recommendation engine
#[derive(Debug)]
struct RecommendationEngine {
    /// Recent performance data for analysis
    performance_history: VecDeque<PerformanceSnapshot>,
    /// Generated recommendations cache
    recommendation_cache: Vec<OptimizationRecommendation>,
    /// Last recommendation generation time
    last_generation: Instant,
}

/// Performance snapshot for recommendations
#[derive(Debug, Clone)]
struct PerformanceSnapshot {
    timestamp: Instant,
    throughput: f32,
    efficiency: f32,
    cpu_util: f32,
    gpu_util: f32,
    memory_util: f32,
    communication_latency: u64,
}

impl RecommendationEngine {
    fn new() -> Self {
        Self {
            performance_history: VecDeque::with_capacity(100),
            recommendation_cache: Vec::new(),
            last_generation: Instant::now(),
        }
    }

    fn update_performance(
        &mut self,
        throughput: f32,
        efficiency: f32,
        cpu_util: f32,
        gpu_util: f32,
        memory_util: f32,
        communication_latency: u64,
    ) {
        let snapshot = PerformanceSnapshot {
            timestamp: Instant::now(),
            throughput,
            efficiency,
            cpu_util,
            gpu_util,
            memory_util,
            communication_latency,
        };

        self.performance_history.push_back(snapshot);
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }
    }

    fn generate_recommendations(&mut self) -> Vec<OptimizationRecommendation> {
        // Only regenerate recommendations every 5 minutes
        if self.last_generation.elapsed().as_secs() < 300 {
            return self.recommendation_cache.clone();
        }

        let mut recommendations = Vec::new();

        if self.performance_history.len() < 10 {
            return recommendations;
        }

        // Analyze recent performance
        let recent_perf: Vec<&PerformanceSnapshot> =
            self.performance_history.iter().rev().take(10).collect();

        // Check for low GPU utilization
        let avg_gpu_util =
            recent_perf.iter().map(|p| p.gpu_util).sum::<f32>() / recent_perf.len() as f32;
        if avg_gpu_util < 70.0 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Performance,
                title: "Increase GPU Utilization".to_string(),
                description: format!("GPU utilization is at {:.1}%. Consider increasing batch size or adjusting data loading.", avg_gpu_util),
                expected_impact: 0.8,
                difficulty: 0.3,
                priority: 4,
            });
        }

        // Check for high memory utilization
        let avg_memory_util =
            recent_perf.iter().map(|p| p.memory_util).sum::<f32>() / recent_perf.len() as f32;
        if avg_memory_util > 90.0 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Efficiency,
                title: "Optimize Memory Usage".to_string(),
                description: format!("Memory utilization is at {:.1}%. Consider enabling gradient checkpointing or reducing batch size.", avg_memory_util),
                expected_impact: 0.6,
                difficulty: 0.4,
                priority: 3,
            });
        }

        // Check for high communication latency
        let avg_latency = recent_perf
            .iter()
            .map(|p| p.communication_latency)
            .sum::<u64>()
            / recent_perf.len() as u64;
        if avg_latency > 5000 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Performance,
                title: "Optimize Communication".to_string(),
                description: format!("Communication latency is {}μs. Consider gradient compression or improving network connectivity.", avg_latency),
                expected_impact: 0.7,
                difficulty: 0.6,
                priority: 3,
            });
        }

        // Check for efficiency trends
        let efficiency_trend = self.calculate_efficiency_trend(&recent_perf);
        if efficiency_trend < -0.1 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Efficiency,
                title: "Address Efficiency Decline".to_string(),
                description: "Training efficiency is declining. Review resource allocation and check for bottlenecks.".to_string(),
                expected_impact: 0.9,
                difficulty: 0.7,
                priority: 5,
            });
        }

        // Check for load balancing issues
        let throughput_variance = self.calculate_throughput_variance(&recent_perf);
        if throughput_variance > 0.2 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Scalability,
                title: "Improve Load Balancing".to_string(),
                description: "High throughput variance detected. Consider redistributing workload across nodes.".to_string(),
                expected_impact: 0.5,
                difficulty: 0.8,
                priority: 2,
            });
        }

        // Sort by priority
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));

        self.recommendation_cache = recommendations.clone();
        self.last_generation = Instant::now();

        recommendations
    }

    fn calculate_efficiency_trend(&self, recent_perf: &[&PerformanceSnapshot]) -> f32 {
        if recent_perf.len() < 6 {
            return 0.0;
        }

        let early_efficiency: f32 = recent_perf[3..].iter().map(|p| p.efficiency).sum::<f32>()
            / (recent_perf.len() - 3) as f32;
        let late_efficiency: f32 = recent_perf[..3].iter().map(|p| p.efficiency).sum::<f32>() / 3.0;

        (late_efficiency - early_efficiency) / early_efficiency.max(0.001)
    }

    fn calculate_throughput_variance(&self, recent_perf: &[&PerformanceSnapshot]) -> f32 {
        if recent_perf.len() < 5 {
            return 0.0;
        }

        let throughputs: Vec<f32> = recent_perf.iter().map(|p| p.throughput).collect();
        let mean = throughputs.iter().sum::<f32>() / throughputs.len() as f32;
        let variance =
            throughputs.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / throughputs.len() as f32;

        if mean > 0.001 {
            variance.sqrt() / mean // Coefficient of variation
        } else {
            0.0
        }
    }
}

impl TrainingAnalyticsDashboard {
    /// Create new training analytics dashboard
    pub fn new(
        config: DashboardConfig,
        monitor: Arc<DistributedMonitor>,
        fault_tolerance: Arc<EnhancedFaultTolerance>,
        memory_optimizer: Arc<DistributedMemoryOptimizer>,
    ) -> Self {
        Self {
            config: config.clone(),
            monitor,
            fault_tolerance,
            memory_optimizer,
            current_analytics: Arc::new(RwLock::new(None)),
            analytics_history: Arc::new(Mutex::new(VecDeque::new())),
            trend_analyzer: Arc::new(Mutex::new(TrendAnalyzer::new(config.aggregation_window))),
            recommendation_engine: Arc::new(Mutex::new(RecommendationEngine::new())),
            last_update: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Update analytics with latest data
    pub fn update_analytics(&self) -> TorshResult<()> {
        // Check if enough time has passed since last update
        {
            let last_update = self.last_update.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "last_update",
                    format!("Lock error: {}", e),
                )
            })?;
            if last_update.elapsed() < self.config.update_interval {
                return Ok(());
            }
        }

        // Gather data from all systems
        let cluster_summary = if let Ok(summary) = self.monitor.get_cluster_summary() {
            Some(summary)
        } else {
            None
        };

        let fault_tolerance_status = self.fault_tolerance.get_status()?;
        let memory_optimization_status = self.memory_optimizer.get_optimization_status()?;

        // Generate comprehensive analytics
        let analytics = self.generate_training_analytics(
            cluster_summary,
            fault_tolerance_status,
            memory_optimization_status,
        )?;

        // Update trend analyzer
        {
            let mut trend_analyzer = self.trend_analyzer.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "trend_analyzer",
                    format!("Lock error: {}", e),
                )
            })?;

            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            trend_analyzer.update_loss(timestamp, analytics.performance.avg_loss);
            trend_analyzer.update_throughput(timestamp, analytics.performance.cluster_throughput);
            trend_analyzer.update_resources(
                timestamp,
                analytics.resource_utilization.avg_cpu_utilization,
                analytics.resource_utilization.avg_gpu_utilization,
                analytics.resource_utilization.avg_memory_utilization,
            );
        }

        // Update recommendation engine
        if self.config.enable_recommendations {
            let mut recommendation_engine = self.recommendation_engine.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "recommendation_engine",
                    format!("Lock error: {}", e),
                )
            })?;

            recommendation_engine.update_performance(
                analytics.performance.cluster_throughput,
                analytics.efficiency.overall_efficiency,
                analytics.resource_utilization.avg_cpu_utilization,
                analytics.resource_utilization.avg_gpu_utilization,
                analytics.resource_utilization.avg_memory_utilization,
                analytics.communication.avg_latency_us,
            );
        }

        // Store current analytics
        {
            let mut current_analytics = self.current_analytics.write().map_err(|e| {
                TorshDistributedError::communication_error(
                    "current_analytics",
                    format!("Lock error: {}", e),
                )
            })?;
            *current_analytics = Some(analytics.clone());
        }

        // Add to history
        {
            let mut analytics_history = self.analytics_history.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "analytics_history",
                    format!("Lock error: {}", e),
                )
            })?;
            analytics_history.push_back(analytics);

            // Cleanup old data
            let retention_cutoff = (SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64)
                .saturating_sub(self.config.retention_period.as_millis() as u64);

            analytics_history.retain(|a| a.timestamp_ms >= retention_cutoff);
        }

        // Update last update time
        {
            let mut last_update = self.last_update.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "last_update",
                    format!("Lock error: {}", e),
                )
            })?;
            *last_update = Instant::now();
        }

        Ok(())
    }

    /// Generate comprehensive training analytics
    fn generate_training_analytics(
        &self,
        cluster_summary: Option<ClusterSummary>,
        fault_tolerance_status: FaultToleranceStatus,
        memory_optimization_status: MemoryOptimizationStatus,
    ) -> TorshResult<TrainingAnalytics> {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Get current monitoring data
        let current_metrics = self.monitor.get_current_metrics()?;

        // Generate performance analytics
        let performance =
            self.generate_performance_analytics(&current_metrics, &cluster_summary)?;

        // Generate resource utilization analytics
        let resource_utilization =
            self.generate_resource_analytics(&cluster_summary, &memory_optimization_status)?;

        // Generate communication analytics
        let communication = self.generate_communication_analytics(&current_metrics)?;

        // Generate system health analytics
        let system_health = self.generate_system_health_analytics(&fault_tolerance_status)?;

        // Generate convergence analytics
        let convergence = self.generate_convergence_analytics(&current_metrics)?;

        // Generate efficiency analytics
        let efficiency = self.generate_efficiency_analytics(
            &performance,
            &resource_utilization,
            &communication,
        )?;

        Ok(TrainingAnalytics {
            performance,
            resource_utilization,
            communication,
            system_health,
            convergence,
            efficiency,
            timestamp_ms,
        })
    }

    /// Generate training performance analytics
    fn generate_performance_analytics(
        &self,
        current_metrics: &Option<NodeMetrics>,
        cluster_summary: &Option<ClusterSummary>,
    ) -> TorshResult<TrainingPerformanceAnalytics> {
        let (current_epoch, avg_loss, cluster_throughput, avg_batch_time_ms) =
            if let Some(metrics) = current_metrics {
                (
                    metrics.training_metrics.epoch,
                    metrics.training_metrics.loss,
                    metrics.training_metrics.throughput_samples_per_sec
                        * cluster_summary.as_ref().map(|s| s.total_nodes).unwrap_or(1) as f32,
                    metrics.training_metrics.batch_time_ms,
                )
            } else {
                (0, 0.0, 0.0, 0)
            };

        // Calculate trends using trend analyzer
        let (loss_trend, training_stability) = {
            let trend_analyzer = self.trend_analyzer.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "trend_analyzer",
                    format!("Lock error: {}", e),
                )
            })?;
            (
                trend_analyzer.calculate_loss_trend(),
                trend_analyzer.calculate_stability(),
            )
        };

        // Estimate throughput efficiency (simplified)
        let theoretical_max_throughput =
            cluster_summary.as_ref().map(|s| s.total_nodes).unwrap_or(1) as f32 * 100.0; // Assume 100 samples/sec per node max
        let throughput_efficiency = if theoretical_max_throughput > 0.0 {
            (cluster_throughput / theoretical_max_throughput).min(1.0)
        } else {
            0.0
        };

        // Calculate batch time variance (simplified estimate)
        let batch_time_variance = 0.1; // Would be calculated from actual node data

        // Estimate completion time (simplified)
        let estimated_completion_time = if cluster_throughput > 0.0 && current_epoch < 100 {
            let remaining_epochs = 100 - current_epoch;
            let samples_per_epoch = 10000; // Assume 10K samples per epoch
            let remaining_samples = remaining_epochs as f32 * samples_per_epoch as f32;
            let remaining_seconds = remaining_samples / cluster_throughput;
            Some(Duration::from_secs(remaining_seconds as u64))
        } else {
            None
        };

        Ok(TrainingPerformanceAnalytics {
            current_epoch,
            avg_loss,
            loss_trend,
            cluster_throughput,
            throughput_efficiency,
            avg_batch_time_ms,
            batch_time_variance,
            training_stability,
            estimated_completion_time,
        })
    }

    /// Generate resource utilization analytics
    fn generate_resource_analytics(
        &self,
        cluster_summary: &Option<ClusterSummary>,
        memory_status: &MemoryOptimizationStatus,
    ) -> TorshResult<ResourceUtilizationAnalytics> {
        let (avg_cpu_utilization, avg_gpu_utilization) = if let Some(summary) = cluster_summary {
            (summary.avg_cpu_utilization, summary.avg_gpu_utilization)
        } else {
            (0.0, 0.0)
        };

        let avg_memory_utilization = memory_status.avg_memory_utilization;

        // Calculate utilization balance (how evenly resources are used)
        let utilizations = [
            avg_cpu_utilization,
            avg_gpu_utilization,
            avg_memory_utilization,
        ];
        let max_util = utilizations.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_util = utilizations.iter().fold(100.0f32, |a, &b| a.min(b));
        let utilization_balance = if max_util > 0.0 {
            1.0 - (max_util - min_util) / max_util
        } else {
            1.0
        };

        // Simulate peak usage (would be tracked from history)
        let peak_cpu = avg_cpu_utilization * 1.2;
        let peak_gpu = avg_gpu_utilization * 1.15;
        let peak_memory = avg_memory_utilization * 1.1;

        // Calculate resource efficiency
        let resource_efficiency =
            (avg_cpu_utilization + avg_gpu_utilization + avg_memory_utilization) / 300.0;

        // Identify primary bottleneck
        let primary_bottleneck = if avg_gpu_utilization < 60.0 {
            ResourceBottleneck::GPU
        } else if avg_memory_utilization > 90.0 {
            ResourceBottleneck::Memory
        } else if avg_cpu_utilization > 95.0 {
            ResourceBottleneck::CPU
        } else {
            ResourceBottleneck::None
        };

        Ok(ResourceUtilizationAnalytics {
            avg_cpu_utilization,
            avg_gpu_utilization,
            avg_memory_utilization,
            utilization_balance,
            peak_cpu,
            peak_gpu,
            peak_memory,
            resource_efficiency,
            primary_bottleneck,
        })
    }

    /// Generate communication analytics
    fn generate_communication_analytics(
        &self,
        current_metrics: &Option<NodeMetrics>,
    ) -> TorshResult<CommunicationAnalytics> {
        let (avg_latency_us, bandwidth_utilization, efficiency_score, failed_operations_rate) =
            if let Some(metrics) = current_metrics {
                let comm = &metrics.communication_metrics;
                (
                    comm.avg_latency_us,
                    comm.comm_bandwidth_mbps / 1000.0, // Assume 1Gbps max
                    comm.efficiency_score,
                    comm.failed_ops_count as f32 / 100.0, // Normalize
                )
            } else {
                (0, 0.0, 0.0, 0.0)
            };

        // Simulate communication patterns
        let communication_patterns = CommunicationPatterns {
            allreduce_frequency: 10.0,
            allgather_frequency: 5.0,
            p2p_frequency: 2.0,
            gradient_sync_frequency: 8.0,
            hotspots: vec![CommunicationHotspot {
                source_node: "node_0".to_string(),
                target_node: "node_1".to_string(),
                traffic_volume: 50.0,
                congestion_score: 0.3,
            }],
        };

        let congestion_level = if avg_latency_us > 5000 { 0.7 } else { 0.2 };

        Ok(CommunicationAnalytics {
            avg_latency_us,
            bandwidth_utilization,
            efficiency_score,
            failed_operations_rate,
            communication_patterns,
            congestion_level,
        })
    }

    /// Generate system health analytics
    fn generate_system_health_analytics(
        &self,
        fault_tolerance_status: &FaultToleranceStatus,
    ) -> TorshResult<SystemHealthAnalytics> {
        let cluster_health_score = fault_tolerance_status.system_health_score;
        let healthy_nodes = fault_tolerance_status.healthy_nodes;
        let degraded_nodes = fault_tolerance_status.excluded_nodes; // Simplified mapping
        let critical_nodes = 0; // Would be calculated from actual health data
        let failed_nodes = fault_tolerance_status
            .total_nodes
            .saturating_sub(fault_tolerance_status.healthy_nodes);
        let active_incidents = fault_tolerance_status.active_incidents;

        // Calculate stability trend (simplified)
        let stability_trend = if cluster_health_score > 0.8 {
            0.1
        } else {
            -0.1
        };

        // Calculate failure probability based on health score
        let failure_probability = (1.0 - cluster_health_score).max(0.0);

        Ok(SystemHealthAnalytics {
            cluster_health_score,
            healthy_nodes,
            degraded_nodes,
            critical_nodes,
            failed_nodes,
            active_incidents,
            stability_trend,
            failure_probability,
        })
    }

    /// Generate convergence analytics
    fn generate_convergence_analytics(
        &self,
        current_metrics: &Option<NodeMetrics>,
    ) -> TorshResult<ConvergenceAnalytics> {
        let (loss, gradient_norm) = if let Some(metrics) = current_metrics {
            (
                metrics.training_metrics.loss,
                metrics.training_metrics.gradient_norm,
            )
        } else {
            (0.0, 0.0)
        };

        // Calculate convergence rate from trend analyzer
        let convergence_rate = {
            let trend_analyzer = self.trend_analyzer.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "trend_analyzer",
                    format!("Lock error: {}", e),
                )
            })?;
            -trend_analyzer.calculate_loss_trend() // Negative loss trend = positive convergence rate
        };

        // Calculate convergence confidence (simplified)
        let convergence_confidence = if convergence_rate > 0.0 { 0.8 } else { 0.3 };

        // Estimate training progress (simplified)
        let training_progress = if loss > 0.0 {
            (1.0 / (loss + 1.0)).min(0.95)
        } else {
            0.0
        };

        // Calculate loss smoothness (simplified)
        let loss_smoothness = 0.7; // Would be calculated from loss variance

        // Gradient norm statistics
        let gradient_norm_stats = GradientNormStats {
            avg_norm: gradient_norm,
            norm_variance: gradient_norm * 0.1, // Simplified
            norm_trend: 0.0,                    // Would be calculated from history
            explosion_risk: if gradient_norm > 10.0 { 0.8 } else { 0.1 },
        };

        // Learning rate effectiveness (simplified)
        let lr_effectiveness = if convergence_rate > 0.0 { 0.7 } else { 0.3 };

        // Overfitting risk (simplified)
        let overfitting_risk = if training_progress > 0.8 { 0.6 } else { 0.2 };

        Ok(ConvergenceAnalytics {
            convergence_rate,
            convergence_confidence,
            training_progress,
            loss_smoothness,
            gradient_norm_stats,
            lr_effectiveness,
            overfitting_risk,
        })
    }

    /// Generate efficiency analytics
    fn generate_efficiency_analytics(
        &self,
        performance: &TrainingPerformanceAnalytics,
        resource_utilization: &ResourceUtilizationAnalytics,
        communication: &CommunicationAnalytics,
    ) -> TorshResult<EfficiencyAnalytics> {
        // Calculate component efficiencies
        let compute_efficiency = resource_utilization.resource_efficiency;
        let communication_efficiency = communication.efficiency_score;
        let memory_efficiency =
            1.0 - (resource_utilization.avg_memory_utilization / 100.0 - 0.8).max(0.0) * 5.0; // Penalty for high memory usage

        // Calculate overall efficiency
        let overall_efficiency =
            (compute_efficiency + communication_efficiency + memory_efficiency) / 3.0;

        // Estimate energy efficiency (simplified)
        let energy_efficiency = overall_efficiency * 0.8; // Assume some energy overhead

        // Estimate cost efficiency (simplified)
        let cost_efficiency = overall_efficiency * performance.throughput_efficiency;

        // Generate recommendations
        let recommendations = if self.config.enable_recommendations {
            let mut recommendation_engine = self.recommendation_engine.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "recommendation_engine",
                    format!("Lock error: {}", e),
                )
            })?;
            recommendation_engine.generate_recommendations()
        } else {
            Vec::new()
        };

        Ok(EfficiencyAnalytics {
            overall_efficiency,
            compute_efficiency,
            communication_efficiency,
            memory_efficiency,
            energy_efficiency,
            cost_efficiency,
            recommendations,
        })
    }

    /// Get current analytics
    pub fn get_current_analytics(&self) -> TorshResult<Option<TrainingAnalytics>> {
        let current_analytics = self.current_analytics.read().map_err(|e| {
            TorshDistributedError::communication_error(
                "get_current_analytics",
                format!("Lock error: {}", e),
            )
        })?;
        Ok(current_analytics.clone())
    }

    /// Get analytics history
    pub fn get_analytics_history(&self) -> TorshResult<Vec<TrainingAnalytics>> {
        let analytics_history = self.analytics_history.lock().map_err(|e| {
            TorshDistributedError::communication_error(
                "get_analytics_history",
                format!("Lock error: {}", e),
            )
        })?;
        Ok(analytics_history.iter().cloned().collect())
    }

    /// Export dashboard data for external visualization
    pub fn export_dashboard_data(&self) -> TorshResult<DashboardExport> {
        let current_analytics = self.get_current_analytics()?;
        let analytics_history = self.get_analytics_history()?;

        Ok(DashboardExport {
            current_analytics,
            analytics_history,
            config: self.config.clone(),
            export_timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }

    /// Generate training summary report
    pub fn generate_training_summary(&self) -> TorshResult<TrainingSummaryReport> {
        let current_analytics = self.get_current_analytics()?.ok_or_else(|| {
            TorshDistributedError::communication_error(
                "summary",
                "No analytics data available".to_string(),
            )
        })?;

        let analytics_history = self.get_analytics_history()?;

        // Calculate summary statistics
        let total_runtime = if !analytics_history.is_empty() {
            let start_time = analytics_history.first().unwrap().timestamp_ms;
            let end_time = current_analytics.timestamp_ms;
            Duration::from_millis(end_time - start_time)
        } else {
            Duration::from_secs(0)
        };

        let avg_efficiency = if !analytics_history.is_empty() {
            analytics_history
                .iter()
                .map(|a| a.efficiency.overall_efficiency)
                .sum::<f32>()
                / analytics_history.len() as f32
        } else {
            0.0
        };

        let peak_throughput = analytics_history
            .iter()
            .map(|a| a.performance.cluster_throughput)
            .fold(0.0f32, |a, b| a.max(b));

        Ok(TrainingSummaryReport {
            current_epoch: current_analytics.performance.current_epoch,
            current_loss: current_analytics.performance.avg_loss,
            total_runtime,
            avg_efficiency,
            peak_throughput,
            total_incidents: analytics_history
                .iter()
                .map(|a| a.system_health.active_incidents)
                .sum(),
            convergence_rate: current_analytics.convergence.convergence_rate,
            resource_utilization_summary: ResourceUtilizationSummary {
                avg_cpu: current_analytics.resource_utilization.avg_cpu_utilization,
                avg_gpu: current_analytics.resource_utilization.avg_gpu_utilization,
                avg_memory: current_analytics
                    .resource_utilization
                    .avg_memory_utilization,
                peak_cpu: current_analytics.resource_utilization.peak_cpu,
                peak_gpu: current_analytics.resource_utilization.peak_gpu,
                peak_memory: current_analytics.resource_utilization.peak_memory,
            },
            optimization_recommendations: current_analytics.efficiency.recommendations,
            generated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }
}

/// Dashboard data export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardExport {
    pub current_analytics: Option<TrainingAnalytics>,
    pub analytics_history: Vec<TrainingAnalytics>,
    pub config: DashboardConfig,
    pub export_timestamp_ms: u64,
}

/// Training summary report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummaryReport {
    pub current_epoch: u32,
    pub current_loss: f32,
    pub total_runtime: Duration,
    pub avg_efficiency: f32,
    pub peak_throughput: f32,
    pub total_incidents: usize,
    pub convergence_rate: f32,
    pub resource_utilization_summary: ResourceUtilizationSummary,
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    pub generated_at: u64,
}

/// Resource utilization summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationSummary {
    pub avg_cpu: f32,
    pub avg_gpu: f32,
    pub avg_memory: f32,
    pub peak_cpu: f32,
    pub peak_gpu: f32,
    pub peak_memory: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed_memory_optimization::{
        DistributedMemoryOptimizer, MemoryOptimizationConfig,
    };
    use crate::distributed_monitoring::{DistributedMonitor, MonitoringConfig};
    use crate::enhanced_fault_tolerance::{EnhancedFaultTolerance, FaultToleranceConfig};

    #[tokio::test]
    async fn test_dashboard_creation() -> TorshResult<()> {
        let monitor_config = MonitoringConfig::default();
        let monitor = Arc::new(DistributedMonitor::new(monitor_config, false));

        let ft_config = FaultToleranceConfig::default();
        let fault_tolerance = Arc::new(EnhancedFaultTolerance::new(ft_config, monitor.clone()));

        let mem_config = MemoryOptimizationConfig::default();
        let memory_optimizer =
            Arc::new(DistributedMemoryOptimizer::new(mem_config, monitor.clone()));

        let dashboard_config = DashboardConfig::default();
        let dashboard = TrainingAnalyticsDashboard::new(
            dashboard_config,
            monitor,
            fault_tolerance,
            memory_optimizer,
        );

        let current_analytics = dashboard.get_current_analytics()?;
        assert!(current_analytics.is_none()); // No data initially

        Ok(())
    }

    #[tokio::test]
    async fn test_trend_analyzer() -> TorshResult<()> {
        let mut analyzer = TrendAnalyzer::new(Duration::from_secs(60));

        // Add some data points
        for i in 0..20 {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64
                + i * 1000;
            analyzer.update_loss(timestamp, 2.0 - i as f32 * 0.1); // Decreasing loss
        }

        let loss_trend = analyzer.calculate_loss_trend();
        assert!(loss_trend < 0.0); // Should detect decreasing trend

        let stability = analyzer.calculate_stability();
        assert!(stability > 0.5); // Should be relatively stable

        Ok(())
    }

    #[tokio::test]
    async fn test_recommendation_engine() -> TorshResult<()> {
        let mut engine = RecommendationEngine::new();

        // Add performance data indicating low GPU utilization
        for _ in 0..15 {
            engine.update_performance(
                100.0, // throughput
                0.6,   // efficiency
                80.0,  // CPU util
                50.0,  // GPU util (low)
                70.0,  // memory util
                2000,  // latency
            );
        }

        let recommendations = engine.generate_recommendations();
        assert!(!recommendations.is_empty());

        // Should recommend increasing GPU utilization
        let gpu_rec = recommendations.iter().find(|r| r.title.contains("GPU"));
        assert!(gpu_rec.is_some());

        Ok(())
    }
}

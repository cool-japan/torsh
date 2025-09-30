//! Enhanced Fault Tolerance for Distributed Training
//!
//! This module provides advanced fault detection, recovery, and resilience mechanisms
//! for production distributed training environments. It includes automatic failure
//! detection, smart recovery strategies, and predictive failure prevention.

use crate::distributed_monitoring::{Alert, AlertSeverity, DistributedMonitor, NodeHealthStatus};
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};

/// Types of failures that can occur in distributed training
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FailureType {
    /// Node completely unresponsive
    NodeUnresponsive { node_id: String, last_seen: u64 },
    /// Communication timeout or failure
    CommunicationFailure {
        source: String,
        target: String,
        error: String,
    },
    /// Out of memory error
    OutOfMemory {
        node_id: String,
        available_mb: u64,
        requested_mb: u64,
    },
    /// GPU failure or unavailability
    GpuFailure {
        node_id: String,
        gpu_id: u32,
        error: String,
    },
    /// Training divergence (loss explosion/NaN)
    TrainingDivergence { loss_value: f32, gradient_norm: f32 },
    /// Checkpoint corruption or unavailability
    CheckpointFailure {
        checkpoint_path: String,
        error: String,
    },
    /// Network partition
    NetworkPartition { affected_nodes: Vec<String> },
    /// Storage failure
    StorageFailure { path: String, error: String },
    /// Resource exhaustion
    ResourceExhaustion {
        resource_type: String,
        node_id: String,
    },
    /// Custom application-specific failure
    Custom {
        failure_name: String,
        details: String,
    },
}

impl std::fmt::Display for FailureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FailureType::NodeUnresponsive { node_id, .. } => {
                write!(f, "Node {} is unresponsive", node_id)
            }
            FailureType::CommunicationFailure { source, target, .. } => {
                write!(f, "Communication failure from {} to {}", source, target)
            }
            FailureType::OutOfMemory { node_id, .. } => {
                write!(f, "Out of memory on node {}", node_id)
            }
            FailureType::GpuFailure {
                node_id, gpu_id, ..
            } => write!(f, "GPU {} failure on node {}", gpu_id, node_id),
            FailureType::TrainingDivergence { loss_value, .. } => {
                write!(f, "Training divergence detected (loss: {})", loss_value)
            }
            FailureType::CheckpointFailure {
                checkpoint_path, ..
            } => write!(f, "Checkpoint failure: {}", checkpoint_path),
            FailureType::NetworkPartition { affected_nodes } => write!(
                f,
                "Network partition affecting {} nodes",
                affected_nodes.len()
            ),
            FailureType::StorageFailure { path, .. } => write!(f, "Storage failure: {}", path),
            FailureType::ResourceExhaustion {
                resource_type,
                node_id,
            } => write!(f, "{} exhaustion on node {}", resource_type, node_id),
            FailureType::Custom { failure_name, .. } => {
                write!(f, "Custom failure: {}", failure_name)
            }
        }
    }
}

/// Recovery strategy for different types of failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Restart the failed node
    RestartNode { node_id: String, max_attempts: u32 },
    /// Exclude the failed node and continue with remaining nodes
    ExcludeNode { node_id: String },
    /// Load from latest checkpoint and continue
    LoadCheckpoint {
        checkpoint_path: String,
        rollback_steps: u32,
    },
    /// Reduce batch size to handle memory pressure
    ReduceBatchSize {
        new_batch_size: u32,
        reduction_factor: f32,
    },
    /// Redistribute work to healthy nodes
    RedistributeWork {
        failed_nodes: Vec<String>,
        target_nodes: Vec<String>,
    },
    /// Scale down training to fewer GPUs
    ScaleDown {
        new_world_size: u32,
        keep_nodes: Vec<String>,
    },
    /// Reset training state (lr, optimizer state)
    ResetTrainingState {
        reset_optimizer: bool,
        reset_lr_schedule: bool,
    },
    /// Switch to degraded mode with reduced functionality
    DegradedMode { disabled_features: Vec<String> },
    /// Emergency stop and save current state
    EmergencyStop { save_checkpoint: bool },
    /// Custom recovery action
    Custom {
        action_name: String,
        parameters: HashMap<String, String>,
    },
}

impl std::fmt::Display for RecoveryStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecoveryStrategy::RestartNode { node_id, .. } => write!(f, "Restart node {}", node_id),
            RecoveryStrategy::ExcludeNode { node_id } => write!(f, "Exclude node {}", node_id),
            RecoveryStrategy::LoadCheckpoint {
                checkpoint_path, ..
            } => write!(f, "Load checkpoint: {}", checkpoint_path),
            RecoveryStrategy::ReduceBatchSize { new_batch_size, .. } => {
                write!(f, "Reduce batch size to {}", new_batch_size)
            }
            RecoveryStrategy::RedistributeWork { failed_nodes, .. } => write!(
                f,
                "Redistribute work from {} failed nodes",
                failed_nodes.len()
            ),
            RecoveryStrategy::ScaleDown { new_world_size, .. } => {
                write!(f, "Scale down to {} nodes", new_world_size)
            }
            RecoveryStrategy::ResetTrainingState { .. } => write!(f, "Reset training state"),
            RecoveryStrategy::DegradedMode { disabled_features } => write!(
                f,
                "Enter degraded mode (disable {} features)",
                disabled_features.len()
            ),
            RecoveryStrategy::EmergencyStop { .. } => write!(f, "Emergency stop"),
            RecoveryStrategy::Custom { action_name, .. } => {
                write!(f, "Custom action: {}", action_name)
            }
        }
    }
}

/// Status of a recovery attempt
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecoveryStatus {
    /// Recovery not yet started
    Pending,
    /// Recovery is currently in progress
    InProgress { progress: f32, stage: String },
    /// Recovery completed successfully
    Completed { duration_ms: u64 },
    /// Recovery failed
    Failed { error: String, retry_count: u32 },
    /// Recovery was cancelled
    Cancelled { reason: String },
}

impl std::fmt::Display for RecoveryStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecoveryStatus::Pending => write!(f, "Pending"),
            RecoveryStatus::InProgress { progress, stage } => {
                write!(f, "In Progress ({:.1}%): {}", progress * 100.0, stage)
            }
            RecoveryStatus::Completed { duration_ms } => {
                write!(f, "Completed in {}ms", duration_ms)
            }
            RecoveryStatus::Failed { error, retry_count } => {
                write!(f, "Failed (attempt {}): {}", retry_count, error)
            }
            RecoveryStatus::Cancelled { reason } => write!(f, "Cancelled: {}", reason),
        }
    }
}

/// A recorded failure incident
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureIncident {
    /// Unique incident identifier
    pub id: String,
    /// Type of failure
    pub failure_type: FailureType,
    /// When the failure was detected
    pub detected_at: u64,
    /// Severity of the failure
    pub severity: AlertSeverity,
    /// Chosen recovery strategy
    pub recovery_strategy: RecoveryStrategy,
    /// Current recovery status
    pub recovery_status: RecoveryStatus,
    /// Time when recovery started
    pub recovery_started_at: Option<u64>,
    /// Time when recovery completed
    pub recovery_completed_at: Option<u64>,
    /// Affected nodes
    pub affected_nodes: Vec<String>,
    /// Additional context and logs
    pub context: HashMap<String, String>,
}

/// Configuration for fault tolerance system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Node heartbeat timeout
    pub node_timeout: Duration,
    /// Communication timeout
    pub communication_timeout: Duration,
    /// Maximum recovery attempts per incident
    pub max_recovery_attempts: u32,
    /// Whether to enable automatic recovery
    pub enable_automatic_recovery: bool,
    /// Whether to enable predictive failure detection
    pub enable_predictive_detection: bool,
    /// Checkpoint interval for fault tolerance
    pub checkpoint_interval: Duration,
    /// Maximum incidents to retain in history
    pub max_incident_history: usize,
    /// Failure detection sensitivity (0.0 to 1.0)
    pub detection_sensitivity: f32,
    /// Recovery timeout
    pub recovery_timeout: Duration,
    /// Whether to enable proactive node health checks
    pub enable_health_checks: bool,
    /// Health check interval
    pub health_check_interval: Duration,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            node_timeout: Duration::from_secs(30),
            communication_timeout: Duration::from_secs(10),
            max_recovery_attempts: 3,
            enable_automatic_recovery: true,
            enable_predictive_detection: true,
            checkpoint_interval: Duration::from_secs(300), // 5 minutes
            max_incident_history: 1000,
            detection_sensitivity: 0.8,
            recovery_timeout: Duration::from_secs(300), // 5 minutes
            enable_health_checks: true,
            health_check_interval: Duration::from_secs(10),
        }
    }
}

/// Enhanced fault tolerance system
pub struct EnhancedFaultTolerance {
    /// Configuration
    config: FaultToleranceConfig,
    /// Distributed monitoring system for health checks
    monitor: Arc<DistributedMonitor>,
    /// Active failure incidents
    active_incidents: Arc<RwLock<HashMap<String, FailureIncident>>>,
    /// Incident history
    incident_history: Arc<Mutex<VecDeque<FailureIncident>>>,
    /// Node status tracking
    node_status: Arc<RwLock<HashMap<String, NodeStatus>>>,
    /// Predictive failure model
    failure_predictor: Arc<Mutex<FailurePredictor>>,
    /// Recovery executor
    recovery_executor: Arc<Mutex<RecoveryExecutor>>,
    /// Last health check time
    last_health_check: Arc<Mutex<Instant>>,
}

/// Node status information
#[derive(Debug, Clone)]
struct NodeStatus {
    /// Last heartbeat timestamp
    last_heartbeat: Instant,
    /// Current health status
    health_status: NodeHealthStatus,
    /// Number of consecutive failures
    consecutive_failures: u32,
    /// Whether node is currently excluded
    is_excluded: bool,
    /// Last known metrics
    last_metrics: Option<crate::distributed_monitoring::NodeMetrics>,
}

/// Predictive failure detection system
#[derive(Debug)]
struct FailurePredictor {
    /// Historical failure patterns
    failure_patterns: HashMap<String, Vec<f32>>,
    /// Risk scores for different nodes
    node_risk_scores: HashMap<String, f32>,
    /// Failure prediction models
    prediction_models: HashMap<String, PredictionModel>,
}

/// Simple prediction model using trend analysis
#[derive(Debug)]
struct PredictionModel {
    /// Historical values for trend analysis
    historical_values: VecDeque<f32>,
    /// Trend slope
    trend_slope: f32,
    /// Variance estimate
    variance: f32,
    /// Last update time
    last_update: Instant,
}

impl PredictionModel {
    fn new() -> Self {
        Self {
            historical_values: VecDeque::with_capacity(100),
            trend_slope: 0.0,
            variance: 0.0,
            last_update: Instant::now(),
        }
    }

    fn update(&mut self, value: f32) {
        self.historical_values.push_back(value);
        if self.historical_values.len() > 100 {
            self.historical_values.pop_front();
        }

        if self.historical_values.len() >= 10 {
            self.calculate_trend();
            self.calculate_variance();
        }

        self.last_update = Instant::now();
    }

    fn calculate_trend(&mut self) {
        let values: Vec<f32> = self.historical_values.iter().cloned().collect();
        if values.len() < 2 {
            return;
        }

        // Simple linear regression for trend
        let n = values.len() as f32;
        let sum_x: f32 = (0..values.len()).map(|i| i as f32).sum();
        let sum_y: f32 = values.iter().sum();
        let sum_xy: f32 = values.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let sum_x2: f32 = (0..values.len()).map(|i| (i as f32).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() > 0.001 {
            self.trend_slope = (n * sum_xy - sum_x * sum_y) / denominator;
        }
    }

    fn calculate_variance(&mut self) {
        if self.historical_values.len() < 2 {
            return;
        }

        let mean: f32 =
            self.historical_values.iter().sum::<f32>() / self.historical_values.len() as f32;
        self.variance = self
            .historical_values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / self.historical_values.len() as f32;
    }

    fn predict_failure_risk(&self) -> f32 {
        if self.historical_values.len() < 10 {
            return 0.0; // Not enough data
        }

        // Risk based on upward trend in error metrics or downward trend in performance
        let trend_risk = if self.trend_slope > 0.1 {
            self.trend_slope * 2.0
        } else {
            0.0
        };

        // Risk based on high variance (instability)
        let variance_risk = if self.variance > 1.0 {
            (self.variance - 1.0) * 0.5
        } else {
            0.0
        };

        // Risk based on staleness of data
        let staleness_risk = if self.last_update.elapsed().as_secs() > 60 {
            0.3
        } else {
            0.0
        };

        (trend_risk + variance_risk + staleness_risk).min(1.0)
    }
}

impl FailurePredictor {
    fn new() -> Self {
        Self {
            failure_patterns: HashMap::new(),
            node_risk_scores: HashMap::new(),
            prediction_models: HashMap::new(),
        }
    }

    fn update_node_metrics(
        &mut self,
        node_id: &str,
        metrics: &crate::distributed_monitoring::NodeMetrics,
    ) {
        // Update prediction models for key metrics
        {
            let cpu_model = self
                .prediction_models
                .entry(format!("{}_cpu", node_id))
                .or_insert_with(PredictionModel::new);
            cpu_model.update(metrics.system_metrics.cpu_utilization);
        }

        {
            let memory_model = self
                .prediction_models
                .entry(format!("{}_memory", node_id))
                .or_insert_with(PredictionModel::new);
            memory_model.update(metrics.system_metrics.memory_usage_mb as f32);
        }

        {
            let latency_model = self
                .prediction_models
                .entry(format!("{}_latency", node_id))
                .or_insert_with(PredictionModel::new);
            latency_model.update(metrics.communication_metrics.avg_latency_us as f32);
        }

        // Calculate overall risk score
        let cpu_risk = self
            .prediction_models
            .get(&format!("{}_cpu", node_id))
            .map_or(0.0, |model| model.predict_failure_risk());
        let memory_risk = self
            .prediction_models
            .get(&format!("{}_memory", node_id))
            .map_or(0.0, |model| model.predict_failure_risk());
        let latency_risk = self
            .prediction_models
            .get(&format!("{}_latency", node_id))
            .map_or(0.0, |model| model.predict_failure_risk());

        let overall_risk = (cpu_risk + memory_risk + latency_risk) / 3.0;
        self.node_risk_scores
            .insert(node_id.to_string(), overall_risk);
    }

    fn get_node_risk_score(&self, node_id: &str) -> f32 {
        self.node_risk_scores.get(node_id).copied().unwrap_or(0.0)
    }

    fn get_high_risk_nodes(&self, threshold: f32) -> Vec<String> {
        self.node_risk_scores
            .iter()
            .filter(|(_, &risk)| risk > threshold)
            .map(|(node_id, _)| node_id.clone())
            .collect()
    }
}

/// Recovery execution system
#[derive(Debug)]
struct RecoveryExecutor {
    /// Currently executing recoveries
    active_recoveries: HashMap<String, RecoveryExecution>,
}

/// A recovery execution instance
#[derive(Debug)]
struct RecoveryExecution {
    /// Incident being recovered
    incident_id: String,
    /// Recovery strategy being executed
    strategy: RecoveryStrategy,
    /// Start time
    start_time: Instant,
    /// Current progress (0.0 to 1.0)
    progress: f32,
    /// Current stage description
    current_stage: String,
    /// Retry count
    retry_count: u32,
}

impl RecoveryExecutor {
    fn new() -> Self {
        Self {
            active_recoveries: HashMap::new(),
        }
    }

    fn start_recovery(
        &mut self,
        incident_id: String,
        strategy: RecoveryStrategy,
    ) -> TorshResult<()> {
        let execution = RecoveryExecution {
            incident_id: incident_id.clone(),
            strategy: strategy.clone(),
            start_time: Instant::now(),
            progress: 0.0,
            current_stage: "Initializing recovery".to_string(),
            retry_count: 0,
        };

        self.active_recoveries
            .insert(incident_id.clone(), execution);
        info!(
            "Started recovery for incident {}: {}",
            incident_id, strategy
        );

        Ok(())
    }

    fn update_recovery_progress(
        &mut self,
        incident_id: &str,
        progress: f32,
        stage: String,
    ) -> TorshResult<()> {
        if let Some(execution) = self.active_recoveries.get_mut(incident_id) {
            execution.progress = progress.min(1.0).max(0.0);
            execution.current_stage = stage;
            debug!(
                "Recovery {} progress: {:.1}% - {}",
                incident_id,
                progress * 100.0,
                execution.current_stage
            );
        }
        Ok(())
    }

    fn complete_recovery(
        &mut self,
        incident_id: &str,
        success: bool,
        error: Option<String>,
    ) -> TorshResult<RecoveryStatus> {
        if let Some(execution) = self.active_recoveries.remove(incident_id) {
            let duration_ms = execution.start_time.elapsed().as_millis() as u64;

            if success {
                info!(
                    "Recovery {} completed successfully in {}ms",
                    incident_id, duration_ms
                );
                Ok(RecoveryStatus::Completed { duration_ms })
            } else {
                let error_msg = error.unwrap_or_else(|| "Unknown error".to_string());
                warn!("Recovery {} failed: {}", incident_id, error_msg);
                Ok(RecoveryStatus::Failed {
                    error: error_msg,
                    retry_count: execution.retry_count,
                })
            }
        } else {
            Err(TorshDistributedError::communication_error(
                "recovery_complete",
                format!("Recovery {} not found", incident_id),
            ))
        }
    }

    fn get_recovery_status(&self, incident_id: &str) -> Option<RecoveryStatus> {
        self.active_recoveries
            .get(incident_id)
            .map(|execution| RecoveryStatus::InProgress {
                progress: execution.progress,
                stage: execution.current_stage.clone(),
            })
    }
}

impl EnhancedFaultTolerance {
    /// Create new enhanced fault tolerance system
    pub fn new(config: FaultToleranceConfig, monitor: Arc<DistributedMonitor>) -> Self {
        Self {
            config: config.clone(),
            monitor,
            active_incidents: Arc::new(RwLock::new(HashMap::new())),
            incident_history: Arc::new(Mutex::new(VecDeque::with_capacity(
                config.max_incident_history,
            ))),
            node_status: Arc::new(RwLock::new(HashMap::new())),
            failure_predictor: Arc::new(Mutex::new(FailurePredictor::new())),
            recovery_executor: Arc::new(Mutex::new(RecoveryExecutor::new())),
            last_health_check: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Detect failures based on monitoring data and alerts
    pub fn detect_failures(&self) -> TorshResult<Vec<FailureType>> {
        let mut detected_failures = Vec::new();

        // Get active alerts from monitoring system
        let alerts = self.monitor.get_active_alerts()?;

        // Get current node metrics
        if let Some(current_metrics) = self.monitor.get_current_metrics()? {
            // Check for node unresponsiveness
            self.check_node_responsiveness(&current_metrics, &mut detected_failures)?;

            // Check for training divergence
            self.check_training_divergence(&current_metrics, &mut detected_failures)?;

            // Check for resource exhaustion
            self.check_resource_exhaustion(&current_metrics, &mut detected_failures)?;
        }

        // Process monitoring alerts
        for alert in alerts {
            match alert.metric_name.as_str() {
                "cpu_utilization" | "gpu_utilization"
                    if alert.severity >= AlertSeverity::Critical =>
                {
                    detected_failures.push(FailureType::ResourceExhaustion {
                        resource_type: alert.metric_name.clone(),
                        node_id: alert.node_id.clone(),
                    });
                }
                "avg_latency_us" if alert.severity >= AlertSeverity::Critical => {
                    detected_failures.push(FailureType::CommunicationFailure {
                        source: alert.node_id.clone(),
                        target: "cluster".to_string(),
                        error: "High latency detected".to_string(),
                    });
                }
                _ => {}
            }
        }

        // Predictive failure detection
        if self.config.enable_predictive_detection {
            self.detect_predictive_failures(&mut detected_failures)?;
        }

        Ok(detected_failures)
    }

    /// Check for node responsiveness issues
    fn check_node_responsiveness(
        &self,
        metrics: &crate::distributed_monitoring::NodeMetrics,
        failures: &mut Vec<FailureType>,
    ) -> TorshResult<()> {
        let node_status = self.node_status.read().map_err(|e| {
            TorshDistributedError::communication_error("node_status", format!("Lock error: {}", e))
        })?;

        if let Some(status) = node_status.get(&metrics.node_id) {
            if status.last_heartbeat.elapsed() > self.config.node_timeout {
                failures.push(FailureType::NodeUnresponsive {
                    node_id: metrics.node_id.clone(),
                    last_seen: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                });
            }
        }

        Ok(())
    }

    /// Check for training divergence
    fn check_training_divergence(
        &self,
        metrics: &crate::distributed_monitoring::NodeMetrics,
        failures: &mut Vec<FailureType>,
    ) -> TorshResult<()> {
        let loss = metrics.training_metrics.loss;
        let gradient_norm = metrics.training_metrics.gradient_norm;

        // Check for NaN or infinite values
        if !loss.is_finite() || !gradient_norm.is_finite() {
            failures.push(FailureType::TrainingDivergence {
                loss_value: loss,
                gradient_norm,
            });
        }

        // Check for extremely high loss values (indicating divergence)
        if loss > 1000.0 || gradient_norm > 100.0 {
            failures.push(FailureType::TrainingDivergence {
                loss_value: loss,
                gradient_norm,
            });
        }

        Ok(())
    }

    /// Check for resource exhaustion
    fn check_resource_exhaustion(
        &self,
        metrics: &crate::distributed_monitoring::NodeMetrics,
        failures: &mut Vec<FailureType>,
    ) -> TorshResult<()> {
        // Check for memory exhaustion
        if metrics.system_metrics.memory_usage_mb > 32000 {
            // Assume 32GB threshold
            failures.push(FailureType::OutOfMemory {
                node_id: metrics.node_id.clone(),
                available_mb: 32000 - metrics.system_metrics.memory_usage_mb,
                requested_mb: metrics.system_metrics.memory_usage_mb,
            });
        }

        // Check for GPU memory exhaustion
        if metrics.system_metrics.gpu_memory_mb > 20000 {
            // Assume 20GB GPU threshold
            failures.push(FailureType::GpuFailure {
                node_id: metrics.node_id.clone(),
                gpu_id: 0, // Simplified to GPU 0
                error: "GPU memory exhaustion".to_string(),
            });
        }

        Ok(())
    }

    /// Detect predictive failures using ML models
    fn detect_predictive_failures(&self, failures: &mut Vec<FailureType>) -> TorshResult<()> {
        let predictor = self.failure_predictor.lock().map_err(|e| {
            TorshDistributedError::communication_error("predictor", format!("Lock error: {}", e))
        })?;

        let high_risk_nodes = predictor.get_high_risk_nodes(0.7); // 70% risk threshold

        for node_id in high_risk_nodes {
            let risk_score = predictor.get_node_risk_score(&node_id);
            warn!(
                "Predictive failure detection: node {} has high risk score {:.2}",
                node_id, risk_score
            );

            failures.push(FailureType::Custom {
                failure_name: "Predictive Failure Risk".to_string(),
                details: format!("Node {} has risk score {:.2}", node_id, risk_score),
            });
        }

        Ok(())
    }

    /// Handle a detected failure with appropriate recovery strategy
    pub fn handle_failure(&self, failure: FailureType) -> TorshResult<String> {
        let incident_id = format!(
            "incident_{}_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                % 100000
        );

        // Determine recovery strategy based on failure type
        let recovery_strategy = self.determine_recovery_strategy(&failure)?;

        // Create incident record
        let incident = FailureIncident {
            id: incident_id.clone(),
            failure_type: failure.clone(),
            detected_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            severity: self.determine_failure_severity(&failure),
            recovery_strategy: recovery_strategy.clone(),
            recovery_status: RecoveryStatus::Pending,
            recovery_started_at: None,
            recovery_completed_at: None,
            affected_nodes: self.get_affected_nodes(&failure),
            context: HashMap::new(),
        };

        // Store incident
        {
            let mut active_incidents = self.active_incidents.write().map_err(|e| {
                TorshDistributedError::communication_error(
                    "incidents",
                    format!("Lock error: {}", e),
                )
            })?;
            active_incidents.insert(incident_id.clone(), incident);
        }

        info!(
            "Handling failure: {} with strategy: {}",
            failure, recovery_strategy
        );

        // Execute recovery if automatic recovery is enabled
        if self.config.enable_automatic_recovery {
            self.execute_recovery(&incident_id, recovery_strategy)?;
        }

        Ok(incident_id)
    }

    /// Determine appropriate recovery strategy for a failure
    fn determine_recovery_strategy(&self, failure: &FailureType) -> TorshResult<RecoveryStrategy> {
        let strategy = match failure {
            FailureType::NodeUnresponsive { node_id, .. } => RecoveryStrategy::RestartNode {
                node_id: node_id.clone(),
                max_attempts: 3,
            },
            FailureType::CommunicationFailure { source, .. } => RecoveryStrategy::ExcludeNode {
                node_id: source.clone(),
            },
            FailureType::OutOfMemory { .. } => {
                RecoveryStrategy::ReduceBatchSize {
                    new_batch_size: 16, // Reduce to smaller batch
                    reduction_factor: 0.5,
                }
            }
            FailureType::GpuFailure { node_id, .. } => RecoveryStrategy::ExcludeNode {
                node_id: node_id.clone(),
            },
            FailureType::TrainingDivergence { .. } => RecoveryStrategy::LoadCheckpoint {
                checkpoint_path: "latest_checkpoint".to_string(),
                rollback_steps: 100,
            },
            FailureType::CheckpointFailure { .. } => RecoveryStrategy::EmergencyStop {
                save_checkpoint: true,
            },
            FailureType::NetworkPartition { affected_nodes } => {
                let healthy_nodes = self.get_healthy_nodes()?;
                RecoveryStrategy::RedistributeWork {
                    failed_nodes: affected_nodes.clone(),
                    target_nodes: healthy_nodes,
                }
            }
            FailureType::StorageFailure { .. } => RecoveryStrategy::DegradedMode {
                disabled_features: vec!["checkpointing".to_string(), "logging".to_string()],
            },
            FailureType::ResourceExhaustion {
                resource_type,
                node_id,
            } => match resource_type.as_str() {
                "cpu_utilization" | "gpu_utilization" => RecoveryStrategy::ReduceBatchSize {
                    new_batch_size: 8,
                    reduction_factor: 0.75,
                },
                _ => RecoveryStrategy::ExcludeNode {
                    node_id: node_id.clone(),
                },
            },
            FailureType::Custom { failure_name, .. } => {
                if failure_name.contains("Predictive") {
                    RecoveryStrategy::Custom {
                        action_name: "PreventiveAction".to_string(),
                        parameters: [("monitoring".to_string(), "increased".to_string())]
                            .iter()
                            .cloned()
                            .collect(),
                    }
                } else {
                    RecoveryStrategy::EmergencyStop {
                        save_checkpoint: true,
                    }
                }
            }
        };

        Ok(strategy)
    }

    /// Determine severity of a failure
    fn determine_failure_severity(&self, failure: &FailureType) -> AlertSeverity {
        match failure {
            FailureType::NodeUnresponsive { .. } => AlertSeverity::Critical,
            FailureType::CommunicationFailure { .. } => AlertSeverity::Warning,
            FailureType::OutOfMemory { .. } => AlertSeverity::Critical,
            FailureType::GpuFailure { .. } => AlertSeverity::Critical,
            FailureType::TrainingDivergence { .. } => AlertSeverity::Emergency,
            FailureType::CheckpointFailure { .. } => AlertSeverity::Warning,
            FailureType::NetworkPartition { .. } => AlertSeverity::Critical,
            FailureType::StorageFailure { .. } => AlertSeverity::Warning,
            FailureType::ResourceExhaustion { .. } => AlertSeverity::Warning,
            FailureType::Custom { .. } => AlertSeverity::Info,
        }
    }

    /// Get nodes affected by a failure
    fn get_affected_nodes(&self, failure: &FailureType) -> Vec<String> {
        match failure {
            FailureType::NodeUnresponsive { node_id, .. } => vec![node_id.clone()],
            FailureType::CommunicationFailure { source, target, .. } => {
                if target == "cluster" {
                    vec![source.clone()]
                } else {
                    vec![source.clone(), target.clone()]
                }
            }
            FailureType::OutOfMemory { node_id, .. } => vec![node_id.clone()],
            FailureType::GpuFailure { node_id, .. } => vec![node_id.clone()],
            FailureType::TrainingDivergence { .. } => vec![], // Affects all nodes
            FailureType::CheckpointFailure { .. } => vec![],  // Affects all nodes
            FailureType::NetworkPartition { affected_nodes } => affected_nodes.clone(),
            FailureType::StorageFailure { .. } => vec![], // Affects all nodes
            FailureType::ResourceExhaustion { node_id, .. } => vec![node_id.clone()],
            FailureType::Custom { .. } => vec![],
        }
    }

    /// Get list of healthy nodes
    fn get_healthy_nodes(&self) -> TorshResult<Vec<String>> {
        let node_status = self.node_status.read().map_err(|e| {
            TorshDistributedError::communication_error(
                "healthy_nodes",
                format!("Lock error: {}", e),
            )
        })?;

        let healthy_nodes = node_status
            .iter()
            .filter(|(_, status)| {
                matches!(status.health_status, NodeHealthStatus::Healthy) && !status.is_excluded
            })
            .map(|(node_id, _)| node_id.clone())
            .collect();

        Ok(healthy_nodes)
    }

    /// Execute recovery strategy
    pub fn execute_recovery(
        &self,
        incident_id: &str,
        strategy: RecoveryStrategy,
    ) -> TorshResult<()> {
        let mut executor = self.recovery_executor.lock().map_err(|e| {
            TorshDistributedError::communication_error("executor", format!("Lock error: {}", e))
        })?;

        executor.start_recovery(incident_id.to_string(), strategy.clone())?;

        // Update incident status
        {
            let mut active_incidents = self.active_incidents.write().map_err(|e| {
                TorshDistributedError::communication_error(
                    "incidents",
                    format!("Lock error: {}", e),
                )
            })?;

            if let Some(incident) = active_incidents.get_mut(incident_id) {
                incident.recovery_status = RecoveryStatus::InProgress {
                    progress: 0.0,
                    stage: "Starting recovery".to_string(),
                };
                incident.recovery_started_at = Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                );
            }
        }

        // Simulate recovery execution (in production, this would interface with actual systems)
        self.simulate_recovery_execution(&mut executor, incident_id, strategy)?;

        Ok(())
    }

    /// Simulate recovery execution (placeholder for actual implementation)
    fn simulate_recovery_execution(
        &self,
        executor: &mut RecoveryExecutor,
        incident_id: &str,
        strategy: RecoveryStrategy,
    ) -> TorshResult<()> {
        info!("Executing recovery strategy: {}", strategy);

        // Simulate different recovery stages
        let stages = match strategy {
            RecoveryStrategy::RestartNode { .. } => {
                vec![
                    (0.2, "Stopping node services".to_string()),
                    (0.5, "Restarting node".to_string()),
                    (0.8, "Reinitializing training".to_string()),
                    (1.0, "Recovery complete".to_string()),
                ]
            }
            RecoveryStrategy::LoadCheckpoint { .. } => {
                vec![
                    (0.3, "Loading checkpoint".to_string()),
                    (0.6, "Restoring model state".to_string()),
                    (0.9, "Synchronizing nodes".to_string()),
                    (1.0, "Recovery complete".to_string()),
                ]
            }
            RecoveryStrategy::ReduceBatchSize { .. } => {
                vec![
                    (0.5, "Reducing batch size".to_string()),
                    (0.8, "Redistributing work".to_string()),
                    (1.0, "Recovery complete".to_string()),
                ]
            }
            _ => {
                vec![
                    (0.5, "Executing recovery".to_string()),
                    (1.0, "Recovery complete".to_string()),
                ]
            }
        };

        // Simulate progression through stages
        for (progress, stage) in stages {
            executor.update_recovery_progress(incident_id, progress, stage)?;

            // Simulate time taken for each stage
            std::thread::sleep(Duration::from_millis(100));
        }

        // Complete recovery (simulate 90% success rate)
        let success = (SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            % 10)
            != 0;
        let recovery_status = executor.complete_recovery(
            incident_id,
            success,
            if success {
                None
            } else {
                Some("Simulated failure".to_string())
            },
        )?;

        // Update incident with final status
        {
            let mut active_incidents = self.active_incidents.write().map_err(|e| {
                TorshDistributedError::communication_error(
                    "incidents",
                    format!("Lock error: {}", e),
                )
            })?;

            if let Some(incident) = active_incidents.get_mut(incident_id) {
                incident.recovery_status = recovery_status;
                incident.recovery_completed_at = Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                );

                // Move to history if completed
                if matches!(
                    incident.recovery_status,
                    RecoveryStatus::Completed { .. } | RecoveryStatus::Failed { .. }
                ) {
                    let completed_incident = incident.clone();
                    drop(active_incidents);

                    let mut history = self.incident_history.lock().map_err(|e| {
                        TorshDistributedError::communication_error(
                            "history",
                            format!("Lock error: {}", e),
                        )
                    })?;
                    history.push_back(completed_incident);
                    if history.len() > self.config.max_incident_history {
                        history.pop_front();
                    }
                }
            }
        }

        Ok(())
    }

    /// Update node heartbeat
    pub fn update_node_heartbeat(
        &self,
        node_id: String,
        metrics: crate::distributed_monitoring::NodeMetrics,
    ) -> TorshResult<()> {
        // Update node status
        {
            let mut node_status = self.node_status.write().map_err(|e| {
                TorshDistributedError::communication_error(
                    "node_heartbeat",
                    format!("Lock error: {}", e),
                )
            })?;

            let status = node_status
                .entry(node_id.clone())
                .or_insert_with(|| NodeStatus {
                    last_heartbeat: Instant::now(),
                    health_status: NodeHealthStatus::Healthy,
                    consecutive_failures: 0,
                    is_excluded: false,
                    last_metrics: None,
                });

            status.last_heartbeat = Instant::now();
            status.health_status = metrics.health_status.clone();
            status.last_metrics = Some(metrics.clone());

            // Reset failure count on successful heartbeat
            if matches!(metrics.health_status, NodeHealthStatus::Healthy) {
                status.consecutive_failures = 0;
            }
        }

        // Update predictive failure models
        if self.config.enable_predictive_detection {
            let mut predictor = self.failure_predictor.lock().map_err(|e| {
                TorshDistributedError::communication_error(
                    "predictor_update",
                    format!("Lock error: {}", e),
                )
            })?;
            predictor.update_node_metrics(&node_id, &metrics);
        }

        Ok(())
    }

    /// Get current fault tolerance status
    pub fn get_status(&self) -> TorshResult<FaultToleranceStatus> {
        let active_incidents = self.active_incidents.read().map_err(|e| {
            TorshDistributedError::communication_error("status", format!("Lock error: {}", e))
        })?;

        let node_status = self.node_status.read().map_err(|e| {
            TorshDistributedError::communication_error("status", format!("Lock error: {}", e))
        })?;

        let total_nodes = node_status.len();
        let healthy_nodes = node_status
            .values()
            .filter(|s| matches!(s.health_status, NodeHealthStatus::Healthy))
            .count();
        let excluded_nodes = node_status.values().filter(|s| s.is_excluded).count();

        let active_incident_count = active_incidents.len();
        let recovering_incidents = active_incidents
            .values()
            .filter(|i| matches!(i.recovery_status, RecoveryStatus::InProgress { .. }))
            .count();

        Ok(FaultToleranceStatus {
            total_nodes,
            healthy_nodes,
            excluded_nodes,
            active_incidents: active_incident_count,
            recovering_incidents,
            system_health_score: self.calculate_system_health_score(&node_status)?,
            last_incident_time: active_incidents.values().map(|i| i.detected_at).max(),
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }

    /// Calculate overall system health score
    fn calculate_system_health_score(
        &self,
        node_status: &HashMap<String, NodeStatus>,
    ) -> TorshResult<f32> {
        if node_status.is_empty() {
            return Ok(1.0);
        }

        let total_nodes = node_status.len() as f32;
        let healthy_weight = node_status
            .values()
            .map(|status| match status.health_status {
                NodeHealthStatus::Healthy => 1.0,
                NodeHealthStatus::Degraded { .. } => 0.7,
                NodeHealthStatus::Critical { .. } => 0.3,
                NodeHealthStatus::Failed { .. } => 0.0,
                NodeHealthStatus::Recovering { progress } => 0.5 + progress * 0.3,
            })
            .sum::<f32>();

        Ok((healthy_weight / total_nodes).max(0.0).min(1.0))
    }

    /// Get incident history for analysis
    pub fn get_incident_history(&self) -> TorshResult<Vec<FailureIncident>> {
        let history = self.incident_history.lock().map_err(|e| {
            TorshDistributedError::communication_error(
                "incident_history",
                format!("Lock error: {}", e),
            )
        })?;
        Ok(history.iter().cloned().collect())
    }

    /// Export fault tolerance data for analysis
    pub fn export_fault_tolerance_data(&self) -> TorshResult<FaultToleranceExport> {
        let status = self.get_status()?;
        let incident_history = self.get_incident_history()?;

        let active_incidents = self.active_incidents.read().map_err(|e| {
            TorshDistributedError::communication_error("export", format!("Lock error: {}", e))
        })?;
        let current_incidents: Vec<FailureIncident> = active_incidents.values().cloned().collect();

        Ok(FaultToleranceExport {
            status,
            current_incidents,
            incident_history,
            config: self.config.clone(),
            export_timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }
}

/// Overall fault tolerance system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceStatus {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub excluded_nodes: usize,
    pub active_incidents: usize,
    pub recovering_incidents: usize,
    pub system_health_score: f32,
    pub last_incident_time: Option<u64>,
    pub timestamp_ms: u64,
}

/// Complete fault tolerance data export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceExport {
    pub status: FaultToleranceStatus,
    pub current_incidents: Vec<FailureIncident>,
    pub incident_history: Vec<FailureIncident>,
    pub config: FaultToleranceConfig,
    pub export_timestamp_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed_monitoring::{DistributedMonitor, MonitoringConfig};

    #[tokio::test]
    async fn test_enhanced_fault_tolerance_creation() -> TorshResult<()> {
        let monitor_config = MonitoringConfig::default();
        let monitor = Arc::new(DistributedMonitor::new(monitor_config, false));

        let ft_config = FaultToleranceConfig::default();
        let fault_tolerance = EnhancedFaultTolerance::new(ft_config, monitor);

        let status = fault_tolerance.get_status()?;
        assert_eq!(status.total_nodes, 0);
        assert_eq!(status.active_incidents, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_failure_detection() -> TorshResult<()> {
        let monitor_config = MonitoringConfig::default();
        let monitor = Arc::new(DistributedMonitor::new(monitor_config, false));

        let ft_config = FaultToleranceConfig::default();
        let fault_tolerance = EnhancedFaultTolerance::new(ft_config, monitor);

        let failures = fault_tolerance.detect_failures()?;
        // Should be empty initially
        assert!(failures.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_failure_handling() -> TorshResult<()> {
        let monitor_config = MonitoringConfig::default();
        let monitor = Arc::new(DistributedMonitor::new(monitor_config, false));

        let ft_config = FaultToleranceConfig::default();
        let fault_tolerance = EnhancedFaultTolerance::new(ft_config, monitor);

        let failure = FailureType::NodeUnresponsive {
            node_id: "test_node".to_string(),
            last_seen: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        let incident_id = fault_tolerance.handle_failure(failure)?;
        assert!(!incident_id.is_empty());

        let status = fault_tolerance.get_status()?;
        assert_eq!(status.active_incidents, 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_prediction_model() -> TorshResult<()> {
        let mut model = PredictionModel::new();

        // Feed normal values
        for i in 0..20 {
            model.update(50.0 + (i as f32 % 5.0) as f32);
        }

        let normal_risk = model.predict_failure_risk();
        assert!(normal_risk < 0.5);

        // Feed increasing values (trending up)
        for i in 20..40 {
            model.update(60.0 + i as f32);
        }

        let high_risk = model.predict_failure_risk();
        assert!(high_risk > normal_risk);

        Ok(())
    }

    #[tokio::test]
    async fn test_recovery_strategy_determination() -> TorshResult<()> {
        let monitor_config = MonitoringConfig::default();
        let monitor = Arc::new(DistributedMonitor::new(monitor_config, false));

        let ft_config = FaultToleranceConfig::default();
        let fault_tolerance = EnhancedFaultTolerance::new(ft_config, monitor);

        let failure = FailureType::OutOfMemory {
            node_id: "test_node".to_string(),
            available_mb: 1000,
            requested_mb: 8000,
        };

        let strategy = fault_tolerance.determine_recovery_strategy(&failure)?;
        match strategy {
            RecoveryStrategy::ReduceBatchSize { .. } => {} // Expected
            _ => panic!("Unexpected recovery strategy for OOM failure"),
        }

        Ok(())
    }
}

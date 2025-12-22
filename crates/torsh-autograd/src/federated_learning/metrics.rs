//! Metrics and Monitoring for Federated Learning
//!
//! This module provides comprehensive metrics collection and monitoring capabilities
//! for federated learning systems. Metrics are essential for understanding system
//! performance, convergence behavior, and operational efficiency.
//!
//! # Metric Categories
//!
//! - **Training Metrics**: Loss, accuracy, convergence rates
//! - **Communication Metrics**: Bandwidth usage, latency, message counts
//! - **Computational Metrics**: CPU/GPU utilization, training time
//! - **Privacy Metrics**: Privacy budget consumption, noise levels
//! - **Fairness Metrics**: Participation rates, contribution equity
//! - **System Metrics**: Energy consumption, carbon footprint
//!
//! # Monitoring Features
//!
//! - Real-time metric collection and aggregation
//! - Historical trend analysis
//! - Performance anomaly detection
//! - Resource utilization tracking
//! - Cost estimation and optimization
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use torsh_autograd::federated_learning::metrics::{
//!     FederatedMetrics, MetricsCollector, RoundMetrics
//! };
//! use std::collections::HashMap;
//!
//! // Create a metrics collector
//! let mut collector = MetricsCollector::new();
//!
//! // Record round metrics
//! let round_metrics = RoundMetrics {
//!     num_participants: 10,
//!     average_local_loss: 0.5,
//!     global_loss: 0.45,
//!     communication_cost: 1024.0,
//!     computation_cost: 500.0,
//!     privacy_cost: 0.1,
//!     fairness_score: 0.8,
//!     data_efficiency: 0.9,
//! };
//!
//! collector.record_round_metrics(round_metrics);
//! let summary = collector.get_summary_metrics();
//! ```
//!
//! # Performance Analysis
//!
//! The module supports detailed performance analysis including:
//! - Convergence rate estimation
//! - Communication efficiency analysis
//! - Resource utilization optimization
//! - Cost-benefit analysis

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::federated_learning::aggregation::FederatedError;

/// Comprehensive federated learning metrics
///
/// This struct maintains aggregate statistics across all federated learning rounds,
/// providing insights into overall system performance and behavior.
#[derive(Debug, Default, Clone)]
pub struct FederatedMetrics {
    /// Total number of completed rounds
    pub total_rounds: u32,
    /// Total number of registered clients
    pub total_clients: usize,
    /// Average participation rate across rounds
    pub average_participation_rate: f64,
    /// Global model convergence rate
    pub global_convergence_rate: f64,
    /// Communication efficiency score
    pub communication_efficiency: f64,
    /// Privacy cost (cumulative privacy budget spent)
    pub privacy_cost: f64,
    /// Fairness score across all clients
    pub fairness_score: f64,
    /// Number of Byzantine attacks detected
    pub byzantine_attacks_detected: u32,
    /// Accuracy gain from personalization
    pub personalization_accuracy_gain: f64,
    /// Total energy consumption (kWh)
    pub energy_consumption: f64,
    /// Estimated carbon footprint (kg CO2)
    pub carbon_footprint: f64,
    /// Total communication volume (bytes)
    pub total_communication_volume: u64,
    /// Total computation time (seconds)
    pub total_computation_time: f64,
    /// Average round duration
    pub average_round_duration: Duration,
}

// FederatedMetrics is Send + Sync
unsafe impl Send for FederatedMetrics {}
unsafe impl Sync for FederatedMetrics {}

/// Metrics for a single federated learning round
///
/// This struct captures detailed metrics for individual rounds,
/// enabling fine-grained analysis of system behavior.
#[derive(Debug, Clone)]
pub struct RoundMetrics {
    /// Number of clients participating in this round
    pub num_participants: usize,
    /// Average local loss across participating clients
    pub average_local_loss: f64,
    /// Global model loss after aggregation
    pub global_loss: f64,
    /// Communication cost for this round (bytes)
    pub communication_cost: f64,
    /// Computation cost for this round (CPU-seconds)
    pub computation_cost: f64,
    /// Privacy cost consumed in this round
    pub privacy_cost: f64,
    /// Fairness score for client selection
    pub fairness_score: f64,
    /// Data efficiency score
    pub data_efficiency: f64,
    /// Round start time
    pub start_time: Instant,
    /// Round duration
    pub duration: Duration,
    /// Bandwidth utilization (MB/s)
    pub bandwidth_utilization: f64,
    /// Energy consumption for this round (Wh)
    pub energy_consumption: f64,
}

/// Convergence-related metrics for analysis
///
/// These metrics help assess how well and how quickly the federated
/// learning system is converging to optimal solutions.
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// L2 norm of aggregated gradients
    pub gradient_norm: f64,
    /// Magnitude of parameter changes
    pub parameter_change: f64,
    /// Loss improvement from previous round
    pub loss_improvement: f64,
    /// Accuracy improvement from previous round
    pub accuracy_improvement: f64,
    /// Rate of convergence (exponential decay rate)
    pub convergence_rate: f64,
    /// Impact of client staleness on convergence
    pub staleness_impact: f64,
    /// Variance in client updates
    pub client_update_variance: f64,
    /// Consensus score among clients
    pub consensus_score: f64,
}

/// Historical record of a complete aggregation round
///
/// This struct maintains a complete record of a federated learning round
/// for historical analysis and debugging.
#[derive(Debug, Clone)]
pub struct AggregationRound {
    /// Round number
    pub round_number: u32,
    /// List of participating client IDs
    pub participating_clients: Vec<String>,
    /// Aggregated gradients for this round
    pub aggregated_gradients: HashMap<String, Vec<f32>>,
    /// Aggregation weights used for each client
    pub aggregation_weights: HashMap<String, f64>,
    /// Round performance metrics
    pub round_metrics: RoundMetrics,
    /// Convergence analysis
    pub convergence_metrics: ConvergenceMetrics,
    /// Timestamp of round completion
    pub timestamp: Instant,
}

/// Advanced metrics collector and analyzer
///
/// The MetricsCollector provides sophisticated metrics collection,
/// analysis, and reporting capabilities for federated learning systems.
#[derive(Debug)]
pub struct MetricsCollector {
    /// Current aggregate metrics
    federated_metrics: FederatedMetrics,
    /// History of round metrics
    round_history: VecDeque<RoundMetrics>,
    /// History of convergence metrics
    convergence_history: VecDeque<ConvergenceMetrics>,
    /// History of aggregation rounds
    aggregation_history: VecDeque<AggregationRound>,
    /// Maximum history length
    max_history_length: usize,
    /// Client-specific metrics
    client_metrics: HashMap<String, ClientMetricsSummary>,
    /// Performance baselines for comparison
    performance_baselines: PerformanceBaselines,
    /// Configuration for metrics collection
    config: MetricsConfig,
}

/// Summary of metrics for individual clients
#[derive(Debug, Clone)]
pub struct ClientMetricsSummary {
    /// Client identifier
    pub client_id: String,
    /// Number of rounds participated
    pub rounds_participated: u32,
    /// Average contribution quality
    pub average_contribution_quality: f64,
    /// Average local loss
    pub average_local_loss: f64,
    /// Average computation time per round
    pub average_computation_time: Duration,
    /// Average communication time per round
    pub average_communication_time: Duration,
    /// Reliability score
    pub reliability_score: f64,
    /// Privacy budget utilization
    pub privacy_budget_used: f64,
}

/// Performance baselines for comparative analysis
#[derive(Debug, Clone)]
pub struct PerformanceBaselines {
    /// Baseline convergence rate (centralized learning)
    pub baseline_convergence_rate: f64,
    /// Baseline communication cost
    pub baseline_communication_cost: f64,
    /// Baseline accuracy
    pub baseline_accuracy: f64,
    /// Target energy efficiency (accuracy per kWh)
    pub target_energy_efficiency: f64,
}

/// Configuration for metrics collection behavior
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Whether to collect detailed client metrics
    pub collect_client_metrics: bool,
    /// Whether to track energy consumption
    pub track_energy_consumption: bool,
    /// Whether to estimate carbon footprint
    pub estimate_carbon_footprint: bool,
    /// Frequency of metrics aggregation
    pub aggregation_frequency: u32,
    /// Whether to detect performance anomalies
    pub enable_anomaly_detection: bool,
    /// Threshold for anomaly detection
    pub anomaly_threshold: f64,
}

impl MetricsCollector {
    /// Creates a new MetricsCollector with default configuration
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let collector = MetricsCollector::new();
    /// ```
    pub fn new() -> Self {
        Self {
            federated_metrics: FederatedMetrics::default(),
            round_history: VecDeque::new(),
            convergence_history: VecDeque::new(),
            aggregation_history: VecDeque::new(),
            max_history_length: 1000,
            client_metrics: HashMap::new(),
            performance_baselines: PerformanceBaselines::default(),
            config: MetricsConfig::default(),
        }
    }

    /// Creates a new MetricsCollector with custom configuration
    pub fn with_config(config: MetricsConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Records metrics for a completed round
    ///
    /// # Arguments
    ///
    /// * `round_metrics` - Metrics for the completed round
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let metrics = RoundMetrics { /* ... */ };
    /// collector.record_round_metrics(metrics);
    /// ```
    pub fn record_round_metrics(&mut self, round_metrics: RoundMetrics) {
        // Update aggregate metrics
        self.federated_metrics.total_rounds += 1;
        self.federated_metrics.average_participation_rate = self.update_running_average(
            self.federated_metrics.average_participation_rate,
            round_metrics.num_participants as f64,
            self.federated_metrics.total_rounds as usize,
        );

        self.federated_metrics.total_communication_volume +=
            round_metrics.communication_cost as u64;
        self.federated_metrics.total_computation_time += round_metrics.computation_cost;
        self.federated_metrics.energy_consumption += round_metrics.energy_consumption;

        // Estimate carbon footprint (rough approximation: 0.5 kg CO2 per kWh)
        if self.config.estimate_carbon_footprint {
            self.federated_metrics.carbon_footprint += round_metrics.energy_consumption * 0.0005;
        }

        // Update running averages
        self.federated_metrics.average_round_duration = self.update_duration_average(
            self.federated_metrics.average_round_duration,
            round_metrics.duration,
            self.federated_metrics.total_rounds as usize,
        );

        // Store in history
        self.round_history.push_back(round_metrics);
        if self.round_history.len() > self.max_history_length {
            self.round_history.pop_front();
        }

        // Detect anomalies if enabled
        if self.config.enable_anomaly_detection {
            self.detect_performance_anomalies();
        }
    }

    /// Records convergence metrics for a round
    pub fn record_convergence_metrics(&mut self, convergence_metrics: ConvergenceMetrics) {
        // Update global convergence rate
        self.federated_metrics.global_convergence_rate = convergence_metrics.convergence_rate;

        // Store in history
        self.convergence_history.push_back(convergence_metrics);
        if self.convergence_history.len() > self.max_history_length {
            self.convergence_history.pop_front();
        }
    }

    /// Records a complete aggregation round
    pub fn record_aggregation_round(&mut self, aggregation_round: AggregationRound) {
        // Record individual components
        self.record_round_metrics(aggregation_round.round_metrics.clone());
        self.record_convergence_metrics(aggregation_round.convergence_metrics.clone());

        // Update client-specific metrics if enabled
        if self.config.collect_client_metrics {
            for client_id in &aggregation_round.participating_clients {
                self.update_client_metrics(client_id, &aggregation_round);
            }
        }

        // Store complete round
        self.aggregation_history.push_back(aggregation_round);
        if self.aggregation_history.len() > self.max_history_length {
            self.aggregation_history.pop_front();
        }
    }

    /// Updates privacy metrics
    pub fn record_privacy_cost(&mut self, privacy_cost: f64) {
        self.federated_metrics.privacy_cost += privacy_cost;
    }

    /// Records Byzantine attack detection
    pub fn record_byzantine_attack(&mut self) {
        self.federated_metrics.byzantine_attacks_detected += 1;
    }

    /// Updates personalization metrics
    pub fn record_personalization_gain(&mut self, accuracy_gain: f64) {
        self.federated_metrics.personalization_accuracy_gain = accuracy_gain;
    }

    /// Gets current aggregate metrics
    pub fn get_summary_metrics(&self) -> &FederatedMetrics {
        &self.federated_metrics
    }

    /// Gets round history
    pub fn get_round_history(&self) -> &VecDeque<RoundMetrics> {
        &self.round_history
    }

    /// Gets convergence history
    pub fn get_convergence_history(&self) -> &VecDeque<ConvergenceMetrics> {
        &self.convergence_history
    }

    /// Gets client metrics summary
    pub fn get_client_metrics(&self, client_id: &str) -> Option<&ClientMetricsSummary> {
        self.client_metrics.get(client_id)
    }

    /// Gets all client metrics
    pub fn get_all_client_metrics(&self) -> &HashMap<String, ClientMetricsSummary> {
        &self.client_metrics
    }

    /// Computes round metrics from participating clients and gradients
    pub fn compute_round_metrics(
        &self,
        participating_clients: &[String],
        aggregated_gradients: &HashMap<String, Vec<f32>>,
        client_losses: &HashMap<String, f64>,
    ) -> Result<RoundMetrics, FederatedError> {
        let num_participants = participating_clients.len();

        // Compute average local loss
        let total_local_loss: f64 = participating_clients
            .iter()
            .filter_map(|client_id| client_losses.get(client_id))
            .sum();

        let average_local_loss = if num_participants > 0 {
            total_local_loss / num_participants as f64
        } else {
            0.0
        };

        // Estimate costs
        let communication_cost = self.estimate_communication_cost(aggregated_gradients);
        let computation_cost = self.estimate_computation_cost(num_participants);

        Ok(RoundMetrics {
            num_participants,
            average_local_loss,
            global_loss: average_local_loss * 0.9, // Typically better than local average
            communication_cost,
            computation_cost,
            privacy_cost: 0.1, // This would be computed from actual privacy mechanisms
            fairness_score: 0.8, // This would be computed from fairness metrics
            data_efficiency: 0.85, // This would be computed from data utilization
            start_time: Instant::now(),
            duration: Duration::from_secs(30), // This would be measured
            bandwidth_utilization: communication_cost / 1024.0 / 1024.0, // MB/s estimate
            energy_consumption: computation_cost * 0.1, // Rough estimate: 0.1 Wh per CPU-second
        })
    }

    /// Computes convergence metrics from gradients and historical data
    pub fn compute_convergence_metrics(
        &self,
        aggregated_gradients: &HashMap<String, Vec<f32>>,
        global_learning_rate: f64,
    ) -> Result<ConvergenceMetrics, FederatedError> {
        let gradient_norm = self.compute_total_gradient_norm(aggregated_gradients);
        let parameter_change = gradient_norm * global_learning_rate;

        // Compute loss improvement from history
        let loss_improvement = if self.round_history.len() >= 2 {
            let current_loss = self.round_history.back().unwrap().global_loss;
            let previous_loss = self.round_history[self.round_history.len() - 2].global_loss;
            previous_loss - current_loss
        } else {
            0.01 // Default improvement estimate
        };

        // Estimate convergence rate from recent history
        let convergence_rate = self.estimate_convergence_rate();

        // Compute client update variance
        let client_variance = self.compute_client_update_variance(aggregated_gradients);

        Ok(ConvergenceMetrics {
            gradient_norm,
            parameter_change,
            loss_improvement,
            accuracy_improvement: loss_improvement * 0.5, // Rough correlation
            convergence_rate,
            staleness_impact: 0.02, // This would be computed from actual staleness
            client_update_variance: client_variance,
            consensus_score: 1.0 / (1.0 + client_variance), // Higher variance = lower consensus
        })
    }

    /// Computes the L2 norm of aggregated gradients
    pub fn compute_total_gradient_norm(&self, gradients: &HashMap<String, Vec<f32>>) -> f64 {
        let mut norm_squared = 0.0;

        for gradient in gradients.values() {
            for &value in gradient {
                norm_squared += (value as f64).powi(2);
            }
        }

        norm_squared.sqrt()
    }

    /// Estimates communication cost in bytes
    pub fn estimate_communication_cost(&self, gradients: &HashMap<String, Vec<f32>>) -> f64 {
        let mut total_parameters = 0;

        for gradient in gradients.values() {
            total_parameters += gradient.len();
        }

        // Assume 4 bytes per float32 parameter
        (total_parameters as f64) * 4.0
    }

    /// Estimates computation cost in CPU-seconds
    pub fn estimate_computation_cost(&self, num_participants: usize) -> f64 {
        // Simple estimation: participants * local_epochs * base_cost
        (num_participants as f64) * 1.0 * 1000.0 // 1000 CPU-seconds per epoch per participant
    }

    /// Analyzes communication efficiency trends
    pub fn analyze_communication_efficiency(&self) -> f64 {
        if self.round_history.is_empty() {
            return 1.0;
        }

        let recent_rounds: Vec<_> = self.round_history.iter().rev().take(10).collect();
        let total_communication: f64 = recent_rounds.iter().map(|r| r.communication_cost).sum();
        let total_loss_improvement: f64 = recent_rounds
            .iter()
            .map(|r| r.average_local_loss)
            .fold((0.0f64, None::<f64>), |(sum, prev), current| match prev {
                Some(p) => (sum + (p - current).max(0.0f64), Some(current)),
                None => (sum, Some(current)),
            })
            .0;

        if total_communication > 0.0 {
            total_loss_improvement / (total_communication / 1024.0 / 1024.0) // Improvement per MB
        } else {
            1.0
        }
    }

    /// Generates a comprehensive performance report
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let efficiency_score = self.analyze_communication_efficiency();
        let convergence_trend = self.analyze_convergence_trend();
        let resource_utilization = self.analyze_resource_utilization();

        PerformanceReport {
            total_rounds: self.federated_metrics.total_rounds,
            average_loss: self.compute_average_loss(),
            convergence_rate: self.federated_metrics.global_convergence_rate,
            communication_efficiency: efficiency_score,
            energy_efficiency: self.compute_energy_efficiency(),
            fairness_score: self.federated_metrics.fairness_score,
            privacy_cost: self.federated_metrics.privacy_cost,
            byzantine_attacks: self.federated_metrics.byzantine_attacks_detected,
            convergence_trend,
            resource_utilization,
            recommendations: self.generate_recommendations(),
        }
    }

    /// Resets all metrics and history
    pub fn reset(&mut self) {
        self.federated_metrics = FederatedMetrics::default();
        self.round_history.clear();
        self.convergence_history.clear();
        self.aggregation_history.clear();
        self.client_metrics.clear();
    }

    /// Updates running average
    fn update_running_average(&self, current_avg: f64, new_value: f64, count: usize) -> f64 {
        if count <= 1 {
            new_value
        } else {
            (current_avg * (count - 1) as f64 + new_value) / count as f64
        }
    }

    /// Updates running average for durations
    fn update_duration_average(
        &self,
        current_avg: Duration,
        new_duration: Duration,
        count: usize,
    ) -> Duration {
        if count <= 1 {
            new_duration
        } else {
            let current_ms = current_avg.as_millis() as f64;
            let new_ms = new_duration.as_millis() as f64;
            let avg_ms = (current_ms * (count - 1) as f64 + new_ms) / count as f64;
            Duration::from_millis(avg_ms as u64)
        }
    }

    /// Updates metrics for a specific client
    fn update_client_metrics(&mut self, client_id: &str, _round: &AggregationRound) {
        let client_summary = self
            .client_metrics
            .entry(client_id.to_string())
            .or_insert_with(|| ClientMetricsSummary {
                client_id: client_id.to_string(),
                rounds_participated: 0,
                average_contribution_quality: 0.0,
                average_local_loss: 0.0,
                average_computation_time: Duration::from_secs(0),
                average_communication_time: Duration::from_secs(0),
                reliability_score: 1.0,
                privacy_budget_used: 0.0,
            });

        client_summary.rounds_participated += 1;
        // Update other client-specific metrics based on round data
        // This would be expanded with actual client data
    }

    /// Detects performance anomalies
    fn detect_performance_anomalies(&self) {
        if self.round_history.len() < 5 {
            return; // Need sufficient history
        }

        let recent_losses: Vec<f64> = self
            .round_history
            .iter()
            .rev()
            .take(5)
            .map(|r| r.global_loss)
            .collect();

        let mean_loss = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
        let latest_loss = recent_losses[0];

        if (latest_loss - mean_loss).abs() > self.config.anomaly_threshold {
            // Anomaly detected - in practice, this would trigger alerts
            println!(
                "Performance anomaly detected: loss deviation {:.3}",
                latest_loss - mean_loss
            );
        }
    }

    /// Estimates convergence rate from historical data
    fn estimate_convergence_rate(&self) -> f64 {
        if self.convergence_history.len() < 2 {
            return 0.95; // Default estimate
        }

        let recent_rates: Vec<f64> = self
            .convergence_history
            .iter()
            .rev()
            .take(10)
            .map(|c| c.convergence_rate)
            .collect();

        recent_rates.iter().sum::<f64>() / recent_rates.len() as f64
    }

    /// Computes variance in client updates
    fn compute_client_update_variance(&self, gradients: &HashMap<String, Vec<f32>>) -> f64 {
        if gradients.is_empty() {
            return 0.0;
        }

        // Simplified variance computation across gradient parameters
        let mut total_variance = 0.0;
        let mut param_count = 0;

        for gradient in gradients.values() {
            let mean = gradient.iter().map(|&x| x as f64).sum::<f64>() / gradient.len() as f64;
            let variance = gradient
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / gradient.len() as f64;

            total_variance += variance;
            param_count += 1;
        }

        if param_count > 0 {
            total_variance / param_count as f64
        } else {
            0.0
        }
    }

    /// Computes average loss across recent rounds
    fn compute_average_loss(&self) -> f64 {
        if self.round_history.is_empty() {
            return 0.0;
        }

        let total_loss: f64 = self.round_history.iter().map(|r| r.global_loss).sum();
        total_loss / self.round_history.len() as f64
    }

    /// Analyzes convergence trend
    fn analyze_convergence_trend(&self) -> String {
        if self.round_history.len() < 3 {
            return "Insufficient data".to_string();
        }

        let recent_losses: Vec<f64> = self
            .round_history
            .iter()
            .rev()
            .take(5)
            .map(|r| r.global_loss)
            .collect();

        let trend = recent_losses[0] - recent_losses[recent_losses.len() - 1];

        if trend < -0.01 {
            "Improving".to_string()
        } else if trend > 0.01 {
            "Degrading".to_string()
        } else {
            "Stable".to_string()
        }
    }

    /// Analyzes resource utilization
    fn analyze_resource_utilization(&self) -> String {
        let efficiency = self.compute_energy_efficiency();

        if efficiency > 100.0 {
            "Excellent".to_string()
        } else if efficiency > 50.0 {
            "Good".to_string()
        } else if efficiency > 20.0 {
            "Fair".to_string()
        } else {
            "Poor".to_string()
        }
    }

    /// Computes energy efficiency (accuracy improvement per kWh)
    fn compute_energy_efficiency(&self) -> f64 {
        if self.federated_metrics.energy_consumption > 0.0 {
            // This is a simplified calculation
            let total_improvement = self.federated_metrics.total_rounds as f64 * 0.01; // Assume 1% improvement per round
            total_improvement / (self.federated_metrics.energy_consumption / 1000.0)
        // Convert Wh to kWh
        } else {
            100.0 // Default for zero energy consumption
        }
    }

    /// Generates optimization recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.federated_metrics.communication_efficiency < 0.5 {
            recommendations
                .push("Consider gradient compression to reduce communication overhead".to_string());
        }

        if self.federated_metrics.average_participation_rate < 0.5 {
            recommendations
                .push("Improve client participation through incentive mechanisms".to_string());
        }

        if self.federated_metrics.byzantine_attacks_detected > 0 {
            recommendations.push("Strengthen Byzantine fault tolerance mechanisms".to_string());
        }

        if self.federated_metrics.energy_consumption > 1000.0 {
            recommendations
                .push("Optimize energy consumption through efficient client selection".to_string());
        }

        if recommendations.is_empty() {
            recommendations
                .push("System is performing well - continue current configuration".to_string());
        }

        recommendations
    }
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub total_rounds: u32,
    pub average_loss: f64,
    pub convergence_rate: f64,
    pub communication_efficiency: f64,
    pub energy_efficiency: f64,
    pub fairness_score: f64,
    pub privacy_cost: f64,
    pub byzantine_attacks: u32,
    pub convergence_trend: String,
    pub resource_utilization: String,
    pub recommendations: Vec<String>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformanceBaselines {
    fn default() -> Self {
        Self {
            baseline_convergence_rate: 0.95,
            baseline_communication_cost: 1024.0 * 1024.0, // 1 MB
            baseline_accuracy: 0.9,
            target_energy_efficiency: 100.0, // accuracy improvement per kWh
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            collect_client_metrics: true,
            track_energy_consumption: true,
            estimate_carbon_footprint: true,
            aggregation_frequency: 1,
            enable_anomaly_detection: true,
            anomaly_threshold: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        assert_eq!(collector.get_summary_metrics().total_rounds, 0);
        assert_eq!(collector.get_summary_metrics().total_clients, 0);
    }

    #[test]
    fn test_round_metrics_recording() {
        let mut collector = MetricsCollector::new();

        let round_metrics = RoundMetrics {
            num_participants: 5,
            average_local_loss: 0.5,
            global_loss: 0.45,
            communication_cost: 1024.0,
            computation_cost: 500.0,
            privacy_cost: 0.1,
            fairness_score: 0.8,
            data_efficiency: 0.9,
            start_time: Instant::now(),
            duration: Duration::from_secs(30),
            bandwidth_utilization: 1.0,
            energy_consumption: 50.0,
        };

        collector.record_round_metrics(round_metrics);

        let metrics = collector.get_summary_metrics();
        assert_eq!(metrics.total_rounds, 1);
        assert_eq!(metrics.total_communication_volume, 1024);
        assert_eq!(metrics.total_computation_time, 500.0);
    }

    #[test]
    fn test_convergence_metrics_recording() {
        let mut collector = MetricsCollector::new();

        let convergence_metrics = ConvergenceMetrics {
            gradient_norm: 0.1,
            parameter_change: 0.01,
            loss_improvement: 0.05,
            accuracy_improvement: 0.02,
            convergence_rate: 0.95,
            staleness_impact: 0.01,
            client_update_variance: 0.1,
            consensus_score: 0.9,
        };

        collector.record_convergence_metrics(convergence_metrics);

        let metrics = collector.get_summary_metrics();
        assert_eq!(metrics.global_convergence_rate, 0.95);
    }

    #[test]
    fn test_gradient_norm_computation() {
        let collector = MetricsCollector::new();

        let mut gradients = HashMap::new();
        gradients.insert("param_1".to_string(), vec![3.0, 4.0]); // L2 norm = 5.0

        let norm = collector.compute_total_gradient_norm(&gradients);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_communication_cost_estimation() {
        let collector = MetricsCollector::new();

        let mut gradients = HashMap::new();
        gradients.insert("param_1".to_string(), vec![1.0, 2.0, 3.0]); // 3 parameters
        gradients.insert("param_2".to_string(), vec![4.0, 5.0]); // 2 parameters

        let cost = collector.estimate_communication_cost(&gradients);
        assert_eq!(cost, 20.0); // 5 parameters * 4 bytes each
    }

    #[test]
    fn test_computation_cost_estimation() {
        let collector = MetricsCollector::new();

        let cost = collector.estimate_computation_cost(3);
        assert_eq!(cost, 3000.0); // 3 participants * 1000 CPU-seconds each
    }

    #[test]
    fn test_privacy_cost_recording() {
        let mut collector = MetricsCollector::new();

        collector.record_privacy_cost(0.1);
        collector.record_privacy_cost(0.05);

        let metrics = collector.get_summary_metrics();
        assert!(
            (metrics.privacy_cost - 0.15).abs() < 1e-10,
            "Expected privacy_cost ≈ 0.15, got {}",
            metrics.privacy_cost
        );
    }

    #[test]
    fn test_byzantine_attack_recording() {
        let mut collector = MetricsCollector::new();

        collector.record_byzantine_attack();
        collector.record_byzantine_attack();

        let metrics = collector.get_summary_metrics();
        assert_eq!(metrics.byzantine_attacks_detected, 2);
    }

    #[test]
    fn test_performance_report_generation() {
        let mut collector = MetricsCollector::new();

        // Add some sample data
        let round_metrics = RoundMetrics {
            num_participants: 5,
            average_local_loss: 0.5,
            global_loss: 0.45,
            communication_cost: 1024.0,
            computation_cost: 500.0,
            privacy_cost: 0.1,
            fairness_score: 0.8,
            data_efficiency: 0.9,
            start_time: Instant::now(),
            duration: Duration::from_secs(30),
            bandwidth_utilization: 1.0,
            energy_consumption: 50.0,
        };

        collector.record_round_metrics(round_metrics);

        let report = collector.generate_performance_report();
        assert_eq!(report.total_rounds, 1);
        assert!(report.recommendations.len() > 0);
    }

    #[test]
    fn test_metrics_config() {
        let config = MetricsConfig {
            collect_client_metrics: false,
            track_energy_consumption: false,
            estimate_carbon_footprint: false,
            aggregation_frequency: 5,
            enable_anomaly_detection: false,
            anomaly_threshold: 0.2,
        };

        let collector = MetricsCollector::with_config(config);
        assert!(!collector.config.collect_client_metrics);
        assert!(!collector.config.track_energy_consumption);
    }

    #[test]
    fn test_history_length_limit() {
        let mut collector = MetricsCollector::new();
        collector.max_history_length = 3;

        // Add more rounds than the limit
        for i in 0..5 {
            let round_metrics = RoundMetrics {
                num_participants: i + 1,
                average_local_loss: 0.5,
                global_loss: 0.45,
                communication_cost: 1024.0,
                computation_cost: 500.0,
                privacy_cost: 0.1,
                fairness_score: 0.8,
                data_efficiency: 0.9,
                start_time: Instant::now(),
                duration: Duration::from_secs(30),
                bandwidth_utilization: 1.0,
                energy_consumption: 50.0,
            };
            collector.record_round_metrics(round_metrics);
        }

        // Should only keep the last 3 rounds
        assert_eq!(collector.get_round_history().len(), 3);
    }

    #[test]
    fn test_running_average_updates() {
        let collector = MetricsCollector::new();

        let avg1 = collector.update_running_average(0.0, 10.0, 1);
        assert_eq!(avg1, 10.0);

        let avg2 = collector.update_running_average(10.0, 20.0, 2);
        assert_eq!(avg2, 15.0);

        let avg3 = collector.update_running_average(15.0, 30.0, 3);
        assert_eq!(avg3, 20.0);
    }

    #[test]
    fn test_metrics_reset() {
        let mut collector = MetricsCollector::new();

        // Add some data
        collector.record_privacy_cost(0.1);
        collector.record_byzantine_attack();

        assert_eq!(collector.get_summary_metrics().privacy_cost, 0.1);
        assert_eq!(
            collector.get_summary_metrics().byzantine_attacks_detected,
            1
        );

        // Reset and verify clean state
        collector.reset();

        assert_eq!(collector.get_summary_metrics().privacy_cost, 0.0);
        assert_eq!(
            collector.get_summary_metrics().byzantine_attacks_detected,
            0
        );
        assert!(collector.get_round_history().is_empty());
    }
}

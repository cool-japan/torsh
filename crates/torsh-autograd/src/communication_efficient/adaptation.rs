//! Adaptive Communication Control and Performance Monitoring
//!
//! This module provides adaptive control mechanisms for communication-efficient distributed
//! training, including performance monitoring, adaptation strategies, and dynamic optimization.
//! The adaptive controller monitors system performance and dynamically adjusts communication
//! strategies to optimize throughput, minimize latency, and maintain system stability.
//!
//! # Key Components
//!
//! - **AdaptationController**: Main controller for adaptive behavior and performance monitoring
//! - **Performance Monitoring**: Real-time collection and analysis of communication metrics
//! - **Adaptation Strategies**: Multiple strategies for responding to performance changes
//! - **Control Parameters**: Configuration for adaptation behavior and thresholds
//! - **Metrics Collection**: Comprehensive tracking of communication performance
//!
//! # Adaptation Strategies
//!
//! The system supports five adaptation strategies:
//! - **Reactive**: Responds to performance degradation after it occurs
//! - **Proactive**: Anticipates issues and adapts before problems occur
//! - **Predictive**: Uses historical data to predict and prevent issues
//! - **Reinforcement Learning**: Uses ML to optimize adaptation decisions
//! - **Hybrid Adaptation**: Combines multiple strategies for optimal performance
//!
//! # Examples
//!
//! ## Basic Adaptation Controller
//! ```rust
//! use torsh_autograd::communication_efficient::adaptation::*;
//!
//! let controller = AdaptationController::new();
//! let performance = PerformanceSnapshot {
//!     timestamp: std::time::Instant::now(),
//!     throughput: 150.0,
//!     latency: std::time::Duration::from_millis(200),
//!     packet_loss_rate: 0.02,
//!     energy_consumption: 45.0,
//!     compression_ratio: 0.7,
//!     communication_efficiency: 0.85,
//! };
//!
//! if controller.should_adapt(&performance).unwrap() {
//!     println!("Adaptation needed based on performance metrics");
//! }
//! ```
//!
//! ## Custom Adaptation Strategy
//! ```rust
//! use torsh_autograd::communication_efficient::adaptation::*;
//! use std::time::Duration;
//!
//! let mut controller = AdaptationController::with_strategy(
//!     AdaptationStrategy::Predictive
//! );
//! controller.set_control_parameters(ControlParameters {
//!     adaptation_rate: 0.15,
//!     stability_threshold: 0.03,
//!     response_time: Duration::from_secs(5),
//!     convergence_tolerance: 0.005,
//! });
//! ```

use crate::communication_efficient::{
    config::{
        CommunicationTopology, CompressedGradient, CompressionStrategy, ProtocolOptimization,
        QualityOfService,
    },
    fault_tolerance::{FaultSeverity, FaultType, RecoveryStrategy},
    management::{BandwidthMeasurement, CongestionLevel, OptimizationObjective},
    transmission::{CommunicationProtocol, NetworkMetrics, ProtocolMetrics},
    CommunicationError,
};

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Main adaptive controller for communication optimization and performance monitoring.
///
/// The AdaptationController monitors system performance, detects degradation patterns,
/// and triggers adaptive responses to maintain optimal communication efficiency.
/// It supports multiple adaptation strategies and maintains historical performance data
/// for informed decision-making.
#[allow(dead_code)]
#[derive(Debug)]
pub struct AdaptationController {
    /// Current adaptation strategy being used
    adaptation_strategy: AdaptationStrategy,
    /// Historical performance data for trend analysis
    performance_history: VecDeque<PerformanceSnapshot>,
    /// Configured triggers that can initiate adaptation
    adaptation_triggers: Vec<AdaptationTrigger>,
    /// Control parameters governing adaptation behavior
    control_parameters: ControlParameters,
}

/// Enumeration of available adaptation strategies.
///
/// Each strategy represents a different approach to monitoring and responding
/// to system performance changes, allowing for flexibility in optimization objectives.
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationStrategy {
    /// Reactive strategy: responds to performance degradation after detection
    Reactive,
    /// Proactive strategy: anticipates issues and prevents degradation
    Proactive,
    /// Predictive strategy: uses historical data and trends for prediction
    Predictive,
    /// Reinforcement Learning strategy: learns optimal responses over time
    ReinforcementLearning,
    /// Hybrid strategy: combines multiple approaches for balanced optimization
    HybridAdaptation,
}

/// Snapshot of system performance at a specific point in time.
///
/// Contains comprehensive metrics for evaluating communication efficiency
/// and determining if adaptation is necessary.
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp when the snapshot was taken
    pub timestamp: Instant,
    /// Current throughput in messages/operations per second
    pub throughput: f64,
    /// Current communication latency
    pub latency: Duration,
    /// Rate of packet loss (0.0 to 1.0)
    pub packet_loss_rate: f64,
    /// Energy consumption rate
    pub energy_consumption: f64,
    /// Current compression ratio being achieved
    pub compression_ratio: f64,
    /// Overall communication efficiency metric (0.0 to 1.0)
    pub communication_efficiency: f64,
}

/// Triggers that can initiate adaptive behavior.
///
/// These triggers represent different conditions or events that the adaptation
/// controller monitors to determine when system adjustments are needed.
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationTrigger {
    /// Detected performance degradation below thresholds
    PerformanceDegradation,
    /// Network congestion affecting communication
    NetworkCongestion,
    /// Energy consumption constraints
    EnergyConstraints,
    /// Time constraints requiring faster adaptation
    TimeConstraints,
    /// Quality metrics falling below acceptable thresholds
    QualityThreshold,
    /// External event requiring system adjustment
    ExternalEvent,
}

/// Configuration parameters for adaptation control behavior.
///
/// These parameters control how aggressively and quickly the system responds
/// to performance changes, balancing responsiveness with stability.
#[derive(Debug)]
pub struct ControlParameters {
    /// Rate at which adaptations are applied (0.0 to 1.0)
    pub adaptation_rate: f64,
    /// Threshold for considering performance stable
    pub stability_threshold: f64,
    /// Maximum time allowed for adaptation to take effect
    pub response_time: Duration,
    /// Tolerance for considering adaptation convergence complete
    pub convergence_tolerance: f64,
}

/// Comprehensive metrics tracking communication performance.
///
/// Tracks detailed statistics about message transmission, protocol usage,
/// fault occurrences, and adaptation events for performance analysis.
#[derive(Debug, Default, Clone)]
pub struct CommunicationMetrics {
    /// Total number of messages sent by this node
    pub total_messages_sent: u64,
    /// Total number of messages received by this node
    pub total_messages_received: u64,
    /// Total bytes transmitted by this node
    pub total_bytes_sent: u64,
    /// Total bytes received by this node
    pub total_bytes_received: u64,
    /// Average communication latency across all messages
    pub average_latency: Duration,
    /// Average throughput in messages per second
    pub average_throughput: f64,
    /// Overall packet loss rate (0.0 to 1.0)
    pub packet_loss_rate: f64,
    /// Compression efficiency achieved
    pub compression_efficiency: f64,
    /// Energy consumption rate
    pub energy_consumption: f64,
    /// Distribution of protocol usage across optimizations
    pub protocol_distribution: HashMap<ProtocolOptimization, u64>,
    /// Count of different fault types encountered
    pub fault_count: HashMap<FaultType, u64>,
    /// Number of adaptations performed
    pub adaptation_count: u64,
}

impl Default for AdaptationStrategy {
    fn default() -> Self {
        AdaptationStrategy::Reactive
    }
}

impl Default for ControlParameters {
    fn default() -> Self {
        Self {
            adaptation_rate: 0.1,
            stability_threshold: 0.05,
            response_time: Duration::from_secs(10),
            convergence_tolerance: 0.01,
        }
    }
}

impl AdaptationController {
    /// Creates a new adaptation controller with default settings.
    ///
    /// Initializes with reactive adaptation strategy and standard control parameters.
    /// The controller starts with common adaptation triggers enabled.
    pub fn new() -> Self {
        Self {
            adaptation_strategy: AdaptationStrategy::Reactive,
            performance_history: VecDeque::new(),
            adaptation_triggers: vec![
                AdaptationTrigger::PerformanceDegradation,
                AdaptationTrigger::NetworkCongestion,
                AdaptationTrigger::EnergyConstraints,
            ],
            control_parameters: ControlParameters::default(),
        }
    }

    /// Creates a new adaptation controller with a specific strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The adaptation strategy to use for decision-making
    pub fn with_strategy(strategy: AdaptationStrategy) -> Self {
        let mut controller = Self::new();
        controller.adaptation_strategy = strategy;
        controller
    }

    /// Sets custom control parameters for adaptation behavior.
    ///
    /// # Arguments
    ///
    /// * `parameters` - New control parameters to configure adaptation
    pub fn set_control_parameters(&mut self, parameters: ControlParameters) {
        self.control_parameters = parameters;
    }

    /// Adds a performance snapshot to the historical record.
    ///
    /// Maintains a sliding window of performance data for trend analysis.
    /// The history is automatically trimmed to prevent excessive memory usage.
    ///
    /// # Arguments
    ///
    /// * `snapshot` - Performance data to add to history
    pub fn record_performance(&mut self, snapshot: PerformanceSnapshot) {
        self.performance_history.push_back(snapshot);

        // Maintain a maximum history size of 100 snapshots
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }
    }

    /// Determines if adaptation is needed based on current performance.
    ///
    /// Analyzes current performance against historical data and configured
    /// thresholds to determine if system adaptation is warranted.
    ///
    /// # Arguments
    ///
    /// * `performance` - Current performance snapshot to evaluate
    ///
    /// # Returns
    ///
    /// * `Ok(true)` if adaptation is recommended
    /// * `Ok(false)` if current performance is acceptable
    /// * `Err` if analysis fails
    pub fn should_adapt(
        &self,
        performance: &PerformanceSnapshot,
    ) -> Result<bool, CommunicationError> {
        // Need sufficient history for meaningful analysis
        if self.performance_history.len() < 5 {
            return Ok(false);
        }

        match self.adaptation_strategy {
            AdaptationStrategy::Reactive => self.reactive_adaptation_check(performance),
            AdaptationStrategy::Proactive => self.proactive_adaptation_check(performance),
            AdaptationStrategy::Predictive => self.predictive_adaptation_check(performance),
            AdaptationStrategy::ReinforcementLearning => self.ml_adaptation_check(performance),
            AdaptationStrategy::HybridAdaptation => self.hybrid_adaptation_check(performance),
        }
    }

    /// Reactive adaptation check based on performance degradation.
    fn reactive_adaptation_check(
        &self,
        performance: &PerformanceSnapshot,
    ) -> Result<bool, CommunicationError> {
        let recent_performance: Vec<_> = self.performance_history.iter().rev().take(5).collect();
        let avg_throughput = recent_performance.iter().map(|p| p.throughput).sum::<f64>()
            / recent_performance.len() as f64;

        let throughput_degradation = (avg_throughput - performance.throughput) / avg_throughput;

        Ok(throughput_degradation > self.control_parameters.stability_threshold)
    }

    /// Proactive adaptation check based on trend analysis.
    fn proactive_adaptation_check(
        &self,
        performance: &PerformanceSnapshot,
    ) -> Result<bool, CommunicationError> {
        // Check if performance is trending downward
        let recent_snapshots: Vec<_> = self.performance_history.iter().rev().take(10).collect();

        if recent_snapshots.len() < 5 {
            return Ok(false);
        }

        let first_half_avg = recent_snapshots[5..]
            .iter()
            .map(|p| p.throughput)
            .sum::<f64>()
            / (recent_snapshots.len() - 5) as f64;

        let second_half_avg = recent_snapshots[..5]
            .iter()
            .map(|p| p.throughput)
            .sum::<f64>()
            / 5.0;

        let trend_degradation = (first_half_avg - second_half_avg) / first_half_avg;

        Ok(trend_degradation > self.control_parameters.stability_threshold * 0.5)
    }

    /// Predictive adaptation check using historical patterns.
    fn predictive_adaptation_check(
        &self,
        performance: &PerformanceSnapshot,
    ) -> Result<bool, CommunicationError> {
        // Simple linear extrapolation for prediction
        let recent_snapshots: Vec<_> = self.performance_history.iter().rev().take(10).collect();

        if recent_snapshots.len() < 8 {
            return self.reactive_adaptation_check(performance);
        }

        let time_deltas: Vec<f64> = recent_snapshots
            .windows(2)
            .map(|w| (w[0].timestamp.duration_since(w[1].timestamp)).as_secs_f64())
            .collect();

        let throughput_changes: Vec<f64> = recent_snapshots
            .windows(2)
            .map(|w| w[0].throughput - w[1].throughput)
            .collect();

        let avg_time_delta = time_deltas.iter().sum::<f64>() / time_deltas.len() as f64;
        let avg_throughput_change =
            throughput_changes.iter().sum::<f64>() / throughput_changes.len() as f64;

        // Predict throughput after next interval
        let predicted_throughput =
            performance.throughput + (avg_throughput_change / avg_time_delta) * 10.0;
        let current_baseline = recent_snapshots.iter().map(|p| p.throughput).sum::<f64>()
            / recent_snapshots.len() as f64;

        let predicted_degradation = (current_baseline - predicted_throughput) / current_baseline;

        Ok(predicted_degradation > self.control_parameters.stability_threshold)
    }

    /// Machine learning-based adaptation check.
    fn ml_adaptation_check(
        &self,
        performance: &PerformanceSnapshot,
    ) -> Result<bool, CommunicationError> {
        // Placeholder for ML-based decision making
        // In a full implementation, this would use a trained model
        // For now, use a weighted combination of reactive and predictive
        let reactive_result = self.reactive_adaptation_check(performance)?;
        let predictive_result = self.predictive_adaptation_check(performance)?;

        // Simple weighted voting
        Ok(reactive_result || predictive_result)
    }

    /// Hybrid adaptation check combining multiple strategies.
    fn hybrid_adaptation_check(
        &self,
        performance: &PerformanceSnapshot,
    ) -> Result<bool, CommunicationError> {
        let reactive = self.reactive_adaptation_check(performance)?;
        let proactive = self.proactive_adaptation_check(performance)?;
        let predictive = self.predictive_adaptation_check(performance)?;

        // Adapt if majority of strategies recommend it
        let vote_count = [reactive, proactive, predictive]
            .iter()
            .filter(|&&x| x)
            .count();

        Ok(vote_count >= 2)
    }

    /// Gets the current adaptation strategy.
    pub fn strategy(&self) -> &AdaptationStrategy {
        &self.adaptation_strategy
    }

    /// Sets a new adaptation strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - New adaptation strategy to use
    pub fn set_strategy(&mut self, strategy: AdaptationStrategy) {
        self.adaptation_strategy = strategy;
    }

    /// Gets the current control parameters.
    pub fn control_parameters(&self) -> &ControlParameters {
        &self.control_parameters
    }

    /// Gets the performance history.
    pub fn performance_history(&self) -> &VecDeque<PerformanceSnapshot> {
        &self.performance_history
    }

    /// Clears the performance history.
    pub fn clear_history(&mut self) {
        self.performance_history.clear();
    }

    /// Gets the current adaptation triggers.
    pub fn adaptation_triggers(&self) -> &[AdaptationTrigger] {
        &self.adaptation_triggers
    }

    /// Sets new adaptation triggers.
    ///
    /// # Arguments
    ///
    /// * `triggers` - New set of triggers to monitor
    pub fn set_adaptation_triggers(&mut self, triggers: Vec<AdaptationTrigger>) {
        self.adaptation_triggers = triggers;
    }
}

impl Default for AdaptationController {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceSnapshot {
    /// Creates a new performance snapshot with current timestamp.
    ///
    /// # Arguments
    ///
    /// * `throughput` - Current system throughput
    /// * `latency` - Current communication latency
    /// * `packet_loss_rate` - Current packet loss rate (0.0 to 1.0)
    /// * `energy_consumption` - Current energy consumption rate
    /// * `compression_ratio` - Current compression ratio
    /// * `communication_efficiency` - Overall efficiency metric (0.0 to 1.0)
    pub fn new(
        throughput: f64,
        latency: Duration,
        packet_loss_rate: f64,
        energy_consumption: f64,
        compression_ratio: f64,
        communication_efficiency: f64,
    ) -> Self {
        Self {
            timestamp: Instant::now(),
            throughput,
            latency,
            packet_loss_rate,
            energy_consumption,
            compression_ratio,
            communication_efficiency,
        }
    }

    /// Calculates a composite performance score.
    ///
    /// Returns a single metric (0.0 to 1.0) representing overall performance,
    /// with higher values indicating better performance.
    pub fn performance_score(&self) -> f64 {
        // Weighted combination of metrics
        let throughput_score = (self.throughput / 1000.0).min(1.0); // Normalize to 1000 max
        let latency_score = 1.0 - (self.latency.as_secs_f64() / 10.0).min(1.0); // 10s max penalty
        let loss_score = 1.0 - self.packet_loss_rate;
        let efficiency_score = self.communication_efficiency;

        (throughput_score * 0.3 + latency_score * 0.3 + loss_score * 0.2 + efficiency_score * 0.2)
    }
}

impl CommunicationMetrics {
    /// Creates a new metrics collection with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a message transmission.
    ///
    /// # Arguments
    ///
    /// * `bytes_sent` - Number of bytes transmitted
    /// * `protocol` - Protocol optimization used
    pub fn record_transmission(&mut self, bytes_sent: u64, protocol: ProtocolOptimization) {
        self.total_messages_sent += 1;
        self.total_bytes_sent += bytes_sent;
        *self.protocol_distribution.entry(protocol).or_insert(0) += 1;
    }

    /// Records a message reception.
    ///
    /// # Arguments
    ///
    /// * `bytes_received` - Number of bytes received
    pub fn record_reception(&mut self, bytes_received: u64) {
        self.total_messages_received += 1;
        self.total_bytes_received += bytes_received;
    }

    /// Records a fault occurrence.
    ///
    /// # Arguments
    ///
    /// * `fault_type` - Type of fault that occurred
    pub fn record_fault(&mut self, fault_type: FaultType) {
        *self.fault_count.entry(fault_type).or_insert(0) += 1;
    }

    /// Records an adaptation event.
    pub fn record_adaptation(&mut self) {
        self.adaptation_count += 1;
    }

    /// Updates average latency with a new measurement.
    ///
    /// # Arguments
    ///
    /// * `new_latency` - Latest latency measurement
    pub fn update_latency(&mut self, new_latency: Duration) {
        if self.total_messages_received > 0 {
            let current_total =
                self.average_latency.as_nanos() as f64 * self.total_messages_received as f64;
            let new_total = current_total + new_latency.as_nanos() as f64;
            self.average_latency = Duration::from_nanos(
                (new_total / (self.total_messages_received + 1) as f64) as u64,
            );
        } else {
            self.average_latency = new_latency;
        }
    }

    /// Updates average throughput with a new measurement.
    ///
    /// # Arguments
    ///
    /// * `new_throughput` - Latest throughput measurement
    pub fn update_throughput(&mut self, new_throughput: f64) {
        if self.total_messages_sent > 0 {
            let current_total = self.average_throughput * self.total_messages_sent as f64;
            let new_total = current_total + new_throughput;
            self.average_throughput = new_total / (self.total_messages_sent + 1) as f64;
        } else {
            self.average_throughput = new_throughput;
        }
    }

    /// Calculates the overall communication efficiency.
    pub fn calculate_efficiency(&self) -> f64 {
        if self.total_messages_sent == 0 {
            return 0.0;
        }

        let success_rate = self.total_messages_received as f64 / self.total_messages_sent as f64;
        let latency_factor = 1.0 / (1.0 + self.average_latency.as_secs_f64());
        let throughput_factor = self.average_throughput / 1000.0; // Normalize to 1000

        success_rate * latency_factor * throughput_factor.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptation_controller_creation() {
        let controller = AdaptationController::new();
        assert_eq!(controller.strategy(), &AdaptationStrategy::Reactive);
        assert_eq!(controller.performance_history().len(), 0);
        assert_eq!(controller.adaptation_triggers().len(), 3);
    }

    #[test]
    fn test_adaptation_controller_with_strategy() {
        let controller = AdaptationController::with_strategy(AdaptationStrategy::Predictive);
        assert_eq!(controller.strategy(), &AdaptationStrategy::Predictive);
    }

    #[test]
    fn test_performance_snapshot_creation() {
        let snapshot =
            PerformanceSnapshot::new(150.0, Duration::from_millis(100), 0.02, 50.0, 0.8, 0.9);

        assert_eq!(snapshot.throughput, 150.0);
        assert_eq!(snapshot.latency, Duration::from_millis(100));
        assert_eq!(snapshot.packet_loss_rate, 0.02);
    }

    #[test]
    fn test_performance_score_calculation() {
        let snapshot = PerformanceSnapshot::new(
            500.0,                      // Good throughput
            Duration::from_millis(100), // Good latency
            0.01,                       // Low packet loss
            30.0,                       // Energy consumption
            0.8,                        // Good compression
            0.9,                        // High efficiency
        );

        let score = snapshot.performance_score();
        assert!(score > 0.7); // Should be a good score
        assert!(score <= 1.0);
    }

    #[test]
    fn test_adaptation_without_sufficient_history() {
        let controller = AdaptationController::new();
        let snapshot =
            PerformanceSnapshot::new(100.0, Duration::from_millis(200), 0.05, 40.0, 0.7, 0.8);

        let result = controller.should_adapt(&snapshot).unwrap();
        assert!(!result); // Should not adapt without sufficient history
    }

    #[test]
    fn test_communication_metrics_recording() {
        let mut metrics = CommunicationMetrics::new();

        metrics.record_transmission(1024, ProtocolOptimization::TCP);
        metrics.record_reception(1024);
        metrics.record_adaptation();

        assert_eq!(metrics.total_messages_sent, 1);
        assert_eq!(metrics.total_messages_received, 1);
        assert_eq!(metrics.total_bytes_sent, 1024);
        assert_eq!(metrics.total_bytes_received, 1024);
        assert_eq!(metrics.adaptation_count, 1);
    }

    #[test]
    fn test_control_parameters_defaults() {
        let params = ControlParameters::default();
        assert_eq!(params.adaptation_rate, 0.1);
        assert_eq!(params.stability_threshold, 0.05);
        assert_eq!(params.response_time, Duration::from_secs(10));
        assert_eq!(params.convergence_tolerance, 0.01);
    }

    #[test]
    fn test_metrics_efficiency_calculation() {
        let mut metrics = CommunicationMetrics::new();

        // Perfect scenario
        metrics.total_messages_sent = 100;
        metrics.total_messages_received = 100;
        metrics.average_latency = Duration::from_millis(10);
        metrics.average_throughput = 500.0;

        let efficiency = metrics.calculate_efficiency();
        assert!(efficiency > 0.0);
        assert!(efficiency <= 1.0);
    }
}

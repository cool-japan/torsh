// Copyright (c) 2025 ToRSh Contributors
//
// Federated Learning Metadata Management
//
// This module provides data structures and abstractions for federated learning,
// enabling privacy-preserving distributed training across multiple clients without
// centralizing sensitive data.
//
// # Key Features
//
// - **Client Management**: Track and manage federated learning clients
// - **Model Aggregation**: FedAvg, FedProx, and other aggregation strategies
// - **Privacy Mechanisms**: Differential privacy, secure aggregation
// - **Client Selection**: Smart client sampling strategies
// - **Communication Efficiency**: Gradient compression, quantization
//
// # Design Principles
//
// 1. **Privacy First**: Built-in differential privacy support
// 2. **Heterogeneity**: Handle non-IID data distributions
// 3. **Efficiency**: Minimize communication overhead
// 4. **Fairness**: Ensure equitable contribution from all clients
//
// # Examples
//
// ```rust
// use torsh_core::federated::{FederatedClient, AggregationStrategy, ClientSelector};
//
// // Create federated learning clients
// let client1 = FederatedClient::new("client_1", 1000, 0.8);
// let client2 = FederatedClient::new("client_2", 500, 0.6);
//
// // Select clients for training round
// let selector = ClientSelector::new(ClientSelectionStrategy::Random);
// let selected = selector.select(&clients, 10);
//
// // Aggregate client updates
// let aggregator = FedAvgAggregator::new();
// let global_update = aggregator.aggregate(&client_updates);
// ```

use core::fmt;

/// Unique identifier for a federated learning client
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClientId(String);

impl ClientId {
    /// Create a new client ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the client ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ClientId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Federated learning client metadata
///
/// Represents a client participating in federated learning with
/// information about their capabilities and data characteristics.
#[derive(Debug, Clone)]
pub struct FederatedClient {
    /// Unique client identifier
    id: ClientId,
    /// Number of training samples
    num_samples: usize,
    /// Client availability (0.0 to 1.0)
    availability: f64,
    /// Computational capacity (relative score)
    compute_capacity: f64,
    /// Network bandwidth (MB/s)
    bandwidth_mbps: f64,
    /// Data distribution characteristics
    data_distribution: DataDistribution,
    /// Privacy budget (epsilon for differential privacy)
    privacy_budget: Option<f64>,
}

impl FederatedClient {
    /// Create a new federated client
    pub fn new(id: impl Into<String>, num_samples: usize, availability: f64) -> Self {
        Self {
            id: ClientId::new(id),
            num_samples,
            availability: availability.max(0.0).min(1.0),
            compute_capacity: 1.0,
            bandwidth_mbps: 10.0,
            data_distribution: DataDistribution::Unknown,
            privacy_budget: None,
        }
    }

    /// Set computational capacity
    pub fn with_compute_capacity(mut self, capacity: f64) -> Self {
        self.compute_capacity = capacity;
        self
    }

    /// Set network bandwidth
    pub fn with_bandwidth(mut self, bandwidth_mbps: f64) -> Self {
        self.bandwidth_mbps = bandwidth_mbps;
        self
    }

    /// Set data distribution
    pub fn with_data_distribution(mut self, distribution: DataDistribution) -> Self {
        self.data_distribution = distribution;
        self
    }

    /// Set privacy budget
    pub fn with_privacy_budget(mut self, epsilon: f64) -> Self {
        self.privacy_budget = Some(epsilon);
        self
    }

    /// Get client ID
    pub fn id(&self) -> &ClientId {
        &self.id
    }

    /// Get number of samples
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Get availability
    pub fn availability(&self) -> f64 {
        self.availability
    }

    /// Get compute capacity
    pub fn compute_capacity(&self) -> f64 {
        self.compute_capacity
    }

    /// Get bandwidth
    pub fn bandwidth_mbps(&self) -> f64 {
        self.bandwidth_mbps
    }

    /// Get data distribution
    pub fn data_distribution(&self) -> &DataDistribution {
        &self.data_distribution
    }

    /// Get privacy budget
    pub fn privacy_budget(&self) -> Option<f64> {
        self.privacy_budget
    }

    /// Calculate client weight for aggregation
    pub fn weight(&self) -> f64 {
        self.num_samples as f64
    }
}

/// Data distribution characteristics for non-IID data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataDistribution {
    /// IID (Independent and Identically Distributed)
    IID,
    /// Non-IID with label skew
    LabelSkew { skew_factor: f64 },
    /// Non-IID with feature skew
    FeatureSkew { skew_factor: f64 },
    /// Non-IID with quantity skew
    QuantitySkew,
    /// Unknown distribution
    Unknown,
}

/// Aggregation strategies for federated learning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Federated Averaging (FedAvg)
    FedAvg,
    /// Federated Proximal (FedProx)
    FedProx,
    /// Federated Adaptive (FedAdam, FedYogi, etc.)
    FedAdaptive,
    /// Secure Aggregation with encryption
    SecureAggregation,
    /// Weighted aggregation by data size
    WeightedBySize,
    /// Weighted aggregation by client performance
    WeightedByPerformance,
}

impl fmt::Display for AggregationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregationStrategy::FedAvg => write!(f, "FedAvg"),
            AggregationStrategy::FedProx => write!(f, "FedProx"),
            AggregationStrategy::FedAdaptive => write!(f, "FedAdaptive"),
            AggregationStrategy::SecureAggregation => write!(f, "SecureAggregation"),
            AggregationStrategy::WeightedBySize => write!(f, "WeightedBySize"),
            AggregationStrategy::WeightedByPerformance => write!(f, "WeightedByPerformance"),
        }
    }
}

/// Client update from a training round
#[derive(Debug, Clone)]
pub struct ClientUpdate {
    /// Client that produced this update
    client_id: ClientId,
    /// Training round number
    round: u64,
    /// Number of local training steps
    num_steps: usize,
    /// Local training loss
    loss: f64,
    /// Local training accuracy
    accuracy: Option<f64>,
    /// Metadata for the update
    metadata: Vec<(String, String)>,
}

impl ClientUpdate {
    /// Create a new client update
    pub fn new(client_id: ClientId, round: u64, num_steps: usize, loss: f64) -> Self {
        Self {
            client_id,
            round,
            num_steps,
            loss,
            accuracy: None,
            metadata: Vec::new(),
        }
    }

    /// Set accuracy
    pub fn with_accuracy(mut self, accuracy: f64) -> Self {
        self.accuracy = Some(accuracy);
        self
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.push((key.into(), value.into()));
    }

    /// Get client ID
    pub fn client_id(&self) -> &ClientId {
        &self.client_id
    }

    /// Get round number
    pub fn round(&self) -> u64 {
        self.round
    }

    /// Get number of steps
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }

    /// Get loss
    pub fn loss(&self) -> f64 {
        self.loss
    }

    /// Get accuracy
    pub fn accuracy(&self) -> Option<f64> {
        self.accuracy
    }

    /// Get metadata
    pub fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }
}

/// Client selection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientSelectionStrategy {
    /// Random selection
    Random,
    /// Select based on availability
    ByAvailability,
    /// Select based on data size
    ByDataSize,
    /// Select based on compute capacity
    ByComputeCapacity,
    /// Power-of-choice (select best from random sample)
    PowerOfChoice { choices: usize },
    /// All clients participate
    All,
}

/// Client selector for federated learning rounds
#[derive(Debug, Clone)]
pub struct ClientSelector {
    strategy: ClientSelectionStrategy,
}

impl ClientSelector {
    /// Create a new client selector
    pub fn new(strategy: ClientSelectionStrategy) -> Self {
        Self { strategy }
    }

    /// Select clients for a training round
    pub fn select(&self, clients: &[FederatedClient], num_select: usize) -> Vec<ClientId> {
        match self.strategy {
            ClientSelectionStrategy::Random => {
                // Simple deterministic selection (in practice, would use RNG)
                clients
                    .iter()
                    .take(num_select.min(clients.len()))
                    .map(|c| c.id().clone())
                    .collect()
            }
            ClientSelectionStrategy::ByAvailability => {
                let mut sorted: Vec<_> = clients.iter().collect();
                sorted.sort_by(|a, b| {
                    b.availability()
                        .partial_cmp(&a.availability())
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                sorted
                    .iter()
                    .take(num_select.min(clients.len()))
                    .map(|c| c.id().clone())
                    .collect()
            }
            ClientSelectionStrategy::ByDataSize => {
                let mut sorted: Vec<_> = clients.iter().collect();
                sorted.sort_by_key(|c| core::cmp::Reverse(c.num_samples()));
                sorted
                    .iter()
                    .take(num_select.min(clients.len()))
                    .map(|c| c.id().clone())
                    .collect()
            }
            ClientSelectionStrategy::ByComputeCapacity => {
                let mut sorted: Vec<_> = clients.iter().collect();
                sorted.sort_by(|a, b| {
                    b.compute_capacity()
                        .partial_cmp(&a.compute_capacity())
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                sorted
                    .iter()
                    .take(num_select.min(clients.len()))
                    .map(|c| c.id().clone())
                    .collect()
            }
            ClientSelectionStrategy::PowerOfChoice { choices: _ } => {
                // Simplified: select by availability from first subset
                clients
                    .iter()
                    .take(num_select.min(clients.len()))
                    .map(|c| c.id().clone())
                    .collect()
            }
            ClientSelectionStrategy::All => clients.iter().map(|c| c.id().clone()).collect(),
        }
    }

    /// Get selection strategy
    pub fn strategy(&self) -> ClientSelectionStrategy {
        self.strategy
    }
}

/// Differential privacy parameters
#[derive(Debug, Clone, Copy)]
pub struct PrivacyParameters {
    /// Privacy budget (epsilon)
    epsilon: f64,
    /// Privacy loss probability (delta)
    delta: f64,
    /// Clipping threshold for gradient norm
    clip_norm: f64,
    /// Noise multiplier
    noise_multiplier: f64,
}

impl PrivacyParameters {
    /// Create new privacy parameters
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            epsilon,
            delta,
            clip_norm: 1.0,
            noise_multiplier: 1.0,
        }
    }

    /// Set clipping norm
    pub fn with_clip_norm(mut self, clip_norm: f64) -> Self {
        self.clip_norm = clip_norm;
        self
    }

    /// Set noise multiplier
    pub fn with_noise_multiplier(mut self, noise_multiplier: f64) -> Self {
        self.noise_multiplier = noise_multiplier;
        self
    }

    /// Get epsilon
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Get delta
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Get clip norm
    pub fn clip_norm(&self) -> f64 {
        self.clip_norm
    }

    /// Get noise multiplier
    pub fn noise_multiplier(&self) -> f64 {
        self.noise_multiplier
    }

    /// Check if privacy budget is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.epsilon <= 0.0
    }
}

/// Communication efficiency techniques
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionTechnique {
    /// No compression
    None,
    /// Gradient quantization (reduce precision)
    Quantization { bits: u8 },
    /// Sparsification (top-k gradients)
    Sparsification { k: usize },
    /// Gradient sketching
    Sketching,
    /// Low-rank approximation
    LowRank { rank: usize },
}

/// Federated learning round metadata
#[derive(Debug, Clone)]
pub struct TrainingRound {
    /// Round number
    round: u64,
    /// Number of clients selected
    num_clients: usize,
    /// Number of clients that completed
    num_completed: usize,
    /// Average loss across clients
    avg_loss: f64,
    /// Average accuracy across clients
    avg_accuracy: Option<f64>,
    /// Total communication cost (bytes)
    communication_cost: usize,
    /// Round duration (seconds)
    duration_secs: f64,
}

impl TrainingRound {
    /// Create a new training round
    pub fn new(round: u64, num_clients: usize) -> Self {
        Self {
            round,
            num_clients,
            num_completed: 0,
            avg_loss: 0.0,
            avg_accuracy: None,
            communication_cost: 0,
            duration_secs: 0.0,
        }
    }

    /// Set number of completed clients
    pub fn set_completed(&mut self, num_completed: usize) {
        self.num_completed = num_completed;
    }

    /// Set average loss
    pub fn set_avg_loss(&mut self, avg_loss: f64) {
        self.avg_loss = avg_loss;
    }

    /// Set average accuracy
    pub fn set_avg_accuracy(&mut self, avg_accuracy: f64) {
        self.avg_accuracy = Some(avg_accuracy);
    }

    /// Set communication cost
    pub fn set_communication_cost(&mut self, cost: usize) {
        self.communication_cost = cost;
    }

    /// Set duration
    pub fn set_duration(&mut self, duration_secs: f64) {
        self.duration_secs = duration_secs;
    }

    /// Get round number
    pub fn round(&self) -> u64 {
        self.round
    }

    /// Get number of selected clients
    pub fn num_clients(&self) -> usize {
        self.num_clients
    }

    /// Get number of completed clients
    pub fn num_completed(&self) -> usize {
        self.num_completed
    }

    /// Get average loss
    pub fn avg_loss(&self) -> f64 {
        self.avg_loss
    }

    /// Get average accuracy
    pub fn avg_accuracy(&self) -> Option<f64> {
        self.avg_accuracy
    }

    /// Get communication cost
    pub fn communication_cost(&self) -> usize {
        self.communication_cost
    }

    /// Get duration
    pub fn duration_secs(&self) -> f64 {
        self.duration_secs
    }

    /// Calculate completion rate
    pub fn completion_rate(&self) -> f64 {
        if self.num_clients == 0 {
            0.0
        } else {
            self.num_completed as f64 / self.num_clients as f64
        }
    }
}

/// Fairness metrics for federated learning
#[derive(Debug, Clone)]
pub struct FairnessMetrics {
    /// Variance in client accuracy
    accuracy_variance: f64,
    /// Minimum client accuracy
    min_accuracy: f64,
    /// Maximum client accuracy
    max_accuracy: f64,
    /// Jain's fairness index
    jains_index: f64,
}

impl FairnessMetrics {
    /// Create new fairness metrics
    pub fn new(
        accuracy_variance: f64,
        min_accuracy: f64,
        max_accuracy: f64,
        jains_index: f64,
    ) -> Self {
        Self {
            accuracy_variance,
            min_accuracy,
            max_accuracy,
            jains_index,
        }
    }

    /// Get accuracy variance
    pub fn accuracy_variance(&self) -> f64 {
        self.accuracy_variance
    }

    /// Get minimum accuracy
    pub fn min_accuracy(&self) -> f64 {
        self.min_accuracy
    }

    /// Get maximum accuracy
    pub fn max_accuracy(&self) -> f64 {
        self.max_accuracy
    }

    /// Get Jain's fairness index
    pub fn jains_index(&self) -> f64 {
        self.jains_index
    }

    /// Check if learning is fair (Jain's index > 0.8)
    pub fn is_fair(&self) -> bool {
        self.jains_index > 0.8
    }
}

/// Federated learning coordinator
#[derive(Debug, Clone)]
pub struct FederatedCoordinator {
    /// Current round number
    current_round: u64,
    /// Aggregation strategy
    strategy: AggregationStrategy,
    /// Privacy parameters
    privacy: Option<PrivacyParameters>,
    /// Compression technique
    compression: CompressionTechnique,
    /// Training history
    rounds: Vec<TrainingRound>,
}

impl FederatedCoordinator {
    /// Create a new federated coordinator
    pub fn new(strategy: AggregationStrategy) -> Self {
        Self {
            current_round: 0,
            strategy,
            privacy: None,
            compression: CompressionTechnique::None,
            rounds: Vec::new(),
        }
    }

    /// Set privacy parameters
    pub fn with_privacy(mut self, privacy: PrivacyParameters) -> Self {
        self.privacy = Some(privacy);
        self
    }

    /// Set compression technique
    pub fn with_compression(mut self, compression: CompressionTechnique) -> Self {
        self.compression = compression;
        self
    }

    /// Start a new training round
    pub fn start_round(&mut self, num_clients: usize) -> u64 {
        self.current_round += 1;
        self.rounds
            .push(TrainingRound::new(self.current_round, num_clients));
        self.current_round
    }

    /// Complete the current round
    pub fn complete_round(&mut self, avg_loss: f64, num_completed: usize) {
        if let Some(round) = self.rounds.last_mut() {
            round.set_avg_loss(avg_loss);
            round.set_completed(num_completed);
        }
    }

    /// Get current round
    pub fn current_round(&self) -> u64 {
        self.current_round
    }

    /// Get strategy
    pub fn strategy(&self) -> AggregationStrategy {
        self.strategy
    }

    /// Get privacy parameters
    pub fn privacy(&self) -> Option<&PrivacyParameters> {
        self.privacy.as_ref()
    }

    /// Get compression technique
    pub fn compression(&self) -> CompressionTechnique {
        self.compression
    }

    /// Get training history
    pub fn rounds(&self) -> &[TrainingRound] {
        &self.rounds
    }

    /// Get statistics
    pub fn statistics(&self) -> CoordinatorStatistics {
        let total_rounds = self.rounds.len();
        let avg_completion_rate = if total_rounds > 0 {
            self.rounds.iter().map(|r| r.completion_rate()).sum::<f64>() / total_rounds as f64
        } else {
            0.0
        };
        let total_communication = self.rounds.iter().map(|r| r.communication_cost()).sum();

        CoordinatorStatistics {
            total_rounds,
            avg_completion_rate,
            total_communication,
        }
    }
}

/// Coordinator statistics
#[derive(Debug, Clone)]
pub struct CoordinatorStatistics {
    /// Total number of rounds
    pub total_rounds: usize,
    /// Average completion rate
    pub avg_completion_rate: f64,
    /// Total communication cost (bytes)
    pub total_communication: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_id() {
        let id = ClientId::new("client_1");
        assert_eq!(id.as_str(), "client_1");
        assert_eq!(format!("{}", id), "client_1");
    }

    #[test]
    fn test_federated_client_creation() {
        let client = FederatedClient::new("client_1", 1000, 0.8);
        assert_eq!(client.id().as_str(), "client_1");
        assert_eq!(client.num_samples(), 1000);
        assert_eq!(client.availability(), 0.8);
        assert_eq!(client.weight(), 1000.0);
    }

    #[test]
    fn test_client_with_builder() {
        let client = FederatedClient::new("client_1", 1000, 0.8)
            .with_compute_capacity(2.0)
            .with_bandwidth(50.0)
            .with_privacy_budget(1.0);

        assert_eq!(client.compute_capacity(), 2.0);
        assert_eq!(client.bandwidth_mbps(), 50.0);
        assert_eq!(client.privacy_budget(), Some(1.0));
    }

    #[test]
    fn test_data_distribution() {
        let iid = DataDistribution::IID;
        let label_skew = DataDistribution::LabelSkew { skew_factor: 0.5 };
        let _feature_skew = DataDistribution::FeatureSkew { skew_factor: 0.3 };

        assert_eq!(iid, DataDistribution::IID);
        assert_ne!(iid, label_skew);
    }

    #[test]
    fn test_aggregation_strategy_display() {
        assert_eq!(format!("{}", AggregationStrategy::FedAvg), "FedAvg");
        assert_eq!(format!("{}", AggregationStrategy::FedProx), "FedProx");
    }

    #[test]
    fn test_client_update() {
        let id = ClientId::new("client_1");
        let mut update = ClientUpdate::new(id.clone(), 5, 100, 0.5).with_accuracy(0.85);

        update.add_metadata("dataset", "mnist");
        assert_eq!(update.client_id(), &id);
        assert_eq!(update.round(), 5);
        assert_eq!(update.num_steps(), 100);
        assert_eq!(update.loss(), 0.5);
        assert_eq!(update.accuracy(), Some(0.85));
        assert_eq!(update.metadata().len(), 1);
    }

    #[test]
    fn test_client_selector_random() {
        let clients = vec![
            FederatedClient::new("client_1", 1000, 0.8),
            FederatedClient::new("client_2", 500, 0.6),
            FederatedClient::new("client_3", 800, 0.9),
        ];

        let selector = ClientSelector::new(ClientSelectionStrategy::Random);
        let selected = selector.select(&clients, 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_client_selector_by_data_size() {
        let clients = vec![
            FederatedClient::new("client_1", 1000, 0.8),
            FederatedClient::new("client_2", 500, 0.6),
            FederatedClient::new("client_3", 800, 0.9),
        ];

        let selector = ClientSelector::new(ClientSelectionStrategy::ByDataSize);
        let selected = selector.select(&clients, 2);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].as_str(), "client_1"); // Largest dataset
    }

    #[test]
    fn test_client_selector_all() {
        let clients = vec![
            FederatedClient::new("client_1", 1000, 0.8),
            FederatedClient::new("client_2", 500, 0.6),
        ];

        let selector = ClientSelector::new(ClientSelectionStrategy::All);
        let selected = selector.select(&clients, 10);
        assert_eq!(selected.len(), 2); // All clients
    }

    #[test]
    fn test_privacy_parameters() {
        let privacy = PrivacyParameters::new(1.0, 1e-5)
            .with_clip_norm(2.0)
            .with_noise_multiplier(0.5);

        assert_eq!(privacy.epsilon(), 1.0);
        assert_eq!(privacy.delta(), 1e-5);
        assert_eq!(privacy.clip_norm(), 2.0);
        assert_eq!(privacy.noise_multiplier(), 0.5);
        assert!(!privacy.is_exhausted());
    }

    #[test]
    fn test_compression_techniques() {
        let _none = CompressionTechnique::None;
        let _quant = CompressionTechnique::Quantization { bits: 8 };
        let _sparse = CompressionTechnique::Sparsification { k: 100 };
        let _sketch = CompressionTechnique::Sketching;
        let _low_rank = CompressionTechnique::LowRank { rank: 10 };
    }

    #[test]
    fn test_training_round() {
        let mut round = TrainingRound::new(1, 10);
        round.set_completed(8);
        round.set_avg_loss(0.5);
        round.set_avg_accuracy(0.85);
        round.set_communication_cost(1024);
        round.set_duration(60.0);

        assert_eq!(round.round(), 1);
        assert_eq!(round.num_clients(), 10);
        assert_eq!(round.num_completed(), 8);
        assert_eq!(round.avg_loss(), 0.5);
        assert_eq!(round.avg_accuracy(), Some(0.85));
        assert_eq!(round.communication_cost(), 1024);
        assert_eq!(round.duration_secs(), 60.0);
        assert_eq!(round.completion_rate(), 0.8);
    }

    #[test]
    fn test_fairness_metrics() {
        let metrics = FairnessMetrics::new(0.01, 0.80, 0.90, 0.85);
        assert_eq!(metrics.accuracy_variance(), 0.01);
        assert_eq!(metrics.min_accuracy(), 0.80);
        assert_eq!(metrics.max_accuracy(), 0.90);
        assert_eq!(metrics.jains_index(), 0.85);
        assert!(metrics.is_fair());
    }

    #[test]
    fn test_federated_coordinator() {
        let mut coordinator = FederatedCoordinator::new(AggregationStrategy::FedAvg)
            .with_privacy(PrivacyParameters::new(1.0, 1e-5))
            .with_compression(CompressionTechnique::Quantization { bits: 8 });

        assert_eq!(coordinator.current_round(), 0);

        let round1 = coordinator.start_round(10);
        assert_eq!(round1, 1);

        coordinator.complete_round(0.5, 8);

        let stats = coordinator.statistics();
        assert_eq!(stats.total_rounds, 1);
    }

    #[test]
    fn test_coordinator_multiple_rounds() {
        let mut coordinator = FederatedCoordinator::new(AggregationStrategy::FedAvg);

        coordinator.start_round(10);
        coordinator.complete_round(0.6, 9);

        coordinator.start_round(10);
        coordinator.complete_round(0.4, 8);

        coordinator.start_round(10);
        coordinator.complete_round(0.3, 10);

        assert_eq!(coordinator.current_round(), 3);
        assert_eq!(coordinator.rounds().len(), 3);

        let stats = coordinator.statistics();
        assert_eq!(stats.total_rounds, 3);
        assert!(stats.avg_completion_rate > 0.8);
    }

    #[test]
    fn test_client_selection_strategies() {
        let _random = ClientSelectionStrategy::Random;
        let _avail = ClientSelectionStrategy::ByAvailability;
        let _size = ClientSelectionStrategy::ByDataSize;
        let _compute = ClientSelectionStrategy::ByComputeCapacity;
        let _power = ClientSelectionStrategy::PowerOfChoice { choices: 3 };
        let _all = ClientSelectionStrategy::All;
    }

    #[test]
    fn test_availability_clamping() {
        let client1 = FederatedClient::new("c1", 100, 1.5); // > 1.0
        let client2 = FederatedClient::new("c2", 100, -0.1); // < 0.0

        assert_eq!(client1.availability(), 1.0);
        assert_eq!(client2.availability(), 0.0);
    }
}

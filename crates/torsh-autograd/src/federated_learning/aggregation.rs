//! Aggregation strategies and coordination for federated learning
//!
//! This module provides the core aggregation functionality for federated learning,
//! including various aggregation strategies, round management, and coordination
//! between clients. It implements the FederatedAggregator which orchestrates
//! the entire federated learning process.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use super::byzantine::ByzantineDetector;
use super::client::FederatedClient;
use super::metrics::FederatedMetrics;
use super::personalization::{PersonalizationManager, PersonalizationStrategy};
use super::privacy::PrivacyEngine;
use super::selection::ClientSelector;
use super::types::{AggregationStrategy, FederatedLearningConfig, PrivacyMechanism};

/// Error type for federated learning operations
#[derive(Debug)]
pub enum FederatedError {
    /// Client not found in the system
    ClientNotFound,
    /// Aggregation operation failed
    AggregationFailed,
    /// Parameter not found in model
    ParameterNotFound,
    /// Dimension mismatch between tensors
    DimensionMismatch,
    /// Insufficient clients for operation
    InsufficientClients,
    /// Privacy budget exceeded
    PrivacyBudgetExceeded,
    /// Byzantine behavior detected
    ByzantineDetected,
    /// General computation error
    ComputationError(String),
}

impl std::fmt::Display for FederatedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FederatedError::ClientNotFound => write!(f, "Client not found"),
            FederatedError::AggregationFailed => write!(f, "Aggregation failed"),
            FederatedError::ParameterNotFound => write!(f, "Parameter not found"),
            FederatedError::DimensionMismatch => write!(f, "Dimension mismatch"),
            FederatedError::InsufficientClients => write!(f, "Insufficient clients"),
            FederatedError::PrivacyBudgetExceeded => write!(f, "Privacy budget exceeded"),
            FederatedError::ByzantineDetected => write!(f, "Byzantine behavior detected"),
            FederatedError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl std::error::Error for FederatedError {}

/// Main coordinator for federated learning aggregation
///
/// The `FederatedAggregator` orchestrates the entire federated learning process,
/// including client selection, update collection, aggregation, and model updates.
/// It supports various aggregation strategies and privacy mechanisms.
///
/// # Examples
/// use std::time::Duration;
///
/// ```rust,ignore
/// use torsh_autograd::federated_learning::aggregation::*;
/// use torsh_autograd::federated_learning::types::*;
///
/// let config = FederatedLearningConfig::default();
/// let aggregator = FederatedAggregator::new(config);
///
/// // Register clients
/// let client = FederatedClient::new("client_001".to_string());
/// aggregator.register_client(client)?;
///
/// // Run federated learning
/// aggregator.run_federated_learning()?;
/// ```
#[derive(Debug)]
pub struct FederatedAggregator {
    config: FederatedLearningConfig,
    global_model: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    clients: Arc<RwLock<HashMap<String, FederatedClient>>>,
    aggregation_history: Arc<Mutex<VecDeque<AggregationRound>>>,
    client_selector: Arc<Mutex<ClientSelector>>,
    privacy_engine: Arc<Mutex<PrivacyEngine>>,
    byzantine_detector: Arc<Mutex<ByzantineDetector>>,
    personalization_manager: Arc<Mutex<PersonalizationManager>>,
    metrics: Arc<Mutex<FederatedMetrics>>,
    current_round: Arc<Mutex<u32>>,
}

// FederatedAggregator is Send + Sync
unsafe impl Send for FederatedAggregator {}
unsafe impl Sync for FederatedAggregator {}

impl FederatedAggregator {
    /// Create a new federated aggregator with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the federated learning system
    ///
    /// # Returns
    ///
    /// A new `FederatedAggregator` instance
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = FederatedLearningConfig {
    ///     aggregation_strategy: AggregationStrategy::FederatedAveraging,
    ///     client_selection_strategy: ClientSelectionStrategy::Random,
    ///     communication_rounds: 100,
    ///     clients_per_round: 10,
    ///     ..Default::default()
    /// };
    /// let aggregator = FederatedAggregator::new(config);
    /// ```
    pub fn new(config: FederatedLearningConfig) -> Self {
        Self {
            config: config.clone(),
            global_model: Arc::new(RwLock::new(HashMap::new())),
            clients: Arc::new(RwLock::new(HashMap::new())),
            aggregation_history: Arc::new(Mutex::new(VecDeque::new())),
            client_selector: Arc::new(Mutex::new(ClientSelector::new(
                config.client_selection_strategy.clone(),
            ))),
            privacy_engine: Arc::new(Mutex::new(PrivacyEngine::new(
                config.privacy_mechanism.clone(),
                config.differential_privacy_epsilon,
                config.differential_privacy_delta,
            ))),
            byzantine_detector: Arc::new(Mutex::new(ByzantineDetector::new())),
            personalization_manager: Arc::new(Mutex::new(PersonalizationManager::new(
                PersonalizationStrategy::None,
            ))),
            metrics: Arc::new(Mutex::new(FederatedMetrics::default())),
            current_round: Arc::new(Mutex::new(0)),
        }
    }

    /// Register a new client in the federated learning system
    ///
    /// # Arguments
    ///
    /// * `client` - The client to register
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error otherwise
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let client = FederatedClient::new("client_001".to_string());
    /// aggregator.register_client(client)?;
    /// ```
    pub fn register_client(&self, client: FederatedClient) -> Result<(), FederatedError> {
        let mut clients = self.clients.write().expect("lock should not be poisoned");
        clients.insert(client.client_id.clone(), client);

        let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
        metrics.total_clients = clients.len();

        Ok(())
    }

    /// Run the complete federated learning process
    ///
    /// Executes the specified number of communication rounds, coordinating
    /// client selection, update collection, aggregation, and model updates.
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful completion, error otherwise
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// aggregator.run_federated_learning()?;
    /// ```
    pub fn run_federated_learning(&self) -> Result<(), FederatedError> {
        let mut round = 0;

        while round < self.config.communication_rounds {
            self.execute_federated_round(round)?;
            round += 1;

            *self
                .current_round
                .lock()
                .expect("lock should not be poisoned") = round;
        }

        Ok(())
    }

    /// Execute a single federated learning round
    ///
    /// Coordinates all steps of a federated learning round including client
    /// selection, update collection, aggregation, and model updates.
    ///
    /// # Arguments
    ///
    /// * `round_number` - The current round number
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error otherwise
    fn execute_federated_round(&self, round_number: u32) -> Result<(), FederatedError> {
        println!("Executing federated round {}", round_number);

        let selected_clients = self.select_clients_for_round()?;
        let client_updates = self.collect_client_updates(&selected_clients)?;

        if self.config.byzantine_tolerance {
            self.detect_byzantine_clients(&client_updates)?;
        }

        let aggregated_gradients = self.aggregate_gradients(&client_updates)?;

        if self.config.privacy_mechanism != PrivacyMechanism::None {
            self.apply_privacy_mechanism(&aggregated_gradients)?;
        }

        self.update_global_model(&aggregated_gradients)?;

        if self.config.personalization_enabled {
            self.update_personalized_models(&selected_clients, &client_updates)?;
        }

        self.record_aggregation_round(round_number, selected_clients, aggregated_gradients)?;

        Ok(())
    }

    /// Select clients for the current round
    ///
    /// Uses the configured client selection strategy to choose clients
    /// for participation in the current round.
    ///
    /// # Returns
    ///
    /// Vector of selected client IDs
    fn select_clients_for_round(&self) -> Result<Vec<String>, FederatedError> {
        // Get client IDs snapshot to avoid holding locks simultaneously
        let client_ids: Vec<String> = {
            let clients = self.clients.read().expect("lock should not be poisoned");
            clients.keys().cloned().collect()
        };

        // Now safely acquire selector lock
        let mut selector = self
            .client_selector
            .lock()
            .expect("lock should not be poisoned");
        let selected = selector.select_clients(&client_ids, self.config.clients_per_round)?;

        println!("Selected {} clients for this round", selected.len());

        Ok(selected)
    }

    /// Collect updates from selected clients
    ///
    /// Coordinates with selected clients to compute and collect their
    /// local model updates, applying local privacy if configured.
    ///
    /// # Arguments
    ///
    /// * `selected_clients` - List of client IDs to collect from
    ///
    /// # Returns
    ///
    /// HashMap mapping client IDs to their gradient updates
    fn collect_client_updates(
        &self,
        selected_clients: &[String],
    ) -> Result<HashMap<String, HashMap<String, Vec<f32>>>, FederatedError> {
        let mut client_updates = HashMap::new();

        // Get current round value before main loop to avoid nested locks
        let current_round_value = *self
            .current_round
            .lock()
            .expect("lock should not be poisoned");

        // Create snapshot of global model to reduce lock scope
        let global_model_snapshot = {
            let global_model = self
                .global_model
                .read()
                .expect("lock should not be poisoned");
            global_model.clone()
        };

        // Now work with clients with minimal lock duration
        {
            let mut clients = self.clients.write().expect("lock should not be poisoned");

            for client_id in selected_clients {
                if let Some(client) = clients.get_mut(client_id) {
                    let local_update = client
                        .compute_local_update(&global_model_snapshot, self.config.local_epochs)?;

                    if self.config.privacy_mechanism == PrivacyMechanism::LocalDifferentialPrivacy {
                        let sensitivity = self.estimate_gradient_sensitivity(&local_update);
                        client.apply_differential_privacy(
                            self.config.differential_privacy_epsilon,
                            sensitivity,
                        )?;
                    }

                    client_updates.insert(client_id.clone(), client.local_gradients.clone());
                    client.last_participation_round = current_round_value;
                }
            }
        }

        Ok(client_updates)
    }

    /// Estimate the sensitivity of gradients for differential privacy
    ///
    /// # Arguments
    ///
    /// * `gradients` - Gradient updates to analyze
    ///
    /// # Returns
    ///
    /// Estimated sensitivity value
    fn estimate_gradient_sensitivity(&self, gradients: &HashMap<String, Vec<f32>>) -> f64 {
        let mut max_norm: f64 = 0.0;

        for gradient in gradients.values() {
            let norm = gradient.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt() as f64;
            max_norm = max_norm.max(norm);
        }

        max_norm.max(1.0)
    }

    /// Detect Byzantine clients based on their updates
    ///
    /// # Arguments
    ///
    /// * `client_updates` - Updates from all clients
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error otherwise
    fn detect_byzantine_clients(
        &self,
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Result<(), FederatedError> {
        let mut byzantine_count = 0;

        // Scope the detector lock to limit its duration
        {
            let mut detector = self
                .byzantine_detector
                .lock()
                .expect("lock should not be poisoned");

            for (client_id, gradients) in client_updates {
                let is_byzantine = detector.detect_byzantine_behavior(client_id, gradients)?;

                if is_byzantine {
                    println!("Detected Byzantine behavior from client: {}", client_id);
                    byzantine_count += 1;
                }
            }
        } // detector lock is released here

        // Now safely acquire metrics lock
        if byzantine_count > 0 {
            self.metrics
                .lock()
                .expect("lock should not be poisoned")
                .byzantine_attacks_detected += byzantine_count;
        }

        Ok(())
    }

    /// Aggregate client updates using the configured strategy
    ///
    /// # Arguments
    ///
    /// * `client_updates` - Updates from all participating clients
    ///
    /// # Returns
    ///
    /// Aggregated gradient updates
    fn aggregate_gradients(
        &self,
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Result<HashMap<String, Vec<f32>>, FederatedError> {
        let clients = self.clients.read().expect("lock should not be poisoned");

        match self.config.aggregation_strategy {
            AggregationStrategy::FederatedAveraging => {
                self.federated_averaging(client_updates, &clients)
            }
            AggregationStrategy::FederatedProx => self.federated_prox(client_updates, &clients),
            AggregationStrategy::MedianAggregation => self.median_aggregation(client_updates),
            AggregationStrategy::TrimmedMean => self.trimmed_mean_aggregation(client_updates, 0.1),
            AggregationStrategy::Krum => self.krum_aggregation(client_updates, 2),
            AggregationStrategy::AdaptiveWeighted => {
                self.adaptive_weighted_aggregation(client_updates, &clients)
            }
            _ => self.federated_averaging(client_updates, &clients),
        }
    }

    /// Federated averaging aggregation strategy
    ///
    /// Computes weighted average of client updates based on data size.
    ///
    /// # Arguments
    ///
    /// * `client_updates` - Updates from participating clients
    /// * `clients` - Client metadata for weighting
    ///
    /// # Returns
    ///
    /// Aggregated gradients
    fn federated_averaging(
        &self,
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
        clients: &HashMap<String, FederatedClient>,
    ) -> Result<HashMap<String, Vec<f32>>, FederatedError> {
        let mut aggregated = HashMap::new();
        let mut total_data_size = 0;

        for client_id in client_updates.keys() {
            if let Some(client) = clients.get(client_id) {
                total_data_size += client.data_size;
            }
        }

        for (client_id, gradients) in client_updates {
            let client = clients
                .get(client_id)
                .ok_or(FederatedError::ClientNotFound)?;
            let weight = client.data_size as f64 / total_data_size as f64;

            for (param_name, gradient) in gradients {
                let aggregated_gradient = aggregated
                    .entry(param_name.clone())
                    .or_insert_with(|| vec![0.0; gradient.len()]);

                for (i, &grad_value) in gradient.iter().enumerate() {
                    aggregated_gradient[i] += (grad_value as f64 * weight) as f32;
                }
            }
        }

        Ok(aggregated)
    }

    /// FedProx aggregation strategy with proximal term
    ///
    /// # Arguments
    ///
    /// * `client_updates` - Updates from participating clients
    /// * `clients` - Client metadata for weighting
    ///
    /// # Returns
    ///
    /// Aggregated gradients with proximal regularization
    fn federated_prox(
        &self,
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
        clients: &HashMap<String, FederatedClient>,
    ) -> Result<HashMap<String, Vec<f32>>, FederatedError> {
        let mu = 0.01;
        let global_model = self
            .global_model
            .read()
            .expect("lock should not be poisoned");
        let mut aggregated = self.federated_averaging(client_updates, clients)?;

        for (param_name, aggregated_gradient) in &mut aggregated {
            if let Some(global_param) = global_model.get(param_name) {
                for (i, grad_value) in aggregated_gradient.iter_mut().enumerate() {
                    *grad_value += mu * global_param[i];
                }
            }
        }

        Ok(aggregated)
    }

    /// Median aggregation for Byzantine robustness
    ///
    /// # Arguments
    ///
    /// * `client_updates` - Updates from participating clients
    ///
    /// # Returns
    ///
    /// Median-aggregated gradients
    fn median_aggregation(
        &self,
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Result<HashMap<String, Vec<f32>>, FederatedError> {
        let mut aggregated = HashMap::new();

        if let Some((_, first_gradients)) = client_updates.iter().next() {
            for (param_name, gradient) in first_gradients {
                let mut param_values = Vec::new();

                for _ in 0..gradient.len() {
                    param_values.push(Vec::new());
                }

                for gradients in client_updates.values() {
                    if let Some(gradient) = gradients.get(param_name) {
                        for (i, &value) in gradient.iter().enumerate() {
                            param_values[i].push(value);
                        }
                    }
                }

                let mut median_gradient = Vec::new();
                for mut values in param_values {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = values.len() / 2;
                    let median = if values.len() % 2 == 0 {
                        (values[mid - 1] + values[mid]) / 2.0
                    } else {
                        values[mid]
                    };
                    median_gradient.push(median);
                }

                aggregated.insert(param_name.clone(), median_gradient);
            }
        }

        Ok(aggregated)
    }

    /// Trimmed mean aggregation for robustness
    ///
    /// # Arguments
    ///
    /// * `client_updates` - Updates from participating clients
    /// * `trim_ratio` - Ratio of updates to trim from extremes
    ///
    /// # Returns
    ///
    /// Trimmed mean aggregated gradients
    fn trimmed_mean_aggregation(
        &self,
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
        trim_ratio: f64,
    ) -> Result<HashMap<String, Vec<f32>>, FederatedError> {
        let mut aggregated = HashMap::new();
        let trim_count = ((client_updates.len() as f64) * trim_ratio).floor() as usize;

        if let Some((_, first_gradients)) = client_updates.iter().next() {
            for (param_name, gradient) in first_gradients {
                let mut param_values = Vec::new();

                for _ in 0..gradient.len() {
                    param_values.push(Vec::new());
                }

                for gradients in client_updates.values() {
                    if let Some(gradient) = gradients.get(param_name) {
                        for (i, &value) in gradient.iter().enumerate() {
                            param_values[i].push(value);
                        }
                    }
                }

                let mut trimmed_mean_gradient = Vec::new();
                for mut values in param_values {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    // Remove extreme values
                    for _ in 0..trim_count {
                        if !values.is_empty() {
                            values.remove(0);
                        }
                        if !values.is_empty() {
                            values.pop();
                        }
                    }

                    let mean = if !values.is_empty() {
                        values.iter().sum::<f32>() / values.len() as f32
                    } else {
                        0.0
                    };
                    trimmed_mean_gradient.push(mean);
                }

                aggregated.insert(param_name.clone(), trimmed_mean_gradient);
            }
        }

        Ok(aggregated)
    }

    /// Krum aggregation for Byzantine robustness
    ///
    /// # Arguments
    ///
    /// * `client_updates` - Updates from participating clients
    /// * `num_byzantine` - Expected number of Byzantine clients
    ///
    /// # Returns
    ///
    /// Krum-aggregated gradients
    fn krum_aggregation(
        &self,
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
        num_byzantine: usize,
    ) -> Result<HashMap<String, Vec<f32>>, FederatedError> {
        let client_ids: Vec<_> = client_updates.keys().collect();
        let num_clients = client_ids.len();

        if num_clients <= 2 * num_byzantine {
            return Err(FederatedError::InsufficientClients);
        }

        let mut scores = HashMap::new();

        for &client_id in &client_ids {
            let mut distances = Vec::new();

            for &other_client_id in &client_ids {
                if client_id != other_client_id {
                    let distance = self.compute_gradient_distance(
                        client_updates.get(client_id).unwrap(),
                        client_updates.get(other_client_id).unwrap(),
                    )?;
                    distances.push(distance);
                }
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let score: f64 = distances.iter().take(num_clients - num_byzantine - 1).sum();

            scores.insert(client_id, score);
        }

        // Find client with minimum score
        let best_client = scores
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| *k)
            .ok_or(FederatedError::AggregationFailed)?;

        // Return the gradients of the best client
        Ok(client_updates.get(best_client).unwrap().clone())
    }

    /// Adaptive weighted aggregation based on client quality
    ///
    /// # Arguments
    ///
    /// * `client_updates` - Updates from participating clients
    /// * `clients` - Client metadata for adaptive weighting
    ///
    /// # Returns
    ///
    /// Adaptively weighted aggregated gradients
    fn adaptive_weighted_aggregation(
        &self,
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
        clients: &HashMap<String, FederatedClient>,
    ) -> Result<HashMap<String, Vec<f32>>, FederatedError> {
        let mut aggregated = HashMap::new();
        let mut total_weight = 0.0;

        for (client_id, gradients) in client_updates {
            let client = clients
                .get(client_id)
                .ok_or(FederatedError::ClientNotFound)?;

            let data_weight = (client.data_size as f64).ln();
            let quality_weight = client.client_metrics.data_quality_score;
            let trust_weight = client.trust_score;
            let combined_weight = data_weight * quality_weight * trust_weight;

            total_weight += combined_weight;

            for (param_name, gradient) in gradients {
                let aggregated_gradient = aggregated
                    .entry(param_name.clone())
                    .or_insert_with(|| vec![0.0; gradient.len()]);

                for (i, &grad_value) in gradient.iter().enumerate() {
                    aggregated_gradient[i] += (grad_value as f64 * combined_weight) as f32;
                }
            }
        }

        for gradient in aggregated.values_mut() {
            for value in gradient.iter_mut() {
                *value = (*value as f64 / total_weight) as f32;
            }
        }

        Ok(aggregated)
    }

    /// Compute Euclidean distance between two gradient sets
    ///
    /// # Arguments
    ///
    /// * `gradients1` - First gradient set
    /// * `gradients2` - Second gradient set
    ///
    /// # Returns
    ///
    /// Euclidean distance between the gradients
    fn compute_gradient_distance(
        &self,
        gradients1: &HashMap<String, Vec<f32>>,
        gradients2: &HashMap<String, Vec<f32>>,
    ) -> Result<f64, FederatedError> {
        let mut distance_squared = 0.0;

        for (param_name, grad1) in gradients1 {
            if let Some(grad2) = gradients2.get(param_name) {
                if grad1.len() != grad2.len() {
                    return Err(FederatedError::DimensionMismatch);
                }

                for (v1, v2) in grad1.iter().zip(grad2.iter()) {
                    let diff = *v1 as f64 - *v2 as f64;
                    distance_squared += diff * diff;
                }
            }
        }

        Ok(distance_squared.sqrt())
    }

    /// Apply the configured privacy mechanism
    ///
    /// # Arguments
    ///
    /// * `aggregated_gradients` - Gradients to apply privacy to
    ///
    /// # Returns
    ///
    /// `Ok(())` on success
    fn apply_privacy_mechanism(
        &self,
        aggregated_gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<(), FederatedError> {
        let mut privacy_engine = self
            .privacy_engine
            .lock()
            .expect("lock should not be poisoned");
        privacy_engine.apply_privacy(&aggregated_gradients)?;
        Ok(())
    }

    /// Update the global model with aggregated gradients
    ///
    /// # Arguments
    ///
    /// * `aggregated_gradients` - Aggregated gradient updates
    ///
    /// # Returns
    ///
    /// `Ok(())` on success
    fn update_global_model(
        &self,
        aggregated_gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<(), FederatedError> {
        let mut global_model = self
            .global_model
            .write()
            .expect("lock should not be poisoned");

        for (param_name, gradient) in aggregated_gradients {
            let model_param = global_model
                .entry(param_name.clone())
                .or_insert_with(|| vec![0.0; gradient.len()]);

            for (i, &grad_value) in gradient.iter().enumerate() {
                model_param[i] -= (self.config.global_learning_rate * grad_value as f64) as f32;
            }
        }

        Ok(())
    }

    /// Update personalized models if enabled
    ///
    /// # Arguments
    ///
    /// * `selected_clients` - Clients to update
    /// * `client_updates` - Updates from clients
    ///
    /// # Returns
    ///
    /// `Ok(())` on success
    fn update_personalized_models(
        &self,
        selected_clients: &[String],
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Result<(), FederatedError> {
        let mut personalization_manager = self
            .personalization_manager
            .lock()
            .expect("lock should not be poisoned");
        personalization_manager.update_personalized_models(selected_clients, client_updates)?;
        Ok(())
    }

    /// Record the aggregation round for monitoring and analysis
    ///
    /// # Arguments
    ///
    /// * `round_number` - Round number
    /// * `participating_clients` - Clients that participated
    /// * `aggregated_gradients` - Final aggregated gradients
    ///
    /// # Returns
    ///
    /// `Ok(())` on success
    fn record_aggregation_round(
        &self,
        round_number: u32,
        participating_clients: Vec<String>,
        aggregated_gradients: HashMap<String, Vec<f32>>,
    ) -> Result<(), FederatedError> {
        let round_metrics =
            self.compute_round_metrics(&participating_clients, &aggregated_gradients)?;
        let convergence_metrics = self.compute_convergence_metrics(&aggregated_gradients)?;

        let aggregation_round = AggregationRound {
            round_number,
            participating_clients: participating_clients.clone(),
            aggregated_gradients,
            aggregation_weights: HashMap::new(),
            round_metrics,
            convergence_metrics,
            timestamp: Instant::now(),
        };

        let mut history = self
            .aggregation_history
            .lock()
            .expect("lock should not be poisoned");
        history.push_back(aggregation_round);

        if history.len() > 100 {
            history.pop_front();
        }

        // Get client count first to avoid holding multiple locks
        let client_count = self
            .clients
            .read()
            .expect("lock should not be poisoned")
            .len();

        let mut metrics = self.metrics.lock().expect("lock should not be poisoned");
        metrics.total_rounds += 1;
        metrics.average_participation_rate =
            participating_clients.len() as f64 / client_count as f64;

        Ok(())
    }

    /// Compute metrics for the current round
    fn compute_round_metrics(
        &self,
        participating_clients: &[String],
        aggregated_gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<RoundMetrics, FederatedError> {
        let clients = self.clients.read().expect("lock should not be poisoned");

        let mut total_local_loss = 0.0;
        let mut num_participants = 0;

        for client_id in participating_clients {
            if let Some(client) = clients.get(client_id) {
                total_local_loss += client.client_metrics.local_loss;
                num_participants += 1;
            }
        }

        let average_local_loss = if num_participants > 0 {
            total_local_loss / num_participants as f64
        } else {
            0.0
        };

        let communication_cost = self.estimate_communication_cost(aggregated_gradients);
        let computation_cost = self.estimate_computation_cost(participating_clients.len());

        Ok(RoundMetrics {
            num_participants,
            average_local_loss,
            global_loss: average_local_loss * 0.9,
            communication_cost,
            computation_cost,
            privacy_cost: 0.1,
            fairness_score: 0.8,
            data_efficiency: 0.85,
        })
    }

    /// Compute convergence metrics for analysis
    fn compute_convergence_metrics(
        &self,
        aggregated_gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<ConvergenceMetrics, FederatedError> {
        let gradient_norm = self.compute_total_gradient_norm(aggregated_gradients);

        Ok(ConvergenceMetrics {
            gradient_norm,
            parameter_change: gradient_norm * self.config.global_learning_rate,
            loss_improvement: 0.01,
            accuracy_improvement: 0.005,
            convergence_rate: 0.95,
            staleness_impact: 0.02,
        })
    }

    /// Compute the total norm of all gradients
    fn compute_total_gradient_norm(&self, gradients: &HashMap<String, Vec<f32>>) -> f64 {
        let mut norm_squared = 0.0;

        for gradient in gradients.values() {
            for &value in gradient {
                norm_squared += (value as f64).powi(2);
            }
        }

        norm_squared.sqrt()
    }

    /// Estimate communication cost based on gradient size
    fn estimate_communication_cost(&self, gradients: &HashMap<String, Vec<f32>>) -> f64 {
        let mut total_parameters = 0;

        for gradient in gradients.values() {
            total_parameters += gradient.len();
        }

        (total_parameters as f64) * 4.0 // 4 bytes per float32
    }

    /// Estimate computation cost based on participants and epochs
    fn estimate_computation_cost(&self, num_participants: usize) -> f64 {
        (num_participants as f64) * self.config.local_epochs as f64 * 1000.0
    }

    /// Get a copy of the current global model
    ///
    /// # Returns
    ///
    /// Current global model parameters
    pub fn get_global_model(&self) -> HashMap<String, Vec<f32>> {
        self.global_model
            .read()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Get current aggregation metrics
    ///
    /// # Returns
    ///
    /// Current federated learning metrics
    pub fn get_metrics(&self) -> FederatedMetrics {
        self.metrics
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Get the aggregation history
    ///
    /// # Returns
    ///
    /// Vector of recent aggregation rounds
    pub fn get_aggregation_history(&self) -> Vec<AggregationRound> {
        self.aggregation_history
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect()
    }
}

/// Represents a single aggregation round in federated learning
#[derive(Debug, Clone)]
pub struct AggregationRound {
    /// Round number
    pub round_number: u32,
    /// List of participating client IDs
    pub participating_clients: Vec<String>,
    /// Aggregated gradient updates
    pub aggregated_gradients: HashMap<String, Vec<f32>>,
    /// Weights used for aggregation
    pub aggregation_weights: HashMap<String, f64>,
    /// Metrics for this round
    pub round_metrics: RoundMetrics,
    /// Convergence metrics
    pub convergence_metrics: ConvergenceMetrics,
    /// Timestamp of the round
    pub timestamp: Instant,
}

/// Metrics for a single federated learning round
#[derive(Debug, Clone)]
pub struct RoundMetrics {
    /// Number of participating clients
    pub num_participants: usize,
    /// Average local loss across participants
    pub average_local_loss: f64,
    /// Global model loss
    pub global_loss: f64,
    /// Communication cost for this round
    pub communication_cost: f64,
    /// Computation cost for this round
    pub computation_cost: f64,
    /// Privacy cost incurred
    pub privacy_cost: f64,
    /// Fairness score for client selection
    pub fairness_score: f64,
    /// Data efficiency metric
    pub data_efficiency: f64,
}

/// Metrics related to convergence behavior
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    /// Norm of the aggregated gradients
    pub gradient_norm: f64,
    /// Magnitude of parameter changes
    pub parameter_change: f64,
    /// Improvement in loss this round
    pub loss_improvement: f64,
    /// Improvement in accuracy this round
    pub accuracy_improvement: f64,
    /// Convergence rate estimate
    pub convergence_rate: f64,
    /// Impact of staleness on convergence
    pub staleness_impact: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregator_creation() {
        let config = FederatedLearningConfig::default();
        let aggregator = FederatedAggregator::new(config);

        let global_model = aggregator.get_global_model();
        assert!(global_model.is_empty());
    }

    #[test]
    fn test_client_registration() {
        let config = FederatedLearningConfig::default();
        let aggregator = FederatedAggregator::new(config);

        let client = FederatedClient::new("test_client".to_string());
        assert!(aggregator.register_client(client).is_ok());

        let metrics = aggregator.get_metrics();
        assert_eq!(metrics.total_clients, 1);
    }

    #[test]
    fn test_federated_averaging() {
        let config = FederatedLearningConfig::default();
        let aggregator = FederatedAggregator::new(config);

        let mut client_updates = HashMap::new();
        let mut gradients1 = HashMap::new();
        gradients1.insert("param1".to_string(), vec![1.0, 2.0, 3.0]);
        client_updates.insert("client1".to_string(), gradients1);

        let mut clients = HashMap::new();
        let client1 = FederatedClient::new("client1".to_string());
        clients.insert("client1".to_string(), client1);

        let result = aggregator.federated_averaging(&client_updates, &clients);
        assert!(result.is_ok());
    }

    #[test]
    fn test_median_aggregation() {
        let config = FederatedLearningConfig::default();
        let aggregator = FederatedAggregator::new(config);

        let mut client_updates = HashMap::new();

        let mut gradients1 = HashMap::new();
        gradients1.insert("param1".to_string(), vec![1.0, 2.0, 3.0]);
        client_updates.insert("client1".to_string(), gradients1);

        let mut gradients2 = HashMap::new();
        gradients2.insert("param1".to_string(), vec![2.0, 3.0, 4.0]);
        client_updates.insert("client2".to_string(), gradients2);

        let result = aggregator.median_aggregation(&client_updates);
        assert!(result.is_ok());

        let aggregated = result.unwrap();
        assert!(aggregated.contains_key("param1"));
        assert_eq!(aggregated["param1"], vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_gradient_distance() {
        let config = FederatedLearningConfig::default();
        let aggregator = FederatedAggregator::new(config);

        let mut grad1 = HashMap::new();
        grad1.insert("param1".to_string(), vec![1.0, 2.0, 3.0]);

        let mut grad2 = HashMap::new();
        grad2.insert("param1".to_string(), vec![2.0, 3.0, 4.0]);

        let distance = aggregator.compute_gradient_distance(&grad1, &grad2);
        assert!(distance.is_ok());
        assert!((distance.unwrap() - 1.732).abs() < 0.01); // sqrt(3) ≈ 1.732
    }
}

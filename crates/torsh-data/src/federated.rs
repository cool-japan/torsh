//! Federated dataset support for distributed training
//!
//! This module provides data loading mechanisms for federated learning scenarios
//! where data is distributed across multiple clients and cannot be centralized.

use crate::dataset::Dataset;
use crate::error::{DataError, Result};
use crate::sampler::{Sampler, SamplerIterator};
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec, vec::Vec};

/// Federated dataset that manages data distribution across clients
pub struct FederatedDataset {
    clients: HashMap<ClientId, Box<dyn Dataset<Item = torsh_tensor::Tensor>>>,
    client_weights: HashMap<ClientId, f64>,
    aggregation_strategy: AggregationStrategy,
    // Placeholder for future privacy-preserving federated learning
    #[allow(dead_code)]
    privacy_budget: Option<f64>,
    client_selection_strategy: ClientSelectionStrategy,
    current_round: usize,
    min_clients_per_round: usize,
    max_clients_per_round: usize,
}

/// Unique identifier for federated learning clients
pub type ClientId = String;

/// Strategies for aggregating data from multiple clients
#[derive(Clone, Debug)]
pub enum AggregationStrategy {
    /// Weighted average based on client data sizes
    WeightedAverage,
    /// Uniform averaging (each client has equal weight)
    Uniform,
    /// Custom weights provided by user
    Custom(HashMap<ClientId, f64>),
    /// Adaptive weighting based on client performance
    Adaptive { learning_rate: f64 },
}

/// Strategies for selecting clients in each federated round
#[derive(Clone, Debug)]
pub enum ClientSelectionStrategy {
    /// Random selection of clients
    Random,
    /// Select all available clients
    All,
    /// Select clients based on data quality/quantity
    QualityBased,
    /// Select clients to maximize diversity
    DiversityBased,
    /// Custom selection function
    Custom,
}

/// Configuration for federated learning setup
#[derive(Clone, Debug)]
pub struct FederatedConfig {
    /// Minimum number of clients required per round
    pub min_clients_per_round: usize,
    /// Maximum number of clients per round
    pub max_clients_per_round: usize,
    /// Privacy budget for differential privacy (optional)
    pub privacy_budget: Option<f64>,
    /// Client selection strategy
    pub client_selection_strategy: ClientSelectionStrategy,
    /// Data aggregation strategy
    pub aggregation_strategy: AggregationStrategy,
    /// Enable secure aggregation
    pub secure_aggregation: bool,
    /// Timeout for client responses (in seconds)
    pub client_timeout: u64,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            min_clients_per_round: 2,
            max_clients_per_round: 100,
            privacy_budget: None,
            client_selection_strategy: ClientSelectionStrategy::Random,
            aggregation_strategy: AggregationStrategy::WeightedAverage,
            secure_aggregation: false,
            client_timeout: 300, // 5 minutes
        }
    }
}

/// Information about a federated learning client
#[derive(Clone, Debug)]
pub struct ClientInfo {
    pub client_id: ClientId,
    pub data_size: usize,
    pub last_seen: std::time::SystemTime,
    pub performance_metrics: HashMap<String, f64>,
    pub is_available: bool,
    pub capabilities: ClientCapabilities,
}

/// Client capabilities and constraints
#[derive(Clone, Debug)]
pub struct ClientCapabilities {
    pub compute_power: ComputePower,
    pub memory_gb: f64,
    pub network_bandwidth: NetworkBandwidth,
    pub supports_gpu: bool,
    pub max_batch_size: usize,
}

#[derive(Clone, Debug)]
pub enum ComputePower {
    Low,
    Medium,
    High,
    Custom(f64), // FLOPS or relative score
}

#[derive(Clone, Debug)]
pub enum NetworkBandwidth {
    Low,         // < 1 Mbps
    Medium,      // 1-10 Mbps
    High,        // > 10 Mbps
    Custom(f64), // Mbps
}

impl FederatedDataset {
    /// Create a new federated dataset
    pub fn new(config: FederatedConfig) -> Self {
        Self {
            clients: HashMap::new(),
            client_weights: HashMap::new(),
            aggregation_strategy: config.aggregation_strategy,
            privacy_budget: config.privacy_budget,
            client_selection_strategy: config.client_selection_strategy,
            current_round: 0,
            min_clients_per_round: config.min_clients_per_round,
            max_clients_per_round: config.max_clients_per_round,
        }
    }

    /// Add a client to the federated dataset
    pub fn add_client(
        &mut self,
        client_id: ClientId,
        dataset: Box<dyn Dataset<Item = torsh_tensor::Tensor>>,
        weight: Option<f64>,
    ) -> Result<()> {
        if self.clients.contains_key(&client_id) {
            return Err(DataError::config(
                crate::error::ConfigErrorKind::ConflictingValues,
                "client_id",
                &client_id,
                "Client ID already exists in federated dataset",
            ));
        }

        let client_weight = weight.unwrap_or_else(|| match &self.aggregation_strategy {
            AggregationStrategy::WeightedAverage => dataset.len() as f64,
            AggregationStrategy::Uniform => 1.0,
            AggregationStrategy::Custom(weights) => weights.get(&client_id).copied().unwrap_or(1.0),
            AggregationStrategy::Adaptive { .. } => 1.0,
        });

        self.clients.insert(client_id.clone(), dataset);
        self.client_weights.insert(client_id, client_weight);

        Ok(())
    }

    /// Remove a client from the federated dataset
    pub fn remove_client(&mut self, client_id: &ClientId) -> Result<()> {
        if !self.clients.contains_key(client_id) {
            return Err(DataError::config(
                crate::error::ConfigErrorKind::InvalidValue,
                "client_id",
                client_id,
                "Client ID not found in federated dataset",
            ));
        }

        self.clients.remove(client_id);
        self.client_weights.remove(client_id);

        Ok(())
    }

    /// Select clients for the current round
    pub fn select_clients_for_round(&mut self) -> Result<Vec<ClientId>> {
        let available_clients: Vec<ClientId> = self.clients.keys().cloned().collect();

        if available_clients.len() < self.min_clients_per_round {
            return Err(DataError::config(
                crate::error::ConfigErrorKind::InvalidValue,
                "min_clients_per_round",
                &self.min_clients_per_round.to_string(),
                format!(
                    "Not enough clients available: {} < {}",
                    available_clients.len(),
                    self.min_clients_per_round
                ),
            ));
        }

        let selected_clients = match &self.client_selection_strategy {
            ClientSelectionStrategy::Random => self.select_random_clients(&available_clients),
            ClientSelectionStrategy::All => available_clients,
            ClientSelectionStrategy::QualityBased => {
                self.select_quality_based_clients(&available_clients)
            }
            ClientSelectionStrategy::DiversityBased => {
                self.select_diversity_based_clients(&available_clients)
            }
            ClientSelectionStrategy::Custom => {
                // Default to random for now
                self.select_random_clients(&available_clients)
            }
        };

        Ok(selected_clients)
    }

    /// Select random clients for the round
    fn select_random_clients(&self, available_clients: &[ClientId]) -> Vec<ClientId> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::Random;

        let mut rng = Random::seed(42);
        let num_clients = self.max_clients_per_round.min(available_clients.len());

        // Use slice shuffle method instead
        let mut selected_clients = available_clients.to_vec();
        use scirs2_core::rand_prelude::SliceRandom;
        selected_clients.shuffle(&mut rng);
        selected_clients.truncate(num_clients);
        selected_clients
    }

    /// Select clients based on data quality/quantity
    fn select_quality_based_clients(&self, available_clients: &[ClientId]) -> Vec<ClientId> {
        let mut client_scores: Vec<(ClientId, f64)> = available_clients
            .iter()
            .map(|client_id| {
                let dataset_size = self
                    .clients
                    .get(client_id)
                    .map(|dataset| dataset.len() as f64)
                    .unwrap_or(0.0);
                let weight = self.client_weights.get(client_id).copied().unwrap_or(1.0);
                let score = dataset_size * weight;
                (client_id.clone(), score)
            })
            .collect();

        // Sort by score (descending)
        client_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let num_clients = self.max_clients_per_round.min(available_clients.len());
        client_scores
            .into_iter()
            .take(num_clients)
            .map(|(client_id, _)| client_id)
            .collect()
    }

    /// Select clients to maximize diversity
    fn select_diversity_based_clients(&self, available_clients: &[ClientId]) -> Vec<ClientId> {
        // Simplified diversity selection based on data size variation
        let mut selected = Vec::new();
        let mut remaining: Vec<_> = available_clients.iter().cloned().collect();

        // Start with the client with largest dataset
        if let Some(largest_client) = remaining
            .iter()
            .max_by_key(|client_id| {
                self.clients
                    .get(*client_id)
                    .map(|dataset| dataset.len())
                    .unwrap_or(0)
            })
            .cloned()
        {
            selected.push(largest_client.clone());
            remaining.retain(|id| id != &largest_client);
        }

        // Add clients that maximize diversity
        while selected.len() < self.max_clients_per_round && !remaining.is_empty() {
            if let Some(next_client) = self.find_most_diverse_client(&selected, &remaining) {
                selected.push(next_client.clone());
                remaining.retain(|id| id != &next_client);
            } else {
                break;
            }
        }

        selected
    }

    /// Find the client that maximizes diversity with already selected clients
    fn find_most_diverse_client(
        &self,
        selected: &[ClientId],
        remaining: &[ClientId],
    ) -> Option<ClientId> {
        let selected_sizes: Vec<usize> = selected
            .iter()
            .map(|client_id| {
                self.clients
                    .get(client_id)
                    .map(|dataset| dataset.len())
                    .unwrap_or(0)
            })
            .collect();

        remaining
            .iter()
            .max_by_key(|client_id| {
                let client_size = self
                    .clients
                    .get(*client_id)
                    .map(|dataset| dataset.len())
                    .unwrap_or(0);

                // Calculate diversity score as minimum distance to selected clients
                selected_sizes
                    .iter()
                    .map(|&size| (client_size as i64 - size as i64).abs())
                    .min()
                    .unwrap_or(client_size as i64)
            })
            .cloned()
    }

    /// Update client weights based on performance
    pub fn update_client_weights(
        &mut self,
        client_performances: &HashMap<ClientId, f64>,
    ) -> Result<()> {
        if let AggregationStrategy::Adaptive { learning_rate } = &self.aggregation_strategy {
            for (client_id, performance) in client_performances {
                if let Some(current_weight) = self.client_weights.get_mut(client_id) {
                    // Update weight based on performance (higher is better)
                    let adjustment = learning_rate * (performance - 0.5); // Assuming performance is in [0, 1]
                    *current_weight = (*current_weight * (1.0 + adjustment)).max(0.1);
                }
            }
        }

        Ok(())
    }

    /// Get the current round number
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Advance to the next round
    pub fn next_round(&mut self) {
        self.current_round += 1;
    }

    /// Get the number of clients
    pub fn num_clients(&self) -> usize {
        self.clients.len()
    }

    /// Get client information
    pub fn get_client_info(&self, client_id: &ClientId) -> Option<ClientInfo> {
        self.clients.get(client_id).map(|dataset| ClientInfo {
            client_id: client_id.clone(),
            data_size: dataset.len(),
            last_seen: std::time::SystemTime::now(),
            performance_metrics: HashMap::new(),
            is_available: true,
            capabilities: ClientCapabilities {
                compute_power: ComputePower::Medium,
                memory_gb: 4.0,
                network_bandwidth: NetworkBandwidth::Medium,
                supports_gpu: false,
                max_batch_size: 32,
            },
        })
    }

    /// Get all client information
    pub fn get_all_client_info(&self) -> Vec<ClientInfo> {
        self.clients
            .keys()
            .filter_map(|client_id| self.get_client_info(client_id))
            .collect()
    }
}

impl Dataset for FederatedDataset {
    type Item = torsh_tensor::Tensor;

    fn len(&self) -> usize {
        self.clients.values().map(|dataset| dataset.len()).sum()
    }

    fn get(&self, _index: usize) -> std::result::Result<Self::Item, torsh_core::TorshError> {
        // For federated datasets, direct indexing doesn't make sense
        // Users should work with individual client datasets
        Err(torsh_core::TorshError::Other(
            "Direct indexing not supported for federated datasets. Use client-specific access."
                .to_string(),
        ))
    }
}

/// Federated sampler that coordinates sampling across multiple clients
pub struct FederatedSampler {
    client_samplers: HashMap<ClientId, Box<dyn Sampler<Iter = crate::sampler::SamplerIterator>>>,
    // Placeholder for future client selection logic
    #[allow(dead_code)]
    selection_strategy: ClientSelectionStrategy,
    aggregation_weights: HashMap<ClientId, f64>,
    current_round: usize,
    samples_per_client: HashMap<ClientId, usize>,
}

impl FederatedSampler {
    /// Create a new federated sampler
    pub fn new(selection_strategy: ClientSelectionStrategy) -> Self {
        Self {
            client_samplers: HashMap::new(),
            selection_strategy,
            aggregation_weights: HashMap::new(),
            current_round: 0,
            samples_per_client: HashMap::new(),
        }
    }

    /// Add a sampler for a specific client
    pub fn add_client_sampler(
        &mut self,
        client_id: ClientId,
        sampler: Box<dyn Sampler<Iter = crate::sampler::SamplerIterator>>,
        weight: f64,
    ) {
        self.client_samplers.insert(client_id.clone(), sampler);
        self.aggregation_weights.insert(client_id.clone(), weight);
        self.samples_per_client.insert(client_id, 0);
    }

    /// Get samples for a specific client
    pub fn get_client_samples(&mut self, client_id: &ClientId) -> Option<Vec<usize>> {
        if let Some(sampler) = self.client_samplers.get(client_id) {
            let samples: Vec<usize> = sampler.iter().collect();
            self.samples_per_client
                .insert(client_id.clone(), samples.len());
            Some(samples)
        } else {
            None
        }
    }

    /// Get the number of samples for each client
    pub fn get_samples_per_client(&self) -> &HashMap<ClientId, usize> {
        &self.samples_per_client
    }

    /// Advance to the next round
    pub fn next_round(&mut self) {
        self.current_round += 1;
        self.samples_per_client.clear();
    }
}

impl Sampler for FederatedSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        // Return combined samples from all clients
        let mut all_indices = Vec::new();
        let mut offset = 0;

        for (_client_id, sampler) in &self.client_samplers {
            let client_samples: Vec<usize> = sampler.iter().map(|idx| idx + offset).collect();

            all_indices.extend(client_samples);

            // Update offset for next client
            offset += sampler.len();
        }

        SamplerIterator::new(all_indices)
    }

    fn len(&self) -> usize {
        self.client_samplers
            .values()
            .map(|sampler| sampler.len())
            .sum()
    }
}

/// Builder for federated dataset configuration
pub struct FederatedDatasetBuilder {
    config: FederatedConfig,
    clients: Vec<(
        ClientId,
        Box<dyn Dataset<Item = torsh_tensor::Tensor>>,
        Option<f64>,
    )>,
}

impl FederatedDatasetBuilder {
    /// Create a new federated dataset builder
    pub fn new() -> Self {
        Self {
            config: FederatedConfig::default(),
            clients: Vec::new(),
        }
    }

    /// Set the minimum number of clients per round
    pub fn min_clients_per_round(mut self, min_clients: usize) -> Self {
        self.config.min_clients_per_round = min_clients;
        self
    }

    /// Set the maximum number of clients per round
    pub fn max_clients_per_round(mut self, max_clients: usize) -> Self {
        self.config.max_clients_per_round = max_clients;
        self
    }

    /// Set the client selection strategy
    pub fn client_selection_strategy(mut self, strategy: ClientSelectionStrategy) -> Self {
        self.config.client_selection_strategy = strategy;
        self
    }

    /// Set the aggregation strategy
    pub fn aggregation_strategy(mut self, strategy: AggregationStrategy) -> Self {
        self.config.aggregation_strategy = strategy;
        self
    }

    /// Enable privacy budget
    pub fn privacy_budget(mut self, budget: f64) -> Self {
        self.config.privacy_budget = Some(budget);
        self
    }

    /// Enable secure aggregation
    pub fn secure_aggregation(mut self, enabled: bool) -> Self {
        self.config.secure_aggregation = enabled;
        self
    }

    /// Set client timeout
    pub fn client_timeout(mut self, timeout_seconds: u64) -> Self {
        self.config.client_timeout = timeout_seconds;
        self
    }

    /// Add a client dataset
    pub fn add_client(
        mut self,
        client_id: impl Into<String>,
        dataset: Box<dyn Dataset<Item = torsh_tensor::Tensor>>,
        weight: Option<f64>,
    ) -> Self {
        self.clients.push((client_id.into(), dataset, weight));
        self
    }

    /// Build the federated dataset
    pub fn build(self) -> Result<FederatedDataset> {
        let mut federated_dataset = FederatedDataset::new(self.config);

        for (client_id, dataset, weight) in self.clients {
            federated_dataset.add_client(client_id, dataset, weight)?;
        }

        Ok(federated_dataset)
    }
}

impl Default for FederatedDatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for federated learning
pub mod federated_utils {
    use super::*;

    /// Calculate client contribution weights based on data size
    pub fn calculate_data_size_weights(
        client_datasets: &HashMap<ClientId, &dyn Dataset<Item = torsh_tensor::Tensor>>,
    ) -> HashMap<ClientId, f64> {
        let total_size: usize = client_datasets.values().map(|dataset| dataset.len()).sum();

        client_datasets
            .iter()
            .map(|(client_id, dataset)| {
                let weight = if total_size > 0 {
                    dataset.len() as f64 / total_size as f64
                } else {
                    1.0 / client_datasets.len() as f64
                };
                (client_id.clone(), weight)
            })
            .collect()
    }

    /// Simulate client availability based on probability
    pub fn simulate_client_availability(
        client_ids: &[ClientId],
        availability_prob: f64,
    ) -> Vec<ClientId> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::{Random, Rng};

        let mut rng = Random::seed(42);
        client_ids
            .iter()
            .filter(|_| rng.random::<f64>() < availability_prob)
            .cloned()
            .collect()
    }

    /// Calculate diversity score for a set of clients
    pub fn calculate_diversity_score(
        client_datasets: &HashMap<ClientId, &dyn Dataset<Item = torsh_tensor::Tensor>>,
    ) -> f64 {
        if client_datasets.is_empty() {
            return 0.0;
        }

        let sizes: Vec<f64> = client_datasets
            .values()
            .map(|dataset| dataset.len() as f64)
            .collect();

        let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let variance =
            sizes.iter().map(|size| (size - mean).powi(2)).sum::<f64>() / sizes.len() as f64;

        variance.sqrt() / mean.max(1.0) // Coefficient of variation
    }
}

/// Wrapper dataset that adapts TensorDataset to return a single tensor
/// This is used for federated learning where we expect single tensors
#[allow(dead_code)]
struct SingleTensorDataset<T: torsh_core::dtype::TensorElement> {
    inner: crate::dataset::TensorDataset<T>,
}

#[allow(dead_code)]
impl<T: torsh_core::dtype::TensorElement> SingleTensorDataset<T> {
    fn new(inner: crate::dataset::TensorDataset<T>) -> Self {
        Self { inner }
    }
}

impl<T: torsh_core::dtype::TensorElement> Dataset for SingleTensorDataset<T> {
    type Item = torsh_tensor::Tensor<T>;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Item, torsh_core::TorshError> {
        let items = self.inner.get(index).map_err(|e| {
            torsh_core::TorshError::Other(format!("Failed to get item from dataset: {}", e))
        })?;
        if items.is_empty() {
            return Err(torsh_core::TorshError::Other(
                "Dataset contains no tensors".to_string(),
            ));
        }
        // Return the first tensor (assuming single tensor datasets for federated learning)
        Ok(items
            .into_iter()
            .next()
            .expect("items is not empty as checked above"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TensorDataset;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_federated_config() {
        let config = FederatedConfig::default();
        assert_eq!(config.min_clients_per_round, 2);
        assert_eq!(config.max_clients_per_round, 100);
        assert!(!config.secure_aggregation);
    }

    #[test]
    fn test_federated_dataset_builder() -> Result<()> {
        let data1 = ones::<f32>(&[10, 5])?;
        let data2 = ones::<f32>(&[15, 5])?;

        let dataset1 = Box::new(SingleTensorDataset::new(TensorDataset::from_tensor(data1)));
        let dataset2 = Box::new(SingleTensorDataset::new(TensorDataset::from_tensor(data2)));

        let federated_dataset = FederatedDatasetBuilder::new()
            .min_clients_per_round(1)
            .max_clients_per_round(2)
            .add_client("client1", dataset1, Some(1.0))
            .add_client("client2", dataset2, Some(1.5))
            .build()?;

        assert_eq!(federated_dataset.num_clients(), 2);
        assert_eq!(federated_dataset.len(), 25); // 10 + 15

        Ok(())
    }

    #[test]
    fn test_client_selection() -> Result<()> {
        let data1 = ones::<f32>(&[10, 5])?;
        let data2 = ones::<f32>(&[15, 5])?;
        let data3 = ones::<f32>(&[20, 5])?;

        let dataset1 = Box::new(SingleTensorDataset::new(TensorDataset::from_tensor(data1)));
        let dataset2 = Box::new(SingleTensorDataset::new(TensorDataset::from_tensor(data2)));
        let dataset3 = Box::new(SingleTensorDataset::new(TensorDataset::from_tensor(data3)));

        let mut federated_dataset = FederatedDatasetBuilder::new()
            .min_clients_per_round(2)
            .max_clients_per_round(2)
            .client_selection_strategy(ClientSelectionStrategy::QualityBased)
            .add_client("client1", dataset1, None)
            .add_client("client2", dataset2, None)
            .add_client("client3", dataset3, None)
            .build()?;

        let selected_clients = federated_dataset.select_clients_for_round()?;
        assert_eq!(selected_clients.len(), 2);

        // Quality-based should select client3 (largest) and client2 (second largest)
        assert!(selected_clients.contains(&"client3".to_string()));
        assert!(selected_clients.contains(&"client2".to_string()));

        Ok(())
    }

    #[test]
    fn test_federated_utils() {
        use federated_utils::*;

        let data1 = ones::<f32>(&[10, 5]).unwrap();
        let data2 = ones::<f32>(&[20, 5]).unwrap();

        let dataset1 = SingleTensorDataset::new(TensorDataset::from_tensor(data1));
        let dataset2 = SingleTensorDataset::new(TensorDataset::from_tensor(data2));

        let mut client_datasets = HashMap::new();
        client_datasets.insert(
            "client1".to_string(),
            &dataset1 as &dyn Dataset<Item = torsh_tensor::Tensor>,
        );
        client_datasets.insert(
            "client2".to_string(),
            &dataset2 as &dyn Dataset<Item = torsh_tensor::Tensor>,
        );

        let weights = calculate_data_size_weights(&client_datasets);

        // client1 should have 1/3 weight, client2 should have 2/3 weight
        assert!((weights["client1"] - 1.0 / 3.0).abs() < 1e-6);
        assert!((weights["client2"] - 2.0 / 3.0).abs() < 1e-6);

        let diversity = calculate_diversity_score(&client_datasets);
        assert!(diversity > 0.0); // Should have some diversity due to different sizes
    }
}

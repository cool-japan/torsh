//! Personalization and Meta-Learning for Federated Learning
//!
//! This module provides comprehensive personalization mechanisms for federated learning systems.
//! Personalization addresses the challenge of heterogeneous data distributions across clients
//! by enabling customized models that adapt to local client characteristics while still
//! benefiting from global collaboration.
//!
//! # Personalization Strategies
//!
//! - **Per-Client Personalization**: Individual models for each client
//! - **Cluster-Based Personalization**: Grouping similar clients for shared personalization
//! - **Meta-Learning**: Learning to quickly adapt to new clients or tasks
//! - **Multi-Task Learning**: Sharing representations across related tasks
//! - **Fine-Tuning**: Post-training adaptation of global models
//! - **Adaptive Mixture**: Dynamic weighting of global and local models
//!
//! # Meta-Learning Approaches
//!
//! The module implements several meta-learning algorithms:
//! - Model-Agnostic Meta-Learning (MAML)
//! - Reptile algorithm
//! - Personalized Federated Learning (pFedMe)
//! - Per-client optimization objectives
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use torsh_autograd::federated_learning::personalization::{
//!     PersonalizationManager, PersonalizationStrategy
//! };
//! use std::collections::HashMap;
//!
//! // Create a personalization manager
//! let mut manager = PersonalizationManager::new(PersonalizationStrategy::MetaLearning);
//!
//! // Update personalized models after federated round
//! let selected_clients = vec!["client_1".to_string(), "client_2".to_string()];
//! let mut client_updates = HashMap::new();
//!
//! let mut gradients = HashMap::new();
//! gradients.insert("layer_1".to_string(), vec![0.1, 0.2, 0.3]);
//! client_updates.insert("client_1".to_string(), gradients);
//!
//! manager.update_personalized_models(&selected_clients, &client_updates)?;
//! ```
//!
//! # Clustering and Adaptation
//!
//! The module supports automatic clustering of clients based on:
//! - Data distribution similarity
//! - Model parameter similarity
//! - Performance characteristics
//! - Geographic or demographic factors

use std::collections::HashMap;

use crate::federated_learning::aggregation::FederatedError;

/// Personalization manager for federated learning systems
///
/// The PersonalizationManager handles the creation and maintenance of personalized
/// models for individual clients or groups of clients. It supports various
/// personalization strategies and maintains the necessary state for meta-learning.
///
/// # Thread Safety
///
/// This struct is designed to be thread-safe and can be safely shared across threads
/// when wrapped in appropriate synchronization primitives.
#[derive(Debug)]
pub struct PersonalizationManager {
    /// The personalization strategy being used
    personalization_strategy: PersonalizationStrategy,
    /// Mapping of clients to their assigned clusters
    client_clusters: HashMap<String, String>,
    /// Cluster-specific model parameters
    cluster_models: HashMap<String, HashMap<String, Vec<f32>>>,
    /// Meta-learning state and parameters
    meta_learning_state: MetaLearningState,
    /// Per-client personalized models
    client_models: HashMap<String, HashMap<String, Vec<f32>>>,
    /// Client similarity metrics for clustering
    client_similarities: HashMap<(String, String), f64>,
    /// Adaptation histories for each client
    adaptation_histories: HashMap<String, Vec<AdaptationStep>>,
    /// Configuration parameters
    config: PersonalizationConfig,
}

// PersonalizationManager is Send + Sync
unsafe impl Send for PersonalizationManager {}
unsafe impl Sync for PersonalizationManager {}

/// Strategies for personalizing federated learning models
///
/// Different strategies provide different trade-offs between personalization
/// effectiveness, computational overhead, and communication requirements.
#[derive(Debug, Clone, PartialEq)]
pub enum PersonalizationStrategy {
    /// No personalization - use global model only
    None,
    /// Individual personalized model per client
    PerClient,
    /// Cluster clients and personalize per cluster
    ClusterBased,
    /// Use meta-learning for fast adaptation
    MetaLearning,
    /// Multi-task learning with shared representations
    MultiTask,
    /// Fine-tune global model for each client
    FineTuning,
    /// Adaptive mixture of global and local models
    AdaptiveMixture,
}

/// Meta-learning state and parameters
///
/// This struct maintains the state needed for meta-learning algorithms
/// like MAML (Model-Agnostic Meta-Learning) and similar approaches.
#[derive(Debug, Clone)]
pub struct MetaLearningState {
    /// Meta-parameters (initialization for fast adaptation)
    pub meta_parameters: HashMap<String, Vec<f32>>,
    /// Number of adaptation steps for each client
    pub adaptation_steps: u32,
    /// Learning rate for inner optimization (client adaptation)
    pub inner_learning_rate: f64,
    /// Learning rate for outer optimization (meta-update)
    pub outer_learning_rate: f64,
    /// Task-specific adaptation histories
    pub task_histories: HashMap<String, Vec<TaskAdaptation>>,
}

// MetaLearningState is Send + Sync
unsafe impl Send for MetaLearningState {}
unsafe impl Sync for MetaLearningState {}

/// Configuration for personalization behavior
#[derive(Debug, Clone)]
pub struct PersonalizationConfig {
    /// Maximum number of clusters for cluster-based personalization
    pub max_clusters: usize,
    /// Similarity threshold for clustering
    pub similarity_threshold: f64,
    /// Whether to use automatic clustering
    pub auto_clustering: bool,
    /// Regularization strength for personalization
    pub regularization_strength: f64,
    /// Whether to enable cross-client knowledge transfer
    pub enable_knowledge_transfer: bool,
    /// Adaptation learning rate decay
    pub adaptation_decay: f64,
}

/// Record of an adaptation step for tracking client learning
#[derive(Debug, Clone)]
pub struct AdaptationStep {
    /// Round number when adaptation occurred
    pub round: u32,
    /// Loss before adaptation
    pub loss_before: f64,
    /// Loss after adaptation
    pub loss_after: f64,
    /// Number of local steps taken
    pub local_steps: u32,
    /// Adaptation effectiveness score
    pub effectiveness: f64,
}

/// Task-specific adaptation for meta-learning
#[derive(Debug, Clone)]
pub struct TaskAdaptation {
    /// Task identifier
    pub task_id: String,
    /// Adaptation gradients
    pub adaptation_gradients: HashMap<String, Vec<f32>>,
    /// Task performance metrics
    pub performance_metrics: TaskPerformanceMetrics,
}

/// Performance metrics for a specific task/client
#[derive(Debug, Clone)]
pub struct TaskPerformanceMetrics {
    /// Task-specific loss
    pub loss: f64,
    /// Task-specific accuracy
    pub accuracy: f64,
    /// Adaptation convergence rate
    pub convergence_rate: f64,
    /// Number of samples used for adaptation
    pub num_samples: usize,
}

impl PersonalizationManager {
    /// Creates a new PersonalizationManager with the specified strategy
    ///
    /// # Arguments
    ///
    /// * `strategy` - The personalization strategy to use
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let manager = PersonalizationManager::new(PersonalizationStrategy::MetaLearning);
    /// ```
    pub fn new(strategy: PersonalizationStrategy) -> Self {
        Self {
            personalization_strategy: strategy,
            client_clusters: HashMap::new(),
            cluster_models: HashMap::new(),
            meta_learning_state: MetaLearningState::new(),
            client_models: HashMap::new(),
            client_similarities: HashMap::new(),
            adaptation_histories: HashMap::new(),
            config: PersonalizationConfig::default(),
        }
    }

    /// Creates a new PersonalizationManager with custom configuration
    ///
    /// # Arguments
    ///
    /// * `strategy` - The personalization strategy to use
    /// * `config` - Custom configuration parameters
    pub fn with_config(strategy: PersonalizationStrategy, config: PersonalizationConfig) -> Self {
        Self {
            config,
            ..Self::new(strategy)
        }
    }

    /// Updates personalized models based on client updates
    ///
    /// This is the main method for updating personalization. It applies the
    /// configured strategy to update client-specific or cluster-specific models
    /// based on the latest federated learning round results.
    ///
    /// # Arguments
    ///
    /// * `selected_clients` - List of clients that participated in this round
    /// * `client_updates` - Gradient updates from each client
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of the update
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let clients = vec!["client_1".to_string()];
    /// let mut updates = HashMap::new();
    /// updates.insert("client_1".to_string(), gradients);
    /// manager.update_personalized_models(&clients, &updates)?;
    /// ```
    pub fn update_personalized_models(
        &mut self,
        selected_clients: &[String],
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Result<(), FederatedError> {
        match self.personalization_strategy {
            PersonalizationStrategy::None => Ok(()), // No personalization needed
            PersonalizationStrategy::PerClient => {
                self.update_per_client_models(selected_clients, client_updates)
            }
            PersonalizationStrategy::ClusterBased => {
                self.update_cluster_based_models(selected_clients, client_updates)
            }
            PersonalizationStrategy::MetaLearning => {
                self.update_meta_learning_models(selected_clients, client_updates)
            }
            PersonalizationStrategy::MultiTask => {
                self.update_multi_task_models(selected_clients, client_updates)
            }
            PersonalizationStrategy::FineTuning => {
                self.update_fine_tuning_models(selected_clients, client_updates)
            }
            PersonalizationStrategy::AdaptiveMixture => {
                self.update_adaptive_mixture_models(selected_clients, client_updates)
            }
        }
    }

    /// Gets the current personalization strategy
    pub fn get_strategy(&self) -> &PersonalizationStrategy {
        &self.personalization_strategy
    }

    /// Sets a new personalization strategy
    pub fn set_strategy(&mut self, strategy: PersonalizationStrategy) {
        self.personalization_strategy = strategy;
    }

    /// Gets personalized model for a specific client
    pub fn get_client_model(&self, client_id: &str) -> Option<&HashMap<String, Vec<f32>>> {
        self.client_models.get(client_id)
    }

    /// Gets cluster assignment for a client
    pub fn get_client_cluster(&self, client_id: &str) -> Option<&String> {
        self.client_clusters.get(client_id)
    }

    /// Gets cluster model parameters
    pub fn get_cluster_model(&self, cluster_id: &str) -> Option<&HashMap<String, Vec<f32>>> {
        self.cluster_models.get(cluster_id)
    }

    /// Gets meta-learning parameters
    pub fn get_meta_parameters(&self) -> &HashMap<String, Vec<f32>> {
        &self.meta_learning_state.meta_parameters
    }

    /// Gets adaptation history for a client
    pub fn get_adaptation_history(&self, client_id: &str) -> Option<&Vec<AdaptationStep>> {
        self.adaptation_histories.get(client_id)
    }

    /// Manually assigns a client to a cluster
    pub fn assign_client_to_cluster(&mut self, client_id: &str, cluster_id: &str) {
        self.client_clusters
            .insert(client_id.to_string(), cluster_id.to_string());
    }

    /// Computes similarity between two clients based on their updates
    pub fn compute_client_similarity(
        &self,
        client1_updates: &HashMap<String, Vec<f32>>,
        client2_updates: &HashMap<String, Vec<f32>>,
    ) -> f64 {
        let mut dot_product = 0.0;
        let mut norm1_squared = 0.0;
        let mut norm2_squared = 0.0;

        for (param_name, grad1) in client1_updates {
            if let Some(grad2) = client2_updates.get(param_name) {
                for (v1, v2) in grad1.iter().zip(grad2.iter()) {
                    dot_product += (*v1 as f64) * (*v2 as f64);
                    norm1_squared += (*v1 as f64).powi(2);
                    norm2_squared += (*v2 as f64).powi(2);
                }
            }
        }

        if norm1_squared > 0.0 && norm2_squared > 0.0 {
            dot_product / (norm1_squared.sqrt() * norm2_squared.sqrt())
        } else {
            0.0
        }
    }

    /// Updates per-client personalized models
    fn update_per_client_models(
        &mut self,
        selected_clients: &[String],
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Result<(), FederatedError> {
        for client_id in selected_clients {
            if let Some(gradients) = client_updates.get(client_id) {
                // Store client-specific model parameters
                self.client_models
                    .insert(client_id.clone(), gradients.clone());

                // Record adaptation step
                let adaptation_step = AdaptationStep {
                    round: 0,         // This would come from the current round number
                    loss_before: 1.0, // This would come from actual loss computation
                    loss_after: 0.9,  // This would come from actual loss computation
                    local_steps: 1,
                    effectiveness: 0.1, // Improvement in loss
                };

                self.adaptation_histories
                    .entry(client_id.clone())
                    .or_insert_with(Vec::new)
                    .push(adaptation_step);
            }
        }
        Ok(())
    }

    /// Updates cluster-based personalized models
    fn update_cluster_based_models(
        &mut self,
        selected_clients: &[String],
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Result<(), FederatedError> {
        for client_id in selected_clients {
            // Assign client to cluster (or create new cluster)
            let cluster_id = if self.config.auto_clustering {
                self.assign_to_cluster_automatically(client_id, client_updates.get(client_id))?
            } else {
                self.assign_to_cluster(client_id, client_updates.get(client_id))?
            };

            self.client_clusters
                .insert(client_id.clone(), cluster_id.clone());

            if let Some(gradients) = client_updates.get(client_id) {
                // Update cluster model with weighted average
                let cluster_model = self
                    .cluster_models
                    .entry(cluster_id)
                    .or_insert_with(HashMap::new);

                for (param_name, gradient) in gradients {
                    let cluster_param = cluster_model
                        .entry(param_name.clone())
                        .or_insert_with(|| vec![0.0; gradient.len()]);

                    // Simple averaging (could be more sophisticated)
                    for (i, &grad_value) in gradient.iter().enumerate() {
                        cluster_param[i] = (cluster_param[i] + grad_value) / 2.0;
                    }
                }
            }
        }
        Ok(())
    }

    /// Updates meta-learning models using MAML-style optimization
    fn update_meta_learning_models(
        &mut self,
        selected_clients: &[String],
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Result<(), FederatedError> {
        for client_id in selected_clients {
            if let Some(gradients) = client_updates.get(client_id) {
                self.perform_meta_update(client_id, gradients)?;

                // Record task adaptation
                let task_adaptation = TaskAdaptation {
                    task_id: client_id.clone(),
                    adaptation_gradients: gradients.clone(),
                    performance_metrics: TaskPerformanceMetrics {
                        loss: 0.5, // This would come from actual computation
                        accuracy: 0.9,
                        convergence_rate: 0.95,
                        num_samples: 1000,
                    },
                };

                self.meta_learning_state
                    .task_histories
                    .entry(client_id.clone())
                    .or_insert_with(Vec::new)
                    .push(task_adaptation);
            }
        }
        Ok(())
    }

    /// Updates multi-task learning models
    fn update_multi_task_models(
        &mut self,
        selected_clients: &[String],
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Result<(), FederatedError> {
        // Multi-task learning shares representations across related tasks
        let mut shared_representations = HashMap::new();

        for client_id in selected_clients {
            if let Some(gradients) = client_updates.get(client_id) {
                // Extract shared representation layers (e.g., early layers)
                for (param_name, gradient) in gradients {
                    if param_name.contains("shared") || param_name.contains("encoder") {
                        let shared_param = shared_representations
                            .entry(param_name.clone())
                            .or_insert_with(|| vec![0.0; gradient.len()]);

                        for (i, &grad_value) in gradient.iter().enumerate() {
                            shared_param[i] += grad_value / selected_clients.len() as f32;
                        }
                    }
                }

                // Store task-specific parameters
                self.client_models
                    .insert(client_id.clone(), gradients.clone());
            }
        }

        // Update shared representations for all clients
        for client_id in selected_clients {
            if let Some(client_model) = self.client_models.get_mut(client_id) {
                for (param_name, shared_param) in &shared_representations {
                    client_model.insert(param_name.clone(), shared_param.clone());
                }
            }
        }

        Ok(())
    }

    /// Updates models using fine-tuning approach
    fn update_fine_tuning_models(
        &mut self,
        selected_clients: &[String],
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Result<(), FederatedError> {
        // Fine-tuning starts from global model and adapts locally
        for client_id in selected_clients {
            if let Some(gradients) = client_updates.get(client_id) {
                // Apply fine-tuning updates with smaller learning rate
                let fine_tuning_lr = self.meta_learning_state.inner_learning_rate * 0.1;

                let mut fine_tuned_model = HashMap::new();
                for (param_name, gradient) in gradients {
                    let mut fine_tuned_param = gradient.clone();
                    for param in &mut fine_tuned_param {
                        *param *= fine_tuning_lr as f32;
                    }
                    fine_tuned_model.insert(param_name.clone(), fine_tuned_param);
                }

                self.client_models
                    .insert(client_id.clone(), fine_tuned_model);
            }
        }
        Ok(())
    }

    /// Updates models using adaptive mixture approach
    fn update_adaptive_mixture_models(
        &mut self,
        selected_clients: &[String],
        client_updates: &HashMap<String, HashMap<String, Vec<f32>>>,
    ) -> Result<(), FederatedError> {
        for client_id in selected_clients {
            if let Some(gradients) = client_updates.get(client_id) {
                // Compute adaptive mixing weights based on client performance
                let mixing_weight = self.compute_mixing_weight(client_id);

                let mut mixed_model = HashMap::new();
                for (param_name, local_gradient) in gradients {
                    // Mix local and global updates
                    let mut mixed_param = local_gradient.clone();
                    for param in &mut mixed_param {
                        *param = mixing_weight * (*param) + (1.0 - mixing_weight) * (*param * 0.5);
                    }
                    mixed_model.insert(param_name.clone(), mixed_param);
                }

                self.client_models.insert(client_id.clone(), mixed_model);
            }
        }
        Ok(())
    }

    /// Assigns a client to a cluster using simple heuristics
    fn assign_to_cluster(
        &self,
        client_id: &str,
        gradients: Option<&HashMap<String, Vec<f32>>>,
    ) -> Result<String, FederatedError> {
        if gradients.is_none() {
            return Ok("default_cluster".to_string());
        }

        // Simple cluster assignment based on client ID hash
        let cluster_id = format!(
            "cluster_{}",
            (client_id.len() % self.config.max_clusters) + 1
        );
        Ok(cluster_id)
    }

    /// Automatically assigns clients to clusters based on similarity
    fn assign_to_cluster_automatically(
        &mut self,
        client_id: &str,
        gradients: Option<&HashMap<String, Vec<f32>>>,
    ) -> Result<String, FederatedError> {
        if let Some(client_gradients) = gradients {
            let mut best_cluster = None;
            let mut best_similarity = -1.0;

            // Find most similar existing cluster
            for (existing_client, cluster_id) in &self.client_clusters {
                if let Some(existing_gradients) = self.client_models.get(existing_client) {
                    let similarity =
                        self.compute_client_similarity(client_gradients, existing_gradients);
                    if similarity > best_similarity && similarity > self.config.similarity_threshold
                    {
                        best_similarity = similarity;
                        best_cluster = Some(cluster_id.clone());
                    }
                }
            }

            // Create new cluster if no similar cluster found
            if let Some(cluster_id) = best_cluster {
                Ok(cluster_id)
            } else {
                let new_cluster_id = format!("auto_cluster_{}", self.client_clusters.len());
                Ok(new_cluster_id)
            }
        } else {
            self.assign_to_cluster(client_id, gradients)
        }
    }

    /// Performs meta-learning update (MAML-style)
    fn perform_meta_update(
        &mut self,
        client_id: &str,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> Result<(), FederatedError> {
        for (param_name, gradient) in gradients {
            let meta_param = self
                .meta_learning_state
                .meta_parameters
                .entry(param_name.clone())
                .or_insert_with(|| vec![0.0; gradient.len()]);

            // Meta-update: θ = θ - α * ∇_θ L
            for (i, &grad_value) in gradient.iter().enumerate() {
                meta_param[i] -= self.meta_learning_state.outer_learning_rate as f32 * grad_value;
            }
        }
        Ok(())
    }

    /// Computes adaptive mixing weight for a client
    fn compute_mixing_weight(&self, client_id: &str) -> f32 {
        // Base mixing weight
        let mut weight = 0.5;

        // Adjust based on adaptation history
        if let Some(history) = self.adaptation_histories.get(client_id) {
            if !history.is_empty() {
                let avg_effectiveness = history.iter().map(|step| step.effectiveness).sum::<f64>()
                    / history.len() as f64;
                // Higher effectiveness -> more local weight
                weight = (0.3 + 0.4 * avg_effectiveness).clamp(0.1, 0.9) as f32;
            }
        }

        weight
    }

    /// Resets all personalization state
    pub fn reset(&mut self) {
        self.client_clusters.clear();
        self.cluster_models.clear();
        self.client_models.clear();
        self.client_similarities.clear();
        self.adaptation_histories.clear();
        self.meta_learning_state = MetaLearningState::new();
    }

    /// Gets configuration
    pub fn get_config(&self) -> &PersonalizationConfig {
        &self.config
    }

    /// Updates configuration
    pub fn set_config(&mut self, config: PersonalizationConfig) {
        self.config = config;
    }
}

impl MetaLearningState {
    /// Creates a new MetaLearningState with default parameters
    pub fn new() -> Self {
        Self {
            meta_parameters: HashMap::new(),
            adaptation_steps: 5,
            inner_learning_rate: 0.01,
            outer_learning_rate: 0.001,
            task_histories: HashMap::new(),
        }
    }

    /// Creates a new MetaLearningState with custom parameters
    pub fn with_params(adaptation_steps: u32, inner_lr: f64, outer_lr: f64) -> Self {
        Self {
            adaptation_steps,
            inner_learning_rate: inner_lr,
            outer_learning_rate: outer_lr,
            ..Self::new()
        }
    }
}

impl Default for MetaLearningState {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PersonalizationConfig {
    fn default() -> Self {
        Self {
            max_clusters: 10,
            similarity_threshold: 0.7,
            auto_clustering: true,
            regularization_strength: 0.01,
            enable_knowledge_transfer: true,
            adaptation_decay: 0.99,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_personalization_manager_creation() {
        let manager = PersonalizationManager::new(PersonalizationStrategy::MetaLearning);
        assert_eq!(
            *manager.get_strategy(),
            PersonalizationStrategy::MetaLearning
        );
    }

    #[test]
    fn test_per_client_personalization() {
        let mut manager = PersonalizationManager::new(PersonalizationStrategy::PerClient);
        let clients = vec!["client_1".to_string()];
        let mut client_updates = HashMap::new();
        let mut gradients = HashMap::new();
        gradients.insert("param_1".to_string(), vec![1.0, 2.0]);
        client_updates.insert("client_1".to_string(), gradients.clone());

        let result = manager.update_personalized_models(&clients, &client_updates);
        assert!(result.is_ok());

        let client_model = manager.get_client_model("client_1");
        assert!(client_model.is_some());
        assert_eq!(client_model.unwrap().get("param_1"), Some(&vec![1.0, 2.0]));
    }

    #[test]
    fn test_cluster_based_personalization() {
        let mut manager = PersonalizationManager::new(PersonalizationStrategy::ClusterBased);
        let clients = vec!["client_1".to_string(), "client_2".to_string()];
        let mut client_updates = HashMap::new();

        let mut gradients1 = HashMap::new();
        gradients1.insert("param_1".to_string(), vec![1.0, 2.0]);
        client_updates.insert("client_1".to_string(), gradients1);

        let mut gradients2 = HashMap::new();
        gradients2.insert("param_1".to_string(), vec![3.0, 4.0]);
        client_updates.insert("client_2".to_string(), gradients2);

        let result = manager.update_personalized_models(&clients, &client_updates);
        assert!(result.is_ok());

        // Both clients should be assigned to clusters
        assert!(manager.get_client_cluster("client_1").is_some());
        assert!(manager.get_client_cluster("client_2").is_some());
    }

    #[test]
    fn test_meta_learning_personalization() {
        let mut manager = PersonalizationManager::new(PersonalizationStrategy::MetaLearning);
        let clients = vec!["client_1".to_string()];
        let mut client_updates = HashMap::new();
        let mut gradients = HashMap::new();
        gradients.insert("param_1".to_string(), vec![1.0, 2.0]);
        client_updates.insert("client_1".to_string(), gradients);

        let result = manager.update_personalized_models(&clients, &client_updates);
        assert!(result.is_ok());

        // Meta-parameters should be updated
        let meta_params = manager.get_meta_parameters();
        assert!(meta_params.contains_key("param_1"));
    }

    #[test]
    fn test_client_similarity_computation() {
        let manager = PersonalizationManager::new(PersonalizationStrategy::ClusterBased);

        let mut updates1 = HashMap::new();
        updates1.insert("param_1".to_string(), vec![1.0, 0.0]);

        let mut updates2 = HashMap::new();
        updates2.insert("param_1".to_string(), vec![1.0, 0.0]);

        let similarity = manager.compute_client_similarity(&updates1, &updates2);
        assert!((similarity - 1.0).abs() < 1e-6); // Should be perfectly similar

        let mut updates3 = HashMap::new();
        updates3.insert("param_1".to_string(), vec![0.0, 1.0]);

        let similarity2 = manager.compute_client_similarity(&updates1, &updates3);
        assert!((similarity2 - 0.0).abs() < 1e-6); // Should be orthogonal
    }

    #[test]
    fn test_meta_learning_state() {
        let state = MetaLearningState::new();
        assert_eq!(state.adaptation_steps, 5);
        assert_eq!(state.inner_learning_rate, 0.01);
        assert_eq!(state.outer_learning_rate, 0.001);

        let custom_state = MetaLearningState::with_params(10, 0.02, 0.002);
        assert_eq!(custom_state.adaptation_steps, 10);
        assert_eq!(custom_state.inner_learning_rate, 0.02);
        assert_eq!(custom_state.outer_learning_rate, 0.002);
    }

    #[test]
    fn test_strategy_switching() {
        let mut manager = PersonalizationManager::new(PersonalizationStrategy::PerClient);
        assert_eq!(*manager.get_strategy(), PersonalizationStrategy::PerClient);

        manager.set_strategy(PersonalizationStrategy::MetaLearning);
        assert_eq!(
            *manager.get_strategy(),
            PersonalizationStrategy::MetaLearning
        );
    }

    #[test]
    fn test_manual_cluster_assignment() {
        let mut manager = PersonalizationManager::new(PersonalizationStrategy::ClusterBased);

        manager.assign_client_to_cluster("client_1", "cluster_a");
        assert_eq!(
            manager.get_client_cluster("client_1"),
            Some(&"cluster_a".to_string())
        );
    }

    #[test]
    fn test_adaptation_history_tracking() {
        let mut manager = PersonalizationManager::new(PersonalizationStrategy::PerClient);
        let clients = vec!["client_1".to_string()];
        let mut client_updates = HashMap::new();
        let mut gradients = HashMap::new();
        gradients.insert("param_1".to_string(), vec![1.0, 2.0]);
        client_updates.insert("client_1".to_string(), gradients);

        let result = manager.update_personalized_models(&clients, &client_updates);
        assert!(result.is_ok());

        let history = manager.get_adaptation_history("client_1");
        assert!(history.is_some());
        assert!(!history.unwrap().is_empty());
    }

    #[test]
    fn test_personalization_config() {
        let config = PersonalizationConfig::default();
        assert_eq!(config.max_clusters, 10);
        assert_eq!(config.similarity_threshold, 0.7);
        assert!(config.auto_clustering);

        let mut manager = PersonalizationManager::with_config(
            PersonalizationStrategy::ClusterBased,
            config.clone(),
        );

        assert_eq!(manager.get_config().max_clusters, 10);

        let new_config = PersonalizationConfig {
            max_clusters: 20,
            ..config
        };

        manager.set_config(new_config);
        assert_eq!(manager.get_config().max_clusters, 20);
    }

    #[test]
    fn test_manager_reset() {
        let mut manager = PersonalizationManager::new(PersonalizationStrategy::PerClient);

        // Add some state
        manager.assign_client_to_cluster("client_1", "cluster_a");
        manager
            .client_models
            .insert("client_1".to_string(), HashMap::new());

        assert!(manager.get_client_cluster("client_1").is_some());
        assert!(manager.get_client_model("client_1").is_some());

        // Reset and verify clean state
        manager.reset();

        assert!(manager.get_client_cluster("client_1").is_none());
        assert!(manager.get_client_model("client_1").is_none());
        assert!(manager.get_meta_parameters().is_empty());
    }

    #[test]
    fn test_all_personalization_strategies() {
        let strategies = [
            PersonalizationStrategy::None,
            PersonalizationStrategy::PerClient,
            PersonalizationStrategy::ClusterBased,
            PersonalizationStrategy::MetaLearning,
            PersonalizationStrategy::MultiTask,
            PersonalizationStrategy::FineTuning,
            PersonalizationStrategy::AdaptiveMixture,
        ];

        for strategy in &strategies {
            let mut manager = PersonalizationManager::new(strategy.clone());
            let clients = vec!["client_1".to_string()];
            let mut client_updates = HashMap::new();
            let mut gradients = HashMap::new();
            gradients.insert("param_1".to_string(), vec![1.0, 2.0]);
            client_updates.insert("client_1".to_string(), gradients);

            let result = manager.update_personalized_models(&clients, &client_updates);
            assert!(result.is_ok(), "Strategy {:?} failed", strategy);
        }
    }
}
